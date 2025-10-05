"""
Repro vs Fix: GLUE with BERT-base-uncased on dev split

 pip install -U transformers evaluate

Repro (WRONG on purpose):
- Single seed
- Naive "overall" = plain average of whatever numbers come out
- Ignores per-task metric definitions

Fix (CORRECT):
- Uses official GLUE metrics via `evaluate.load("glue", subset)`
  Docs: https://huggingface.co/spaces/evaluate-metric/glue/blob/main/README.md
- Aggregates metrics per GLUE rules (CoLA=Matthews; STS-B=Pearson/Spearman avg;
  MRPC/QQP=avg(F1,Acc); others=Accuracy)
  Task list/metrics: https://gluebenchmark.com/tasks ; paper: https://openreview.net/pdf?id=rJ4km2R5t7
- Runs multiple seeds and reports mean/median/std
- Uses `Trainer.model_init` so each run reinitializes heads deterministically
  Docs: https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.seed

BERT paper recipe for context: ~3 epochs, bs≈32, LR∈{2e-5,3e-5,4e-5,5e-5} chosen on dev.
Refs: https://arxiv.org/pdf/1810.04805 ; https://aclanthology.org/N19-1423.pdf
"""
import os
import argparse
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed, )
import evaluate

from model import BertConfig


# --------------------
# Config
# --------------------
MODEL_NAME = "bert-large-uncased"  # stable public checkpoint
# Keep the demo fast by default. You can add more tasks later (e.g., "rte","qnli","mnli","stsb","cola","wnli","qqp")
TASKS = ['stsb', "sst2", "mrpc"]  # includes both single-metric and two-metric tasks
SEEDS = [13, 21, 34, 55, 89]  # five seeds as in many HF examples
NUM_EPOCHS = 3
LR = 2e-5
BATCH = 32
MAX_LEN = 128

parser = argparse.ArgumentParser()
args = parser.parse_args()
class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # # ========== wike2 The configuration related to the dataset.
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_uncased")
        self.data_name = 'wiki2'

        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.model_save_path = os.path.join(self.model_save_dir, f'LAbase_model_{self.data_name}_1.bin')
        self.is_sample_shuffle = True
        self.use_embedding_weight = True
        self.batch_size = 32
        self.max_sen_len = None  # When set to None, padding will be applied to the samples in the batch based on the longest sample in that batch.
        self.pad_index = 0
        self.random_state = 2022
        self.learning_rate = 1e-4 #4e-5
        self.weight_decay = 0.01
        self.masked_rate = 0.15
        self.masked_token_rate = 0.8
        self.masked_token_unchanged_rate = 0.5
        self.log_level = 30 #logging.DEBUG
        self.use_torch_multi_head = False  # False means using the multi-head implementation from `model/BasicBert/MyTransformer`.
        self.epochs = 40
        self.model_val_per_epoch = 1
        self.model_val_per_iter = 20000
        bert_config_path = os.path.join('~/BERT/bert-base-uncased', "config_large.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value



model_config = ModelConfig()


# --------------------
# Tokenization per task
# --------------------
def get_task_fields(task: str) -> Tuple[str, str]:
    # GLUE text fields vary per task.
    if task in {"sst2",  "cola"}:
        return "sentence", None
    if task in {"qnli",}:
        return "question", "sentence"
    if task in {"mrpc", "rte", "stsb"}:  #"wnli",
        return "sentence1", "sentence2"
    if task in {"qqp"}:  #"wnli",
        return "question1", "question2"
    if task == "mnli":  # we evaluate on validation_matched by default below
        return "premise", "hypothesis"
    raise ValueError(f"Unsupported task: {task}")


# --------------------
# Official GLUE metric adapters
# --------------------
def glue_task_score(task: str, metric_dict: Dict[str, float]) -> float:
    """
    Map raw metric dict to the single GLUE task score.
    - CoLA: matthews_correlation
    - STS-B: mean of pearson and spearmanr
    - MRPC/QQP: mean of f1 and accuracy
    - Others: accuracy
    """
    # Metric definitions per GLUE: https://gluebenchmark.com/tasks ; paper: https://openreview.net/pdf?id=rJ4km2R5t7
    if task == "cola":
        return metric_dict["eval_matthews_correlation"]
    if task == "stsb":
        return 0.5 * (metric_dict["eval_pearson"] + metric_dict["eval_spearmanr"])
    if task in {"mrpc", "qqp"}:
        return 0.5 * (metric_dict["eval_f1"] + metric_dict["eval_accuracy"])
    # default accuracy tasks
    return metric_dict["eval_accuracy"]

# --------------------
# Data + preprocess
# --------------------
def load_task(task: str):
    ds = load_dataset("glue", task)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    t1, t2 = get_task_fields(task)

    def preprocess(batch):
        if t2 is None:
            return t
            ok(batch[t1], truncation=True, max_length=MAX_LEN)
        return tok(batch[t1], batch[t2], truncation=True, max_length=MAX_LEN)

    # For MNLI explicitly use matched dev set for comparability
    valid_split = "validation_matched" if task == "mnli" else "validation"  #validation_mismatched
    enc = ds.map(preprocess, batched=True)
    return enc["train"], enc[valid_split], tok, ds["train"].features["label"].num_classes if task != "stsb" else 1


# --------------------
# Metric computation hook
# --------------------
def make_compute_metrics(task: str):
    metr = evaluate.load("glue", task)  # https://huggingface.co/spaces/evaluate-metric/glue
    def _fn(eval_pred):
        logits, labels = eval_pred
        if task == "stsb":
            preds = np.squeeze(logits)  # regression
        else:
            preds = np.argmax(logits, axis=1)
        raw = metr.compute(predictions=preds, references=labels)
        return raw
    return _fn


# --------------------
# Model factory to ensure fresh, seeded heads each run
# --------------------
def make_model_init(task: str, num_labels: int):
    def _init():
        if task == "stsb":
            return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1, problem_type="regression")
        return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    return _init


# --------------------
# TRAIN + EVAL once
# --------------------
def run_once(task: str, seed: int) -> Dict[str, float]:
    set_seed(seed)
    train_ds, val_ds, tok, num_labels = load_task(task)
    args = TrainingArguments(
        output_dir=f"./out/{task}-s{seed}",
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        learning_rate=LR,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="no",
        seed=seed,  # Trainer will set this at train start
        report_to=[],
    )
    trainer = Trainer(
        model_init=make_model_init(task, num_labels),  # critical for reproducibility across seeds
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        compute_metrics=make_compute_metrics(task),
    )
    trainer.train()
    raw = trainer.evaluate()
    # raw contains task-specific metrics; reduce to single GLUE task score
    score = glue_task_score(task, raw)
    return {"seed": seed, "task": task, "score": float(score), **{k: float(v) for k, v in raw.items()}}


# --------------------
# REPRO: single-seed and naive averaging (WRONG)
# --------------------
def repro_single_seed(tasks: List[str], seed: int = 42) -> Dict[str, float]:
    """
    What people often do by mistake:
    - One seed only
    - Take the 'accuracy' field if present, ignore task rules
    - Then average across tasks directly
    This can overstate results vs GLUE conventions.
    """
    per_task = []
    for t in tasks:
        res = run_once(t, seed)
        # WRONG: pick 'accuracy' if present else fallback to 'score' (mixes metrics)
        wrong_task_val = float(res.get("eval_accuracy", res["score"]))
        per_task.append(wrong_task_val)
    wrong_overall = float(np.mean(per_task))
    return {"overall_naive": wrong_overall}


# --------------------
# FIX: multi-seed with correct per-task metrics and aggregation
# --------------------
@dataclass
class Aggregate:
    mean: float
    median: float
    std: float
    per_seed: List[float]


def aggregate_scores(xs: List[float]) -> Aggregate:
    return Aggregate(mean=float(np.mean(xs)), median=float(np.median(xs)), std=float(np.std(xs, ddof=1) if len(xs) > 1 else 0.0), per_seed=[float(x) for x in xs])


def fix_multi_seed(tasks: List[str], seeds: List[int]) -> Dict[str, Aggregate]:
    """
    Correct protocol:
    - For each task: compute official GLUE task score per seed
    - Aggregate per task across seeds (mean/median/std)
    - Overall GLUE(dev) = unweighted macro-average of per-task *means*
      (other choices like median are fine if you report them explicitly)
    """
    per_task_scores: Dict[str, List[float]] = {t: [] for t in tasks}
    for t in tasks:
        for s in seeds:
            res = run_once(t, s)
            per_task_scores[t].append(res["score"])

    per_task_agg = {t: aggregate_scores(v) for t, v in per_task_scores.items()}
    overall_mean = float(np.mean([per_task_agg[t].mean for t in tasks]))
    overall_median = float(np.median([np.median(per_task_scores[t]) for t in tasks]))
    return {"overall_mean": Aggregate(overall_mean, overall_median, 0.0, []), **per_task_agg}


# --------------------
# Example usage
# --------------------
if __name__ == "__main__":
    # REPRO (fast): one seed, naive averaging
    # repro = repro_single_seed(TASKS, seed=42)
    # print("[REPRO] Naive single-seed overall (WRONG):", repro["overall_naive"])

    # FIX (slower): multi-seed, correct metrics and macro-average
    fix = fix_multi_seed(TASKS, SEEDS)
    print("[FIX] Overall GLUE(dev) mean across tasks:", fix["overall_mean"].mean)
    print("[FIX] Per-task aggregates:")
    for t in TASKS:
        agg = fix[t]
        print(f"  {t}: mean={agg.mean:.4f} median={agg.median:.4f} std={agg.std:.4f} seeds={agg.per_seed}")