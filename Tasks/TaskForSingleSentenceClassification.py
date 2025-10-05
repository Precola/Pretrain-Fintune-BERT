import sys
import argparse

sys.path.append('../')
from model import BertForSentenceClassification
from model import BertConfig
from utils import LoadSingleSentenceClassificationDataset
from utils import logger_init
from utils import preprocess_function, tuple_collate_fn_glue
from transformers import BertTokenizer
import torch.distributed as dist
from torch.utils.data import DataLoader

from datasets import load_dataset
from datasets.features import ClassLabel, Value
from functools import partial
from transformers import get_polynomial_decay_schedule_with_warmup
import logging
import torch
import os
import time
import random
import numpy as np
import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed to set")
parser.add_argument("--task_name", type=str, default="sst2", help="task name")  #stsb,mrpc,qqp,cola, sst2, qnli,rte, mnli
parser.add_argument("--learning_rate", type=float, default=3e-5, help="5e-5, 3e-5, 2e-5")   #

args = parser.parse_args()
def set_seed():
    print("!!!!!Random seed: ", args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    # "wnli": ("sentence1", "sentence2"),
}

f1_metric = evaluate.load("f1") #F1 scores
accuracy_metric = evaluate.load("accuracy")
spearmanr_metric = evaluate.load("spearmanr")
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

class ModelConfig:
    def __init__(self):
        self.task_name = args.task_name   #"mnli" #TODO

        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'SingleSentenceClassification')
        self.pretrained_cache_dir = os.path.join(self.project_dir, "cache/LAbase_model_wiki2_2.bin")
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert-base-uncased")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.split_sep = '_!_'
        self.is_sample_shuffle = True
        self.batch_size = 16
        self.max_sen_len = 128
        self.num_labels = 15
        self.epochs = 10
        self.model_val_per_epoch = 1
        self.learning_rate = args.learning_rate#1e-4

        self.gradient_accumulation_steps = 2# max 8
        self.grad_clip = 1.0
        self.resume = False
        logger_init(log_file_name='single', log_level=logging.INFO,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        bert_config_path = os.path.join(self.pretrained_model_dir, "config_large.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        logging.info(" ### Print the current configuration to the log file. ")
        for key, value in self.__dict__.items():
            logging.info(f"###  {key} = {value}")


def train(config):
    if config.task_name == "stsb":
        config.pooler_type = 'all_token_average'

    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_dir)

    # Preprocessing the raw_datasets
    sentence1_key, sentence2_key = task_to_keys[config.task_name]
    wrapped_func = partial(preprocess_function, tokenizer=tokenizer, sentence1_key=sentence1_key, sentence2_key=sentence2_key)

    task = load_dataset('glue', config.task_name) #TODO
    print(f"Loading dataset GLUE:{config.task_name}.........")
    print(f"learning rate: {config.learning_rate}")

    if is_main_process():
        print("Only main process will run map()")
        dataset = task.map(wrapped_func, batched=True)    #, remove_columns=[sentence1_key, sentence2_key, 'idx']

    # Then broadcast or reload the result for other processes
    dataset.set_format(type='torch', columns=[
        'input_ids', 'token_type_ids', 'attention_mask',
        'labels'
    ])
    # the number of label
    label_feature = dataset["train"].features["label"]
    if isinstance(label_feature, Value):
        print("label's dtype is Value  =", label_feature.dtype)
        num_labels = 1
    else:
        num_labels = label_feature.num_classes
    config.num_labels = num_labels

    train_iter = DataLoader(dataset['train'],
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=lambda x: tuple_collate_fn_glue(x, pad_idx=0, num_labels=num_labels))
    if config.task_name == 'mnli':
        val_iter_m = DataLoader(dataset['validation_matched'],
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=lambda x: tuple_collate_fn_glue(x, pad_idx=0, num_labels=num_labels))
        val_iter_mm = DataLoader(dataset['validation_mismatched'],
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=lambda x: tuple_collate_fn_glue(x, pad_idx=0, num_labels=num_labels))
    else:
        val_iter = DataLoader(dataset['validation'],
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=lambda x: tuple_collate_fn_glue(x, pad_idx=0, num_labels=num_labels))

    model = BertForSentenceClassification(config=config,
                                          bert_pretrained_model_dir=True)    #loading from HF to test
    # Freezing all parameters in the BERT model
    # for param in model.bert.parameters():
    #     param.requires_grad = False
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "initial_lr": config.learning_rate

        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "initial_lr": config.learning_rate
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
    )

    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                          num_warmup_steps=int(len(train_iter) * 0.1),    #int(len(train_iter) * 0)
                                                          num_training_steps=int(config.epochs * len(train_iter)))

    # load the model
    model_save_path = os.path.join(config.model_save_dir, f'{config.task_name}_model.pt')
    if os.path.exists(model_save_path) and config.resume:
        checkpoint = torch.load(model_save_path)
        loaded_paras = checkpoint['model']
        # new_state_dict = {}
        # for k, v in loaded_paras.items():
        #     new_key = k.replace('module.', '', 1)
        #     new_state_dict[new_key] = v
        # model.load_state_dict(new_state_dict)
        model.load_state_dict(loaded_paras)

        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(config.device)

        scheduler.load_state_dict(checkpoint['scheduler'])

    model = model.to(config.device)

    model.train()

    max_eva = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (b_token_ids, b_segs, b_mask, b_label) in enumerate(train_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_token_type_ids = b_segs.to(config.device)
            b_mask = b_mask.to(config.device)  ## [batch_size, src_len]
            b_label = b_label.to(config.device)
            loss, logits = model(
                input_ids=b_token_ids,
                attention_mask=b_mask,
                token_type_ids=b_token_type_ids,
                position_ids=None,
                labels=b_label)

            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if ((idx  % config.gradient_accumulation_steps) == (config.gradient_accumulation_steps - 1)):
                if config.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            losses += (loss.item() * config.gradient_accumulation_steps)

            # evalute
            if config.task_name in ["qqp", "mrpc"]:
                f1_scores = f1_metric.compute(predictions=logits.argmax(1).tolist(), references=b_label.tolist()).get('f1')
                if idx % 50 == 0:
                    logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                                 f"Train loss :{loss.item():.3f}, Train f1_scores: {f1_scores:.3f}")
            elif config.task_name == "stsb":
                spearmanr_co = spearmanr_metric.compute(predictions=(logits).squeeze().tolist(), references=b_label.tolist()).get("spearmanr", 0)
                if idx % 50 == 0:
                    logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                                 f"Train loss :{loss.item():.3f}, Train spearmanr_co: {spearmanr_co:.3f}")
            else:
                acc = (logits.argmax(1) == b_label).float().mean()
                if idx % 50 == 0:
                    logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                                 f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")

        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_val_per_epoch == 0:
            if config.task_name == "mnli":
                eva = evaluate(val_iter_m, model, config.device, config.task_name)
                logging.info(f"Accuracy on val_m {eva:.3f}")
                eva = evaluate(val_iter_mm, model, config.device, config.task_name)
                logging.info(f"Accuracy on val_mm {eva:.3f}")
            else:
                eva = evaluate(val_iter, model, config.device, config.task_name)
                logging.info(f"Accuracy on val {eva:.3f}")
            if eva > max_eva:
                max_eva = eva
                # torch.save(model.state_dict(), model_save_path)
                checkpoint = {
                    'model': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': vars(config),
                }
                # print(f"saving checkpoint...........")
                # torch.save(checkpoint, model_save_path)

def evaluate(data_iter, model, device, task_name):
    model.eval()
    with ((torch.no_grad())):
        eva, n = 0.0, 0
        all_preds = []
        all_labels = []
        for idx, (b_token_ids, b_segs, b_mask, b_label) in enumerate(data_iter):
            b_token_ids = b_token_ids.to(device)  # [src_len, batch_size]
            b_token_type_ids = b_segs.to(device)
            b_mask = b_mask.to(device)  ## [batch_size, src_len]
            b_label = b_label.to(device)
            loss, logits = model(
                input_ids=b_token_ids,
                attention_mask=b_mask,
                token_type_ids=b_token_type_ids,
                position_ids=None,
                labels=b_label)

            n += len(b_label)
            if task_name == 'stsb':
                all_preds.append(logits.squeeze().cpu())
            else:
                all_preds.append(logits.argmax(1).cpu())
            all_labels.append(b_label)

        all_preds = torch.cat(all_preds).tolist()
        all_labels = torch.cat(all_labels).tolist()

        if task_name in ["qqp", "mrpc"]:
            # f1_scores
            eva += f1_metric.compute(predictions=all_preds, references=all_labels).get("f1", 0)

        elif task_name == "stsb":
            # spearmanr_co
            eva += spearmanr_metric.compute(predictions=all_preds, references=all_labels).get("spearmanr", 0)
        else:
            # acc_sum
            eva += accuracy_metric.compute(predictions=all_preds, references=all_labels).get("accuracy", 0)
        model.train()
        return eva


if __name__ == '__main__':
    # set_seed()
    model_config = ModelConfig()
    train(model_config)