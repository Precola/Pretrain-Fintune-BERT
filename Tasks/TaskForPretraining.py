import os
import logging
import sys
from itertools import chain
import nltk
from nltk.tokenize import sent_tokenize

try:
    sent_tokenize("Test sentence.")
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
import random
import numpy as np
from utils import logger_init
from model import BertConfig
from model import BertForPretrainingModel
from transformers import BertTokenizer, BertTokenizerFast, DataCollatorForLanguageModeling
from torch.optim import AdamW
from datetime import datetime
from datasets import load_dataset, concatenate_datasets #, load_from_disk
from datasets.distributed import split_dataset_by_node
from torch import amp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch import amp
from utils import LoadBertPretrainingDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import get_polynomial_decay_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import torch
import time
# print("!!CPUs per task:", os.environ.get("SLURM_CPUS_PER_TASK"))
# print("Total tasks:", os.environ.get("SLURM_NTASKS"))
# cpu_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK"))
# num_workers = min(6, cpu_per_task)
# # num_workers = 5
# print("num_workers", num_workers)
# date_str = datetime.now().strftime("%Y-%m-%d")
# print(f'cuda:{torch.cuda.is_available()}')
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # # ========== wike2 The configuration related to the dataset.
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_uncased_english")
        self.data_name = 'wiki2'

        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.model_save_path = os.path.join(self.model_save_dir, f'LAbase_model_{self.data_name}_1.bin')
        self.writer = SummaryWriter(f"runs/LAbase_{self.data_name}_{date_str}_test")   # print result
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
        self.gradient_accumulation_steps = 8# max 8
        self.gradient_accumulation_steps_max = 8# max 8 TODO

        self.ddp = True # True #False TODO
        self.backend = 'nccl' # 'nccl', 'gloo', etc.
        self.use_amp = False #True
        self.resume = True #False  TODO
        self.grad_clip = 1.0
        logger_init(log_file_name=self.data_name, log_level=self.log_level,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        bert_config_path = os.path.join(self.pretrained_model_dir, "config_large.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        logging.info(" ### Print the current configuration to the log file. ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")

max_position_embeddings = 512
tokenizer = BertTokenizerFast.from_pretrained("~/BERT/bert_base_uncased")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
    pad_to_multiple_of=8,
)
## paragraph -> senten
def preprocess_text(text):
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences

def preprocess_examples(examples):
    processed_texts = []
    for text in examples["text"]:
        processed_texts.extend(preprocess_text(text))
    return {"text": processed_texts}

## tokenize
def tokenize_function(examples):
    # Remove empty lines
    examples['text'] = [
        line for line in examples['text'] if len(line) > 0 and not line.isspace()
    ]
    return tokenizer(
        examples['text'],
        # padding=padding,
        truncation=True,
        max_length=max_position_embeddings,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // max_position_embeddings) * max_position_embeddings
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + max_position_embeddings] for i in range(0, total_length, max_position_embeddings)]
        for k, t in concatenated_examples.items()
    }
    return result


def train(config):
    scaler = amp.GradScaler( enabled=config.use_amp) # # Gradient scaler to prevent underflow/overflow during fp16 training #TODO: enabled=False
    model = BertForPretrainingModel(config,)

    if config.ddp:
        init_process_group(backend=config.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        config.device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(config.device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        # assert config.gradient_accumulation_steps % ddp_world_size == 0
        # config.gradient_accumulation_steps //= ddp_world_size

        # training!
        if ddp_local_rank == 0:
            writer = SummaryWriter(f"runs/G_{config.data_name}_{date_str}_LA", comment='ddp-training')
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 10  # 113 #    #0
        ddp_world_size = 1
        writer = SummaryWriter(f"runs/G_{config.data_name}_{date_str}_LA___")
        config.gradient_accumulation_steps = 1
        # config.gradient_accumulation_steps = config.gradient_accumulation_steps * 2

    model = model.to(config.device)

    if os.path.exists(config.model_save_path) and config.resume:
        checkpoint = torch.load(config.model_save_path)
        last_epoch = checkpoint['last_epoch']
        print(f"loading last epoch: {last_epoch}...............")

    # Loading dataset
    wiki_dataset = load_dataset("wikipedia", "20220301.en")
    bookdataset = load_dataset("bookcorpus", split="train")
    dataset = concatenate_datasets([wiki_dataset['train'], bookdataset])    #dataset.num_shards:10
    dataset = dataset.to_iterable_dataset(num_shards=24).shuffle(buffer_size=20000, seed=100)  # shuffle first
    column_names = list(dataset.features)

    dataset = dataset.map(
        preprocess_examples,
        batched=True,
        remove_columns=column_names,
        batch_size=2000,
    )

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=2000,
        remove_columns=['text'],
    )

    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=2000,
    )

    if config.ddp:
        ds = split_dataset_by_node(tokenized_datasets, rank=ddp_rank, world_size=ddp_world_size)
        train_iter = StatefulDataLoader(ds,
                                batch_size=config.batch_size,
                                num_workers=num_workers,
                                collate_fn=lambda x: data_collator(x))

    else:
        train_iter = StatefulDataLoader(tokenized_datasets,
                                batch_size=config.batch_size,
                                num_workers=num_workers,
                                collate_fn=lambda x: data_collator(x))
    last_epoch = -1
    if os.path.exists(config.model_save_path) and config.resume:
        print("loading...")
        checkpoint = torch.load(config.model_save_path)
        last_epoch = checkpoint['last_epoch']
        # train_iter.load_state_dict(checkpoint['state_dict'])

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
            "initial_lr": config.learning_rate

        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "initial_lr": config.learning_rate
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                        betas=(0.9, 0.999))
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                          num_warmup_steps=10000,    #int(len(train_iter) * 0)
                                                          num_training_steps=int(config.epochs * 8280005),
                                                          last_epoch=last_epoch)

    if os.path.exists(config.model_save_path) and config.resume:
        loaded_paras = checkpoint['model']
        model.load_state_dict(loaded_paras)

        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(config.device)

        scheduler.load_state_dict(checkpoint['scheduler'])
        if config.use_amp:
            scaler.load_state_dict(checkpoint['scaler'])
            for attr in ['_scale', '_scale_growth_tracker']:
                val = getattr(scaler, attr, None)
                if isinstance(val, torch.Tensor):
                    setattr(scaler, attr, val.to(config.device))

        logging.info("## Successfully loaded the existing model and continue training. ................")

    if config.ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    model = torch.compile(model)
    model.train()
    max_acc = 0
    state_dict = None
    for epoch in range(config.epochs):
        if os.path.exists(config.model_save_path) and config.resume:
            losses = checkpoint['loss']
        else:
            losses = 0
        start_time = time.time()

        for idx, batch in enumerate(train_iter):
            if config.ddp:
                model.require_backward_grad_sync = (idx % config.gradient_accumulation_steps == config.gradient_accumulation_steps - 1)

            b_token_ids = batch['input_ids'].t().to(config.device)  # [src_len, batch_size]
            b_segs = batch['token_type_ids'].t().to(config.device)  # [src_len, batch_size]
            b_mask = batch['attention_mask'].to(config.device)  ## [batch_size, src_len]
            b_mlm_label = batch['labels'].t().to(config.device)  # [src_len, batch_size]
            with amp.autocast(device_type='cuda',dtype=torch.bfloat16):    # for forward function! bfloat16
                loss, mlm_logits = model(input_ids=b_token_ids,
                                         attention_mask=b_mask,
                                         token_type_ids=b_segs,
                                         masked_lm_labels=b_mlm_label,
                                         )
                loss = loss / config.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
            scaler.scale(loss).backward()

            if ((idx  % config.gradient_accumulation_steps) == (config.gradient_accumulation_steps - 1)):
                # with record_function("## optimizer ##"):
                # Step optimizer with scaled gradients
                if config.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            losses += (loss.item() * config.gradient_accumulation_steps)
            mlm_acc, _, _= accuracy(mlm_logits, b_mlm_label, config.pad_index)
            if idx % 1000 == 0 and (not config.ddp or ddp_local_rank == 0):
                # print(f"occupied memory: {torch.cuda.memory_allocated(0)}")
                # print(f"reserved memory: {torch.cuda.memory_reserved(0)}")
                logging.info(f"Epoch: [{epoch + 1}/{config.epochs}], "  #, Batch[{idx}/{len(train_iter)}]
                             f"Train loss :{loss.item():.3f}, Train mlm acc: {mlm_acc:.3f},")
                config.writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
                config.writer.add_scalar('Training/Learning Rate', scheduler.get_last_lr()[0], scheduler.last_epoch)
                config.writer.add_scalars(main_tag='Training/Accuracy',
                                          tag_scalar_dict={'MLM': mlm_acc},
                                          global_step=scheduler.last_epoch)

                print(f"Last_epoch:{scheduler.last_epoch},Iter:{idx},Epoch: [{epoch + 1}/{config.epochs}]")
                print('Training/Loss', loss.item()*config.gradient_accumulation_steps)

            if (idx % config.model_val_per_iter == 0) and (not config.ddp or ddp_local_rank == 0):
                # mlm_acc, nsp_acc = evaluate(config, train_iter, model, data_loader.PAD_IDX)
                checkpoint = {
                    'last_epoch': scheduler.last_epoch,
                    'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict() if config.use_amp else None,
                    'scheduler': scheduler.state_dict(),
                    'idx_last' : idx,
                    'state_dict': train_iter.state_dict(),
                    'loss': losses.items() if isinstance(losses, torch.Tensor) else losses
                }
                print(f"saving checkpoint...........")
                torch.save(checkpoint, config.model_save_path)

        end_time = time.time()

    if config.ddp:
        destroy_process_group()

def accuracy(mlm_logits, mlm_labels, PAD_IDX):
    """
    :param mlm_logits:  [src_len,batch_size,src_vocab_size]
    :param mlm_labels:  [src_len,batch_size]
    :param nsp_logits:  [batch_size,2]
    :param nsp_label:  [batch_size]
    :param PAD_IDX:
    :return:
    """
    mlm_pred = mlm_logits.transpose(0, 1).argmax(axis=2).reshape(-1)
    # 将 [src_len,batch_size,src_vocab_size] 转成 [batch_size, src_len,src_vocab_size]
    mlm_true = mlm_labels.transpose(0, 1).reshape(-1)
    # 将 [src_len,batch_size] 转成 [batch_size， src_len]
    mlm_acc = mlm_pred.eq(mlm_true)
    mask = torch.logical_not(mlm_true.eq(PAD_IDX))
    mlm_acc = mlm_acc.logical_and(mask)
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()
    mlm_acc = float(mlm_correct) / mlm_total

    # nsp_correct = (nsp_logits.argmax(1) == nsp_label).float().sum()
    # nsp_total = len(nsp_label)
    # nsp_acc = float(nsp_correct) / nsp_total
    return [mlm_acc, mlm_correct, mlm_total]#, nsp_acc, nsp_correct, nsp_total


def evaluate(config, data_iter, model, PAD_IDX):
    model.eval()
    mlm_corrects, mlm_totals, nsp_corrects, nsp_totals = 0, 0, 0, 0
    with torch.no_grad():
        for idx, (b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label) in enumerate(data_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_segs = b_segs.to(config.device)
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            b_nsp_label = b_nsp_label.to(config.device)
            mlm_logits, nsp_logits = model(input_ids=b_token_ids,
                                           attention_mask=b_mask,
                                           token_type_ids=b_segs)
            result = accuracy(mlm_logits, nsp_logits, b_mlm_label, b_nsp_label, PAD_IDX)
            _, mlm_cor, mlm_tot, _, nsp_cor, nsp_tot = result
            mlm_corrects += mlm_cor
            mlm_totals += mlm_tot
            nsp_corrects += nsp_cor
            nsp_totals += nsp_tot
    model.train()
    return [float(mlm_corrects) / mlm_totals, float(nsp_corrects) / nsp_totals]

if __name__ == '__main__':
    set_seed(1010)
    config = ModelConfig()
    train(config)

