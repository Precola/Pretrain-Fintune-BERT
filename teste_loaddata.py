import os
from datasets import load_dataset, concatenate_datasets, load_from_disk
from datasets import get_dataset_config_names
import torch
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
from torch.utils.data import DataLoader
from utils import LoadBertPretrainingDataset, tuple_collate_fn
import psutil

from torch.nn.utils.rnn import pad_sequence
import torch

final_dataset = load_from_disk("/home/zoe/TTFS-BERT/data_processed_nsp_mlm4gb_small")
split_dataset = final_dataset.train_test_split(test_size=0.02, seed=42)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']
columns = ['input_ids', 'token_type_ids', 'attention_mask', 'masked_lm_labels', 'next_sentence_labels']
train_dataset.set_format(type='torch', columns=columns)
val_dataset.set_format(type='torch', columns=columns)

from torch.utils.data import Dataset

class MyCustomWrapperDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # 自定义返回内容
        return (
            item['input_ids'],
            item['attention_mask'],
            item['token_type_ids'],
            item['masked_lm_labels'],
            item['next_sentence_labels'],
        )

# 使用方法
wrapped_train_dataset = MyCustomWrapperDataset(train_dataset)
wrapped_val_dataset = MyCustomWrapperDataset(val_dataset)

# 然后交给 DataLoader：
# train_loader = DataLoader(wrapped_train_dataset, ...)


train_iter = DataLoader(train_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=4,
                        collate_fn=lambda x: tuple_collate_fn(x, pad_idx=0))
val_iter = DataLoader(val_dataset,
                      batch_size=1,
                      shuffle=True,
                      num_workers=4,
                      collate_fn=lambda x: tuple_collate_fn(x, pad_idx=0))

for idx, (b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label) in enumerate(train_iter):
    vm = psutil.virtual_memory()
    print(f"[{idx}] Mem Used: {vm.used / 1e9:.2f}GB\n")