import logging
import random
from tqdm import tqdm
from .data_helpers import build_vocab
from .data_helpers import pad_sequence, pad_sequence_ex
from .data_helpers import process_cache
import torch
from torch.utils.data import DataLoader
import os


def read_wiki2(filepath=None, seps='.'):
    """
    This function is used to format the raw wikitext-2 dataset.
    Download link: [https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)

    **Parameters:**
    - `filepath`: Path to the dataset file.

    **Return:**
    The final return is a two-dimensional list, where each element in the outer list is a paragraph, and each element in the inner list is a collection of all the sentences in that paragraph.
    `[ [sentence 1, sentence 2, ...], [sentence 1, sentence 2,...], ..., [] ]`

    This return format is also a standard format. If you need to load other datasets (including Chinese), simply format the dataset into this structure.
    Then, you can add the preprocessing function to the `get_format_data()` method in the `LoadBertPretrainingDataset` class to complete the construction of the pretraining dataset.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()  # Read all lines at once, where each line represents a paragraph.
        # - ① Convert all uppercase letters to lowercase.
        # - ② Only keep paragraphs that contain at least two sentences, as the next sentence needs to be constructed later.
    lines_lower = [line.lower() for line in lines]
    paragraphs = []
    for line in tqdm(lines_lower, ncols=80, desc=" ## loading the raw data"):
        if len(line.split(' . ')) < 2:
            continue
        line = line.strip()
        paragraphs.append([line[0]])
        for w in line[1:]:  # During data preprocessing and splitting, retain the delimiters while filtering out cases where the next sentence is empty.
            if paragraphs[-1][-1][-1] in seps:
                paragraphs[-1].append(w)
            else:
                paragraphs[-1][-1] += w
    random.shuffle(paragraphs)  # Shuffle all the paragraphs.
    return paragraphs


def read_songci(filepath=None, seps='。'):
    """
    本函数的作用是格式化原始的ci.song.xxx.json数据集
    下载地址为：https://github.com/chinese-poetry/chinese-poetry
    掌柜在此感谢该仓库的作者维护与整理
    :param filepath:
    :return: 返回和 read_wiki2() 一样形式的结果
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # 一次读取所有行，每一行为一首词
    paragraphs = []
    for line in tqdm(lines, ncols=80, desc=" ## 正在读取原始数据"):
        if "□" in line or "……" in line or len(line.split('。')) < 2:
            continue
        paragraphs.append([line[0]])
        line = line.strip()  # 去掉换行符和两边的空格
        for w in line[1:]:
            if paragraphs[-1][-1][-1] in seps:
                paragraphs[-1].append(w)
            else:
                paragraphs[-1][-1] += w
    random.shuffle(paragraphs)  # 将所有段落打乱
    return paragraphs


def read_custom(filepath=None):
    raise NotImplementedError("本函数为实现，请参照`read_songci()`或`read_wiki2()`返回格式进行实现")


def cache(func):
    """
    本修饰器的作用是将数据预处理后的结果进行缓存，下次使用时可直接载入！
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        filepath = kwargs['filepath']
        postfix = kwargs['postfix']
        data_path = filepath.split('.')[0] + '_' + postfix + '.pt'
        if not os.path.exists(data_path):
            logging.info(f"The cache file '{data_path}' does not exist, reprocessing and caching it!")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            logging.info(f"Cache file '{data_path}' exists, directly loading the cache file!")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


class LoadBertPretrainingDataset(object):
    r"""

    Arguments:

    """

    def __init__(self,
                 vocab_path='./vocab.txt',
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True,
                 random_state=2021,
                 data_name='wiki2',
                 masked_rate=0.15, # To mask 15% of the words in a sequence
                 masked_token_rate=0.8, # 15%的80%的替换为[MASK]
                 masked_token_unchanged_rate=0.5,# 15%的20%的(50%保持不变，50%替换为随机)
                 seps="."):
        self.tokenizer = tokenizer
        self.seps = seps
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        self.batch_size = batch_size
        self.max_sen_len = max_sen_len
        self.max_position_embeddings = max_position_embeddings
        self.pad_index = pad_index
        self.is_sample_shuffle = is_sample_shuffle
        self.data_name = data_name
        self.masked_rate = masked_rate
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.random_state = random_state
        random.seed(random_state)

    def get_format_data(self, file_path):
        """
        The function's purpose is to format the dataset into a standard form.
        :param file_path:
        :return:  [ [sentence 1, sentence 2, ...], [sentence 1, sentence 2,...],...,[] ]
        """
        if self.data_name == 'wiki2':
            return read_wiki2(file_path, self.seps)
        elif self.data_name == 'custom':
            return read_custom(file_path)
            #Here, you can call your own formatting function corresponding to the dataset,
            # but the return format needs to be consistent with `read_wiki2()`.
        elif self.data_name == 'songci':
            return read_songci(file_path, self.seps)
        else:
            raise ValueError(f"The data {self.data_name} does not have a corresponding formatting function."
                             f"To implement the corresponding formatting function, please refer to the read_wiki(filepath)")

    @staticmethod
    def get_next_sentence_sample(sentence, next_sentence, paragraphs):
        """
        This function is designed to generate sentence pairs and labels for
        the Next Sentence Prediction (NSP) task based on the given two consecutive sentences
        and their corresponding paragraph.
        :param sentence:  str
        :param next_sentence: str
        :param paragraphs: [str,str,...,]
        :return: sentence A, sentence B, True
        """
        if random.random() < 0.5:  # Generate a random number in the range [0,1)
            is_next = True
        else:
            # The purpose of random.choice here is to randomly select an element from a list.
            # The process works as follows:
            # Randomly select a paragraph from the list of all paragraphs.
            # Randomly select a sentence from the chosen paragraph.
            new_next_sentence = next_sentence
            while next_sentence == new_next_sentence:  #To prevent selecting the same next sentence when randomly choosing a negative sample for NSP (even though the probability is very low),
                new_next_sentence = random.choice(random.choice(paragraphs))
            next_sentence = new_next_sentence
            is_next = False
        return sentence, next_sentence, is_next

    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        """
        The purpose of this function is to return the masked token_ids and label information based on
        the given token_ids, the candidate mask positions, and the number of tokens to be masked.
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """
        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # If the number of already masked tokens is greater than or equal to `num_mlm_preds`, stop masking.
            masked_token_id = None
            # 80% of the time: Replace the word with the `[MASK]` token,
            # but here it is directly replaced with the corresponding `[MASK]` token ID.
            if random.random() < self.masked_token_rate:  # 0.8
                masked_token_id = self.MASK_IDS
            else:
                # 10% of the time: Keep the word unchanged.
                if random.random() < self.masked_token_unchanged_rate:  # 0.5
                    masked_token_id = token_ids[mlm_pred_position]
                # 10% of the time: Replace the word with a random word.
                else:
                    masked_token_id = random.randint(0, len(self.vocab.stoi) - 1)
            logging.debug(f"token{mlm_input_tokens_id[mlm_pred_position]} is replaced by {masked_token_id}")
            mlm_input_tokens_id[mlm_pred_position] = masked_token_id
            pred_positions.append(mlm_pred_position)  # Keep the index information of the masked positions.
        # Construct the correct labels for the positions to predict in the MLM task.
        # If the position does not appear in `pred_positions`,
        # it means that the position is not a mask position.
        # In loss calculation, these positions should be ignored (i.e., set to `PAD_IDX`).
        # If the position appears in the masked positions,
        # its label should be the original `token_ids` corresponding to that position.
        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        return mlm_input_tokens_id, mlm_label

    def get_masked_sample(self, token_ids):
        """
        This function's purpose is to apply masking to a portion of the input token_ids.
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        candidate_pred_positions = []  # index of candidate prediction positions
        for i, ids in enumerate(token_ids):
            # n the Masked Language Model (MLM) task, we generally do not predict special tokens
            # such as [CLS], [SEP], and [PAD], since they are not part of the vocabulary to
            # be predicted by the model.
            # The position will not be a candidate for the mask.
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            candidate_pred_positions.append(i)
            # Save the indices of candidate positions, for example, it could be `[2, 3, 4, 5, ....]`.
        random.shuffle(candidate_pred_positions)  # Shuffle all the candidate positions to facilitate randomness for the subsequent steps.
        # The number of masked positions. In the BERT model, by default, **15% of the tokens** are masked.
        num_mlm_preds = max(1, round(len(token_ids) * self.masked_rate))
        logging.debug(f" ##  the number of mask: {num_mlm_preds}")
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label

    @process_cache(unique_key=["max_sen_len", "random_state",
                               "masked_rate", "masked_token_rate", "masked_token_unchanged_rate"])
    def data_process(self, file_path):
        """
        The purpose of this function is to generate the processed data for
        both NSP (Next Sentence Prediction) and MLM (Masked Language Model) tasks based on the formatted data.
        :param file_path:
        :return:
        """
        paragraphs = self.get_format_data(file_path)
        # The return value is a two-dimensional list,
        # where each list can be considered as a paragraph (with each element being a sentence).
        data = []
        max_len = 0
        # The `max_len` here is used to record the length of the longest sequence in the entire dataset,
        # and it can be used as the standard padding length for subsequent processing.
        desc = f" ## Constructing NSP (Next Sentence Prediction) and MLM (Masked Language Model) samples.({file_path.split('.')[1]})"
        for paragraph in tqdm(paragraphs, ncols=80, desc=desc):  # Iterating over each one.
            for i in range(len(paragraph)-1):  # Iterating over each sentence in a paragraph. TODO -1
                sentence, next_sentence, is_next = self.get_next_sentence_sample(
                    paragraph[i], paragraph[i + 1], paragraphs)  # Constructing NSP
                logging.debug(f" ## Current sentence text:{sentence}")
                logging.debug(f" ## Next sentence text:{next_sentence}")
                logging.debug(f" ## Next sentence label:{is_next}")
                if len(next_sentence) < 2:
                    logging.warning(f"The next sentence for the sentence '{sentence}' is empty."
                                    f" Please check the data preprocessing. "
                                    f"The current paragraph text is:{paragraph}")
                    continue
                token_a_ids = [self.vocab[token] for token in self.tokenizer(sentence)]
                token_b_ids = [self.vocab[token] for token in self.tokenizer(next_sentence)]
                token_ids = [self.CLS_IDX] + token_a_ids + [self.SEP_IDX] + token_b_ids
                seg1 = [0] * (len(token_a_ids) + 2)  # 2 Represent the [CLS] and the [SEP] tokens.
                seg2 = [1] * (len(token_b_ids) + 1)
                segs = seg1 + seg2
                if len(token_ids) > self.max_position_embeddings - 1:
                    token_ids = token_ids[:self.max_position_embeddings - 1]  # The BERT pre-trained model only takes the first 512 tokens.
                    segs = segs[:self.max_position_embeddings]
                token_ids += [self.SEP_IDX]
                assert len(token_ids) <= self.max_position_embeddings
                assert len(segs) <= self.max_position_embeddings
                logging.debug(f" ## The token results before masking:{[self.vocab.itos[t] for t in token_ids]}")
                segs = torch.tensor(segs, dtype=torch.long)
                logging.debug(f" ## The token IDs before masking::{token_ids}")
                logging.debug(f" ## segment ids:{segs.tolist()}, The sequence length is:{len(segs)}")
                nsp_lable = torch.tensor(int(is_next), dtype=torch.long)
                mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids)
                token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
                mlm_label = torch.tensor(mlm_label, dtype=torch.long)
                max_len = max(max_len, token_ids.size(0))
                logging.debug(f" ## The token IDs after masking:{token_ids.tolist()}")
                logging.debug(f" ## The token results after masking:{[self.vocab.itos[t] for t in token_ids.tolist()]}")
                logging.debug(f" ## The label IDs after masking:{mlm_label.tolist()}")
                logging.debug(f" ## The construction of the current sample is complete. ================== \n\n")
                data.append([token_ids, segs, nsp_lable, mlm_label])
        all_data = {'data': data, 'max_len': max_len}
        return all_data

    def generate_batch(self, data_batch):
        b_token_ids, b_segs, b_nsp_label, b_mlm_label = [], [], [], []
        for (token_ids, segs, nsp_lable, mlm_label) in data_batch:
            # Start processing each sample in the batch.
            b_token_ids.append(token_ids)
            b_segs.append(segs)
            b_nsp_label.append(nsp_lable)
            b_mlm_label.append(mlm_label)
        b_token_ids = pad_sequence(b_token_ids,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=False,
                                   max_len=self.max_sen_len)
        # b_token_ids:  [src_len,batch_size]

        b_segs = pad_sequence(b_segs,  # [batch_size,max_len]
                              padding_value=self.PAD_IDX,
                              batch_first=False,
                              max_len=self.max_sen_len)
        # b_segs: [src_len,batch_size]

        b_mlm_label = pad_sequence(b_mlm_label,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=False,
                                   max_len=self.max_sen_len)
        # b_mlm_label:  [src_len,batch_size]

        b_mask = (b_token_ids == self.PAD_IDX).transpose(0, 1)
        # b_mask: [batch_size,max_len]

        b_nsp_label = torch.tensor(b_nsp_label, dtype=torch.long)
        # b_nsp_label: [batch_size]
        return b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label

    def load_train_val_test_data(self,
                                 train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 only_test=False):
        # postfix = f"_ml{self.max_sen_len}_rs{self.random_state}_mr{str(self.masked_rate)[2:]}" \
        #           f"_mtr{str(self.masked_token_rate)[2:]}_mtur{str(self.masked_token_unchanged_rate)[2:]}"
        test_data = self.data_process(file_path=test_file_path)['data']
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            logging.info(f"## Successfully returned the test set, which contains a total of {len(test_iter.dataset)} samples.")
            return test_iter
        data = self.data_process(file_path=train_file_path)
        train_data, max_len = data['data'], data['max_len']
        if self.max_sen_len == 'same':
            self.max_sen_len = max_len
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=self.is_sample_shuffle,
                                collate_fn=self.generate_batch)
        val_data = self.data_process(file_path=val_file_path)['data']
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False,
                              collate_fn=self.generate_batch)
        logging.info(f"Successfully returned the training set with {len(train_iter.dataset)} samples and the validation set with {len(val_iter.dataset)} samples."
                     f"The test set contains {len(test_iter.dataset)} samples.")
        return train_iter, test_iter, val_iter

    def make_inference_samples(self, sentences=None, masked=False, language='en', random_state=None):
        """
        制作推理时的数据样本
        :param sentences:
        :param masked:  指传入的句子没有标记mask的位置
        :param language:  判断是中文zh还是英文en
        :param random_state:  控制mask字符时的随机状态
        :return:
        e.g.
        sentences = ["I no longer love her, true,but perhaps I love her.",
                     "Love is so short and oblivion so long."]
        input_tokens_ids.transpose(0,1):
                tensor([[  101,  1045,  2053,   103,  2293,  2014,  1010,  2995,  1010,  2021,
                            3383,   103,  2293,  2014,  1012,   102],
                        [  101,  2293,   103,  2061,  2460,  1998, 24034,  2061,  2146,  1012,
                            102,     0,     0,     0,     0,     0]])
        tokens:
                [CLS] i no [MASK] love her , true , but perhaps [MASK] love her . [SEP]
                [CLS] love [MASK] so short and oblivion so long . [SEP] [PAD] [PAD] [PAD] [PAD] [PAD]
        pred_index:
                [[3, 11], [2]]
        mask:
                tensor([[False, False, False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False],
                      [False, False, False, False, False, False, False, False, False, False,
                        False,  True,  True,  True,  True,  True]])
        """
        if not isinstance(sentences, list):
            sentences = [sentences]
        mask_token = self.vocab.itos[self.MASK_IDS]
        input_tokens_ids = []
        pred_index = []
        for sen in sentences:
            if language == 'en':
                sen_list = sen.split()
            else:
                sen_list = [w for w in sen]
            tmp_token = []
            if not masked:  # 如果传入的样本没有进行mask，则此处进行mask
                candidate_pred_positions = [i for i in range(len(sen_list))]
                random.seed(random_state)
                random.shuffle(candidate_pred_positions)
                num_mlm_preds = max(1, round(len(sen_list) * self.masked_rate))
                for p in candidate_pred_positions[:num_mlm_preds]:
                    sen_list[p] = mask_token
            for item in sen_list:  # 逐个词进行tokenize
                if item == mask_token:
                    tmp_token.append(item)
                else:
                    tmp_token.extend(self.tokenizer(item))
            token_ids = [self.vocab[t] for t in tmp_token]
            token_ids = [self.CLS_IDX] + token_ids + [self.SEP_IDX]
            pred_index.append(self.get_pred_idx(token_ids))  # 得到被mask的Token的位置
            input_tokens_ids.append(torch.tensor(token_ids, dtype=torch.long))
        input_tokens_ids = pad_sequence(input_tokens_ids,
                                        padding_value=self.PAD_IDX,
                                        batch_first=False,
                                        max_len=None)  # 按一个batch中最长的样本进行padding
        mask = (input_tokens_ids == self.PAD_IDX).transpose(0, 1)
        return input_tokens_ids, pred_index, mask

    def get_pred_idx(self, token_ids):
        """
        根据token_ids返回'[MASK]'所在的位置，即需要预测的位置
        :param token_ids:
        :return:
        """
        pred_idx = []
        for i, t in enumerate(token_ids):
            if t == self.MASK_IDS:
                pred_idx.append(i)
        return pred_idx



def tuple_collate_fn(data_batch, pad_idx=0):
    """
    自动动态 pad 到当前 batch 中最长的句子
    """

    b_input_ids = []
    b_token_type_ids = []
    # b_attention_mask = []
    b_mlm_labels = []
    b_nsp_labels = []
#    'input_ids', 'token_type_ids', 'attention_mask',
#     'next_sentence_labels', 'masked_lm_labels'
    for item in data_batch:
        input_ids = item["input_ids"].clone().detach()#.requires_grad_(True)
        token_type_ids = item["token_type_ids"].clone().detach()#.requires_grad_(True)
        # attention_mask =  torch.tensor(item["attention_mask"])
        mlm_labels = item["masked_lm_labels"].clone().detach()#.requires_grad_(True)
        nsp_label = item["next_sentence_labels"]

        b_input_ids.append(input_ids)
        b_token_type_ids.append(token_type_ids)
        # b_attention_mask.append(attention_mask)
        b_mlm_labels.append(mlm_labels)
        b_nsp_labels.append(nsp_label)

    # Dynamic padding: Automatically pad sequences in the current batch to match the length of the longest sequence.
    # b_input_ids = pad_sequence(b_input_ids, batch_first=False, padding_value=pad_idx)
    # b_token_type_ids = pad_sequence(b_token_type_ids, batch_first=False, padding_value=pad_idx)
    # b_mlm_labels = pad_sequence(b_mlm_labels, batch_first=False, padding_value=pad_idx)
    b_input_ids = pad_sequence_ex(b_input_ids, batch_first=False, padding_value=pad_idx, pad_to_multiple_of=8)
    b_token_type_ids = pad_sequence_ex(b_token_type_ids, batch_first=False, padding_value=pad_idx, pad_to_multiple_of=8)
    b_mlm_labels = pad_sequence_ex(b_mlm_labels, batch_first=False, padding_value=pad_idx, pad_to_multiple_of=8)

    # Attention mask: Padding positions are marked with 1.
    b_mask = (b_input_ids == pad_idx).transpose(0, 1)  # shape: [batch_size, seq_len]

    # NSP  tensor
    b_nsp_labels = torch.tensor(b_nsp_labels, dtype=torch.long)  # shape: [batch_size]

    return b_input_ids, b_token_type_ids, b_mask, b_mlm_labels, b_nsp_labels


def tuple_collate_fn_glue(data_batch, num_labels, pad_idx=0):
    """
    自动动态 pad 到当前 batch 中最长的句子
    """

    b_input_ids = []
    b_token_type_ids = []
    # b_attention_mask = []
    b_labels = []
#    'input_ids', 'token_type_ids', 'attention_mask',
#     'next_sentence_labels', 'masked_lm_labels'
    for item in data_batch: #TODO!!!!!!! dim
        input_ids = item["input_ids"].clone().detach().squeeze(0)
        token_type_ids = item["token_type_ids"].clone().detach().squeeze(0)
        # attention_mask =  torch.tensor(item["attention_mask"])
        labels = item["labels"].clone().detach()

        b_input_ids.append(input_ids)
        b_token_type_ids.append(token_type_ids)
        # b_attention_mask.append(attention_mask)
        b_labels.append(labels)

    # Dynamic padding: Automatically pad sequences in the current batch to match the length of the longest sequence.
    b_input_ids = pad_sequence_ex(b_input_ids, batch_first=False, padding_value=pad_idx, pad_to_multiple_of=8)
    b_token_type_ids = pad_sequence_ex(b_token_type_ids, batch_first=False, padding_value=pad_idx, pad_to_multiple_of=8)

    # Attention mask: Padding positions are marked with 1.
    b_mask = (b_input_ids == pad_idx).transpose(0, 1)  # shape: [batch_size, seq_len]

    if num_labels > 1:
        b_labels = torch.tensor(b_labels, dtype=torch.long)  # shape: [batch_size]
    else:
        b_labels = torch.tensor(b_labels)

    return b_input_ids, b_token_type_ids, b_mask, b_labels

def preprocess_function(samples, tokenizer, sentence1_key, sentence2_key):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    label = []

    samples = [dict(zip(samples, t)) for t in zip(*samples.values())]
    for sample in samples:
        # Tokenize the texts
        args = (
            (sample[sentence1_key],) if sentence2_key is None else (sample[sentence1_key], sample[sentence2_key])
        )
        result = tokenizer(*args, padding=False, truncation=True, return_tensors='pt', max_length=128)   #
        # print()
        input_ids.append(result['input_ids'])
        token_type_ids.append(result['token_type_ids'])
        attention_mask.append(result['attention_mask'])
        label.append(sample['label'])

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "labels": label
    }

