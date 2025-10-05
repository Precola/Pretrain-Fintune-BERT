import logging
from ..BasicBert import BertModel, BertModel_TTFS, BertModel_TTFS_relu
from ..BasicBert import get_activation
import torch.nn as nn
import torch

class BertForLMTransformHead(nn.Module):
    """
    用于BertForMaskedLM中的一次变换。 因为在单独的MLM任务中
    和最后NSP与MLM的整体任务中均要用到，所以这里单独抽象为一个类便于复用

    ref: https://github.com/google-research/bert/blob/master/run_pretraining.py
        第248-262行
    """

    def __init__(self, config, bert_model_embedding_weights=None):
        """
        :param config:
        :param bert_model_embedding_weights:
        the output-weights are the same as the input embeddings, but there is
        an output-only bias for each token. 即TokenEmbedding层中的词表矩阵
        """
        super(BertForLMTransformHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        if bert_model_embedding_weights is not None:
            self.decoder.weight = nn.Parameter(bert_model_embedding_weights)
        # [hidden_size, vocab_size]
        self.decoder.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        """
        :param hidden_states: [src_len, batch_size, hidden_size] Bert最后一层的输出
        :return:
        """
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.transform_act_fn(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.LayerNorm(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.decoder(hidden_states)
        # hidden_states:  [src_len, batch_size, vocab_size]
        return hidden_states


class BertForPretrainingModel(nn.Module):
    """
    The BERT pre-trained model includes two tasks: MLM (Masked Language Model) and NSP (Next Sentence Prediction).
    """

    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForPretrainingModel, self).__init__()
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)    #TODO
        else:  # If no pre-trained model path is specified, randomly initialize the entire network weights.
            self.bert = BertModel(config)  #TODO
            # self.bert = BertModel_TTFS_relu(config)
        weights = None
        if 'use_embedding_weight' in config.__dict__ and config.use_embedding_weight:
            weights = self.bert.bert_embeddings.word_embeddings.embedding.weight  #TODo
            logging.info(f"## Use the weight matrix from the token embedding as the weights for the output layer!{weights.shape}")
        self.mlm_prediction = BertForLMTransformHead(config, weights)
        self.nsp_prediction = nn.Linear(config.hidden_size, 2)
        self.config = config

    def forward(self, input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len] mask掉padding部分的内容
                token_type_ids=None,  # [src_len, batch_size]
                masked_lm_labels=None,  # [src_len,batch_size]
                position_ids=None,
                next_sentence_labels=None):  # [batch_size]

        pooled_output, sequence_output, all_encoder_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        # sequence_output = all_encoder_outputs[-1]
        # sequence_output: [src_len, batch_size, hidden_size]
        mlm_prediction_logits = self.mlm_prediction(sequence_output)
        # mlm_prediction_logits: [src_len, batch_size, vocab_size]
        # nsp_pred_logits = self.nsp_prediction(pooled_output)
        # nsp_pred_logits： [batch_size, 2]
        if masked_lm_labels is not None:#and next_sentence_labels is not None:
            loss_fct_mlm = nn.CrossEntropyLoss(ignore_index=-100)
            # loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)

            # In the MLM task, during dataset construction, the padding part of the labels is filled with 0.
            # Therefore, the ignore_index needs to be set to 0.
            # loss_fct_nsp = nn.CrossEntropyLoss()
            # Since the classification labels in NSP contain 0, and the MLM loss has set ignore_index=0,
            # a new CrossEntropyLoss needs to be defined here.
            # If the MLM task uses values like 100 to replace padding and MASK, both tasks can share the same CrossEntropyLoss.

            mlm_loss = loss_fct_mlm(mlm_prediction_logits.reshape(-1, self.config.vocab_size),
                                    masked_lm_labels.reshape(-1))
            # nsp_loss = loss_fct_nsp(nsp_pred_logits.reshape(-1, 2),
            #                         next_sentence_labels.reshape(-1))
            total_loss = mlm_loss #+ nsp_loss
            return total_loss, mlm_prediction_logits#, nsp_pred_logits
        else:
            return mlm_prediction_logits#, nsp_pred_logits
        # [src_len, batch_size, vocab_size], [batch_size, 2]
