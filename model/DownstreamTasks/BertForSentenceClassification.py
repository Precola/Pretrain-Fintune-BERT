from ..BasicBert.Bert import BertModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss


class BertForSentenceClassification(nn.Module):
    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForSentenceClassification, self).__init__()
        self.num_labels = config.num_labels
        self.problem_type = None
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                labels=None):
        """

        :param input_ids:  [src_len, batch_size]
        :param attention_mask: [batch_size, src_len]
        :param token_type_ids: 句子分类时为None
        :param position_ids: [1,src_len]
        :param labels: [batch_size,]
        :return:
        """
        # for Trainer
        input_ids = input_ids.transpose(0, 1)   # [src_len, batch_size]
        token_type_ids = token_type_ids.transpose(0, 1)
        attention_mask = attention_mask  ## [batch_size, src_len]
        labels = labels#.transpose(0, 1)

        pooled_output, sequence_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids)  # [batch_size,hidden_size]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_label]
        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     return loss, logits
        # else:
        #     return logits
        if labels is not None:
            if self.num_labels == 1:
                self.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.problem_type = "single_label_classification"
            else:
                self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                num = labels.sum()
                num_pos = (labels == 1).sum()
                num_neg = (labels == 0).sum()
                w0 = num/num_neg
                w1 = num/num_pos

                weight = torch.tensor([w0, w1], dtype=torch.float).to(labels.device)
                loss_fct = CrossEntropyLoss(weight=weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


            elif self.config.problem_type == "multi_label_classification":
                # num_pos = (labels == 1).sum()
                # num_neg = (labels == 0).sum()
                # pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float)
                # loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

            return loss, logits