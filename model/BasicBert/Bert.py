import torch
from torch.nn.init import normal_
from .BertEmbedding import BertEmbeddings
from .MyTransformer import MyMultiheadAttention
import torch.nn as nn
import os
import logging
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForMaskedLM

class LayerNorm(nn.Module):
    def __init__(self, ndim, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim))
        self.eps = eps

    def forward(self, x):
        x_m = x.mean(-1, keepdim=True)
        self.x_s = x.std(-1, keepdim=True, unbiased=False)
        norm = self.weight * (x - x_m)/(self.x_s + self.eps) + self.bias
        return norm

def get_activation(activation_string):
    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return nn.ReLU()
    elif act == "gelu":
        return nn.GELU()
    elif act == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("Unsupported activation: %s" % act)


class BertSelfAttention(nn.Module):
    """
    https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/modeling.py#L558
    """

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if 'use_torch_multi_head' in config.__dict__ and config.use_torch_multi_head:
            MultiHeadAttention = nn.MultiheadAttention
        else:
            MultiHeadAttention = MyMultiheadAttention
        self.multi_head_attention = MultiHeadAttention(embed_dim=config.hidden_size,
                                                       num_heads=config.num_attention_heads,
                                                       dropout=config.attention_probs_dropout_prob)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """

        :param query: # [tgt_len, batch_size, hidden_size], tgt_len
        :param key:  #  [src_len, batch_size, hidden_size], src_len
        :param value: # [src_len, batch_size, hidden_size], src_len
        :param attn_mask: # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]  causa mask
        :param key_padding_mask: [batch_size, src_len], src_len padding mask
        :return:
        attn_output: [tgt_len, batch_size, hidden_size]
        attn_output_weights: # [batch_size, tgt_len, src_len]
        """
        return self.multi_head_attention(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        :param hidden_states: [src_len, batch_size, hidden_size]
        :param input_tensor: [src_len, batch_size, hidden_size]
        :return: [src_len, batch_size, hidden_size]
        """
        # hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self,
                hidden_states,
                attention_mask=None):
        """

        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len]
        :return: [src_len, batch_size, hidden_size]
        """
        # hidden_states = self.LayerNorm(hidden_states)   # TODO
        self_outputs = self.self(hidden_states,
                                 hidden_states,
                                 hidden_states,
                                 attn_mask=None,
                                 key_padding_mask=attention_mask)
        # self_outputs[0] shape: [src_len, batch_size, hidden_size]
        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        """

        :param hidden_states: [src_len, batch_size, hidden_size]
        :return: [src_len, batch_size, intermediate_size]
        """
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, intermediate_size]
        if self.intermediate_act_fn is None:
            hidden_states = hidden_states
        else:
            hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """

        :param hidden_states: [src_len, batch_size, intermediate_size]
        :param input_tensor: [src_len, batch_size, hidden_size]
        :return: [src_len, batch_size, hidden_size]
        """
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states)  #TOOD
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert_attention = BertAttention(config)
        self.bert_intermediate = BertIntermediate(config)
        self.bert_output = BertOutput(config)

    def forward(self,
                hidden_states,
                attention_mask=None):
        """

        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len] mask掉padding部分的内容
        :return: [src_len, batch_size, hidden_size]
        """
        attention_output = self.bert_attention(hidden_states, attention_mask)
        # [src_len, batch_size, hidden_size]
        intermediate_output = self.bert_intermediate(attention_output)
        # [src_len, batch_size, intermediate_size]
        layer_output = self.bert_output(intermediate_output, attention_output)
        # [src_len, batch_size, hidden_size]
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None):
        """

        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len]
        :return:
        """
        all_encoder_layers = []
        layer_output = hidden_states
        for i, layer_module in enumerate(self.bert_layers):
            layer_output = layer_module(layer_output,
                                        attention_mask)
            #  [src_len, batch_size, hidden_size]
            all_encoder_layers.append(layer_output)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.config = config

    def forward(self, hidden_states):
        """
        :param hidden_states:  [src_len, batch_size, hidden_size]
        :return: [batch_size, hidden_size]
        """
        if 'pooler_type' not in self.config.__dict__:
            raise ValueError("pooler_type must be one of ['first_token_transform', 'all_token_average']. "
                             "Please add a pooler_type parameter in the config.json file.")

        if self.config.pooler_type == "first_token_transform":
            token_tensor = hidden_states[0, :].reshape(-1, self.config.hidden_size)
        elif self.config.pooler_type == "all_token_average":
            token_tensor = torch.mean(hidden_states, dim=0)
        pooled_output = self.dense(token_tensor)  # [batch_size, hidden_size]
        pooled_output = self.activation(pooled_output)
        # pooled_output = self.layer_norm(pooled_output)
        return pooled_output  # [batch_size, hidden_size]


def format_paras_for_torch(loaded_paras_names, loaded_paras):
    """
    torch 2.5.0
    :param loaded_paras_names:
    :param loaded_paras:
    :return:
    """
    qkv_weight_names = ['query.weight', 'key.weight', 'value.weight']
    qkv_bias_names = ['query.bias', 'key.bias', 'value.bias']
    qkv_weight, qkv_bias = [], []
    torch_paras = []
    for i in range(len(loaded_paras_names)):
        para_name_in_pretrained = loaded_paras_names[i]
        para_name = ".".join(para_name_in_pretrained.split('.')[-2:])
        if para_name in qkv_weight_names:
            qkv_weight.append(loaded_paras[para_name_in_pretrained])
        elif para_name in qkv_bias_names:
            qkv_bias.append(loaded_paras[para_name_in_pretrained])
        else:
            torch_paras.append(loaded_paras[para_name_in_pretrained])
        if len(qkv_weight) == 3:
            torch_paras.append(torch.cat(qkv_weight, dim=0))
            qkv_weight = []
        if len(qkv_bias) == 3:
            torch_paras.append(torch.cat(qkv_bias, dim=0))
            qkv_bias = []
    return torch_paras


def replace_512_position(init_embedding, loaded_embedding):
    logging.info(f"max_positional_embedding > 512")
    init_embedding[:512, :] = loaded_embedding[:512, :]
    return init_embedding


class BertModel(nn.Module):
    """
    ***** It is very important to note that in the attention_mask, the masked tokens are represented by 1 (True),
    and the unmasked tokens are represented by 0 (False). This point must be carefully considered.

    :param input_ids: [src_len, batch_size]
    :param attention_mask: [batch_size, src_len] to mask the padding part
    :param token_type_ids: [src_len, batch_size]  # This parameter can be omitted if the model only takes a single sequence.
    :param position_ids: [1, src_len]  # This parameter can actually be omitted in practical modeling.

    :return:
    """

    def __init__(self, config):
        super().__init__()
        self.bert_embeddings = BertEmbeddings(config)
        self.bert_encoder = BertEncoder(config)
        # self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.bert_pooler = BertPooler(config)
        self.config = config
        self._reset_parameters()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None):

        embedding_output = self.bert_embeddings(input_ids=input_ids,
                                                position_ids=position_ids,
                                                token_type_ids=token_type_ids)
        # embedding_output: [src_len, batch_size, hidden_size]
        all_encoder_outputs = self.bert_encoder(embedding_output,
                                                attention_mask=attention_mask)
        # all_encoder_outputs is a collection of outputs from num_hidden_layers layers.
        sequence_output = all_encoder_outputs[-1]  # Take the last layer.
        # sequence_output: [src_len, batch_size, hidden_size]
        pooled_output = self.bert_pooler(sequence_output)
        # By default, it is the result of the dense layer + tanh applied to the first token of the last layer, i.e., the [CLS] token.
        # pooled_output: [batch_size, hidden_size]
        return pooled_output, all_encoder_outputs

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=self.config.initializer_range)

    @classmethod
    def from_pretrained(cls, config, pretrained_model_dir=None):
        model = cls(config)  # Initialize the model, where `cls` is an uninstantiated object, meaning it is an uninstantiated `BertModel` object.
        # pretrained_model_path = os.path.join(pretrained_model_dir, "pytorch_model.bin")   #TODO
        pretrained_model_path = pretrained_model_dir
        if not os.path.exists(pretrained_model_path):
            print("loading from HF")
        else:
            model_body = AutoModelForMaskedLM.from_pretrained("bert-large-uncased", num_labels=2)
            # torch.load()#pathway

        loaded_paras = model_body.state_dict()
        state_dict = deepcopy(model.state_dict())
        # loaded_paras_names = list(loaded_paras.keys())[:-8]
        # model_paras_names = list(state_dict.keys())[1:]
        loaded_paras_names = list(loaded_paras)[:-7]
        model_paras_names = list(state_dict.keys())[1:-5]

        if 'use_torch_multi_head' in config.__dict__ and config.use_torch_multi_head:
            torch_paras = format_paras_for_torch(loaded_paras_names, loaded_paras)
            for i in range(len(model_paras_names)):
                logging.debug(
                    f"## Successfully assigned parameter: {model_paras_names[i]}, with shape: {torch_paras[i].size()}")
                if "position_embeddings" in model_paras_names[i]:
                    # This part of the code is used to remove the limitation of pre-trained models that can only input fewer than 512 characters.
                    if config.max_position_embeddings > 512:
                        new_embedding = replace_512_position(state_dict[model_paras_names[i]],
                                                             loaded_paras[loaded_paras_names[i]])
                        state_dict[model_paras_names[i]] = new_embedding
                        continue
                state_dict[model_paras_names[i]] = torch_paras[i]
            logging.info(f"## Note, using the MultiHeadAttention implementation from the torch framework.")

        else:
            for i in range(len(loaded_paras_names)):
                logging.debug(f"## Successfully assigned parameter: {loaded_paras_names[i]} to {model_paras_names[i]}, "
                              f"with parameter shape: {state_dict[model_paras_names[i]].size()}")

                if "position_embeddings" in model_paras_names[i]:
                    # This part of the code is used to eliminate the restriction that pre-trained models can only input fewer than 512 characters.
                    if config.max_position_embeddings > 512:
                        new_embedding = replace_512_position(state_dict[model_paras_names[i]],
                                                             loaded_paras[loaded_paras_names[i]])
                        state_dict[model_paras_names[i]] = new_embedding
                        continue
                state_dict[model_paras_names[i]] = loaded_paras[loaded_paras_names[i]]
            logging.info(f"## Note, using the MyMultiHeadAttention implementation from local MyTransformer. "
                     f"If you want to use the MultiHeadAttention module from the torch framework, "
                     f"you can enable it by setting config.__dict__['use_torch_multi_head'] = True.")

        model.load_state_dict(state_dict)
        return model
