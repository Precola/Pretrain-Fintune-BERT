from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
import torch.nn as nn
import copy
import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

class MyMultiheadAttention(nn.Module):
    """

    The calculation formula for the multi-head attention mechanism
    (i.e., the formula on page 5 of the paper) is:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super(MyMultiheadAttention, self).__init__()
        """
        :param embed_dim:   The dimension of word embeddings, which corresponds to the d_model parameter mentioned earlier, is set to 512 by default in the paper.
        :param num_heads:   The number of heads in the multi-head attention mechanism, which corresponds to the nhead parameter mentioned earlier, is set to 8 by default in the paper.
        :param dropout:     
        :param bias:        When applying the linear transformation to the output of the multi-head attention (combined output), should a bias be used?
        """
        self.embed_dim = embed_dim  # The d_model parameter mentioned earlier.
        self.head_dim = embed_dim // num_heads  # head_dim refers to d_k,d_v
        self.kdim = self.head_dim
        self.vdim = self.head_dim

        self.num_heads = num_heads  # the number of head
        self.dropout = dropout

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim % num_heads == 0"
        #   d_k = d_v = d_model/n_head

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # embed_dim = kdim * num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # W_k,  embed_dim = kdim * num_heads
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # W_v,  embed_dim = vdim * num_heads
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        :param query: # [tgt_len, batch_size, embed_dim], tgt_len
        :param key:  #  [src_len, batch_size, embed_dim], src_len
        :param value: # [src_len, batch_size, embed_dim], src_len
        :param attn_mask: # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len] for causa  mask

        :param key_padding_mask: [batch_size, src_len], src_len for padding tokens mask
        :return:
        attn_output: [tgt_len, batch_size, embed_dim]
        attn_output_weights: # [batch_size, tgt_len, src_len]
        """
        return multi_head_attention_forward(query, key, value, self.num_heads,
                                            self.dropout,
                                            out_proj=self.out_proj,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            q_proj=self.q_proj,
                                            k_proj=self.k_proj,
                                            v_proj=self.v_proj,
                                            attn_mask=attn_mask)


def multi_head_attention_forward(query,  # [tgt_len,batch_size, embed_dim]
                                 key,  # [src_len, batch_size, embed_dim]
                                 value,  # [src_len, batch_size, embed_dim]
                                 num_heads,
                                 dropout_p,
                                 out_proj,
                                 training=True,
                                 key_padding_mask=None,  # [batch_size,src_len/tgt_len]
                                 q_proj=None,  # weight: [embed_dim,kdim * num_heads]  , bias: [embed_dim]
                                 k_proj=None,  # weight: [embed_dim,kdim * num_heads]  , bias: [embed_dim]
                                 v_proj=None,  # weight: [embed_dim,kdim * num_heads]  , bias: [embed_dim]
                                 attn_mask=None,  # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
                                 ):
    q = q_proj(query)
    #  [tgt_len,batch_size, embed_dim] x [embed_dim,kdim * num_heads] = [tgt_len,batch_size,kdim * num_heads]

    k = k_proj(key)
    # [src_len, batch_size, embed_dim] x [embed_dim, kdim * num_heads] = [src_len, batch_size, kdim * num_heads]

    v = v_proj(value)
    # [src_len, batch_size, embed_dim] x [embed_dim, vdim * num_heads] = [src_len, batch_size, vdim * num_heads]
    tgt_len, bsz, embed_dim = query.size()  # [tgt_len,batch_size, embed_dim]
    src_len = key.size(0)
    head_dim = embed_dim // num_heads  # num_heads * head_dim = embed_dim
    scaling = float(head_dim) ** -0.5
    # q = q * scaling  # [query_len,batch_size,kdim * num_heads]



    ##########for SDPA#################
    q = q.permute(1,0,2).contiguous().view(bsz, -1, num_heads, head_dim).transpose(1, 2) # [batch_size * num_heads,tgt_len,kdim]
    k = k.permute(1,0,2).contiguous().view(bsz, -1, num_heads, head_dim).transpose(1, 2) # [batch_size * num_heads,src_len,kdim]
    v = v.permute(1,0,2).contiguous().view(bsz, -1, num_heads, head_dim).transpose(1, 2)  # [batch_size * num_heads,src_len,vdim]
    extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
        key_padding_mask, torch.float32, tgt_len=tgt_len)
    attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, attn_mask=extended_attention_mask)  ## batch_size, num_heads, seq_length, head_dim
    attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)


    ##########for normal Attention#################
    # if attn_mask is not None:  # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
    #     if attn_mask.dim() == 2:
    #         attn_mask = attn_mask.unsqueeze(0)  # [1, tgt_len,src_len]
    #         if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
    #             raise RuntimeError('The size of the 2D attn_mask is not correct.')
    #     elif attn_mask.dim() == 3:
    #         if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
    #             raise RuntimeError('The size of the 3D attn_mask is not correct.')
    #     # 现在 atten_mask 的维度就变成了3D

    # q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)# [batch_size * num_heads,tgt_len,kdim]
    # k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads,src_len,kdim]
    # v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads,src_len,vdim]
    # attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    # # [batch_size * num_heads,tgt_len,kdim] x [batch_size * num_heads, kdim, src_len]
    # # =  [batch_size * num_heads, tgt_len, src_len]


    # if attn_mask is not None:
    #     attn_output_weights += attn_mask  # [batch_size * num_heads, tgt_len, src_len]

    # if key_padding_mask is not None:
    #     attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    #     # 变成 [batch_size, num_heads, tgt_len, src_len]的形状
    #     attn_output_weights = attn_output_weights.masked_fill(
    #         (1-key_padding_mask).bool().unsqueeze(1).unsqueeze(2),
    #         float('-inf'))  #
    #     # attn_output_weights = attn_output_weights.masked_fill(
    #     #     key_padding_mask.unsqueeze(1).unsqueeze(2),
    #     #     float('-inf'))  #
    #     # 扩展维度，key_padding_mask从[batch_size,src_len]变成[batch_size,1,1,src_len]
    #     # 然后再对attn_output_weights进行填充
    #     attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len,
    #                                                    src_len)  # [batch_size * num_heads, tgt_len, src_len]
    #
    # attn_output_weights = F.softmax(attn_output_weights, dim=-1)  # [batch_size * num_heads, tgt_len, src_len]
    # attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    # attn_output = torch.bmm(attn_output_weights, v)
    # # Z = [batch_size * num_heads, tgt_len, src_len]  x  [batch_size * num_heads,src_len,vdim]
    # # = # [batch_size * num_heads,tgt_len,vdim]
    # # 这就num_heads个Attention(Q,K,V)结果

    # attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    # 先transpose成 [tgt_len, batch_size* num_heads ,kdim]
    # 再view成 [tgt_len,batch_size,num_heads*kdim]
    # attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

    Z = out_proj(attn_output)   # [tgt_len,batch_size,embed_dim]

    return Z, None#attn_output_weights.sum(dim=1) / num_heads  # average attention weights over heads
