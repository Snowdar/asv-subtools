# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2022-07)
# Reference: https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/attention.py


"""Multi-Head Attention layer definition."""

from cmath import pi
import math
from typing import Optional, Tuple, Dict, Any
import torch
from torch import nn
from libs.nnet.activation import Nonlinearity
from .scaling import ScaledLinear,ScaledConv1d,ActivationBalancer

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        add_t5rel_bias (bool): whether apply T5 rel_pos on attention score matrix.
        t5rel_module (torch.nn.Module): T5Rel module instance, if not None, means share a T5 rel position matrix in all layers.
    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float,add_t5rel_bias: bool =False,conv_out: bool =False,attention_norm_args: Optional[Dict[str, Any]]=None,re_scale=False):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head

        self.conv_out = conv_out

        self.att_norm = AttentionNormalize(self.d_k, att_type = "mlh", **attention_norm_args)
        
        project = ScaledLinear if re_scale else nn.Linear

        self.linear_q = project(n_feat, n_feat)
        self.linear_k = project(n_feat, n_feat)
        self.linear_v = project(n_feat, n_feat)
        if self.conv_out:
            conv = ScaledConv1d if re_scale else nn.Conv1d
            self.linear_out = conv(n_feat, n_feat, 3,
                                            stride=1, padding=1)
        else:
            self.linear_out = ScaledLinear(n_feat, n_feat,initial_scale=0.25) if re_scale else  nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.add_t5rel_bias = add_t5rel_bias

        self.t5rel_module = T5RelPositionBias(self.d_k**0.5) if self.add_t5rel_bias else None

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v
        




    def forward_attention(self, value: torch.Tensor, scores: torch.Tensor,
                          mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        attn = self.att_norm(scores,mask)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1,
                                                 self.h * self.d_k)
             )  # (batch, time1, d_model)
        if self.conv_out:
            x = self.linear_out(x.transpose(-1, 1)).transpose(-1, 1)
        else:
            x = self.linear_out(x) 
    
        return x  # (batch, time1, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                pos_emb: torch.Tensor = torch.empty(0)) -> torch.Tensor:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
 
        scores = torch.matmul(q, k.transpose(-2, -1))
        if self.add_t5rel_bias and self.t5rel_module is not None:
            scores+=self.t5rel_module(scores)

        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        add_t5rel_bias (bool): whether apply T5 rel_pos on attention score matrix.
        t5rel_module (torch.nn.Module): T5Rel module instance, if not None, means share a T5 rel position matrix in all layers.
    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float,add_t5rel_bias: bool =False,conv_out: bool =False,attention_norm_args: Optional[Dict[str, Any]]=None,re_scale=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate,add_t5rel_bias,conv_out,attention_norm_args,re_scale)
        # linear transformation for positional encoding
        project = ScaledLinear if re_scale else nn.Linear
        self.linear_pos = project(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu: bool = False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """

        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: torch.ones((0, 0, 0), dtype=torch.bool),
                pos_emb: torch.Tensor = torch.empty(0)):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift(matrix_bd)

        scores = matrix_ac + matrix_bd  # (batch, head, time1, time2)
        if self.add_t5rel_bias and self.t5rel_module is not None:
            scores+=self.t5rel_module(scores)

        return self.forward_attention(v, scores, mask)

# RoPE (Leo 2022-07-25) 
# reference: 
#   RoFormer: Enhanced Transformer with Rotary Position Embedding.
class RoPESelfAttention(MultiHeadedAttention):
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float,add_t5rel_bias: bool =False,conv_out: bool =False, attention_norm_args: Optional[Dict[str, Any]]=None, rotary_value: bool = True,re_scale=False):
        """Construct an RelPositionMultiHeadedAttention object.
        Args:
            n_head (int): The number of heads.
            n_feat (int): The number of features.
            dropout_rate (float): Dropout rate.
            rotary_value (bool): add rotary positon to value tensor.  
            add_t5rel_bias (bool): whether apply T5 rel_pos on attention score matrix.
            t5rel_module (torch.nn.Module): T5Rel module instance, if not None, means share one T5 rel position matrix in all layers.
        """
        super().__init__(n_head, n_feat, dropout_rate,add_t5rel_bias,conv_out,attention_norm_args,re_scale)
        self.rotary_value = rotary_value

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                pos_emb:torch.Tensor = torch.empty(0)):
        """Compute 'Scaled Dot Product Attention' with rotary positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            pos_emb : Positional embedding tensor(#batch, time2, size//2) of query and key_value,
                each is a tuple contains sin part tensor and cos part tensor.
        Returns:
            torch.Tensor: Output tensor (#batch, time1, size).
        """
        q, k, v = self.forward_qkv(query, key, value)
        

        q = self.apply_rotary(q,pos_emb)
        k = self.apply_rotary(k,pos_emb)
        if self.rotary_value:
            v = self.apply_rotary(v,pos_emb)       
        scores = torch.matmul(q, k.transpose(-2, -1))
        if self.add_t5rel_bias and self.t5rel_module is not None:
            scores+=self.t5rel_module(scores)

      
        return self.forward_attention(v, scores, mask)

    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos.chunk(2,dim=-1) # (1, time1, d_model//2)

        x1, x2 = x[..., 0::2], x[..., 1::2]

        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)

# T5RelPE (Leo 2022-07-25)
# reference:
#   Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
#   https://arxiv.org/abs/1910.10683
class T5RelPositionBias(torch.nn.Module):
    """T5 rel_pos, which is trainalble bias matrix added to attention score.
    Args:
        scale (float): scale the bias, usually set to query_dim**0.5.
        causal (bool): true, means omit the future.
        num_buckets (int): relative to the length of sensitive area.
        max_distance (int): when distance > max_distance,they share the same bias.
    """
    def __init__(
        self,
        scale,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = torch.nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        """Get the rel bias from relative_position_bucket.
        Args:
            x (torch.Tensor): attention scores. (#batch, head, time1, time2).
        Returns:
            torch.Tensor: bias tensor. (#time1, time2).
        
        """
        i, j, device = *x.shape[-2:], x.device
        # get q,k position
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        # calculate the relative position distance matrix (i, j)
        rel_pos = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = values.squeeze(-1)
        return bias * self.scale





class OffsetScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1,dim))
        self.beta = nn.Parameter(torch.zeros(1,dim))
        nn.init.xavier_uniform_(self.gamma)

    def forward(self, x):
        out = x*self.gamma + self.beta
        return out

# (Leo 2022-08-10)
class GAU(nn.Module):
    """Gated Attention Unit. now it just support selfatt for the whole sequence.

    Args:
        hidden_dim (int): The size of hidden_dim, recommend 2*n_feat.
        n_feat (int): The number of features.
        d_k (int): Dim of query,key, default (128).
        dropout_rate (float): Dropout rate.
        activation_type (str): activation function.
        add_t5rel_bias (bool): whether apply T5 rel_pos on attention score matrix.
        t5rel_module (torch.nn.Module): T5Rel module instance, if not None, means share a T5 rel position matrix in all layers.
    """
    def __init__(self, n_feat: int, hidden_dim:int ,d_k: int = 128 ,dropout_rate: float = 0., 
                 conv_out: bool =False,
                 attention_norm_args: Optional[Dict[str, Any]]=None,
                 re_scale: bool=False,
                 activation_type='swish',
                 add_t5rel_bias: bool =False, 

        ):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        self.d_k =d_k
        self.conv_out = conv_out
        project = ScaledLinear if re_scale else nn.Linear
        banlancer = nn.Identity if re_scale else ActivationBalancer
        self.to_gate = nn.Sequential(
            project(n_feat, hidden_dim),
            banlancer(),
            Nonlinearity(activation_type)
        )
        self.to_v = nn.Sequential(
            project(n_feat, hidden_dim),
            banlancer(),
            Nonlinearity(activation_type)
        )
        self.to_qk = nn.Sequential(
            project(n_feat, d_k),
            banlancer(),
            Nonlinearity(activation_type)
        )
        if self.conv_out:
            conv = ScaledConv1d if re_scale else nn.Conv1d
            self.to_out = nn.Sequential(
            conv(hidden_dim, n_feat, 3,
                      stride=1, padding=1)
        )
        else:
            self.to_out = nn.Sequential(
            project(hidden_dim, n_feat)
        )
        self.att_norm = AttentionNormalize(self.d_k, att_type = "gau", **attention_norm_args)
        self.dropout = nn.Dropout(dropout_rate)
        self.scale_q = OffsetScale(d_k)
        self.scale_k = OffsetScale(d_k)

        self.add_t5rel_bias = add_t5rel_bias
        self.t5rel_module = T5RelPositionBias(self.d_k**0.5) if self.add_t5rel_bias else None

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            now it assume q = k = v

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, time2, d_k).
            torch.Tensor: gate tensor, size
                (#batch, time1, hidden_dim).
            torch.Tensor: value tensor, size
                (#batch, time2, hidden_dim).
        """

        u = self.to_gate(query) # (batch, time1, hidden_dim)
        v = self.to_v(value)    # (batch, time2, hidden_dim)
        # here q is the whole sequence.
        qk = self.to_qk(key)      # (batch, time1, d_k)
        q = self.scale_q(qk)
        k = self.scale_k(qk)

        return q, k, u, v
        
    def forward_qkuv(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            now it assume q = k = v

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, time2, d_k).
            torch.Tensor: gate tensor, size
                (#batch, time1, hidden_dim).
            torch.Tensor: value tensor, size
                (#batch, time2, hidden_dim).
        """

        u = self.to_gate(query) # (batch, time1, hidden_dim)
        v = self.to_v(query)    # (batch, time2, hidden_dim)
        # here q is the whole sequence.
        qk = self.to_qk(query)      # (batch, time1, d_k)
        q = self.scale_q(qk)
        k = self.scale_k(qk)

        return q, k, u, v

    def forward_attention(self, u: torch.Tensor, value: torch.Tensor, scores: torch.Tensor,
                          mask:torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            u (torch.Tensor): Transformed gate, size
                (#batch, time1, hidden_dim).       
            value (torch.Tensor): Transformed value, size
                (#batch, time2, hidden_dim).
            scores (torch.Tensor): Attention score, size
                (#batch, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        attn = self.att_norm(scores,mask)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, time1, time2) (batch, time2, hidden_dim) ->  (batch, time1, hidden_dim)
        x = u * x
        if self.conv_out:
            x = self.to_out(x.transpose(-1, 1)).transpose(-1, 1)
        else:
            x = self.to_out(x)    
        return x # (batch, time1, n_feat)

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor],
                value: Optional[torch.Tensor],
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                pos_emb: torch.Tensor = torch.empty(0)) -> torch.Tensor:
        """Compute Gated Attention.
        now it just support selfatt for the whole sequence,
        which means q = k = v. 

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.


        Returns:
            torch.Tensor: Output tensor (#batch, time1, n_feat).

        """
        q, k, u, v = self.forward_qkuv(query)

        scores = torch.matmul(q, k.transpose(-2, -1))
        if self.add_t5rel_bias and self.t5rel_module is not None:
            scores+=self.t5rel_module(scores)

        return self.forward_attention(u, v, scores, mask)


class RoPEGAU(GAU):
    def __init__(self, n_feat: int, hidden_dim:int ,d_k: int = 128 ,dropout_rate: float = 0.,
                 conv_out: bool =False,
                 attention_norm_args: Optional[Dict[str, Any]]=None,
                 re_scale: bool=False,
                 activation_type='swish',
                 add_t5rel_bias: bool =False,
    ):
        """Construct an RelPositionMultiHeadedAttention object.
        Args:
            n_head (int): The number of heads.
            n_feat (int): The number of features.
            dropout_rate (float): Dropout rate.
            rotary_value (bool): add rotary positon to value tensor.  
            add_t5rel_bias (bool): whether apply T5 rel_pos on attention score matrix.
            t5rel_module (torch.nn.Module): T5Rel module instance, if not None, means share one T5 rel position matrix in all layers.
        """
        super().__init__(n_feat, hidden_dim, d_k ,dropout_rate,
                         conv_out,
                         attention_norm_args,
                         re_scale,
                         activation_type,
                         add_t5rel_bias,
                         )


    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor],
                value: Optional[torch.Tensor], mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                pos_emb:  torch.Tensor = torch.empty(0)):
        """Compute 'Scaled Dot Product Attention' with rotary positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            pos_emb : Positional embedding tensor(#batch, time2, size//2) of query and key_value,
                each is a tuple contains sin part tensor and cos part tensor.
        Returns:
            torch.Tensor: Output tensor (#batch, time1, size).
        """
        q, k, u, v = self.forward_qkuv(query)

        q = self.apply_rotary(q,pos_emb)
        k = self.apply_rotary(k,pos_emb)

        scores = torch.matmul(q, k.transpose(-2, -1))
        if self.add_t5rel_bias and self.t5rel_module is not None:
            scores+=self.t5rel_module(scores)   
        return self.forward_attention(u,v, scores, mask)

    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos.chunk(2,dim=-1) # (1, time1, d_model//2)
        x1, x2 = x[..., 0::2], x[..., 1::2]

        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)


# (Leo 2022-08-10)
class AttentionNormalize(nn.Module):
    def __init__(self,
                 d_k: int, 
                 att_type = "mlh", 
                 scale_adapt: bool=False, 
                 norm_method: str='softmax', 
                 diag_mask: bool=False, 
                 g_sa: bool=False, 
                 train_len : int = 512,
                 dim: int=-1):
        super().__init__()
        self.method = norm_method
        self.dim = dim
        self.scale_adapt = scale_adapt
        if self.scale_adapt:
            self.scale = nn.Parameter(torch.log(torch.tensor(d_k**-0.5)))
        else:
            self.scale = torch.tensor(math.sqrt(d_k))
        self.diag_mask = diag_mask
        self.att_type = att_type
        self.g_sa = g_sa
        self.omiga = torch.tensor(0.001)
        self.bias = torch.zeros(1)-0.001
        if self.g_sa:
            if 'softmax' not in norm_method:
                raise ValueError("g_sa just support softmax form calculate now")            
            # self.omiga = nn.Parameter(torch.abs(nn.init.trunc_normal_(torch.empty(1)))+0.001)
            # self.bias = nn.Parameter(-torch.abs(nn.init.trunc_normal_(torch.empty(1))))
            self.omiga = nn.Parameter(torch.tensor(0.001))
            self.bias = nn.Parameter(torch.zeros(1)-0.001)
        # self.train_len = torch.tensor(math.log(train_len))

        self.train_len = nn.Parameter(torch.tensor(math.log(train_len))) if self.method =="softmax_plus" else torch.empty(0)
    def forward(self, scores: torch.Tensor, 
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)):        
        if self.g_sa:
            i, j=scores.shape[-2:]
            device = scores.device
            q_pos = torch.arange(j-i,j, dtype = torch.long, device = device)
            k_pos = torch.arange(j, dtype = torch.long, device = device)
            # calculate the relative position distance matrix (i, j)
            dis_matrix = (k_pos.unsqueeze(0) - q_pos.unsqueeze(1))**2
            dis_matrix = -torch.abs(torch.abs(dis_matrix*self.omiga) - torch.abs(self.bias)) # (i, j)
            scores = scores + dis_matrix

        if self.scale_adapt:
            scores = scores*(self.scale.exp())
        else:
            scores = scores/self.scale
        if mask.size(2) > 0 :  # time2 > 0
            mask = mask[..., :scores.size(-1)]
            if self.diag_mask:
                mask = (~torch.eye(scores.shape[-2],scores.shape[-1],dtype=torch.bool,device=scores.device)) & mask
            if self.att_type == "gau":
                mask = mask.eq(0) # (batch,  *, time2)
            else:
                mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)     
            
            scores = scores.masked_fill(mask, -1e4)         
            attn = self.attention_normalize(scores, dim=self.dim,method=self.method).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2) or # (batch,  time1, time2)
        else:
            attn = self.attention_normalize(scores, dim=self.dim,method=self.method)  # (batch, head, time1, time2) or  (batch, time1, time2)   
        return attn

    def attention_normalize(self, 
                            a: torch.Tensor, 
                            dim: int=-1, 
                            method: str='softmax'):
        """attention score normalization
        softmax
        relu_plus: https://arxiv.org/abs/2202.10447 
        softmax_plus: https://kexue.fm/archives/8823 。
        """
        assert method in ['softmax','relu_plus','softmax_plus']
        if method == 'softmax':
            return torch.softmax(a, dim=dim)

        else:
            mask = (a > -1e4).float()
            l = torch.sum(mask ,dim=dim, keepdim=True).clamp_(1.)
   
            if method == 'relu_plus':
                return torch.relu(a)**2 / l
            elif method == 'softmax_plus':

                scale = torch.log(l) / self.train_len * mask + 1 - mask

                return torch.softmax(a * scale, dim=dim)
                
            else:
                raise ValueError("check your attention norm ")   

        return a