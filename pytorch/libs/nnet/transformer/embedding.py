# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2022-07)
# Reference: https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/embedding.py.

"""Positonal Encoding Module."""

import math
from typing import Tuple

import torch

def get_abs_position(dim: int,
                     max_len: int) -> torch.Tensor:
    abs_pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len,
                            dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) *
        -(math.log(10000.0) / dim))
    abs_pe[:, 0::2] = torch.sin(position * div_term)
    abs_pe[:, 1::2] = torch.cos(position * div_term)
    return abs_pe.unsqueeze(0)

class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    :param int att_h: invalid here,for compatibility to RoPositionalEncoding
    :param bool rope_abs_plus: invalid here,for compatibility to RoPositionalEncoding
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """
    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 att_h: int=4,
                 rope_abs_plus: bool=False,
                 max_len: int = 5000,
                 reverse: bool = False):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self,
                x: torch.Tensor,
                offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """
        assert offset + x.size(1) < self.max_len
        self.pe = self.pe.to(x.device)
        pos_emb = self.pe[:, offset:offset + x.size(1)]
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int): start offset
            size (int): requried size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        assert offset + size < self.max_len
        return self.dropout(self.pe[:, offset:offset + size])


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        att_h (int): invalid here,for compatibility to RoPositionalEncoding
        rope_abs_plus (bool): invalid here,for compatibility to RoPositionalEncoding
        max_len (int): Maximum input length.
    """
    def __init__(self, d_model: int, dropout_rate: float, att_h: int=4, max_len: int = 5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len=max_len, reverse=True)

    def forward(self,
                x: torch.Tensor,
                offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        assert offset + x.size(1) < self.max_len
        self.pe = self.pe.to(x.device)
        x = x * self.xscale
        pos_emb = self.pe[:, offset:offset + x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)


class NoPositionalEncoding(torch.nn.Module):
    """ No position encoding
    """
    def __init__(self, d_model: int, dropout_rate: float, att_h: int=4,rope_abs_plus: bool=False):
        super().__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self,
                x: torch.Tensor,
                offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Just return zero vector for interface compatibility
        """
        pos_emb = torch.zeros(1, x.size(1), self.d_model).to(x.device)
        return self.dropout(x), pos_emb

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        return torch.zeros(1, size, self.d_model)


# RoPE (Leo 2022-07-25) 
# reference: 
#   RoFormer: Enhanced Transformer with Rotary Position Embedding.
class RoPositionalEncoding(PositionalEncoding):
    """ Rotary positional encoding module. The cos features of sinusoidal are organized in
        the 2nd half of the vector. 
    Args:
        d_embed (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        att_h (int): attention head num.
        rope_abs_plus (bool): rope plus abs.
        max_len (int): Maximum input length.
    """
    def __init__(self, d_embed: int, dropout_rate: float ,att_h: int=4 ,rope_abs_plus: bool=False, max_len: int = 5000 , d_roembed: int=-1):
        """Initialize class."""
        super().__init__(d_embed, dropout_rate, max_len=max_len)
        if d_roembed < 1:
            assert (d_embed % att_h) % 2 == 0
            d_roembed = d_embed //att_h
        else:
            d_roembed = d_roembed
        abs_rope = get_abs_position(d_roembed, self.max_len)
        freq = torch.zeros_like(abs_rope)
        sentinel = d_roembed // 2
        freq[...,0:sentinel] = abs_rope[...,0::2]
        freq[...,sentinel:] = abs_rope[...,1::2]
        self.abs_pe = self.pe
        self.pe = freq
        self.rope_abs_plus = rope_abs_plus

    def forward(self,
                x: torch.Tensor,
                offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        assert offset + x.size(1) < self.max_len
        self.pe = self.pe.to(x.device)
        x = x * self.xscale
        pos_emb = self.pe[:, offset:offset + x.size(1)]
        if self.rope_abs_plus:
            self.abs_pe = self.abs_pe.to(x.device)
            abs_pe = self.abs_pe[:, offset:offset + x.size(1)]
            x += abs_pe
        return self.dropout(x), pos_emb

