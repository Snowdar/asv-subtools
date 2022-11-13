# -*- coding:utf-8 -*-

# Reference: https://github.com/espnet/espnet.

from re import X
import torch

from torch import nn
from typing import Optional, Tuple
from .layer_norm import LayerNorm,Trans_Bat,BasicNorm
from .scaling import ActivationBalancer,ScaledLinear

class TransformerEncoderLayer(nn.Module):
    """Encoder layer module

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        dropout_rate: float=0.1,
        layer_dropout: float=0.,
        normalize_before: bool = True,
        norm_type: str = "layer_norm",
        positionwise_layer_type: str="linear",
        concat_after: bool = False
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        if norm_type == "batch_norm":
            self.use_layer_norm = False
            
            norm_tp = Trans_Bat
        else:
            self.use_layer_norm = True
            if norm_type == "layer_norm":
                norm_tp = LayerNorm
            else:
                norm_tp = BasicNorm
        self.norm1 = norm_tp(size)
        self.norm2 = norm_tp(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_dropout = layer_dropout
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.concat_linear = None
        self.positionwise_layer_type = positionwise_layer_type
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(
            self, 
            x: torch.Tensor, 
            mask: torch.Tensor,
            pos_emb: torch.Tensor,
            mask_pad: Optional[torch.Tensor] = None,
            warmup: float = 1.0,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoded features

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :param torch.Tensor pos_emb: 
        :param torch.Tensor mask_pad: does not used in transformer layer, just for unified api with conformer.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        warmup_scale = min(0.1 + warmup, 1.0)
        if self.training:
            alpha = (
                warmup_scale
                if torch.rand(()).item() <= (1.0 - self.layer_dropout)
                else 0.1
            )
        else:
            alpha = 1.0
        x_orig = x
        residual = x

        if self.normalize_before:
            if not self.use_layer_norm:
                x = x.transpose(1, 2)
            x = self.norm1(x)
            if not self.use_layer_norm:
                x = x.transpose(1, 2)

        x_att = self.self_attn(x, x, x, mask, pos_emb)

        if self.concat_linear is not None:
            x_concat = torch.cat((x,x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)

        if not self.normalize_before:
            if not self.use_layer_norm:
                x = x.transpose(1, 2)
            x = self.norm1(x)
            if not self.use_layer_norm:
                x = x.transpose(1, 2)

        residual = x

        if self.normalize_before:
            if not self.use_layer_norm:
                x = x.transpose(1, 2)
            x = self.norm2(x)
            if not self.use_layer_norm:
                x = x.transpose(1, 2)
        if self.positionwise_layer_type == 'gau':
            x = self.feed_forward(x, x, x, mask, pos_emb)
        else:
            x = self.feed_forward(x)
        x = residual + self.dropout(x)

        if not self.normalize_before:
            if not self.use_layer_norm:
                x = x.transpose(1, 2)
            x = self.norm2(x)
            if not self.use_layer_norm:
                x = x.transpose(1, 2)

        if alpha != 1.0:
            x = alpha * x + (1 - alpha) * x_orig

        return x, mask

class ConformerEncoderLayer(nn.Module):
    """Encoder layer module

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward module
    :param feed_forward_macaron (torch.nn.Module): Additional feed-forward module
           instance.
           `PositionwiseFeedForward` instance can be used as the argument.
    :param conv_module (torch.nn.Module): Convolution module instance.
           `ConvlutionModule` instance can be used as the argument.
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float=0.1,
        layer_dropout: float=0.,
        normalize_before: bool = True,
        norm_type: str = "layer_norm",
        positionwise_layer_type: str="linear",
        concat_after: bool = False
    ):
        super(ConformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        if norm_type == "batch_norm":
            self.use_layer_norm = False
            
            norm_tp = Trans_Bat
        else:
            self.use_layer_norm = True
            if norm_type == "layer_norm":
                norm_tp = LayerNorm
            else:
                norm_tp = BasicNorm
        self.norm_ff = norm_tp(size)
        self.norm_mha = norm_tp(size)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = norm_tp(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = norm_tp(size)  # for the CNN module
            self.norm_final = norm_tp(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_dropout = layer_dropout
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.positionwise_layer_type = positionwise_layer_type
        self.concat_linear = None
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(
            self, 
            x: torch.Tensor, 
            mask: torch.Tensor,
            pos_emb: torch.Tensor,
            mask_pad: Optional[torch.Tensor] = None,
            warmup: float = 1.0,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoded features

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in, max_time_in)
        :param torch.Tensor pos_emb: 
        :param torch.Tensor mask_pad: batch padding mask used for conv module (batch, 1, time)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        warmup_scale = min(0.1 + warmup, 1.0)
        if self.training:
            alpha = (
                warmup_scale
                if torch.rand(()).item() <= (1.0 - self.layer_dropout)
                else 0.1
            )
        else:
            alpha = 1.0

        x_orig = x
        if self.feed_forward_macaron is not None:
            residual = x

            if self.normalize_before:
                if not self.use_layer_norm:
                    x = x.transpose(1, 2)
                x = self.norm_ff_macaron(x)
                if not self.use_layer_norm:
                    x = x.transpose(1, 2)
            if self.positionwise_layer_type == 'gau':
                x = self.feed_forward_macaron(x, x, x, mask, pos_emb)
            else:
                x = self.feed_forward_macaron(x)
            x = residual + self.ff_scale * self.dropout(x)

            if not self.normalize_before:
                if not self.use_layer_norm:
                    x = x.transpose(1, 2)
                x = self.norm_ff_macaron(x)
                if not self.use_layer_norm:
                    x = x.transpose(1, 2)
        # MHA        
        residual = x

        if self.normalize_before:
            if not self.use_layer_norm:
                x = x.transpose(1, 2)
            x = self.norm_mha(x)
            if not self.use_layer_norm:
                x = x.transpose(1, 2)

        x_att = self.self_attn(x, x, x, mask, pos_emb)

        if self.concat_linear is not None:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)

        if not self.normalize_before:
            if not self.use_layer_norm:
                x = x.transpose(1, 2)
            x = self.norm_mha(x)
            if not self.use_layer_norm:
                x = x.transpose(1, 2)

        # convolution module
        if self.conv_module is not None:
            residual = x

            if self.normalize_before:
                if not self.use_layer_norm:
                    x = x.transpose(1, 2)
                x = self.norm_conv(x)
                if not self.use_layer_norm:
                    x = x.transpose(1, 2)

            x = self.conv_module(x, mask_pad)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                if not self.use_layer_norm:
                    x = x.transpose(1, 2)
                x = self.norm_conv(x)
                if not self.use_layer_norm:
                    x = x.transpose(1, 2)
        # FFN
        residual = x

        if self.normalize_before:
            if not self.use_layer_norm:
                x = x.transpose(1, 2)
            x = self.norm_ff(x)
            if not self.use_layer_norm:
                x = x.transpose(1, 2)
                
        if self.positionwise_layer_type == 'gau':
            x = self.feed_forward(x, x, x, mask, pos_emb)
        else:
            x = self.feed_forward(x)
        x = residual + self.ff_scale * self.dropout(x)


        if not self.normalize_before:
            if not self.use_layer_norm:
                x = x.transpose(1, 2)
            x = self.norm_ff(x)
            if not self.use_layer_norm:
                x = x.transpose(1, 2)
        
        if self.conv_module is not None:

            if not self.use_layer_norm:
                x = x.transpose(1, 2)
            x = self.norm_final(x)
            if not self.use_layer_norm:
                x = x.transpose(1, 2)

        if alpha != 1.0:
            x = alpha * x + (1 - alpha) * x_orig
        return x, mask


class ReConformerEncoderLayer(nn.Module):
    """Encoder layer module

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward module
    :param feed_forward_macaron (torch.nn.Module): Additional feed-forward module
           instance.
           `PositionwiseFeedForward` instance can be used as the argument.
    :param conv_module (torch.nn.Module): Convolution module instance.
           `ConvlutionModule` instance can be used as the argument.
    :param float dropout_rate: dropout rate
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float=0.1,
        layer_dropout: float=0.075,
        positionwise_layer_type: str="linear",
        concat_after: bool = False
    ):
        super(ReConformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_dropout = layer_dropout
        self.size = size
        self.concat_after = concat_after
        self.positionwise_layer_type = positionwise_layer_type
        self.concat_linear = None
        if self.concat_after:
            self.concat_linear = ScaledLinear(size + size, size)
        self.norm_final = BasicNorm(size)
        # try to ensure the output is close to zero-mean (or at least, zero-median).
        self.balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55, max_abs=6.0
        )
    def forward(
            self, 
            x: torch.Tensor, 
            mask: torch.Tensor,
            pos_emb: torch.Tensor,
            mask_pad: Optional[torch.Tensor] = None,
            warmup: float = 1.0,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoded features

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in, max_time_in)
        :param torch.Tensor pos_emb: 
        :param torch.Tensor mask_pad: batch padding mask used for conv module (batch, 1, time)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        warmup_scale = min(0.1 + warmup, 1.0)
        if self.training:
            alpha = (
                warmup_scale
                if torch.rand(()).item() <= (1.0 - self.layer_dropout)
                else 0.1
            )
        else:
            alpha = 1.0

        x_orig = x
        if self.feed_forward_macaron is not None:
            residual = x

            if self.positionwise_layer_type == 'gau':
                x = self.feed_forward_macaron(x, x, x, mask, pos_emb)
            else:
                x = self.feed_forward_macaron(x)
            x = residual + self.dropout(x)


        # MHA        

        x_att = self.self_attn(x, x, x, mask, pos_emb)

        if self.concat_linear is not None:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = x + self.concat_linear(x_concat)
        else:
            x = x + self.dropout(x_att)


        # convolution module
        if self.conv_module is not None:
            residual = x


            x = self.conv_module(x, mask_pad)
            x = residual + self.dropout(x)

        # FFN
        residual = x

                
        if self.positionwise_layer_type == 'gau':
            x = self.feed_forward(x, x, x, mask, pos_emb)
        else:
            x = self.feed_forward(x)
        x = residual + self.dropout(x)


        x = self.norm_final(self.balancer(x))


        if alpha != 1.0:
            x = alpha * x + (1 - alpha) * x_orig
        return x, mask
