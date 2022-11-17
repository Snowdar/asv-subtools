# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2022-07)
# Reference: https://github.com/wenet-e2e/wenet
import torch
import math
from typing import Tuple, List, Optional
from .attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
    RoPESelfAttention,
    RoPEGAU,
    GAU
)
from .embedding import (
    PositionalEncoding,
    RelPositionalEncoding,
    NoPositionalEncoding,
    RoPositionalEncoding
    )
from .encoder_layer import (
    TransformerEncoderLayer,
    ConformerEncoderLayer,
    ReConformerEncoderLayer
)
from .layer_norm import BasicNorm, LayerNorm,Trans_Bat
from .multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from .convolution import ConvolutionModule, ReConvolutionModule
from .positionwise_feed_forward import PositionwiseFeedForward
from .subsampling import (
    LinearNoSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling4,
    ReConv2dSubsampling4,
    SVConv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    SVConv2dSubsampling2,
    FrameMerger
)
import torch.nn as nn
from .mask import add_optional_chunk_mask
from .mask import make_pad_mask
from libs.nnet.activation import Nonlinearity

class BaseEncoder(torch.nn.Module):
    """
    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of encoder blocks
    :param int aux_layer_period: the period for randomcombiner/mfa.
    :param int aux_layer_start: the start position for randomcombiner/mfa.
        default :1. e.g. 3 means from 1/3 of the total block nums suggest for randomcombiner, mfa suggest set to 1+num_blocks.  
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str input_layer: input layer type
    :param str pos_enc_type: Encoder positional encoding layer type.
        opitonal [abs_pos, rel_pos, no_pos, rot_pos]
    :param bool rotary_value: whether apply rot_pos on value vector when use rot_pos, which contains abs position information.
    :param bool rope_abs_plus: whether apply abs_pos when use rot_pos.
    :param bool add_t5rel_bias: whether apply t5_rel position in attention score.   
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param str positionwise_layer_type: positionwise feedforward type
    :param int positionwise_conv_kernel_size: Kernel size of positionwise conv1d layer.
    :param int static_chunk_size: chunk size for static chunk training and decoding.
    :param bool use_dynamic_chunk: whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dynamic chunk size(use_dynamic_chunk = True)
    :param bool use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training    
    :param str comnbiner_type: combine the output of encoder with its sublayers.
        opitonal [norm, mfa, random_frame, random_layer] 
    """
    aux_layers: List[int]
    def __init__(self, idim,
                 attention_dim=256,
                 attention_heads=4,
                 linear_units=2048,
                 mlp_head=False,
                 num_blocks=6,
                 aux_layer_period = 3,
                 aux_layer_start = 1,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0,
                 attention_conv_out = False,
                 attention_norm_args = {},
                 input_layer="conv2d",
                 pos_enc_type="abs_pos",
                 rotary_value=True,
                 rope_abs_plus = False,
                 add_t5rel_bias = False,
                 att_type: str= 'multi',
                 gau_units: int = 512,
                 gau_key: int = 64,
                 normalize_before=True,
                 norm_type = "layer_norm",
                 concat_after=False,
                 positionwise_layer_type="linear",
                 positionwise_conv_kernel_size=1,
                 activation_type='relu',
                 activation_balancer=False,
                 static_chunk_size: int = 0,
                 left_chunk_size: int = -1,
                 use_dynamic_chunk: bool = False,
                 use_dynamic_left_chunk: bool = False,
                 combiner_type: str="norm",
                 re_scale: bool=False):
        super().__init__()

        
        self.att_type = att_type
        self.mlp_head = mlp_head


        if att_type != 'gau' and positionwise_layer_type == 'gau':
            assert gau_key == attention_dim // attention_heads
        pos_enc_dict = {}
        if pos_enc_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_type == "rot_pos":
            pos_enc_class = RoPositionalEncoding
            pos_enc_dict['rope_abs_plus']= rope_abs_plus
            if self.att_type == 'gau':
                pos_enc_dict['d_roembed']= gau_key
            else:
                pos_enc_dict['att_h']= attention_heads
        elif pos_enc_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_type)

        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
        elif input_layer == "conv2d2":
            subsampling_class = SVConv2dSubsampling2
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "re_conv2d":
            subsampling_class = ReConv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.embed = subsampling_class(idim, attention_dim, dropout_rate,mlp_head,pos_enc_class(attention_dim,positional_dropout_rate, **pos_enc_dict))
        self.pos_enc_type = pos_enc_type
        self.positionwise_layer, self.positionwise_layer_args = self.get_positionwise_layer(
            positionwise_layer_type,
            attention_dim,
            linear_units,
            dropout_rate,
            positionwise_conv_kernel_size,
            activation_type,
            activation_balancer,
            re_scale,
            attention_norm_args=attention_norm_args
        )

        if self.att_type == 'gau':
            self.selfattn_layer, self.selfattn_layer_args = self.get_gau_layer(
                pos_enc_type,
                attention_dim,
                gau_units,
                gau_key,
                attention_dropout_rate,
                attention_conv_out,
                attention_norm_args=attention_norm_args,
                re_scale=re_scale
            )
        else:
            self.selfattn_layer, self.selfattn_layer_args = self.get_selfattn_layer(
                pos_enc_type,
                attention_heads,
                attention_dim,
                attention_dropout_rate,
                add_t5rel_bias,
                attention_conv_out,
                rotary_value,
                attention_norm_args=attention_norm_args,
                re_scale=re_scale
            )

        self.aux_layers,self.combiner = self.get_combiner(
            num_blocks,
            aux_layer_period,
            aux_layer_start,
            combiner_type
        )

        self.normalize_before = normalize_before
        self._output_size = attention_dim*len(self.aux_layers) if combiner_type=="mfa" else attention_dim   
        self.after_norm = None
        if self.normalize_before or combiner_type=="mfa":
            if norm_type == "batch_norm":
                self.use_layer_norm=False
                self.after_norm = Trans_Bat(self._output_size)
            else:
                self.use_layer_norm=True
                if norm_type == "layer_norm":
                    self.after_norm = LayerNorm(self._output_size,eps=1e-5)
                else:
                    self.after_norm = BasicNorm(self._output_size) 
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.left_chunk_size = left_chunk_size        

    def get_positionwise_layer(
        self,
        positionwise_layer_type="linear",
        attention_dim=256,
        linear_units=2048,
        dropout_rate=0.1,
        positionwise_conv_kernel_size=1,
        activation_type='relu',
        activation_balancer=False,
        re_scale = False,
        gau_key = 64,
        attention_norm_args = {}
    ):
        """Define positionwise layer."""
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate, activation_type, activation_balancer,re_scale)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
                activation_type,
                activation_balancer,
                re_scale
            )

        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
                activation_type,
                activation_balancer,
                re_scale
            )

        elif positionwise_layer_type == "gau":
            positionwise_layer, positionwise_layer_args = self.get_gau_layer(
                self.pos_enc_type,
                attention_dim,
                linear_units,
                gau_key,
                dropout_rate,
                attention_norm_args = attention_norm_args,
                re_scale=re_scale
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        return positionwise_layer, positionwise_layer_args


    def get_selfattn_layer(
        self,
        pos_enc_type="abs_pos",
        attention_heads=4,
        attention_dim=256,
        attention_dropout_rate=0.0,
        add_t5rel_bias=False,
        conv_out=False,
        rotary_value = False,
        attention_norm_args={},
        re_scale=False
    ):
        """Define selfattn layer."""
        selfattn_layer_args = (attention_heads,attention_dim,attention_dropout_rate,add_t5rel_bias,conv_out,attention_norm_args,re_scale)

        if pos_enc_type == "rel_pos":
            selfattn_layer = RelPositionMultiHeadedAttention
        elif pos_enc_type == "rot_pos":
            selfattn_layer = RoPESelfAttention
            selfattn_layer_args=(attention_heads,attention_dim,attention_dropout_rate,add_t5rel_bias,conv_out,attention_norm_args,rotary_value,re_scale)
        else:
            selfattn_layer = MultiHeadedAttention

        return selfattn_layer, selfattn_layer_args 

    def get_gau_layer(
        self,
        pos_enc_type="abs_pos",
        attention_dim=256,
        hidden_dim = 512,
        d_qk = 64,
        attention_dropout_rate=0.0,
        conv_out=False,
        attention_norm_args={},
        re_scale=False
    ):
        """Define gau layer."""

        if pos_enc_type == "abs_pos":
            selfattn_layer = GAU
        else: 
            selfattn_layer = RoPEGAU
        selfattn_layer_args=(attention_dim,hidden_dim,d_qk,attention_dropout_rate,conv_out,attention_norm_args,re_scale)

        return selfattn_layer, selfattn_layer_args 

    def get_combiner(self,
                     num_blocks: int, 
                     aux_layer_period: int = 3,
                     aux_layer_start: int = 1,
                     combiner_type="norm"
                     ) -> Tuple[List[int],torch.nn.Module]:
        """Define combiner layer."""
        assert combiner_type in ["norm", "mfa", "random_frame", "random_layer"], "unknown combiner_type {}".format(combiner_type)
        assert aux_layer_period,aux_layer_start > 0 
        assert num_blocks > 0
        aux_layers=list(
                range(
                    num_blocks // aux_layer_start,
                    num_blocks - 1,
                    aux_layer_period,
                )
        )        
        assert len(set(aux_layers)) == len(aux_layers)
        assert num_blocks - 1 not in aux_layers
        aux_layers = aux_layers + [num_blocks - 1]
        combiner = RandomCombine(
            aux_layers=aux_layers,
            combiner_type=combiner_type,
            final_weight=0.5,
            pure_prob=0.333,
            stddev=2.0,
        )
        return aux_layers,combiner

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = -1,
        decoding_left_chunk_size: int = -2,
        warmup: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor

        :param torch.Tensor xs: input tensor (B, T, D)
        :param torch.Tensor xs_lens: input length (B)
        :param int decoding_chunk_size: decoding chunk size for dynamic chunk
                 0: use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
        :param int decoding_left_chunk_size: 
            <-1: default , the number of left chunks is self.left_chunk_size.
             -1: use full left chunk
              0: no left chunk
             >0: the number of left chunks
        : param float warmup: 
            Model level warmup, a floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up".  It is used
            to turn modules on sequentially.
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if decoding_left_chunk_size <= -2:
            left_chunk_size = self.left_chunk_size
        else:
            left_chunk_size = decoding_left_chunk_size
        T = xs.size(1)
        if self.mlp_head:
            T+=1

            xs_lens+=1

        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)

        xs, pos_emb, masks = self.embed(xs, masks)

        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              left_chunk_size)
        
        out= []

        for i,layer in enumerate(self.encoders):
            xs, chunk_masks = layer(xs, chunk_masks, pos_emb, mask_pad, warmup = warmup)
            
            if i in self.aux_layers:
                out.append(xs)
        if len(out)>0:
            xs = self.combiner(out)

        if self.after_norm is not None:
            if not self.use_layer_norm:
                xs = xs.transpose(1, 2)    
            xs = self.after_norm(xs)
            if not self.use_layer_norm:
                xs = xs.transpose(1, 2)
        return xs, masks


class TransformerEncoder(BaseEncoder):
    """Transformer encoder module."""
    def __init__(
        self,
        idim: int,
        attention_dim: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        mlp_head: bool=False,
        num_blocks: int = 6,
        aux_layer_period: int = 3,
        aux_layer_start:int = 1,
        dropout_rate: float = 0.1,
        layer_dropout: float = 0.,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        attention_conv_out : bool = False,
        attention_norm_args : dict = {},
        input_layer: str = "conv2d",
        pos_enc_type: str = "abs_pos",
        rotary_value: bool = True,
        rope_abs_plus: bool = False,
        add_t5rel_bias: bool = False,
        att_type: str= 'multi',
        gau_units: int = 512,
        gau_key: int = 64,
        normalize_before: bool = True,
        norm_type: str = "layer_norm",
        concat_after: bool = False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        activation_type='relu',
        activation_balancer: bool=False,
        static_chunk_size: int = 0,
        left_chunk_size: int = -1,
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
        combiner_type: str="norm",
        re_scale: bool=False,
        convfnn_blocks: int=0,
        **args
    ):
        """ Construct TransformerEncoder

        See Encoder for the meaning of each parameter.
        """

        super().__init__(idim, attention_dim, attention_heads, linear_units, 
                         mlp_head, num_blocks, aux_layer_period, aux_layer_start, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate, attention_conv_out,
                         attention_norm_args, input_layer, pos_enc_type, rotary_value, rope_abs_plus, 
                         add_t5rel_bias, att_type, gau_units, gau_key, normalize_before, norm_type, 
                         concat_after, positionwise_layer_type, positionwise_conv_kernel_size, 
                         activation_type, activation_balancer, static_chunk_size, left_chunk_size, 
                         use_dynamic_chunk, use_dynamic_left_chunk,combiner_type, re_scale)
        pre_selfattn_layer, pre_selfattn_layer_args = self.selfattn_layer,self.selfattn_layer_args
        if positionwise_layer_type == "gau":
            pre_positionwise_layer,pre_positionwise_layer_args = self.get_gau_layer(
                self.pos_enc_type,
                attention_dim,
                linear_units,
                gau_key,
                dropout_rate,
                conv_out =True,
                attention_norm_args = attention_norm_args,
                re_scale=re_scale
            )

            pre_selfattn_layer, pre_selfattn_layer_args = self.get_gau_layer(
                pos_enc_type,
                attention_dim,
                gau_units,
                gau_key,
                attention_dropout_rate,
                conv_out =True,
                attention_norm_args=attention_norm_args,
                re_scale=re_scale
            )
        else:
            pre_positionwise_layer = MultiLayeredConv1d
            pre_positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
                activation_type,
                activation_balancer,
                re_scale
            )

        encoders = []
        for _ in range(convfnn_blocks):
            encoders.append(TransformerEncoderLayer(
                attention_dim,
                pre_selfattn_layer(*pre_selfattn_layer_args),
                pre_positionwise_layer(*pre_positionwise_layer_args),
                dropout_rate,
                layer_dropout,
                normalize_before,
                norm_type,
                positionwise_layer_type,
                concat_after,
            ))
        for _ in range(num_blocks-convfnn_blocks):
            encoders.append(TransformerEncoderLayer(
              attention_dim,
                self.selfattn_layer(*self.selfattn_layer_args),
                self.positionwise_layer(*self.positionwise_layer_args),dropout_rate,
                layer_dropout,normalize_before, norm_type,positionwise_layer_type,concat_after))

        self.encoders = torch.nn.ModuleList(encoders)        


class ConformerEncoder(BaseEncoder):
    """Conformer encoder module."""
    def __init__(
        self,
        idim: int,
        attention_dim: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        mlp_head: bool=False,
        num_blocks: int = 6,
        aux_layer_period: int = 3,
        aux_layer_start: int = 1,
        dropout_rate: float = 0.1,
        layer_dropout: float = 0.,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        attention_conv_out : bool = False,
        attention_norm_args : dict = {},
        input_layer: str = "conv2d",
        pos_enc_type: str = "rel_pos",
        rotary_value: bool = True,
        rope_abs_plus: bool = False,
        add_t5rel_bias: bool = False,
        att_type: str= 'multi',
        gau_units: int = 512,
        gau_key: int = 64,
        normalize_before: bool = True,
        norm_type: str = "layer_norm",
        concat_after: bool = False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=3,
        activation_type: str = "swish",
        activation_balancer: bool=False,
        static_chunk_size: int = 0,
        left_chunk_size: int = -1,
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
        macaron_style: bool = True,
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        combiner_type: str="norm",
        re_scale: bool=False,
        convfnn_blocks: int=0,
        **args
    ):
        """Construct ConformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
        """

        super().__init__(idim, attention_dim, attention_heads, linear_units, 
                         mlp_head, num_blocks, aux_layer_period, aux_layer_start, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate, attention_conv_out,
                         attention_norm_args,input_layer, pos_enc_type, rotary_value, rope_abs_plus, 
                         add_t5rel_bias, att_type, gau_units, gau_key, normalize_before, norm_type, 
                         concat_after, positionwise_layer_type, positionwise_conv_kernel_size, 
                         activation_type, activation_balancer,static_chunk_size, left_chunk_size, 
                         use_dynamic_chunk, use_dynamic_left_chunk,combiner_type,re_scale)

        activation = Nonlinearity(activation_type)
        assert activation is not None
        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation,
                                  cnn_module_norm, causal, activation_balancer)
        pre_selfattn_layer, pre_selfattn_layer_args = self.selfattn_layer,self.selfattn_layer_args
        if positionwise_layer_type == "gau":
            pre_positionwise_layer,pre_positionwise_layer_args = self.get_gau_layer(
                self.pos_enc_type,
                attention_dim,
                linear_units,
                gau_key,
                dropout_rate,
                conv_out =True,
                attention_norm_args = attention_norm_args,
                re_scale=re_scale
            )

            pre_selfattn_layer, pre_selfattn_layer_args = self.get_gau_layer(
                pos_enc_type,
                attention_dim,
                gau_units,
                gau_key,
                attention_dropout_rate,
                conv_out =True,
                attention_norm_args=attention_norm_args,
                re_scale=re_scale
            )
        else:
            pre_positionwise_layer = MultiLayeredConv1d
            pre_positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
                activation_type,
                activation_balancer,
                re_scale
            )

        encoders = []
        for _ in range(convfnn_blocks):
            encoders.append(ConformerEncoderLayer(
                attention_dim,
                pre_selfattn_layer(*pre_selfattn_layer_args),
                pre_positionwise_layer(*pre_positionwise_layer_args),
                pre_positionwise_layer(
                    *pre_positionwise_layer_args) if macaron_style else None,
                convolution_layer(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                layer_dropout,
                normalize_before,
                norm_type,
                positionwise_layer_type,
                concat_after,
            ))

        for _ in range(num_blocks-convfnn_blocks):
            encoders.append(ConformerEncoderLayer(
                attention_dim,
                self.selfattn_layer(*self.selfattn_layer_args),
                self.positionwise_layer(*self.positionwise_layer_args),
                self.positionwise_layer(
                    *self.positionwise_layer_args) if macaron_style else None,
                convolution_layer(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                layer_dropout,
                normalize_before,
                norm_type,
                positionwise_layer_type,
                concat_after,
            ))

        self.encoders = torch.nn.ModuleList(encoders)

class ReConformerEncoder(BaseEncoder):
    """Conformer encoder module."""
    def __init__(
        self,
        idim: int,
        attention_dim: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        mlp_head: bool=False,
        num_blocks: int = 6,
        aux_layer_period: int = 3,
        aux_layer_start: int = 1,
        dropout_rate: float = 0.1,
        layer_dropout: float = 0.,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        attention_conv_out : bool = False,
        attention_norm_args : dict = {},
        input_layer: str = "re_conv2d",
        pos_enc_type: str = "rel_pos",
        rotary_value: bool = True,
        rope_abs_plus: bool = False,
        add_t5rel_bias: bool = False,
        att_type: str= 'multi',
        gau_units: int = 512,
        gau_key: int = 64,
        normalize_before: bool = False,
        norm_type: str = "basic_norm",
        concat_after: bool = False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=3,
        activation_type: str = "double_swish",
        activation_balancer: bool=True,
        static_chunk_size: int = 0,
        left_chunk_size: int = -1,
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
        macaron_style: bool = True,
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        combiner_type: str="norm",
        re_scale: bool=True,
        convfnn_blocks: int=0,
        **args
    ):
        """Construct ConformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
        """
        assert normalize_before == False,"set normalize_before=False."
        assert norm_type == "basic_norm","set norm_type=basic_norm."
        assert re_scale == True, "reconformer set res_scale=True."
        assert activation_balancer == True
        super().__init__(idim, attention_dim, attention_heads, linear_units, 
                         mlp_head, num_blocks, aux_layer_period, aux_layer_start, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate, attention_conv_out,
                         attention_norm_args,input_layer, pos_enc_type, rotary_value, rope_abs_plus, 
                         add_t5rel_bias, att_type, gau_units, gau_key, normalize_before, norm_type, 
                         concat_after, positionwise_layer_type, positionwise_conv_kernel_size, 
                         activation_type, activation_balancer,static_chunk_size, left_chunk_size, 
                         use_dynamic_chunk, use_dynamic_left_chunk,combiner_type,re_scale)

        activation = Nonlinearity(activation_type)
        assert activation is not None
        # convolution module definition
        convolution_layer = ReConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation,
                                  causal)
        pre_selfattn_layer, pre_selfattn_layer_args = self.selfattn_layer,self.selfattn_layer_args
        if positionwise_layer_type == "gau":
            pre_positionwise_layer,pre_positionwise_layer_args = self.get_gau_layer(
                self.pos_enc_type,
                attention_dim,
                linear_units,
                gau_key,
                dropout_rate,
                conv_out =True,
                attention_norm_args = attention_norm_args,
                re_scale=re_scale
            )

            pre_selfattn_layer, pre_selfattn_layer_args = self.get_gau_layer(
                pos_enc_type,
                attention_dim,
                gau_units,
                gau_key,
                attention_dropout_rate,
                conv_out =True,
                attention_norm_args=attention_norm_args,
                re_scale=re_scale
            )
        else:
            pre_positionwise_layer = MultiLayeredConv1d
            pre_positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
                activation_type,
                activation_balancer,
                re_scale
            )

        encoders = []
        for _ in range(convfnn_blocks):
            encoders.append(ReConformerEncoderLayer(
                attention_dim,
                pre_selfattn_layer(*pre_selfattn_layer_args),
                pre_positionwise_layer(*pre_positionwise_layer_args),
                pre_positionwise_layer(
                    *pre_positionwise_layer_args) if macaron_style else None,
                convolution_layer(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                layer_dropout,
                positionwise_layer_type,
                concat_after,
            ))

        for _ in range(num_blocks-convfnn_blocks):
            encoders.append(ReConformerEncoderLayer(
                attention_dim,
                self.selfattn_layer(*self.selfattn_layer_args),
                self.positionwise_layer(*self.positionwise_layer_args),
                self.positionwise_layer(
                    *self.positionwise_layer_args) if macaron_style else None,
                convolution_layer(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                layer_dropout,
                positionwise_layer_type,
                concat_after,
            ))

        self.encoders = torch.nn.ModuleList(encoders)



# RandomCombine conformer in k2, for deeper training.
# https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless5/conformer.py
class RandomCombine(nn.Module):
    """
    This module combines a list of Tensors, all with the same shape, to
    produce a single output of that same shape which, in training time,
    is a random combination of all the inputs; but which in test time
    will be just the last input.
    The idea is that the list of Tensors will be a list of outputs of multiple
    conformer layers.  This has a similar effect as iterated loss. (See:
    DEJA-VU: DOUBLE FEATURE PRESENTATION AND ITERATED LOSS IN DEEP TRANSFORMER
    NETWORKS).
    """

    def __init__(
        self,
        aux_layers: list,
        combiner_type: str = "norm",
        final_weight: float = 0.5,
        pure_prob: float = 0.5,
        stddev: float = 2.0,
    ) -> None:
        """
        Args:
          num_inputs:
            The number of tensor inputs, which equals the number of layers'
            outputs that are fed into this module.  E.g. in an 18-layer neural
            net if we output layers 16, 12, 18, num_inputs would be 3.
          final_weight:
            The amount of weight or probability we assign to the
            final layer when randomly choosing layers or when choosing
            continuous layer weights.
          pure_prob:
            The probability, on each frame, with which we choose
            only a single layer to output (rather than an interpolation)
          stddev:
            A standard deviation that we add to log-probs for computing
            randomized weights.
        The method of choosing which layers, or combinations of layers, to use,
        is conceptually as follows::
            With probability `pure_prob`::
               With probability `final_weight`: choose final layer,
               Else: choose random non-final layer.
            Else::
               Choose initial log-weights that correspond to assigning
               weight `final_weight` to the final layer and equal
               weights to other layers; then add Gaussian noise
               with variance `stddev` to these log-weights, and normalize
               to weights (note: the average weight assigned to the
               final layer here will not be `final_weight` if stddev>0).
        """
        super().__init__()
        self.num_inputs = len(aux_layers)
        self.aux_layers = aux_layers

        assert self.num_inputs >= 1
        if combiner_type in ["random_frame", "random_layer"]:
            assert 0 <= pure_prob <= 1, pure_prob
            assert 0 < final_weight < 1, final_weight

            self.final_weight = final_weight
            self.pure_prob = pure_prob
            self.stddev = stddev

            self.final_log_weight = (
                torch.tensor(
                    (final_weight / (1 - final_weight)) * (self.num_inputs - 1)
                )
                .log()
                .item()
            )
        self.combiner_type = combiner_type
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
            
        if self.combiner_type == "mfa":
            return self.forward_mfa(inputs)
        elif self.combiner_type == "random_frame":
            return self.forward_rand_frame(inputs)
        elif self.combiner_type == "random_layer":
            return self.forward_rand_layer(inputs)
        return self.forward_norm(inputs)

    def forward_mfa(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(inputs,dim=-1)

    def forward_norm(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return inputs[-1]

    def forward_rand_frame(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward function.
        Args:
          inputs:
            A list of Tensor, e.g. from various layers of a transformer.
            All must be the same shape, of (*, num_channels)
        Returns:
          A Tensor of shape (*, num_channels).  In test mode
          this is just the final input.
        """
        num_inputs = self.num_inputs
        assert len(inputs) == num_inputs
        if not self.training or torch.jit.is_scripting() or self.num_inputs==1:
            return inputs[-1]

        # Shape of weights: (*, num_inputs)
        num_channels = inputs[0].shape[-1]
        num_frames = inputs[0].numel() // num_channels

        ndim = inputs[0].ndim
        # stacked_inputs: (num_frames, num_channels, num_inputs)
        stacked_inputs = torch.stack(inputs, dim=ndim).reshape(
            (num_frames, num_channels, num_inputs)
        )

        # weights: (num_frames, num_inputs)
        weights = self._get_random_weights(
            inputs[0].dtype, inputs[0].device, num_frames
        )

        weights = weights.reshape(num_frames, num_inputs, 1)
        # ans: (num_frames, num_channels, 1)
        ans = torch.matmul(stacked_inputs, weights)
        # ans: (*, num_channels)

        ans = ans.reshape(inputs[0].shape[:-1] + (num_channels,))

        # The following if causes errors for torch script in torch 1.6.0
        #  if __name__ == "__main__":
        #      # for testing only...
        #      print("Weights = ", weights.reshape(num_frames, num_inputs))
        return ans

    def forward_rand_layer(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward function.
        Args:
          inputs:
            A list of Tensor, e.g. from various layers of a transformer.
            All must be the same shape, of (B, T,  C)
        Returns:
          A Tensor of shape (B, T,  C).  In test mode
          this is just the final input.
        """
        num_inputs = self.num_inputs
        assert len(inputs) == num_inputs
        if not self.training or torch.jit.is_scripting() or self.num_inputs==1:
            return inputs[-1]



        num_channels = inputs[0].shape[-1]
        num_b = inputs[0].shape[0]

        ndim = inputs[0].ndim
        # stacked_inputs: (B, T, C, num_inputs)
        stacked_inputs = torch.stack(inputs, dim=ndim)

        # weights: (B, num_inputs)
        weights = self._get_random_weights(
            inputs[0].dtype, inputs[0].device, num_b
        )

        weights = weights.reshape(num_b,1, num_inputs, 1)

        # ans: (B, T, C, 1)
        ans = torch.matmul(stacked_inputs, weights)

        # ans: (B, T, C)
        ans = ans.reshape(inputs[0].shape[:-1] + (num_channels,))

        # The following if causes errors for torch script in torch 1.6.0
        #  if __name__ == "__main__":
        #      # for testing only...
        #      print("Weights = ", weights.reshape(num_frames, num_inputs))
        return ans

    def _get_random_weights(
        self, dtype: torch.dtype, device: torch.device, num_frames: int
    ) -> torch.Tensor:
        """Return a tensor of random weights, of shape
        `(num_frames, self.num_inputs)`,
        Args:
          dtype:
            The data-type desired for the answer, e.g. float, double.
          device:
            The device needed for the answer.
          num_frames:
            The number of sets of weights desired
        Returns:
          A tensor of shape (num_frames, self.num_inputs), such that
          `ans.sum(dim=1)` is all ones.
        """
        pure_prob = self.pure_prob
        if pure_prob == 0.0:
            return self._get_random_mixed_weights(dtype, device, num_frames)
        elif pure_prob == 1.0:
            return self._get_random_pure_weights(dtype, device, num_frames)
        else:
            p = self._get_random_pure_weights(dtype, device, num_frames)
            m = self._get_random_mixed_weights(dtype, device, num_frames)
            return torch.where(
                torch.rand(num_frames, 1, device=device) < self.pure_prob, p, m
            )

    def _get_random_pure_weights(
        self, dtype: torch.dtype, device: torch.device, num_frames: int
    ):
        """Return a tensor of random one-hot weights, of shape
        `(num_frames, self.num_inputs)`,
        Args:
          dtype:
            The data-type desired for the answer, e.g. float, double.
          device:
            The device needed for the answer.
          num_frames:
            The number of sets of weights desired.
        Returns:
          A one-hot tensor of shape `(num_frames, self.num_inputs)`, with
          exactly one weight equal to 1.0 on each frame.
        """
        final_prob = self.final_weight

        # final contains self.num_inputs - 1 in all elements
        final = torch.full((num_frames,), self.num_inputs - 1, device=device)
        # nonfinal contains random integers in [0..num_inputs - 2], these are for non-final weights.
        nonfinal = torch.randint(
            self.num_inputs - 1, (num_frames,), device=device
        )

        indexes = torch.where(
            torch.rand(num_frames, device=device) < final_prob, final, nonfinal
        )
        ans = torch.nn.functional.one_hot(
            indexes, num_classes=self.num_inputs
        ).to(dtype=dtype)
        return ans

    def _get_random_mixed_weights(
        self, dtype: torch.dtype, device: torch.device, num_frames: int
    ):
        """Return a tensor of random one-hot weights, of shape
        `(num_frames, self.num_inputs)`,
        Args:
          dtype:
            The data-type desired for the answer, e.g. float, double.
          device:
            The device needed for the answer.
          num_frames:
            The number of sets of weights desired.
        Returns:
          A tensor of shape (num_frames, self.num_inputs), which elements
          in [0..1] that sum to one over the second axis, i.e.
          `ans.sum(dim=1)` is all ones.
        """
        logprobs = (
            torch.randn(num_frames, self.num_inputs, dtype=dtype, device=device)
            * self.stddev
        )
        logprobs[:, -1] += self.final_log_weight
        return logprobs.softmax(dim=1)

    def extra_repr(self):
        s = ('{combiner_type}, num_inputs_layer={num_inputs}, aux_layers={aux_layers}')
        if "random" in self.combiner_type:
            s += ', final_weight={final_weight}, pure_prob={pure_prob}, stddev={stddev}, final_log_weight={final_log_weight}'
        return s.format(**self.__dict__)

# def scale_args(layer_args,num=2,scale=2):
    # new_args=[]
    # for i,arg in enumerate(layer_args):
        # if i<num:
            # new_args.append(arg*scale)
        # else:
            # new_args.append(arg)
    # return new_args
