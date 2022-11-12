# -*- coding: utf-8 -*-

# Reference: https://github.com/espnet/espnet.
# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Layer modules for FFT block in FastSpeech (Feed-forward Transformer)."""

import torch
from libs.nnet.activation import Nonlinearity
from .scaling import ActivationBalancer,ScaledConv1d,ScaledLinear

class MultiLayeredConv1d(torch.nn.Module):
    """Multi-layered conv1d for Transformer block.

    This is a module of multi-leyered conv1d designed to replace positionwise feed-forward network
    in Transforner block, which is introduced in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate,activation_type='relu',activation_balancer=False,re_scale=False):
        """Initialize MultiLayeredConv1d module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.

        """
        super(MultiLayeredConv1d, self).__init__()
        conv = ScaledConv1d if re_scale else torch.nn.Conv1d
        self.w_1 = conv(in_chans, hidden_chans, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = conv(hidden_chans, in_chans, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = Nonlinearity(activation_type)
        self.balancer = None
        if activation_balancer:
            self.balancer =  ActivationBalancer(channel_dim=1)

    def forward(self, x,x1:torch.Tensor = torch.empty(0),x2:torch.Tensor = torch.empty(0),mask:torch.Tensor = torch.empty(0),pos_embed:torch.Tensor = torch.empty(0)):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, *, in_chans).

        Returns:
            Tensor: Batch of output tensors (B, *, hidden_chans)

        """
        x = self.w_1(x.transpose(-1, 1))
        if self.balancer is not None:
            x=self.balancer(x)
        x = self.activation(x).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)


class Conv1dLinear(torch.nn.Module):
    """Conv1D + Linear for Transformer block.

    A variant of MultiLayeredConv1d, which replaces second conv-layer to linear.

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate,activation_type='relu',activation_balancer=False,re_scale=False):
        """Initialize Conv1dLinear module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.

        """
        super(Conv1dLinear, self).__init__()
        conv = ScaledConv1d if re_scale else torch.nn.Conv1d

        self.w_1 = conv(in_chans, hidden_chans, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = ScaledLinear(hidden_chans, in_chans,initial_scale=0.25) if re_scale else torch.nn.Linear(hidden_chans, in_chans)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = Nonlinearity(activation_type)
        self.balancer = None
        if activation_balancer:
            self.balancer =  ActivationBalancer(channel_dim=1)
    def forward(self, x,x1:torch.Tensor = torch.empty(0),x2:torch.Tensor = torch.empty(0),mask:torch.Tensor = torch.empty(0),pos_embed:torch.Tensor = torch.empty(0)):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, *, in_chans).

        Returns:
            Tensor: Batch of output tensors (B, *, hidden_chans)

        """
        x = self.w_1(x.transpose(-1, 1))
        if self.balancer is not None:
            x=self.balancer(x)
        x = self.activation(x).transpose(-1, 1)
        return self.w_2(self.dropout(x))

