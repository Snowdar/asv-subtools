# -*- coding:utf-8 -*-

# Reference: https://github.com/espnet/espnet.

from tkinter import N
from tkinter.messagebox import NO
import torch
from libs.nnet.activation import Nonlinearity
from .scaling import ActivationBalancer,ScaledLinear

class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward

    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate
    """

    def __init__(self, idim, hidden_units, dropout_rate,activation_type='relu',activation_balancer=False,re_scale=False):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = ScaledLinear(idim, hidden_units) if re_scale else torch.nn.Linear(idim, hidden_units)
        self.w_2 = ScaledLinear(hidden_units, idim,initial_scale=0.25) if re_scale else torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        self.activation = Nonlinearity(activation_type)
        self.balancer = None
        if activation_balancer:
            self.balancer =  ActivationBalancer(channel_dim=-1)
        assert self.activation is not None
    def forward(self, x,x1:torch.Tensor = torch.empty(0),x2:torch.Tensor = torch.empty(0),mask:torch.Tensor = torch.empty(0),pos_embed:torch.Tensor = torch.empty(0)):
        x = self.w_1(x)
        if self.balancer is not None:
            x=self.balancer(x)
        return self.w_2(self.dropout(self.activation(x)))
        # return self.w_2(self.dropout(self.activation(self.w_1(x))))
