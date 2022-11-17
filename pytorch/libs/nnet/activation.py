# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-05-29)

import numpy as np

import torch
import torch.nn.functional as F

from libs.support.utils import to_device

## Activation ✿
class Mish(torch.nn.Module):
    """A changed ReLU. 
    Mish: A Self Regularized Non-Monotonic Neural Activation Function.
    """
    def __init__(self):
         super(Mish, self).__init__()

    def forward(self, inputs):
        return inputs * torch.tanh(F.softplus(inputs))

class Swish(torch.nn.Module):
    """Construct an Swish object."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swish activation function."""
        return x * torch.sigmoid(x)

class DoubleSwishFunction(torch.autograd.Function):
    """
      double_swish(x) = x * torch.sigmoid(x-1)
    This is a definition, originally motivated by its close numerical
    similarity to swish(swish(x)), where swish(x) =  x * sigmoid(x).
    Memory-efficient derivative computation:
     double_swish(x) = x * s, where s(x) = torch.sigmoid(x-1)
     double_swish'(x) = d/dx double_swish(x) =  x * s'(x) + x' * s(x) = x * s'(x) + s(x).
     Now, s'(x) = s(x) * (1-s(x)).
     double_swish'(x) =  x * s'(x) + s(x).
                      =  x * s(x) * (1-s(x)) + s(x).
                     = double_swish(x) * (1-s(x)) + s(x)
     ... so we just need to remember s(x) but not x itself.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        x = x.detach()
        s = torch.sigmoid(x - 1.0)
        y = x * s
        ctx.save_for_backward(s, y)
        return y

    @staticmethod
    def backward(ctx, y_grad: torch.Tensor) -> torch.Tensor:
        s, y = ctx.saved_tensors
        return (y * (1 - s) + s) * y_grad


class DoubleSwish(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return double-swish activation function which is an approximation to Swish(Swish(x)),
        that we approximate closely with x * sigmoid(x-1).
        """
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return x * torch.sigmoid(x - 1.0)
        else:
            return DoubleSwishFunction.apply(x)

## Wrapper ✿
def Nonlinearity(nonlinearity="relu", inplace=True, negative_slope=0.01):
    """A wrapper for activation.
    """
    if nonlinearity == 'relu' :
        activation = torch.nn.ReLU(inplace=inplace)
    elif nonlinearity == 'leaky_relu' :
        activation = torch.nn.LeakyReLU(inplace=inplace, negative_slope=negative_slope)
    elif nonlinearity == 'selu' :
        activation = torch.nn.SELU(inplace=inplace)
    elif nonlinearity == 'mish' :
        activation = Mish()
    elif nonlinearity == 'tanh' :
        activation = torch.nn.Tanh()
    elif nonlinearity == 'swish' :
        func = getattr(torch.nn, "SiLU", Swish)
        activation = func()
    elif nonlinearity == 'gelu' :
        activation = torch.nn.GELU(inplace=inplace)
    elif nonlinearity == 'double_swish' :
        activation = DoubleSwish()
    elif nonlinearity == "" or nonlinearity is None or nonlinearity == False:
        activation = None
    else:
        raise ValueError("Do not support {0} nonlinearity now.".format(nonlinearity))

    return activation
