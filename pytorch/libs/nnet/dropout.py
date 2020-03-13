# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-03-13)

import torch
import torch.nn.functional as F

import libs.support.utils as utils

## Dropout ✿
class ContextDropout(torch.nn.Module):
    """It dropouts in the context dimensionality to achieve two purposes:
           1.make training with random chunk-length;
           2.decrease the context dependence to augment the training data.
        It is still not available now.
    """
    def __init__(self, p=0.):
        super(ContextDropout, self).__init__()

        self.dropout2d = torch.nn.Dropout2d(p=p)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        outputs = self.dropout2d(inputs.transpose(1,2)).transpose(1,2)
        return outputs


class RandomDropout(torch.nn.Module):
    """Implement random dropout.
    Reference: Bouthillier, X., Konda, K., Vincent, P., & Memisevic, R. (2015). 
               Dropout as data augmentation. arXiv preprint arXiv:1506.08700. 
    """
    def __init__(self, p=0.5, dim=2, inplace=True):
        super(RandomDropout, self).__init__()

        self.p = p
        self.dim = dim
        self.inplace = inplace
        self.init_value = torch.tensor(1.)

        if self.dim != 1 and self.dim != 2 and self.dim != 3:
            raise TypeError("Expected dim = 1, 2 or 3, but got {}".format(self.dim))

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        if self.training and self.p > 0:
            self.init_value.uniform_(0, self.p)
            if self.dim == 1:
                outputs = F.dropout(inputs, self.init_value, inplace=self.inplace)
            elif self.dim == 2:
                outputs = F.dropout2d(inputs, self.init_value, inplace=self.inplace)
            else:
                outputs = F.dropout3d(inputs, self.init_value, inplace=self.inplace)
            return outputs
        else:
            return inputs


## Wrapper ✿
# Simple name for calling.
def get_dropout(p=0., dropout_params={}):
    return get_dropout_from_wrapper(p=p, dropout_params=dropout_params)

def get_dropout_from_wrapper(p=0., dropout_params={}):

    assert 0 <= p <= 1

    default_dropout_params = {
            "type":"default", # default | random
            "dim":2,
            "inplace":True,
        }

    dropout_params = utils.assign_params_dict(default_dropout_params, dropout_params)
    name = dropout_params["type"]

    if p == 0:
        return None

    if name == "default":
        return get_default_dropout(p=p, dim=dropout_params["dim"], inplace=dropout_params["inplace"])
    elif name == "random":
        return RandomDropout(p=p, dim=dropout_params["dim"], inplace=dropout_params["inplace"])
    elif name == "alpha":
        return torch.nn.AlphaDropout(p=p, inplace=dropout_params["inplace"])
    elif name == "context":
        return ContextDropout(p=p)
    else:
        raise TypeError("Do not support {} dropout in current wrapper.".format(name))

def get_default_dropout(p=0., dim=2):
    """Wrapper for torch's dropout.
    """
    if dim == 1:
        return torch.nn.Dropout(p=p)
    elif dim == 2:
        return torch.nn.Dropout2d(p=p)
    elif dim == 3:
        return torch.nn.Dropout3d(p=p)
    else:
        raise TypeError("Expected dim = 1, 2 or 3, but got {}".format(dim))