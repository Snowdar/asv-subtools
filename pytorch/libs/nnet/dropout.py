# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-03-13)

import numpy as np

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
    def __init__(self, p=0.5, start_p=0., dim=2, method="uniform", std=0.1, inplace=True):
        super(RandomDropout, self).__init__()

        assert 0. <= start_p <= p < 1.

        self.start_p = start_p
        self.p = p
        self.dim = dim
        self.method = method
        self.std = std
        self.inplace = inplace
        self.init_value = torch.tensor(1.)

        if self.dim != 1 and self.dim != 2 and self.dim != 3:
            raise TypeError("Expected dim = 1, 2 or 3, but got {}".format(self.dim))

        if self.method != "uniform" and self.method != "normal":
            raise TypeError("Do not support {} method for random dropout.".format(self.method))

    def forward(self, inputs):
        if self.training and self.p > 0.:
            if self.method == "uniform":
                self.start_p = min(self.start_p, self.p)
                self.init_value.uniform_(self.start_p, self.p)
            else:
                self.init_value = self.init_value.normal_(self.p, self.std).clamp(min=0., max=0.5)

            if self.dim == 1:
                outputs = F.dropout(inputs, self.init_value, inplace=self.inplace)
            elif self.dim == 2:
                outputs = F.dropout2d(inputs, self.init_value, inplace=self.inplace)
            else:
                outputs = F.dropout3d(inputs, self.init_value, inplace=self.inplace)
            return outputs
        else:
            return inputs


class NoiseDropout(torch.nn.Module):
    """Implement noise dropout.
    Reference: [1] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). 
               Dropout: a simple way to prevent neural networks from overfitting. The journal of machine 
               learning research, 15(1), 1929-1958. 

               [2] Li, X., Chen, S., Hu, X., & Yang, J. (2019). Understanding the disharmony between dropout 
               and batch normalization by variance shift. Paper presented at the Proceedings of the IEEE 
               Conference on Computer Vision and Pattern Recognition.
    """
    def __init__(self, p=0.5, dim=2, method="uniform", inplace=True):
        super(NoiseDropout, self).__init__()

        assert 0. <= p < 1.

        self.p = p
        self.dim = dim
        self.method = method
        self.inplace = inplace

        if self.dim != 1 and self.dim != 2 and self.dim != 3:
            raise TypeError("Expected dim = 1, 2 or 3, but got {}".format(self.dim))

        if self.method != "uniform" and self.method != "normal":
            raise TypeError("Do not support {} method for random dropout.".format(self.method))

        if self.method == "normal":
            self.std = (self.p / (1 - self.p))**(0.5)

    def forward(self, inputs):
        if self.training and self.p > 0.:
            input_size = inputs.shape
            if self.dim == 1:
                noise_size = input_size
            if self.dim == 2:
                # Apply the same noise for every frames (a.k.a channels) in one sample.
                noise_size = (input_size[0], input_size[1], 1)
            else:
                noise_size = (input_size[0], input_size[1], 1, 1)

            if self.method == "uniform":
                r = np.random.uniform(-self.p, self.p, size=inputs.shape) + 1
            else:
                r = np.random.normal(1, self.std, size=inputs.shape)

            if self.inplace:
                return inputs.mul_(torch.tensor(r, device=inputs.device))
            else:
                return inputs * torch.tensor(r, device=inputs.device)
        else:
            return inputs


## Wrapper ✿
# Simple name for calling.
def get_dropout(p=0., dropout_params={}):
    return get_dropout_from_wrapper(p=p, dropout_params=dropout_params)

def get_dropout_from_wrapper(p=0., dropout_params={}):

    assert 0. <= p < 1.

    default_dropout_params = {
            "type":"default", # default | random
            "start_p":0.,
            "dim":2,
            "method":"uniform",
            "std":0.1,
            "inplace":True
        }

    dropout_params = utils.assign_params_dict(default_dropout_params, dropout_params)
    name = dropout_params["type"]

    if p == 0:
        return None

    if name == "default":
        return get_default_dropout(p=p, dim=dropout_params["dim"], inplace=dropout_params["inplace"])
    elif name == "random":
        return RandomDropout(p=p, start_p=dropout_params["start_p"], dim=dropout_params["dim"], 
                             method=dropout_params["method"], std=dropout_params["std"], 
                             inplace=dropout_params["inplace"])
    elif name == "alpha":
        return torch.nn.AlphaDropout(p=p, inplace=dropout_params["inplace"])
    elif name == "context":
        return ContextDropout(p=p)
    elif name == "noise":
        return NoiseDropout(p=p, dim=dropout_params["dim"], method=dropout_params["method"],
                            inplace=dropout_params["inplace"])
    else:
        raise TypeError("Do not support {} dropout in current wrapper.".format(name))

def get_default_dropout(p=0., dim=2, inplace=True):
    """Wrapper for torch's dropout.
    """
    if dim == 1:
        return torch.nn.Dropout(p=p, inplace=inplace)
    elif dim == 2:
        return torch.nn.Dropout2d(p=p, inplace=inplace)
    elif dim == 3:
        return torch.nn.Dropout3d(p=p, inplace=inplace)
    else:
        raise TypeError("Expected dim = 1, 2 or 3, but got {}".format(dim))
