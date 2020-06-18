# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-03-13)

import numpy as np

import torch
import torch.nn.functional as F

import libs.support.utils as utils

## Dropout ✿
class ContextDropout(torch.nn.Module):
    """It dropouts values in the context (frame/time) dimensionality. 
    Different to specaugment (see libs/egs/augmentation.py), it is not continuous.
    """
    def __init__(self, p=0.):
        super(ContextDropout, self).__init__()

        self.p = p
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
    def __init__(self, p=0.5, start_p=0., dim=2, method="uniform", inplace=True):
        super(RandomDropout, self).__init__()

        assert 0. <= start_p <= p < 1.

        self.start_p = start_p
        self.p = p
        self.dim = dim
        self.method = method
        self.inplace = inplace
        self.init_value = torch.tensor(1.)

        if self.dim != 1 and self.dim != 2 and self.dim != 3:
            raise TypeError("Expected dim = 1, 2 or 3, but got {}".format(self.dim))

        if self.method != "uniform" and self.method != "normal":
            raise TypeError("Do not support {} method for random dropout.".format(self.method))

        if self.method == "normal":
            self.mean = self.p / 2
            self.std = 0.01**self.p
        

    def forward(self, inputs):
        if self.training and self.p > 0.:
            if self.method == "uniform":
                # For step training when p decay to mini value.
                if self.start_p > self.p:
                    self.start_p = self.p
                self.init_value.uniform_(self.start_p, self.p)
            else:
                # Only take (0, self.p) of the gaussian curve.
               self.init_value = self.init_value.normal_(self.mean, self.std).clamp(min=0., max=self.p)

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

               [3] Shen, X., Tian, X., Liu, T., Xu, F., & Tao, D. (2017). Continuous dropout. IEEE transactions 
               on neural networks and learning systems, 29(9), 3926-3937. 

    """
    def __init__(self, p=0.5, dim=2, method="uniform", continuous=False, inplace=True):
        super(NoiseDropout, self).__init__()

        assert 0. <= p < 1.

        self.p = p
        self.dim = dim
        self.method = method
        self.continuous = continuous
        self.inplace = inplace

        if self.dim != 1 and self.dim != 2 and self.dim != 3:
            raise TypeError("Expected dim = 1, 2 or 3, but got {}".format(self.dim))

        if self.method != "uniform" and self.method != "normal":
            raise TypeError("Do not support {} method for random dropout.".format(self.method))

        if self.method == "normal":
            self.std = (self.p / (1 - self.p))**(0.5)
        else:
            self.a = -self.p + 1 # From (-p, p) to (-p+1, p+1) for x. = x(1+r), r~U(-p, p) and 1+r~U(-p+1,p+1)
            self.b = self.p + 1

        self.init = False # To speed up the computing.

    def forward(self, inputs):
        if self.training and self.p > 0.:
            if not self.init:
                input_size = inputs.shape
                if self.dim == 1:
                    noise_size = (input_size[0], input_size[1], input_size[2])
                elif self.dim == 2:
                    # Apply the same noise for every frames (a.k.a channels) in one sample.
                    noise_size = (input_size[0], input_size[1], 1)
                else:
                    noise_size = (input_size[0], input_size[1], 1, 1)

                self.r = torch.randn(noise_size, device=inputs.device)

                self.init = True

            if self.method == "uniform":
                if self.continuous:
                    self.r.uniform_(0, 1)
                else:
                    self.r.uniform_(self.a, self.b)
            else:
                if self.continuous:
                    self.r.normal_(0.5, self.std).clamp_(min=0.,max=1.)
                else:
                    self.r.normal_(1, self.std).clamp_(min=0.)

            if self.inplace:
                return inputs.mul_(self.r)
            else:
                return inputs * self.r
        else:
            return inputs


class SpecAugment(torch.nn.Module):
    """Implement specaugment for acoustics features' augmentation but without time wraping.
    It is different to egs.augmentation.SpecAugment for all egs have a same dropout method in one batch here.

    Reference: Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). 
               Specaugment: A simple data augmentation method for automatic speech recognition. arXiv 
               preprint arXiv:1904.08779.

    Likes in Compute Vision: 
           [1] DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks 
               with cutout. arXiv preprint arXiv:1708.04552.

           [2] Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2017). Random erasing data augmentation. 
               arXiv preprint arXiv:1708.04896. 
    """
    def __init__(self, frequency=0.2, frame=0.2, rows=1, cols=1, random_rows=False, random_cols=False):
        super(SpecAugment, self).__init__()

        assert 0. <= frequency < 1.
        assert 0. <= frame < 1. # a.k.a time axis.

        self.p_f = frequency
        self.p_t = frame

        # Multi-mask.
        self.rows = rows # Mask rows times for frequency.
        self.cols = cols # Mask cols times for frame.

        self.random_rows = random_rows
        self.random_cols = random_cols

        self.init = False

    def __call__(self, inputs):
        """
        @inputs: a 3-dimensional tensor, including [batch, frenquency, time]
        """
        assert len(inputs.shape) == 3

        if not self.training: return inputs

        if self.p_f > 0. or self.p_t > 0.:
            if not self.init:
                input_size = (inputs.shape[1], inputs.shape[2])
                if self.p_f > 0.:
                    self.num_f = input_size[0] # Total channels.
                    self.F = int(self.num_f * self.p_f) # Max channels to drop.
                if self.p_t > 0.:
                    self.num_t = input_size[1] # Total frames. It requires all egs with the same frames.
                    self.T = int(self.num_t * self.p_t) # Max frames to drop.
                self.init = True

            if self.p_f > 0.:
                if self.random_rows:
                    multi = np.random.randint(1, self.rows+1)
                else:
                    multi = self.rows

                for i in range(multi):
                    f = np.random.randint(0, self.F + 1)
                    f_0 = np.random.randint(0, self.num_f - f + 1)
                    inverted_factor = self.num_f / (self.num_f - f)
                    inputs[f_0:f_0+f,:].fill_(0.)
                    inputs.mul_(inverted_factor)

            if self.p_t > 0.:
                if self.random_cols:
                    multi = np.random.randint(1, self.cols+1)
                else:
                    multi = self.cols

                for i in range(multi):
                    t = np.random.randint(0, self.T + 1)
                    t_0 = np.random.randint(0, self.num_t - t + 1)
                    inputs[:,t_0:t_0+t].fill_(0.)

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
            "method":"normal",
            "continuous":False,
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
                             method=dropout_params["method"], inplace=dropout_params["inplace"])
    elif name == "alpha":
        return torch.nn.AlphaDropout(p=p, inplace=dropout_params["inplace"])
    elif name == "context":
        return ContextDropout(p=p)
    elif name == "noise":
        return NoiseDropout(p=p, dim=dropout_params["dim"], method=dropout_params["method"],
                            continuous=dropout_params["continuous"],inplace=dropout_params["inplace"])
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
