# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-08-01)

import math
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

from .optim import *
import libs.support.utils as utils

## Wrapper ✿

class LRSchedulerWrapper():

    def __init__(self, optimizer, params:dict={}):
        # Suggested weight_decay: 1e-4 for l2 regularization (sgd, adam) and 
        #                         1e-1 for decouped weight decay (sgdw, adamw, radam, ralamb, adamod etc.)
        default_params = {
            "name":"warmR",
            "1cycle.learn_rate":0.001,
            "warmR.T_max":10,
            "warmR.T_mult":1,
            "warmR.factor":1.0,
            "warmR.eta_min":4e-8,
            "warmR.log_decay":False,
            "warmR.lr_decay_step":1
        }

        used_params = utils.assign_params_dict(default_params, params, force_check=False, support_unknow=True)
        split_params = utils.split_params(used_params)

        if isinstance(optimizer, Lookahead):
            base_optimizer = optimizer.optimizer
        else:
            base_optimizer = optimizer

        self.name = split_params["public"]["name"]
        if self.name == "1cycle":
            # To do.
            self.lr_scheduler = optim.lr_scheduler.OneCycleLR(base_optimizer, **split_params["1cycle"])
        elif self.name == "warmR":
            T_max = split_params["warmR"].pop("T_max")
            self.lr_decay_step = split_params["warmR"].pop("lr_decay_step")
            self.lr_scheduler = CosineAnnealingWarmRestarts(base_optimizer, T_max, **split_params["warmR"])
        else:
            raise ValueError("Do not support {0} lr_scheduler now.".format(name))

    def step(self, training_point=None):
        if self.name == "warmR":
            if self.lr_decay_step > 0 and training_point[1]%self.lr_decay_step == 0:
                self.lr_scheduler.step(training_point[0]+training_point[1]/training_point[2])
            elif self.lr_decay_step == 0:
                self.lr_scheduler.step(training_point[0])
        elif self.name == "1cycle":
            self.lr_scheduler.step()


## Learn rate scheduler ✿
class CosineAnnealingWarmRestarts(_LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{i}}\pi))

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0`(after restart), set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    Base lr decay has been added. [Snowdar 2019-08-29]
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, factor=1.0, log_decay=False, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult <=0: # or not isinstance(T_mult, int):
            raise ValueError("Expected T_mult > 0, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.factor = factor
        self.this_factor = 1
        self.T_cur = last_epoch
        self.log_decay = log_decay
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.log_decay:
            eta_min = np.log10(self.eta_min)
            return [ 10**(eta_min + (np.log10(base_lr * self.this_factor) - eta_min) * 
                    (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2)
                    for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr * self.this_factor - self.eta_min) * 
                    (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         scheduler.step(epoch + i / iters)
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()

        This function can be called in an interleaved way.

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.this_factor = self.factor ** (epoch // self.T_0)
                else:
                    n = int(math.log(max(0.05, (epoch / self.T_0 * (self.T_mult - 1) + 1)), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
                    self.this_factor = self.factor ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
