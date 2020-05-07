# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-08-01)

import logging
import types
import math
import itertools as it
from torch._six import inf
from functools import partial, wraps
import warnings
from bisect import bisect_right

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer

import libs.support.utils as utils

# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

## Wrapper ✿
def get_optimizer(model, params:dict={}):
    # Suggested weight_decay: 1e-4 for l2 regularization (sgd, adam) and 
    #                         1e-1 for decouped weight decay (sgdw, adamw, radam, ralamb, adamod etc.)
    default_params = {
        "name":"adamW",
        "learn_rate":0.001,
        "beta1":0.9,
        "beta2":0.999,
        "beta3":0.999,
        "weight_decay":1e-4,
        "lookahead.k":5,
        "lookahead.alpha":0.,
        "gc":False
    }

    used_params = utils.assign_params_dict(default_params, params)

    # Base params
    name = used_params["name"]
    learn_rate = used_params["learn_rate"]
    beta1 = used_params["beta1"]
    beta2 = used_params["beta1"]
    beta3 = used_params["beta1"]
    weight_decay = used_params["weight_decay"]
    gc = used_params["gc"]

    extra_params = {}

    # Gradient centralization: 
    # Yong, H., Huang, J., Hua, X., & Zhang, L. (2020). Gradient Centralization: 
    #     A New Optimization Technique for Deep Neural Networks. arXiv e-prints, arXiv:2004.01461. 
    #     Retrieved from https://ui.adsabs.harvard.edu/abs/2020arXiv200401461Y
    # Github: https://github.com/Yonghongwei/Gradient-Centralization
    if gc:
        # Specify this list by developer.
        default_support_gc_list = ["adamW", "ralamb"]

        if name not in default_support_gc_list:
            raise TypeError("Optimizer {} does not support gradient centralization (GC) now.".format(name))

        extra_params["gc"] = True

    # Select optimizer
    if name == "sgd":
        base_optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=beta1, weight_decay=weight_decay)
    elif name == "sgdW":
        base_optimizer = SGDW(model.parameters(), lr=learn_rate, momentum=beta1, weight_decay=weight_decay)
    elif name == "adam":
        base_optimizer = optim.Adam(model.parameters(), lr=learn_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    elif name == "adamW":
        base_optimizer = AdamW(model.parameters(), lr=learn_rate, betas=(beta1, beta2), weight_decay=weight_decay, **extra_params)
    elif name == "radam":
        base_optimizer = RAdam(model.parameters(), lr=learn_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    elif name == "ralamb":
        base_optimizer = Ralamb(model.parameters(), lr=learn_rate, betas=(beta1, beta2), weight_decay=weight_decay, **extra_params)
    elif name == "adamod":
        base_optimizer = AdaMod(model.parameters(), lr=learn_rate, betas=(beta1, beta2), beta3=beta3, weight_decay=weight_decay)
    elif name == "novograd":
        base_optimizer = NovoGrad(model.parameters(), lr=learn_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    else:
        raise ValueError("Do not support {0} optimizer now.".format(name))

    # Using alpha to decide whether to use lookahead
    if used_params["lookahead.alpha"] > 0:
        logger.info("Use lookahead optimizer with alpha={} and k={}".format(used_params["lookahead.alpha"], used_params["lookahead.k"]))
        optimizer = Lookahead(base_optimizer, k=used_params["lookahead.k"], alpha=used_params["lookahead.alpha"])
    else:
        optimizer = base_optimizer

    return optimizer


## Optim-wrapper ✿
class Lookahead(Optimizer):
    """https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py
    """
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        self.optimizer = base_optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        self.is_back_step = False
        self.init_weights = False

        for group in self.param_groups:
            group["step_counter"] = 0

    def step(self, closure=None):
        self.is_back_step = False
        # Init weights after model in a certrain device and keep the device of weights same to model. [Snowdar 2018-09-01]
        if not self.init_weights and self.alpha > 0:
            self.slow_weights = [[p.clone().detach() for p in group['params']]
                                    for group in self.param_groups]

            for w in it.chain(*self.slow_weights):
                w.requires_grad = False
            
            self.init_weights = True

        loss = None
        if closure is not None:
            loss = closure()
        loss = self.optimizer.step()
        if self.alpha > 0:
            for group,slow_weights in zip(self.param_groups,self.slow_weights):
                group['step_counter'] += 1
                if group['step_counter'] % self.k != 0:
                    continue
                else:
                    self.is_back_step = True

                for p,q in zip(group['params'],slow_weights):
                    if p.grad is None:
                        continue
                    q.data.add_(self.alpha,p.data - q.data)
                    p.data.copy_(q.data)
        return loss


## Optimizer ✿
class SGDW(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum) with decouped weight decay.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v_{t+1} = \mu * v_{t} + g_{t+1} \\
                  p_{t+1} = p_{t} - lr * v_{t+1}

        where p, g, v and :math:`\mu` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v_{t+1} = \mu * v_{t} + lr * g_{t+1} \\
             p_{t+1} = p_{t} - v_{t+1}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                p.data.add_(-group['lr'], d_p)

        return loss


class AdamW(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, gc=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        self.gc = gc
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data) #, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data) #, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data) #, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if self.gc:
                    # For linear layer Y=WX+b, the tensor shape of weight is (outplanes, inplanes),
                    # but for CNN layer(1d and 2d etc.), the tensor shape of weight is (outplanes, inplanes, [cnn-core]).
                    # And here the gc is used in both linear and CNN layer.
                    # It is not influenced by weight decay for weight decay directly changes the p.data rather than p.grad.
                    # But when using gc in adam, the order question should be considered for L2 regularization changes 
                    # the p.grad.
                    if len(list(grad.size()))>=2:
                        grad.add_(-grad.mean(dim = tuple(range(1,len(list(grad.size())))), keepdim = True))

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class RAdam(Optimizer):
    '''https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py
    
    a PyTorch implementation of the RAdam Optimizer from th paper
    On the Variance of the Adaptive Learning Rate and Beyond.
    https://arxiv.org/abs/1908.03265
    Example:
        >>> from optimizer import RAdam
        >>> optimizer = RAdam(model.parameters(), lr=0.001)
    Note, here the weight decay is not L2 regularization.
    '''

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), N_sma_threshhold=4, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.N_sma_threshhold = N_sma_threshhold
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class Ralamb(Optimizer):
    '''https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py
    Ralamb optimizer [RAdam + Layer-wise Adaptive Rate Scaling (LARS) trick]
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), N_sma_threshhold=4, eps=1e-8, weight_decay=0, gc=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.N_sma_threshhold = N_sma_threshhold
        self.buffer = [[None, None, None] for ind in range(10)]
        self.gc = gc
        super(Ralamb, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Ralamb, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Ralamb does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                if self.gc:
                    # For linear layer Y=WX+b, the tensor shape of weight is (outplanes, inplanes),
                    # but for CNN layer(1d and 2d etc.), the tensor shape of weight is (outplanes, inplanes, [cnn-core]).
                    # And here the gc is used in both linear and CNN layer.
                    if len(list(grad.size()))>=2:
                        grad.add_(-grad.mean(dim = tuple(range(1,len(list(grad.size())))), keepdim = True))

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, radam_step = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma > self.N_sma_threshhold:
                        radam_step = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        radam_step = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = radam_step

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                radam_norm = p_data_fp32.pow(2).sum().sqrt()
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm

                state['weight_norm'] = weight_norm
                state['adam_norm'] = radam_norm
                state['trust_ratio'] = trust_ratio

                # more conservative since it's an approximated value
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-radam_step * trust_ratio, exp_avg, denom)
                else:
                    p_data_fp32.add_(-radam_step * trust_ratio, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class AdaMod(Optimizer):
    """Implements AdaMod algorithm with Decoupled Weight Decay (arxiv.org/abs/1711.05101)
    It has been proposed in `Adaptive and Momental Bounds for Adaptive Learning Rate Methods`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        beta3 (float, optional): smoothing coefficient for adaptive learning rates (default: 0.9999)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay rather than L2 penalty (default: 0)

        Reference: https://github.com/lancopku/AdaMod.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), beta3=0.999,
                 eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= beta3 < 1.0:
            raise ValueError("Invalid beta3 parameter: {}".format(beta3))
        defaults = dict(lr=lr, betas=betas, beta3=beta3, eps=eps,
                        weight_decay=weight_decay)
        super(AdaMod, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaMod, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaMod does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Exponential moving average of actual learning rates
                    state['exp_avg_lr'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, exp_avg_lr = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_lr']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                # Applies momental bounds on actual learning rates
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom)
                exp_avg_lr.mul_(group['beta3']).add_(1 - group['beta3'], step_size)
                step_size = torch.min(step_size,  exp_avg_lr)
                step_size.mul_(exp_avg)

                p.data.add_(-step_size)

        return loss


class NovoGrad(Optimizer):
    r"""Implements Novograd optimization algorithm.
    It has been proposed in `Stochastic Gradient Methods with Layer-wise
    Adaptive Moments for Training of Deep Networks`__.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (not L2 penalty) (default: 0)
        grad_averaging: gradient averaging (default: False)
        amsgrad: whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`
            (default: False)

    Reference:
        1.https://arxiv.org/abs/1905.11286
        2.https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/novograd.py
        3.https://github.com/NVIDIA/DeepLearningExamples
    """

    def __init__(self, params, lr = 1e-3, betas = (0.95, 0), eps = 1e-8, weight_decay = 0, grad_averaging = False, amsgrad = False):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_averaging=grad_averaging,
            amsgrad=amsgrad,
        )

        super(NovoGrad, self).__init__(params, defaults)

    def __setstate__(self, state: dict) -> None:
        super(NovoGrad, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure = None):
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        'NovoGrad does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros([]).to(
                        state['exp_avg'].device
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq.
                        # grad. values
                        state['max_exp_avg_sq'] = torch.zeros([]).to(
                            state['exp_avg'].device
                        )

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                norm = torch.sum(torch.pow(grad, 2))

                if exp_avg_sq == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, norm)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg.
                    # till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                grad.div_(denom)
                if group['grad_averaging']:
                    grad.mul_(1 - beta1)
                exp_avg.mul_(beta1).add_(grad)

                p.data.add_(-group['lr'], exp_avg)

        return loss
