# MIT License

# Copyright (c) 2019 kaiJIN

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
r"""Synchronized Batch Norm
Author: KJ
Date: 19/09/25
Version: align with pytorch 1.2
Reference: https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
Reference: https://www.zhihu.com/question/282672547
"""
import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import horovod.torch as hvd


class _SynchronizedBatchNorm(_BatchNorm):
    r"""Synchronized BatchNorm align with pytorch 1.2 batchnorm syntax.
        1) training=true, track_running_stats=true: 
        running_mean and running_var are just tracked and do not use.
        mean and var are computed by batch samples.
        2) training=true, track_running_stats=false:
        running_mean and running_var do not track and use.
        mean and var are computed by batch samples.
        3) training=false, track_running_stats=true:
        using running_mean and running_var instead of mean/var by batches.
        4) training=false, track_running_stats=false:
        using batch mean/var instead of running_mean and running_var.
    """

    def __init__(self, num_features,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True):

        super(_SynchronizedBatchNorm, self).__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats)

    def forward(self, inputs):
        self._check_input_dim(inputs)

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        # setting momentum
        if self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                # use cumulative moving average
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                # use exponential moving average
                else:
                    exponential_average_factor = self.momentum

        # Resize the input to (B, C, -1).
        ch = self.num_features
        inputs_shape = inputs.size()
        inputs = inputs.reshape(inputs.size(0), ch, -1)

        # reshape
        if self.affine:
            weight = self.weight.view(1, ch, 1)
            bias = self.bias.view(1, ch, 1)

        # verification inference version (the only is used to test function)
        if not self.training:
            if self.track_running_stats:
                mean = self.running_mean.view(1, ch, 1)
                inv_std = 1 / (self.running_var + self.eps).sqrt().view(1, ch, 1)

                if self.affine:
                    outputs = weight * inv_std * (inputs - mean) + bias
                else:
                    outputs = inv_std * (inputs - mean)
            else:
                var, mean = torch.var_mean(
                inputs, unbiased=False, dim=[0, 2], keepdim=True)
                inv_std = 1 / (var + self.eps).sqrt()

                if self.affine:
                    outputs = weight * inv_std * (inputs - mean) + bias
                else:
                    outputs = inv_std * (inputs - mean)

            return outputs.reshape(inputs_shape)

        # Compute the sum and square-sum.
        sum_size = inputs.size(0) * inputs.size(2) * hvd.size()
        stat_sum = inputs.sum(dim=[0, 2])
        stat_ssum = inputs.pow(2).sum(dim=[0, 2])

        # Reduce-and-broadcast the statistics.
        # concat in order to broadcast once
        stats = torch.stack([stat_sum, stat_ssum]).to(inputs.device)
        sync_sum, sync_ssum = hvd.allreduce(stats, average=False).split(1)

        # VAR = E(X^2) - (EX)^2
        mean = sync_sum.view(1, ch, 1) / sum_size
        var = sync_ssum.view(1, ch, 1) / sum_size - mean.pow(2)
        inv_std = 1. / (var + self.eps).sqrt()

        # track running stat
        if self.track_running_stats:
            with torch.no_grad():
                m = exponential_average_factor
                uvar = sync_ssum / (sum_size - 1) - (sync_sum / (sum_size)).pow(2)
                self.running_mean = (1 - m) * self.running_mean + m * mean.view(-1)
                self.running_var = (1 - m) * self.running_var + m * uvar.view(-1)

        # affine
        if self.affine:
            outputs = weight * inv_std * (inputs - mean) + bias
        else:
            outputs = (inputs - mean) * inv_std

        return outputs.reshape(inputs_shape)


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    def _check_input_dim(self, inputs):
        if inputs.dim() != 2 and inputs.dim() != 3:
            raise ValueError('expected 2D or 3D inputs (got {}D inputs)'
                       .format(inputs.dim()))


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    def _check_input_dim(self, inputs):
        if inputs.dim() != 4:
            raise ValueError('expected 4D inputs (got {}D inputs)'
                       .format(inputs.dim()))


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    def _check_input_dim(self, inputs):
        if inputs.dim() != 5:
            raise ValueError('expected 5D inputs (got {}D inputs)'
                       .format(inputs.dim()))


def convert_sync_batchnorm(module):
    """ Snowdar 2020-04-22.
    Do not use python list to gather module, or it will not correctly convert your model. 
    Just use torch.nn.ModuleList to replace python list.
    Reference: https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#SyncBatchNorm
    """
    module_output = module

    for torch_module, sync_module in zip([torch.nn.BatchNorm1d,
                                      torch.nn.BatchNorm2d,
                                      torch.nn.BatchNorm3d],
                                     [SynchronizedBatchNorm1d,
                                      SynchronizedBatchNorm2d,
                                      SynchronizedBatchNorm3d]):
        
        if isinstance(module, torch_module):
            module_output = sync_module(module.num_features,
                                        module.eps, module.momentum,
                                        module.affine,
                                        module.track_running_stats)
            if module.affine:
                with torch.no_grad():
                    module_output.weight.copy_(module.weight)
                    module_output.bias.copy_(module.bias)
                # keep requires_grad unchanged
                module_output.weight.requires_grad = module.weight.requires_grad
                module_output.bias.requires_grad = module.bias.requires_grad
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
        for name, child in module.named_children():
            module_output.add_module(name, convert_sync_batchnorm(child))

    del module
    return module_output