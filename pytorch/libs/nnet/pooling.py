# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-05-29 2020-06-10)

import numpy as np
from typing import Optional
import torch
import torch.nn.functional as F

from libs.support.utils import to_device

from .components import *

## Pooling ✿
class StatisticsPooling(torch.nn.Module):
    """ An usual mean [+ stddev] poolling layer"""
    def __init__(self, input_dim, stddev=True, unbiased=False, eps=1.0e-10):
        super(StatisticsPooling, self).__init__()

        self.stddev = stddev
        self.input_dim = input_dim

        if self.stddev :
            self.output_dim = 2 * input_dim
        else :
            self.output_dim = input_dim

        self.eps = eps
        # Used for unbiased estimate of stddev
        self.unbiased = unbiased

    def forward(self, inputs, lengths:torch.Tensor = torch.ones((0),dtype=torch.long)):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        if lengths.size(0) > 0:
            mean = []
            std = []
            for i in range(inputs.shape[0]):
                act_len = lengths[i]
                act_counts = act_len
                mean_i = torch.mean(inputs[i,:,:act_len],dim=1,keepdim=True)
                mean.append(mean_i)
                if self.stddev :
                    if self.unbiased and act_len > 1:
                        act_counts = act_len - 1
                    var = torch.sum((inputs[i,:,:act_len]-mean_i)**2, dim=1, keepdim=True)/act_counts
                    std.append(torch.sqrt(var.clamp(min=self.eps))) 

            mean = torch.stack(mean)
            out = mean
            if self.stddev :
                std_o = torch.stack(std)
                out = torch.cat((mean, std_o), dim=1)             
        else:
            counts = inputs.shape[2]
            mean = inputs.mean(dim=2, keepdim=True)
            out = mean
            if self.stddev:
                if self.unbiased and counts > 1:
                    counts = counts - 1
                var = torch.sum((inputs - mean)**2, dim=2, keepdim=True) / counts
                std = torch.sqrt(var.clamp(min=self.eps))
                out = torch.cat((mean, std), dim=1)  
       
        return out


    def get_output_dim(self):
        return self.output_dim
    
    def extra_repr(self):
        return '{input_dim}, {output_dim}, stddev={stddev}, unbiased={unbiased}, eps={eps}'.format(**self.__dict__)

    @classmethod
    def thop_count(self, m, x, y):
        pass
        # To do
        # x = x[0]

        # kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
        # bias_ops = 1 if m.bias is not None else 0

        # # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        # total_ops = y.nelement() * (m.input_dim * kernel_ops + bias_ops)

        # m.total_ops += torch.DoubleTensor([int(total_ops)])

class FreeStatisticsPooling(torch.nn.Module):
    """ An usual mean [+ stddev] poolling layer"""
    def __init__(self, stddev=True, unbiased=False, eps=1.0e-10):
        super(FreeStatisticsPooling, self).__init__()

        self.stddev = stddev

        self.eps = eps
        # Used for unbiased estimate of stddev
        self.unbiased = unbiased

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """

        inputs = inputs.reshape(inputs.shape[0], -1, inputs.shape[len(inputs.shape)-1])

        # Get the num of frames
        counts = inputs.shape[2]

        mean = inputs.sum(dim=2, keepdim=True) / counts

        if self.stddev :
            if self.unbiased and counts > 1:
                counts = counts - 1

            # The sqrt (as follows) is deprecated because it results in Nan problem.
            # std = torch.unsqueeze(torch.sqrt(torch.sum((inputs - mean)**2, dim=2) / counts), dim=2)
            # There is a eps to solve this problem.
            # Another method: Var is equal to std in "cat" way, actually. So, just use Var directly.

            var = torch.sum((inputs - mean)**2, dim=2, keepdim=True) / counts
            std = torch.sqrt(var.clamp(min=self.eps))
            return torch.cat((mean, std), dim=1)
        else:
            return mean

class LDEPooling(torch.nn.Module):
    """A novel learnable dictionary encoding layer.
    Reference: Weicheng Cai, etc., "A NOVEL LEARNABLE DICTIONARY ENCODING LAYER FOR END-TO-END 
               LANGUAGE IDENTIFICATION", icassp, 2018
    """
    def __init__(self, input_dim, c_num=64, eps=1.0e-10):
        super(LDEPooling, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim * c_num
        self.eps = eps

        self.mu = torch.nn.Parameter(torch.randn(input_dim, c_num))
        self.s = torch.nn.Parameter(torch.ones(c_num))

        self.softmax_for_w = torch.nn.Softmax(dim=3)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        r = inputs.transpose(1,2).unsqueeze(3) - self.mu
        # Make sure beta=self.s**2+self.eps > 0
        w = self.softmax_for_w(- (self.s**2 + self.eps) * torch.sum(r**2, dim=2, keepdim=True))
        e = torch.mean(w * r, dim=1)

        return e.reshape(-1, self.output_dim, 1)

    def get_output_dim(self):
        return self.output_dim

## Xi-vector pooling (softplus_prec)
class xivec_stdinit_softplus2_prec_pooling(torch.nn.Module):

    def __init__(self, input_dim, hidden_size=256, context=[0], stddev=False, train_mean=True, train_prec=True):
        super(xivec_stdinit_softplus2_prec_pooling, self).__init__()
 
        self.input_dim = input_dim
        self.stddev = stddev

        if self.stddev:
            self.output_dim = 2 * input_dim
        else:
            self.output_dim = input_dim        

        self.prior_mean = torch.nn.Parameter(torch.zeros(1, input_dim), requires_grad=train_mean)
        self.prior_logprec = torch.nn.Parameter(torch.zeros(1, input_dim), requires_grad=train_prec)
        self.softmax = torch.nn.Softmax(dim=2)
   
        # Log-precision estimator
        self.lin1_relu_bn = ReluBatchNormTdnnLayer(input_dim, hidden_size, context)
        self.lin2 = TdnnAffine(hidden_size, input_dim, context=context)
        self.softplus2 = torch.nn.Softplus(beta=1, threshold=20)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        feat = inputs

        # Log-precision estimator
        logprec = self.softplus2(self.lin2(self.lin1_relu_bn(feat)))   # frame precision estimate
        logprec = 2.0*torch.log(logprec)                               # Square and take log before softmax                           

        ### Gaussian Posterior Inference
        ### Option 1: a_o (prior_mean-phi) included in variance
        weight_attn = self.softmax(torch.cat((logprec, self.prior_logprec.repeat(logprec.shape[0], 1).unsqueeze(dim=2)), 2))
        # Posterior precision
        # Ls = torch.sum(torch.exp(torch.cat((logprec, self.prior_logprec.repeat(logprec.shape[0], 1).unsqueeze(dim=2)), 2)), dim=2)
        # Posterior mean
        phi = torch.sum(torch.cat((feat, self.prior_mean.repeat(feat.shape[0], 1).unsqueeze(dim=2)), 2) * weight_attn, dim=2)

        if self.stddev:
            sigma2 = torch.sum(torch.cat((feat, self.prior_mean.repeat(feat.shape[0], 1).unsqueeze(dim=2)), 2).pow(2) * weight_attn, dim=2)
            sigma = torch.sqrt(torch.clamp(sigma2 - phi ** 2, min=1.0e-10)) 
            return torch.cat((phi, sigma), dim=1).unsqueeze(dim=2)
        else: 
            return phi.unsqueeze(dim=2)

    def get_output_dim(self):
        return self.output_dim


# Attention-based
class AttentionAlphaComponent(torch.nn.Module):
    """Compute the alpha with attention module.
            alpha = softmax(v'·f(w·x + b) + k) or softmax(v'·x + k)
    where f is relu here and bias could be lost.
    Support: 
            1. Single or Multi-head attention
            2. One affine or two affine
            3. Share weight (last affine = vector) or un-shared weight (last affine = matrix)
            4. Self-attention or time context attention (supported by context parameter of TdnnAffine)
            5. Different temperatures for different heads.
    """
    def __init__(self, input_dim, num_head=1, split_input=True, share=True, affine_layers=2, 
                 hidden_size=64, context=[0], bias=True, temperature=False, fixed=True):
        super(AttentionAlphaComponent, self).__init__()
        assert num_head >= 1
        # Multi-head case.
        if num_head > 1:
            if split_input:
                # Make sure fatures/planes with input_dim dims could be splited to num_head parts.
                print("input_dim:",input_dim)
                assert input_dim % num_head == 0
            if temperature:
                if fixed:
                    t_list = []
                    for i in range(num_head):
                        t_list.append([[max(1, (i // 2) * 5)]])
                    # shape [1, num_head, 1, 1]
                    self.register_buffer('t', torch.tensor([t_list]))
                else:
                    # Different heads have different temperature.
                    # Use 1 + self.t**2 in forward to make sure temperature >= 1.
                    self.t = torch.nn.Parameter(torch.zeros(1, num_head, 1, 1))

        self.input_dim = input_dim
        self.num_head = num_head
        self.split_input = split_input
        self.share = share
        self.temperature = temperature
        self.fixed = fixed

        if share:
            # weight: [input_dim, 1] or [input_dim, hidden_size] -> [hidden_size, 1]
            final_dim = 1
        elif split_input:
            # weight: [input_dim, input_dim // num_head] or [input_dim, hidden_size] -> [hidden_size, input_dim // num_head]
            final_dim = input_dim // num_head
        else:
            # weight: [input_dim, input_dim] or [input_dim, hidden_size] -> [hidden_size, input_dim]
            final_dim = input_dim

        first_groups = 1
        last_groups = 1

        if affine_layers == 1:
            last_affine_input_dim = input_dim
            # (x, 1) for global case and (x, h) for split case.
            if num_head > 1 and split_input:
               last_groups = num_head
            self.relu_affine = False
        elif affine_layers == 2:
            last_affine_input_dim = hidden_size * num_head
            if num_head > 1:
                # (1, h) for global case and (h, h) for split case.
                last_groups = num_head
                if split_input:
                    first_groups = num_head
            # Add a relu-affine with affine_layers=2.
            self.relu_affine = True
            self.first_affine = TdnnAffine(input_dim, last_affine_input_dim, context=context, bias=bias, groups=first_groups)
            self.relu = torch.nn.ReLU(inplace=True)
        else:
            raise ValueError("Expected 1 or 2 affine layers, but got {}.",format(affine_layers))

        self.last_affine = TdnnAffine(last_affine_input_dim, final_dim * num_head, context=context, bias=bias, groups=last_groups)
        # Dim=2 means to apply softmax in different frames-index (batch is a 3-dim tensor in this case).
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        if self.temperature:
            batch_size = inputs.shape[0]
            chunk_size = inputs.shape[2]

        x = inputs
        if self.relu_affine:
            x = self.relu(self.first_affine(x))
        if self.num_head > 1 and self.temperature:
            if self.fixed:
                t = self.t
            else:
                t = 1 + self.t**2
            x = self.last_affine(x).reshape(batch_size, self.num_head, -1, chunk_size) / t
            return self.softmax(x.reshape(batch_size, -1, chunk_size))
        else:
            return self.softmax(self.last_affine(x))


class AttentiveStatisticsPooling(torch.nn.Module):
    """ An attentive statistics pooling.
    Reference: Okabe, Koji, Takafumi Koshinaka, and Koichi Shinoda. 2018. "Attentive Statistics Pooling 
               for Deep Speaker Embedding." ArXiv Preprint ArXiv:1803.10963.
    """
    def __init__(self, input_dim, affine_layers=2, hidden_size=64, context=[0], stddev=True, stddev_attention=True, eps=1.0e-10):
        super(AttentiveStatisticsPooling, self).__init__()

        self.stddev = stddev
        self.input_dim = input_dim

        if self.stddev :
            self.output_dim = 2 * input_dim
        else :
            self.output_dim = input_dim

        self.eps = eps
        self.stddev_attention = stddev_attention

        self.attention = AttentionAlphaComponent(input_dim, num_head=1, share=True, affine_layers=affine_layers, 
                                                 hidden_size=hidden_size, context=context)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        alpha = self.attention(inputs)

        # Weight avarage
        mean = torch.sum(alpha * inputs, dim=2, keepdim=True)

        if self.stddev :
            if self.stddev_attention:
                var = torch.sum(alpha * inputs**2, dim=2, keepdim=True) - mean**2
                std = torch.sqrt(var.clamp(min=self.eps))
            else:
                var = torch.mean((inputs - mean)**2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=self.eps))
            return torch.cat((mean, std), dim=1)
        else :
            return mean

    def get_output_dim(self):
        return self.output_dim


class MultiHeadAttentionPooling(torch.nn.Module):
    """Implement multi-head attention pooling based on AttentionAlphaComponent.
    Reference: Safari, Pooyan, and Javier Hernando. 2019. “Self Multi-Head Attention for Speaker 
               Recognition.” ArXiv Preprint ArXiv:1906.09890.
    Note, in this paper, affine_layers is default to 1, and final_dim is 1 which means the weights are shared.
    """
    def __init__(self, input_dim, stddev=True, stddev_attention=True, num_head=4, share=True, affine_layers=1, **options):
        super(MultiHeadAttentionPooling, self).__init__()

        self.input_dim = input_dim
        self.stddev = stddev
        self.stddev_attention = stddev_attention
        self.num_head = num_head

        if self.stddev :
            self.output_dim = 2 * input_dim
        else :
            self.output_dim = input_dim

        if "split_input" in options.keys():
            if not options["split_input"]:
                raise ValueError("split_input==False is not valid for this MultiHeadAttentionPooling.")
            options.pop("split_input")

        # In this pooling, the special point is that inputs will be splited.
        self.attention = AttentionAlphaComponent(input_dim, num_head=num_head, split_input=True, share=share, 
                                                 affine_layers=affine_layers, bias=False, **options)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        batch_size = inputs.shape[0]
        chunk_size = inputs.shape[2] # a.k.a total frames

        # alpha: [batch, weight, frames]
        # When using the conv1d to implement the multi-multiple of multi-head, we can get
        # the weight distribution of multi-head: [h11, h12, h13, h21, h22, h23, ..., hn1, hn2, ...]
        # So, just reshape it to split different heads.
        alpha = self.attention(inputs)

        # In sharing weight case, the shape of alpha is [batch, head, 1, frames] and [batch, head, splited-features, frames]
        # for another case.
        # inputs: [batch, head, splited-features, frames]
        after_mul = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                    inputs.reshape(batch_size, self.num_head, -1, chunk_size)

        # After multi-multipling alpha and inputs for multi-head case, the mean could be got by reshaping back.
        mean = torch.sum(after_mul.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True)

        if self.stddev :
            if self.stddev_attention:
                after_mul_2 = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                        inputs.reshape(batch_size, self.num_head, -1, chunk_size)**2
                var = torch.sum(after_mul_2.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True) - mean**2
                std = torch.sqrt(var.clamp(min=1.0e-10))
            else:
                var = torch.mean((inputs - mean)**2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=1.0e-10))
            return torch.cat((mean, std), dim=1)
        else :
            return mean

    def get_output_dim(self):
        return self.output_dim


class GlobalMultiHeadAttentionPooling(torch.nn.Module):
    """Implement global multi-head attention pooling based on AttentionAlphaComponent.
    Reference: Zhiming Wang, Kaisheng Yao, Xiaolong Li, Shuo Fang. "MULTI-RESOLUTION MULTI-HEAD 
               ATTENTION IN DEEP SPEAKER EMBEDDING." ICASSP, 2020.
    It is not equivalent to multi-head attention pooling even when
               input_dim of global multi-head = 1/num_head * input_dim of multi-head.
    """
    def __init__(self, input_dim, stddev=True, stddev_attention=True, num_head=4, share=True, affine_layers=2, **options):
        super(GlobalMultiHeadAttentionPooling, self).__init__()

        self.input_dim = input_dim
        self.num_head = num_head
        self.stddev = stddev
        self.stddev_attention = stddev_attention

        if self.stddev :
            self.output_dim = 2 * input_dim
        else :
            self.output_dim = input_dim

        if "split_input" in options.keys():
            if options["split_input"]:
                raise ValueError("split_input==True is not valid for GlobalMultiHeadAttentionPooling.")
            options.pop("split_input")
        if "temperature" in options.keys():
            if options["temperature"]:
                raise ValueError("temperature==True is not valid for GlobalMultiHeadAttentionPooling.")
            options.pop("temperature")

        # In this pooling, the special point is that all (global) features of inputs will be used.
        self.attention = AttentionAlphaComponent(input_dim, num_head=num_head, split_input=False, share=share, 
                                                 temperature=False, affine_layers=affine_layers, bias=True, **options)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        batch_size = inputs.shape[0]
        chunk_size = inputs.shape[2] # a.k.a total frames

        # alpha: [batch, weight, frames]
        # When using the conv1d to implement the multi-multiple of multi-head, we can get
        # the weight distribution of multi-head: [h11, h12, h13, h21, h22, h23, ..., hn1, hn2, ...]
        # So, just reshape it to split different heads.
        alpha = self.attention(inputs)

        # In sharing weight case, the shape of alpha is [batch, head, 1, frames] and [batch, head, all-features, frames]
        # for another case.
        # inputs: [batch, 1, all-features, frames]
        after_mul = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                    inputs.reshape(batch_size, 1, -1, chunk_size)

        # After multi-multipling alpha and inputs for multi-head case, the mean could be got by reshaping back.
        mean = torch.sum(after_mul.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True)

        if self.stddev :
            if self.stddev_attention:
                after_mul_2 = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                        inputs.reshape(batch_size, 1, -1, chunk_size)**2
                var = torch.sum(after_mul_2.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True) - mean**2
                std = torch.sqrt(var.clamp(min=1.0e-10))
            else:
                var = torch.mean((inputs - mean)**2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=1.0e-10))
            return torch.cat((mean, std), dim=1)
        else :
            return mean

    def get_output_dim(self):
        return self.output_dim * self.num_head


class MultiResolutionMultiHeadAttentionPooling(torch.nn.Module):
    """Implement multi-resolution global multi-head attention pooling based on AttentionAlphaComponent.
    Reference: Zhiming Wang, Kaisheng Yao, Xiaolong Li, Shuo Fang. "MULTI-RESOLUTION MULTI-HEAD 
               ATTENTION IN DEEP SPEAKER EMBEDDING." ICASSP, 2020.
    """
    def __init__(self, input_dim, stddev=True, stddev_attention=True, num_head=4, share=True, affine_layers=2, **options):
        super(MultiResolutionMultiHeadAttentionPooling, self).__init__()

        self.input_dim = input_dim
        self.num_head = num_head
        self.stddev = stddev
        self.stddev_attention = stddev_attention

        if self.stddev :
            self.output_dim = 2 * input_dim
        else :
            self.output_dim = input_dim

        if "split_input" in options.keys():
            if options["split_input"]:
                raise ValueError("split_input==True is not valid for MultiResolutionMultiHeadAttentionPooling.")
            options.pop("split_input")
        if "temperature" in options.keys():
            if not options["temperature"]:
                raise ValueError("temperature==False is not valid for MultiResolutionMultiHeadAttentionPooling.")
            options.pop("temperature")

        # In this pooling, the special point is that all (global) features of inputs will be used and
        # the temperature will be added.
        self.attention = AttentionAlphaComponent(input_dim, num_head=num_head, split_input=False, temperature=True, 
                                                 share=share, affine_layers=affine_layers, bias=True, **options)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        batch_size = inputs.shape[0]
        chunk_size = inputs.shape[2] # a.k.a total frames

        # alpha: [batch, weight, frames]
        # When using the conv1d to implement the multi-multiple of multi-head, we can get
        # the weight distribution of multi-head: [h11, h12, h13, h21, h22, h23, ..., hn1, hn2, ...]
        # So, just reshape it to split different heads.
        alpha = self.attention(inputs)

        # In sharing weight case, the shape of alpha is [batch, head, 1, frames] and [batch, head, all-features, frames]
        # for another case.
        # inputs: [batch, 1, all-features, frames]
        after_mul = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                    inputs.reshape(batch_size, 1, -1, chunk_size)

        # After multi-multipling alpha and inputs for multi-head case, the mean could be got by reshaping back.
        mean = torch.sum(after_mul.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True)

        if self.stddev :
            if self.stddev_attention:
                after_mul_2 = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * \
                        inputs.reshape(batch_size, 1, -1, chunk_size)**2
                var = torch.sum(after_mul_2.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True) - mean**2
                std = torch.sqrt(var.clamp(min=1.0e-10))
            else:
                var = torch.mean((inputs - mean)**2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=1.0e-10))
            return torch.cat((mean, std), dim=1)
        else :
            return mean

    def get_output_dim(self):
        return self.output_dim * self.num_head

# (Leo 2022-11-17)
class MQMHASP(torch.nn.Module):
    """ 
     Reference:
        Miao Zhao, Yufeng Ma, and Yiwei Ding et al. "Multi-query multi-head attention pooling and Inter-topK penalty for speaker verification".
        https://arxiv.org/pdf/2110.05042.pdf   
    """
    def __init__(self, in_dim,
                 num_q = 2, 
                 num_head=4,
                 hidden_size=128, 
                 stddev=True,
                 share = True,
                 affine_layers=2,
                 time_attention=False,
                 norm_type = 'batch_norm',
                 **kargs
    ):
        super(MQMHASP,self).__init__()
        self.stddev = stddev
        # self.output_dim = in_dim*2 if self.stddev else in_dim
        self.num_head = max(1, num_head)
        self.num_q = max(1, num_q)
        self.time_attention = time_attention 
        assert (in_dim % num_head) == 0 
        att_idim = in_dim //num_head 
        if time_attention:
            att_idim = (in_dim * 3) //num_head  if stddev else (in_dim * 2) //num_head
        att_odim = 1  if share else in_dim //num_head 
        self.attention = self.build_attention(att_idim * num_head, att_odim * num_head * num_q, num_q, num_head, affine_layers, hidden_size, norm_type = norm_type)
        self.out_dim = in_dim * num_q * 2 if stddev else in_dim * num_q
        

    def forward(self, x, mask: torch.Tensor = torch.ones((0, 0, 0))):
        """
            x: input feature [B, F, T] 
            returns: pooling statiscs [B, F * qs, 1]
        """
        B, C ,T = x.shape

        if mask.size(2) == 0 :
            mask = torch.ones((B, 1, T)).to(x.device)
            

        if self.time_attention:

            total = mask.sum(dim=2, keepdim=True) # [B, *, 1]
            mean, std = compute_statistics(x, mask / total,stddev = self.stddev)
            mean = (mean.repeat(1, 1, T)).view(B, self.num_head, -1, T)
            x_in = x.view(B, self.num_head, -1, T)
            if self.stddev:
                std = (std.repeat(1, 1, T)).view(B, self.num_head, -1, T)
                x_in = torch.cat([x_in, mean, std], dim = 2)
            else:
                x_in = torch.cat([x_in, mean], dim = 2)
            x_in = x_in.reshape(B, -1, T)  
        else:
            x_in = x
        alpha = self.attention(x_in)   # [B, head * att_dim, T]

        alpha = alpha.masked_fill(mask == 0, float("-inf"))

        alpha = F.softmax(alpha, dim=2)
        alpha = alpha.reshape(B, self.num_head, self.num_q, -1, T)

        mean, std = compute_statistics(x.reshape(B, self.num_head, 1, -1, T), alpha,stddev = self.stddev)   # mean: [B, head, q, C/head, 1]
        mean = mean.reshape(B, -1, 1)
        if self.stddev:
            std = std.reshape(B, -1, 1)
            out =  torch.cat([mean, std], dim=1)
        else:
            out = mean
        return out
    def get_output_dim(self):
        return self.out_dim

    def build_attention(self,
                        idim,
                        odim,
                        num_q,
                        num_head,
                        affine_layers=1, 
                        hidden_size=128,
                        norm_type = 'batch_norm',
    ):
        assert affine_layers in [1 ,2], "Expected 1 or 2 affine layers."
        assert (idim % num_head) == 0 
        assert (odim % (num_head * num_q)) == 0

        if affine_layers == 2:
            if norm_type == 'batch_norm':
                norm = torch.nn.BatchNorm1d(hidden_size * num_head * num_q) 
            elif norm_type == 'layer_norm':
                norm =  torch.nn.GroupNorm(num_head * num_q, hidden_size * num_head * num_q)
            else:
                raise ValueError("Unsupport norm type".format(norm_type))
            att = torch.nn.Sequential(
                torch.nn.Conv1d(idim, hidden_size * num_head * num_q, kernel_size=1, groups=num_head),
                torch.nn.ReLU(),
                norm,              
                torch.nn.Tanh(),
                torch.nn.Conv1d(hidden_size * num_head * num_q, odim, kernel_size=1, groups=num_head * num_q)
            )
        elif affine_layers == 1 :
            att = torch.nn.Sequential(
                torch.nn.Conv1d(idim, odim, kernel_size=1, groups=num_head),
            )
        else:
            raise ValueError("Expected 1 or 2 affine layers, but got {}.".format(affine_layers))
        return att

    def extra_repr(self):
        return '(stddev={stddev}, num_head={num_head}, num_q={num_q}, out_dim={out_dim}) '.format(**self.__dict__)

# (Leo 2022-11-17)
class MQMHASP_Linear(torch.nn.Module):
    """ Linear version of MQMHASP means computing querys one by one, which can save memory but cost more time. 
     Reference:
        Miao Zhao, Yufeng Ma, and Yiwei Ding et al. "Multi-query multi-head attention pooling and Inter-topK penalty for speaker verification".
        https://arxiv.org/pdf/2110.05042.pdf   
    """
    def __init__(self, in_dim, 
                 num_q = 2,
                 stddev=True,
                 share = True,
                 num_head=4,
                 affine_layers=2,
                 hidden_size=128, 
                 **kargs
    ):
        super(MQMHASP_Linear,self).__init__()
        num_q = max(1, num_q)
        self.num_head = max(1, num_head)
        self.num_q = max(1, num_q)
        self.stddev=stddev
        self.querys = torch.nn.ModuleList(
            [
                MQMHASP(in_dim, 
                        num_q = 1,
                        hidden_size=hidden_size, 
                        stddev=stddev, 
                        share=share, 
                        num_head=num_head, 
                        affine_layers=affine_layers,
                        **kargs) for i in range(num_q)
            ]
        )
        self.out_dim = in_dim * num_q * 2 if stddev else in_dim * num_q

    def forward(self, x, mask: torch.Tensor = torch.ones((0, 0, 0))):
        """
            x: input feature [B, F, T] 
            returns: pooling statiscs [B, F * qs, 1]
        """
        out = []
        for i, layer in enumerate(self.querys):
            out.append(layer(x, mask))
        return torch.cat(out, dim=1)

    def get_output_dim(self):
        return self.out_dim
    def extra_repr(self):
        return '(stddev={stddev}, num_head={num_head}, num_q={num_q}, out_dim={out_dim}) '.format(**self.__dict__)