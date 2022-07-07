# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-05-29)


import numpy as np

import torch
import torch.nn.functional as F

from .activation import Nonlinearity

from libs.support.utils import to_device
import libs.support.utils as utils


### There are some basic custom components/layers. ###

## Base ✿
class TdnnAffine(torch.nn.Module):
    """ An implemented tdnn affine component by conv1d
        y = splice(w * x, context) + b

    @input_dim: number of dims of frame <=> inputs channels of conv
    @output_dim: number of layer nodes <=> outputs channels of conv
    @context: a list of context
        e.g.  [-2,0,2]
    If context is [0], then the TdnnAffine is equal to linear layer.
    """
    def __init__(self, input_dim, output_dim, context=[0], bias=True, pad=True, stride=1, groups=1, norm_w=False, norm_f=False):
        super(TdnnAffine, self).__init__()
        assert input_dim % groups == 0
        # Check to make sure the context sorted and has no duplicated values
        for index in range(0, len(context) - 1):
            if(context[index] >= context[index + 1]):
                raise ValueError("Context tuple {} is invalid, such as the order.".format(context))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context = context
        self.bool_bias = bias
        self.pad = pad
        self.groups = groups

        self.norm_w = norm_w
        self.norm_f = norm_f

        # It is used to subsample frames with this factor
        self.stride = stride

        self.left_context = context[0] if context[0] < 0 else 0 
        self.right_context = context[-1] if context[-1] > 0 else 0 

        self.tot_context = self.right_context - self.left_context + 1

        # Do not support sphereConv now.
        if self.tot_context > 1 and self.norm_f:
            self.norm_f = False
            print("Warning: do not support sphereConv now and set norm_f=False.")

        kernel_size = (self.tot_context,)

        self.weight = torch.nn.Parameter(torch.randn(output_dim, input_dim//groups, *kernel_size))

        # For jit compiling. 
        # 2021-07-08 Leo
        self.bias = torch.nn.Parameter(torch.randn(output_dim)) if self.bool_bias else None

        # if self.bool_bias:
        #     self.bias = torch.nn.Parameter(torch.randn(output_dim))
        # else:
            # self.register_parameter('bias', None)

        # init weight and bias. It is important
        self.init_weight()

        # Save GPU memory for no skiping case
        if len(context) != self.tot_context:
            # Used to skip some frames index according to context
            self.mask = torch.tensor([[[ 1 if index in context else 0 \
                                        for index in range(self.left_context, self.right_context + 1) ]]])
        else:
            self.mask = None

        ## Deprecated: the broadcast method could be used to save GPU memory, 
        # self.mask = torch.randn(output_dim, input_dim, 0)
        # for index in range(self.left_context, self.right_context + 1):
        #     if index in context:
        #         fixed_value = torch.ones(output_dim, input_dim, 1)
        #     else:
        #         fixed_value = torch.zeros(output_dim, input_dim, 1)

        #     self.mask=torch.cat((self.mask, fixed_value), dim = 2)

        # Save GPU memory of thi case.

        self.selected_device = False

    def init_weight(self):
        # Note, var should be small to avoid slow-shrinking
        torch.nn.init.normal_(self.weight, 0., 0.01)

        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.)


    def forward(self, inputs:torch.Tensor)-> torch.Tensor:
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3

        assert inputs.shape[1] == self.input_dim

        # Do not use conv1d.padding for self.left_context + self.right_context != 0 case.
        if self.pad:
            inputs = F.pad(inputs, (-self.left_context, self.right_context), mode="constant", value=0.0)

        assert inputs.shape[2] >=  self.tot_context

        # if not self.selected_device and self.mask is not None:
        #     # To save the CPU -> GPU moving time
        #     # Another simple case, for a temporary tensor, jus specify the device when creating it.
        #     self.mask = to_device(self, self.mask)

        #     
        #     self.selected_device = True
        #    filters = self.weight  * self.mask if self.mask is not None else self.weight


        # For jit compiling. 
        # 2021-07-08 Leo
        if self.mask is not None:
  
            self.mask=self.mask.to(inputs.device)
            filters=self.weight  * self.mask
        else:
            filters=self.weight


        if self.norm_w:
            filters = F.normalize(filters, dim=1)

        if self.norm_f:
            inputs = F.normalize(inputs, dim=1)

        outputs = F.conv1d(inputs, filters, self.bias, stride=self.stride, padding=0, dilation=1, groups=self.groups)

        return outputs

    def extra_repr(self):
        return '{input_dim}, {output_dim}, context={context}, bias={bool_bias}, stride={stride}, ' \
               'pad={pad}, groups={groups}, norm_w={norm_w}, norm_f={norm_f}'.format(**self.__dict__)

    @classmethod
    def thop_count(self, m, x, y):
        x = x[0]

        kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
        bias_ops = 1 if m.bias is not None else 0

        # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        total_ops = y.nelement() * (m.input_dim * kernel_ops + bias_ops)

        m.total_ops += torch.DoubleTensor([int(total_ops)])


class FTdnnBlock(torch.nn.Module):
    """ Factorized TDNN block w.r.t http://danielpovey.com/files/2018_interspeech_tdnnf.pdf.
    Reference: Povey, D., Cheng, G., Wang, Y., Li, K., Xu, H., Yarmohammadi, M., & Khudanpur, S. (2018). 
               Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks. Paper presented at the Interspeech.
    """
    def __init__(self, input_dim, output_dim, bottleneck_dim, context_size=0, bypass_scale=0.66, pad=True):
        super(FTdnnBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bottleneck_dim = bottleneck_dim
        self.context_size = context_size
        self.bypass_scale = bypass_scale
        self.pad = pad

        if context_size > 0:
            context_factor1 = [-context_size, 0]
            context_factor2 = [0, context_size]
        else:
            context_factor1 = [0]
            context_factor2 = [0]
        
        self.factor = TdnnAffine(input_dim, bottleneck_dim, context_factor1, pad=pad, bias=False)
        self.affine = TdnnAffine(bottleneck_dim, output_dim, context_factor2, pad=pad, bias=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.bn =torch.nn.BatchNorm1d(output_dim, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        identity = inputs

        out = self.factor(inputs)
        out = self.affine(out)
        out = self.relu(out)
        out = self.bn(out)
        if self.bypass_scale != 0:
            # assert identity.shape[1] == self.output_dim
            out += self.bypass_scale * identity

        return out


    '''
    Reference: https://github.com/cvqluu/Factorized-TDNN.
    '''
    def step_semi_orth(self):
        with torch.no_grad():
            M = self.get_semi_orth_weight(self.factor)
            self.factor.weight.copy_(M)

    @staticmethod
    def get_semi_orth_weight(conv1dlayer):
        # updates conv1 weight M using update rule to make it more semi orthogonal
        # based off ConstrainOrthonormalInternal in nnet-utils.cc in Kaldi src/nnet3
        # includes the tweaks related to slowing the update speed
        # only an implementation of the 'floating scale' case
        with torch.no_grad():
            update_speed = 0.125
            orig_shape = conv1dlayer.weight.shape
            # a conv weight differs slightly from TDNN formulation:
            # Conv weight: (out_filters, in_filters, kernel_width)
            # TDNN weight M is of shape: (in_dim, out_dim) or [rows, cols]
            # the in_dim of the TDNN weight is equivalent to in_filters * kernel_width of the Conv
            M = conv1dlayer.weight.reshape(
                orig_shape[0], orig_shape[1]*orig_shape[2]).T
            # M now has shape (in_dim[rows], out_dim[cols])
            mshape = M.shape
            if mshape[0] > mshape[1]:  # semi orthogonal constraint for rows > cols
                M = M.T
            P = torch.mm(M, M.T)
            PP = torch.mm(P, P.T)
            trace_P = torch.trace(P)
            trace_PP = torch.trace(PP)
            ratio = trace_PP * P.shape[0] / (trace_P * trace_P)

            # the following is the tweak to avoid divergence (more info in Kaldi)
            assert ratio > 0.99
            if ratio > 1.02:
                update_speed *= 0.5
                if ratio > 1.1:
                    update_speed *= 0.5

            scale2 = trace_PP/trace_P
            update = P - (torch.matrix_power(P, 0) * scale2)
            alpha = update_speed / scale2
            update = (-4.0 * alpha) * torch.mm(update, M)
            updated = M + update
            # updated has shape (cols, rows) if rows > cols, else has shape (rows, cols)
            # Transpose (or not) to shape (cols, rows) (IMPORTANT, s.t. correct dimensions are reshaped)
            # Then reshape to (cols, in_filters, kernel_width)
            return updated.reshape(*orig_shape) if mshape[0] > mshape[1] else updated.T.reshape(*orig_shape)

    @staticmethod
    def get_M_shape(conv_weight):
        orig_shape = conv_weight.shape
        return (orig_shape[1]*orig_shape[2], orig_shape[0])


class GruAffine(torch.nn.Module):
    """A GRU affine component.
    Author: Zheng Li xmuspeech 2020-02-05
    """
    def __init__(self, input_dim, output_dim):
        super(GruAffine, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        hidden_size = output_dim
        num_directions = 1

        self.hidden_size = hidden_size
        self.num_directions = num_directions

        self.gru = torch.nn.GRU(input_dim, hidden_size)


    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        The tensor of inputs in the GRU module is [seq_len, batch, input_size]
        The tensor of outputs in the GRU module is [seq_len, batch, num_directions * hidden_size]
        If the bidirectional is True, num_directions should be 2, else it should be 1.
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        inputs = inputs.permute(2,0,1)

        outputs, hn = self.gru(inputs)

        outputs = outputs.permute((1,2,0))

        return outputs


## Block ✿
class SoftmaxAffineLayer(torch.nn.Module):
    """ An usual 2-fold softmax layer with an affine transform.
    @dim: which dim to apply softmax on
    """
    def __init__(self, input_dim, output_dim, context=[0], dim=1, log=True, bias=True, groups=1, t=1., special_init=False):
        super(SoftmaxAffineLayer, self).__init__()

        self.affine = TdnnAffine(input_dim, output_dim, context=context, bias=bias, groups=groups)
        # A temperature parameter.
        self.t = t

        if log:
            self.softmax = torch.nn.LogSoftmax(dim=dim)
        else:
            self.softmax = torch.nn.Softmax(dim=dim)

        if special_init :
            torch.nn.init.xavier_uniform_(self.affine.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

    def forward(self, inputs):
        """
        @inputs: any, such as a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        return self.softmax(self.affine(inputs)/self.t)


## ReluBatchNormLayer
class _BaseActivationBatchNorm(torch.nn.Module):
    """[Affine +] Relu + BatchNorm1d.
    Affine could be inserted by a child class.
    """
    def __init__(self):
        super(_BaseActivationBatchNorm, self).__init__()
        self.affine = None
        self.activation = None
        self.batchnorm = None
        self.bn_relu = False
    def add_relu_bn(self, output_dim=None, options:dict={}):
        default_params = {
            "bn-relu":False,
            "nonlinearity":'relu',
            "nonlinearity_params":{"inplace":True, "negative_slope":0.01},
            "bn":True,
            "bn_params":{"momentum":0.1, "affine":True, "track_running_stats":True},
            "special_init":True,
            "mode":'fan_out',
            "jit_compile":False   # For jit compiling
        }

        default_params = utils.assign_params_dict(default_params, options)

        # This 'if else' is used to keep a corrected order when printing model.
        # torch.sequential is not used for I do not want too many layer wrappers and just keep structure as tdnn1.affine 
        # rather than tdnn1.layers.affine or tdnn1.layers[0] etc..
        if not default_params["bn-relu"]:
            # ReLU-BN
            # For speaker recognition, relu-bn seems better than bn-relu. And w/o affine (scale and shift) of bn is 
            # also better than w/ affine.
            # self.after_forward = self._relu_bn_forward
            self.bn_relu = False
            self.activation = Nonlinearity(default_params["nonlinearity"], **default_params["nonlinearity_params"])
            if default_params["bn"]:
                self.batchnorm = torch.nn.BatchNorm1d(output_dim, **default_params["bn_params"])
        else:
            # BN-ReLU
            # self.after_forward = self._bn_relu_forward
            self.bn_relu = True            
            if default_params["bn"]:
                self.batchnorm = torch.nn.BatchNorm1d(output_dim, **default_params["bn_params"])
            self.activation = Nonlinearity(default_params["nonlinearity"], **default_params["nonlinearity_params"])

        if default_params["special_init"] and self.affine is not None and not default_params["jit_compile"]:
            if default_params["nonlinearity"] in ["relu", "leaky_relu", "tanh", "sigmoid"]:
                # Before special_init, there is another initial way been done in TdnnAffine and it 
                # is just equal to use torch.nn.init.normal_(self.affine.weight, 0., 0.01) here. 
                if isinstance(self.affine, ChunkSeparationAffine):
                    torch.nn.init.kaiming_uniform_(self.affine.odd.weight, a=0, mode=default_params["mode"], 
                                               nonlinearity=default_params["nonlinearity"])
                    torch.nn.init.kaiming_uniform_(self.affine.even.weight, a=0, mode=default_params["mode"], 
                                               nonlinearity=default_params["nonlinearity"])
                else:
                    torch.nn.init.kaiming_uniform_(self.affine.weight, a=0, mode=default_params["mode"], 
                                               nonlinearity=default_params["nonlinearity"])
            else:
                torch.nn.init.xavier_normal_(self.affine.weight, gain=1.0)

    def _bn_relu_forward(self, x):
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def _relu_bn_forward(self, x):
        if self.activation is not None:
            x = self.activation(x)

        if self.batchnorm is not None:
            x = self.batchnorm(x)
        return x

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        x = self.affine(inputs)

        if self.bn_relu:
            outputs=self._bn_relu_forward(x)

        else:
            outputs=self._relu_bn_forward(x)

        # outputs = self.after_forward(x)
        return outputs


class ReluBatchNormTdnnLayer(_BaseActivationBatchNorm):
    """ TDNN-ReLU-BN.
    An usual 3-fold layer with TdnnAffine affine.
    """
    def __init__(self, input_dim, output_dim, context=[0], affine_type="tdnn", **options):
        super(ReluBatchNormTdnnLayer, self).__init__()

        affine_options = {
            "bias":True, 
            "groups":1,
            "norm_w":False,
            "norm_f":False
        }

        affine_options = utils.assign_params_dict(affine_options, options)

        # Only keep the order: affine -> layers.insert -> add_relu_bn,
        # the structure order will be right when print(model), such as follows:
        # (tdnn1): ReluBatchNormTdnnLayer(
        #          (affine): TdnnAffine()
        #          (activation): ReLU()
        #          (batchnorm): BatchNorm1d(512, eps=1e-05, momentum=0.5, affine=False, track_running_stats=True)
        if affine_type == "tdnn":
            self.affine = TdnnAffine(input_dim, output_dim, context=context, **affine_options)
        else:
            self.affine = ChunkSeparationAffine(input_dim, output_dim, context=context, **affine_options)

        self.add_relu_bn(output_dim, options=options)

        # Implement forward function extrally if needed when forward-graph is changed.


class ReluBatchNormTdnnfLayer(_BaseActivationBatchNorm):
    """ F-TDNN-ReLU-BN.
    An usual 3-fold layer with TdnnfBlock affine.
    """
    def __init__(self, input_dim, output_dim, inner_size, context_size = 0, **options):
        super(ReluBatchNormTdnnfLayer, self).__init__()

        self.affine = TdnnfBlock(input_dim, output_dim, inner_size, context_size)
        self.add_relu_bn(output_dim, options=options)



## Others ✿
class ImportantScale(torch.nn.Module):
    """A based idea to show importantance of every dim of inputs acoustic features.
    """
    def __init__(self, input_dim):
        super(ImportantScale, self).__init__()

        self.input_dim = input_dim
        self.groups = input_dim
        output_dim = input_dim

        kernel_size = (1,)

        self.weight = torch.nn.Parameter(torch.ones(output_dim, input_dim//self.groups, *kernel_size))

    def forward(self, inputs):
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        outputs = F.conv1d(inputs, self.weight, bias=None, groups=self.groups)
        return outputs


class AdaptivePCMN(torch.nn.Module):
    """ Using adaptive parametric Cepstral Mean Normalization to replace traditional CMN.
        It is implemented according to [Ozlem Kalinli, etc. "Parametric Cepstral Mean Normalization 
        for Robust Automatic Speech Recognition", icassp, 2019.]
    """
    def __init__(self, input_dim, left_context=-10, right_context=10, pad=True):
        super(AdaptivePCMN, self).__init__()

        assert left_context < 0 and right_context > 0

        self.left_context = left_context
        self.right_context = right_context
        self.tot_context = self.right_context - self.left_context + 1

        kernel_size = (self.tot_context,)

        self.input_dim = input_dim
        # Just pad head and end rather than zeros using replicate pad mode 
        # or set pad false with enough context egs. 
        self.pad = pad
        self.pad_mode = "replicate"

        self.groups = input_dim
        output_dim = input_dim

        # The output_dim is equal to input_dim and keep every dims independent by using groups conv.
        self.beta_w = torch.nn.Parameter(torch.randn(output_dim, input_dim//self.groups, *kernel_size))
        self.alpha_w = torch.nn.Parameter(torch.randn(output_dim, input_dim//self.groups, *kernel_size))
        self.mu_n_0_w = torch.nn.Parameter(torch.randn(output_dim, input_dim//self.groups, *kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(output_dim))

        # init weight and bias. It is important
        self.init_weight()

    def init_weight(self):
        torch.nn.init.normal_(self.beta_w, 0., 0.01)
        torch.nn.init.normal_(self.alpha_w, 0., 0.01)
        torch.nn.init.normal_(self.mu_n_0_w, 0., 0.01)
        torch.nn.init.constant_(self.bias, 0.)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim
        assert inputs.shape[2] >= self.tot_context

        if self.pad:
            pad_input = F.pad(inputs, (-self.left_context, self.right_context), mode=self.pad_mode)
        else:
            pad_input = inputs
            inputs = inputs[:,:,-self.left_context:-self.right_context]

        # outputs beta + 1 instead of beta to avoid potentially zeroing out the inputs cepstral features.
        self.beta = F.conv1d(pad_input, self.beta_w, bias=self.bias, groups=self.groups) + 1
        self.alpha = F.conv1d(pad_input, self.alpha_w, bias=self.bias, groups=self.groups)
        self.mu_n_0 = F.conv1d(pad_input, self.mu_n_0_w, bias=self.bias, groups=self.groups)

        outputs = self.beta * inputs - self.alpha * self.mu_n_0

        return outputs


class SEBlock(torch.nn.Module):
    """ A SE Block layer layer which can learn to use global information to selectively emphasise informative 
    features and suppress less useful ones.
    This is a pytorch implementation of SE Block based on the paper:
    Squeeze-and-Excitation Networks
    by JFChou xmuspeech 2019-07-13
       Snowdar xmuspeech 2020-04-28 [Check and update]
    """
    def __init__(self, input_dim, ratio=16, inplace=True):
        '''
        @ratio: a reduction ratio which allows us to vary the capacity and computational cost of the SE blocks 
        in the network.
        '''
        super(SEBlock, self).__init__()

        self.input_dim = input_dim

        self.fc_1 = TdnnAffine(input_dim, input_dim//ratio)
        self.relu = torch.nn.ReLU(inplace=inplace)
        self.fc_2 = TdnnAffine(input_dim//ratio, input_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        x = inputs.mean(dim=2, keepdim=True)
        x = self.relu(self.fc_1(x))
        scale = self.sigmoid(self.fc_2(x))

        return inputs * scale

class SEBlock_2D(torch.nn.Module):
    """ A SE Block layer layer which can learn to use global information to selectively emphasise informative 
    features and suppress less useful ones.
    This is a pytorch implementation of SE Block based on the paper:
    Squeeze-and-Excitation Networks
    by JFChou xmuspeech 2019-07-13
        leo 2020-12-20 [Check and update]
        """
    def __init__(self, in_planes, ratio=16, inplace=True):
        '''
        @ratio: a reduction ratio which allows us to vary the capacity and computational cost of the SE blocks 
        in the network.
        '''
        super(SEBlock_2D, self).__init__()

        self.in_planes = in_planes
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc_1 = torch.nn.Linear(in_planes, in_planes // ratio)
        self.relu = torch.nn.ReLU(inplace=inplace)
        self.fc_2 = torch.nn.Linear(in_planes // ratio, in_planes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 4
        assert inputs.shape[1] == self.in_planes

        b, c, _, _ = inputs.size()
        # print(b,c)
        # exit()
        x = self.avg_pool(inputs).view(b, c)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)

        scale = x.view(b, c, 1, 1)
        return inputs * scale

        
class MultiAffine(torch.nn.Module):
    """To complete.
    """
    def __init__(self, input_dim, output_dim, num=1, split_input=True, bias=True):
        super(MultiAffine, self).__init__()

        if not isinstance(num, int):
            raise TypeError("Expected an integer num, but got {}.".format(type(num).__name__))
        if num < 1:
            raise ValueError("Expected num >= 1, but got num={} .".format(num))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num = num
        self.bool_bias = bias

        if split_input:
            assert self.input_dim % self.num == 0
            self.num_feature_every_part = self.input_dim // self.num
        else:
            self.num_feature_every_part = input_dim

        self.weight = torch.nn.Parameter(torch.randn(1, self.num, self.output_dim, self.num_feature_every_part))

        if self.bool_bias:
            self.bias = torch.nn.Parameter(torch.randn(1, self.num, output_dim, 1))
        else:
            self.register_parameter('bias', None)
        
        self.init_weight()

    def init_weight(self):
        # Note, var should be small to avoid slow-shrinking
        torch.nn.init.normal_(self.weight, 0., 0.01)

        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.)

    def forward(self, inputs):
        # inputs [batch-size, num_head, num_feature_every_part, frames]
        x = inputs.reshape(inputs.shape[0], -1, self.num_feature_every_part, inputs.shape[2])
        y = torch.matmul(self.weight, x)

        if self.bias is not None:
            return (y + self.bias).reshape(inputs.shape[0], -1, inputs.shape[2])
        else:
            return y.reshape(inputs.shape[0], -1, inputs.shape[2])


class ChunkSeparationAffine(torch.nn.Module):
    """By this component, the chunk will be grouped to two parts, odd and even.
    """
    def __init__(self, input_dim, output_dim, **options):
        super(ChunkSeparationAffine, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.odd = TdnnAffine(input_dim, output_dim // 2, stride=2, **options)
        self.even = TdnnAffine(input_dim, output_dim // 2, stride=2, **options)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim
        
        if inputs.shape[2] % 2 != 0:
            # Make sure that the chunk length of inputs is an even number.
            inputs = F.pad(inputs, (0, 1), mode="constant", value=0.0)

        return torch.cat((self.odd(inputs), self.even(inputs[:,:,1:])), dim=1)


class Mixup(torch.nn.Module):
    """Implement a mixup component to augment data and increase the generalization of model training.
    Reference: 
        [1] Zhang, Hongyi, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. n.d. Mixup: BEYOND EMPIRICAL RISK MINIMIZATION.
        [2] Zhu, Yingke, Tom Ko, and Brian Mak. 2019. “Mixup Learning Strategies for Text-Independent Speaker Verification.”

    Github: https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
    """
    def __init__(self, alpha=1.0):
        super(Mixup, self).__init__()

        self.alpha = alpha

    def forward(self, inputs):
        if not self.training: return inputs

        batch_size = inputs.shape[0]
        self.lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0. else 1.
        # Shuffle the original index to generate the pairs, such as
        # Origin:           1 2 3 4 5
        # After Shuffling:  3 4 1 5 2
        # Then the pairs are (1, 3), (2, 4), (3, 1), (4, 5), (5,2).
        self.index = torch.randperm(batch_size, device=inputs.device)

        mixed_data = self.lam * inputs + (1 - self.lam) * inputs[self.index,:]

        return mixed_data




# https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/processing/signal_processing.py
# Leo 2021-09-27

class InputSequenceNormalization(torch.nn.Module):
    """Performs mean and variance normalization of the input tensor.

    Arguments
    ---------
    mean_norm : True
         If True, the mean will be normalized.
    std_norm : True
         If True, the standard deviation will be normalized.

    Example
    -------
    >>> import torch
    >>> norm = InputNormalization()
    >>> inputs = torch.randn([10, 101, 20])
    >>> inp_len = torch.ones([10])
    >>> features = norm(inputs, inp_len)
    """


    def __init__(
        self,
        mean_norm=True,
        std_norm=True,
    ):
        super().__init__()
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.eps = 1e-10
    def forward(self, x, lengths=None):
        """Returns the tensor with the surrounding context.

        Arguments
        ---------
        x : tensor
            A batch of tensors.
        lengths : tensor
            A batch of tensors containing the relative length of each
            sentence (e.g, [0.7, 0.9, 1.0]). It is used to avoid
            computing stats on zero-padded steps.
        """
        N_batches = x.shape[0]

        current_means = []
        current_stds = []
        if lengths is None:
            lengths=torch.ones(N_batches)
        for snt_id in range(N_batches):

            # Avoiding padded time steps
            actual_size = torch.round(lengths[snt_id] * x.shape[2]).int()
            # computing statistics
            current_mean, current_std = self._compute_current_stats(
                x[snt_id, ..., 0:actual_size]
            )

            current_means.append(current_mean)
            current_stds.append(current_std)



            x[snt_id] = (x[snt_id] - current_mean.data) / current_std.data


        return x

    def _compute_current_stats(self, x):
        """Returns the tensor with the surrounding context.

        Arguments
        ---------
        x : tensor
            A batch of tensors.
        """
        # Compute current mean
        if self.mean_norm:
            current_mean = torch.mean(x, dim=1,keepdim=True).detach().data
        else:
            current_mean = torch.tensor([0.0], device=x.device)

        # Compute current std
        if self.std_norm:
            current_std = torch.std(x, dim=1,keepdim=True).detach().data
        else:
            current_std = torch.tensor([1.0], device=x.device)

        # Improving numerical stability of std
        current_std = torch.max(
            current_std, self.eps * torch.ones_like(current_std)
        )

        return current_mean, current_std


    def to(self, device):
        """Puts the needed tensors in the right device.
        """
        self = super(InputSequenceNormalization, self).to(device)

        return self

 