# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2021-12-28)
# refs:
# 1.  Repvgg: Making vgg-style convnets great again
#           https://arxiv.org/abs/2101.03697
# 2.  PyTorch implementation.
#           https://github.com/DingXiaoH/RepVGG
# 3.  RepSPK: Ma Y, Zhao M, Ding Y, et al. Rep Works in Speaker Verification[J]
#           https://arxiv.53yu.com/abs/2110.09720

import torch.nn as nn
import numpy as np
import torch
import copy
import torch.nn.functional as F
from libs.nnet.components import SEBlock_2D


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1,norm_layer_params={}):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels,**norm_layer_params))
    return result



class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False,norm_layer_params={}):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU(inplace=True)

        if use_se:
            self.se = SEBlock_2D(out_channels, 4)

        else:
            self.se = nn.Identity()
        self.rbr_reparam=None
        self.rbr_identity=None
        self.rbr_dense=None
        self.rbr_1x1=None
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:            
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels,**norm_layer_params) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,norm_layer_params=norm_layer_params)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=1, stride=stride, padding=padding_11, groups=groups,norm_layer_params=norm_layer_params)


    def forward(self, inputs):
        if self.deploy and self.rbr_reparam is not None:

            return self.se(self.nonlinearity(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        if self.rbr_dense is not None and self.rbr_1x1 is not None:
            return self.se(self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
        else:
             raise TypeError("It's a training repvgg structure but branch conv not exits.")

    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()

    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var +
              self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var +
              self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        # The equivalent resultant central point of 3x3 kernel.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        # Normalize for an L2 coefficient comparable to regular L2.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()
        return l2_loss_eq_kernel + l2_loss_circle


#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(
                    kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if self.rbr_reparam is not None:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.rbr_dense=None
        self.rbr_1x1=None
        if hasattr(self, 'rbr_identity'):
            self.rbr_identity=None
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

class RepSPKBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, branch_dilation=2, groups=1, padding_mode='zeros', deploy=False, use_se=False,norm_layer_params={}):
        super(RepSPKBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1
        assert dilation == 1
        assert branch_dilation == 2
        self.branch_dilation = branch_dilation
        self.depoly_kernel_size = (kernel_size-1)*(branch_dilation-1)+kernel_size

        self.nonlinearity = nn.ReLU(inplace=True)

        if use_se:
            self.se = SEBlock_2D(out_channels, 4)

        else:
            self.se = nn.Identity()
        self.rbr_reparam=None
        self.rbr_identity=None
        self.rbr_dense=None
        self.rbr_dense_dilation=None
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.depoly_kernel_size, stride=stride,
                                         padding=self.branch_dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:            
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels,**norm_layer_params) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,norm_layer_params=norm_layer_params)
            self.rbr_dense_dilation = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=kernel_size, stride=stride, padding=self.branch_dilation, dilation=self.branch_dilation,groups=groups,norm_layer_params=norm_layer_params)


    def forward(self, inputs):
        if self.deploy and self.rbr_reparam is not None:

            return self.se(self.nonlinearity(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        if self.rbr_dense is not None and self.rbr_dense_dilation is not None:
            return self.se(self.nonlinearity(self.rbr_dense(inputs) + self.rbr_dense_dilation(inputs) + id_out))
        else:
             raise TypeError("It's a training repvgg structure but branch conv not exits.")
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel_dilation_branch, bias_dilation_branch = self._fuse_bn_tensor(self.rbr_dense_dilation)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return self._convert_3x3_dilation_to_5x5_tensor(kernel_dilation_branch) + self._pad_3x3_to_5x5_tensor(kernel3x3) + kernelid, bias3x3 + bias_dilation_branch + biasid

    def _pad_3x3_to_5x5_tensor(self, kernel3x3):
        if kernel3x3 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel3x3, [1, 1, 1, 1])

    def _convert_3x3_dilation_to_5x5_tensor(self,kernel3x3):
        if kernel3x3 is None:
            return 0
        else:
            kernel_value = torch.zeros((kernel3x3.size(0),kernel3x3.size(1),5,5), dtype=kernel3x3.dtype)
            kernel_value[:,:,::2,::2] = kernel3x3
            return kernel_value


    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 5, 5), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 2, 2] = 1
                self.id_tensor = torch.from_numpy(
                    kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if self.rbr_reparam is not None:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.depoly_kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.branch_dilation, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.rbr_dense=None
        self.rbr_dense_dilation=None
        if hasattr(self, 'rbr_identity'):
            self.rbr_identity=None
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RepVGG(nn.Module):

    def __init__(self, head_inplanes,block="RepVGG",num_blocks=[2, 4, 14, 1],strides=[1,1,2,2,2], base_width=64,width_multiplier=None, override_groups_map=None, deploy=False, use_se=False,norm_layer_params={}):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4
        assert len(num_blocks) == 4
        assert len(strides) == 5

        width_multiplier=[w *(base_width/64.) for w in width_multiplier]
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se
        self.norm_layer_params = norm_layer_params
        self.downsample_multiple = 1
        if block == "RepVGG":
            used_block = RepVGGBlock
        elif block == "RepSPK":
            used_block = RepSPKBlock
        else:
            raise TypeError("Do not support {} block.".format(block))
    
        for s in strides:
             self.downsample_multiple*=s

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = used_block(head_inplanes, out_channels=self.in_planes,
                                  kernel_size=3, stride=strides[0], padding=1, deploy=self.deploy, use_se=self.use_se,norm_layer_params=self.norm_layer_params)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(used_block,
            int(64 * width_multiplier[0]), num_blocks[0], stride=strides[1])
        self.stage2 = self._make_stage(used_block,
            int(128 * width_multiplier[1]), num_blocks[1], stride=strides[2])
        self.stage3 = self._make_stage(used_block,
            int(256 * width_multiplier[2]), num_blocks[2], stride=strides[3])
        self.stage4 = self._make_stage(used_block,
            int(512 * width_multiplier[3]), num_blocks[3], stride=strides[4])
        self.output_planes = self.in_planes
        
        if "affine" in norm_layer_params.keys():
            norm_layer_affine = norm_layer_params["affine"]
        else:
            norm_layer_affine = True # torch.nn default it True

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0., 0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)) and norm_layer_affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)        
        
    def _make_stage(self, block,planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(block(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se,norm_layer_params=self.norm_layer_params))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def get_downsample_multiple(self):
        return self.downsample_multiple

    def get_output_planes(self):
        return self.output_planes

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x

def repvgg_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
