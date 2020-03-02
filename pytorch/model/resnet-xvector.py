# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-28)

import torch.nn
import torch.nn.functional as F
import libs.support.utils as utils

from libs.nnet import *


class ResnetXvector(TopVirtualNnet):
    """ A resnet x-vector framework """
    
    def init(self, inputs_dim, num_targets, aug_dropout=0.2, training=True, extracted_embedding="near", 
             resnet_params={}, fc_params={}, margin_loss=False, margin_loss_params={},
             use_step=False, step_params={}, transfer_from="softmax_loss"):

        ## params
        default_resnet_params = {
            "head_conv":True, "head_conv_params":{"kernel_size":3, "stride":1, "padding":1},
            "block":"BasicBlock",
            "layers":[3, 4, 6, 3],
            "planes":[32, 64, 128, 256], # a.k.a channels.
            "full_pre_activation":True,
            "zero_init_residual":False
            }
        
        default_fc_params = {
            "nonlinearity":"relu",
            "bn":True,
            "bias":True,
            "bn_momentum":0.5
            }

        default_margin_loss_params = {
            "method":"am", "m":0.2, "feature_normalize":True, 
            "s":30, "mhe_loss":False, "mhe_w":0.01
            }
        
        default_step_params = {
            "t":False, "s":False, "m":False, 
            "T":None, "record_T":0, "t_tuple":(0.5, 1.2), 
            "s_tuple":(30, 12), "m_tuple":(0, 0.2)
            }

        resnet_params = utils.assign_params_dict(default_resnet_params, resnet_params)
        fc_params = utils.assign_params_dict(default_fc_params, fc_params)
        margin_loss_params =utils.assign_params_dict(default_margin_loss_params, margin_loss_params)
        step_params = utils.assign_params_dict(default_step_params, step_params)


        ## Var.
        self.extracted_embedding = extracted_embedding # only near here.
        self.use_step = use_step
        self.step_params = step_params
        
        ## Nnet.
        self.aug_dropout = torch.nn.Dropout2d(p=aug_dropout) if aug_dropout > 0 else None

        self.resnet = ResNet(**resnet_params)

        # It is just equal to Ceil function.
        resnet_output_dim = (inputs_dim + self.resnet.get_downsample_multiple() - 1) // self.resnet.get_downsample_multiple() \
                            * resnet_params["planes"][3]

        self.stats = StatisticsPooling(resnet_output_dim, stddev=True)
        self.fc = ReluBatchNormTdnnLayer(self.stats.get_output_dim(), resnet_params["planes"][3], **fc_params)

        ## Do not need when extracting embedding.
        if training :
            if margin_loss:
                self.loss = MarginSoftmaxLoss(resnet_params["planes"][3], num_targets, **margin_loss_params)
            else:
                self.loss = SoftmaxLoss(resnet_params["planes"][3], num_targets)

            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["resnet","stats","fc"]

            if margin_loss and transfer_from == "softmax_loss":
                # For softmax_loss to am_softmax_loss
                self.rename_transform_keys = {"loss.affine.weight":"loss.weight"} 

    @utils.for_device_free
    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        x = inputs
        x = self.auto(self.aug_dropout, x) # This auto function is equal to "x = layer(x) if layer is not None else x" for convenience.
        # [samples-index, frames-dim-index, frames-index] -> [samples-index, 1, frames-dim-index, frames-index]
        x = self.resnet(x.unsqueeze(1))
        # [samples-index, channel, frames-dim-index, frames-index] -> [samples-index, channel*frames-dim-index, frames-index]
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        x = self.stats(x) 
        x = self.fc(x)

        return x


    @utils.for_device_free
    def get_loss(self, inputs, targets):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)
        """
        return self.loss(inputs, targets)

    def get_posterior(self):
        """Should call get_posterior after get_loss. This function is to get outputs from loss component.
        @return: return posterior
        """
        return self.loss.get_posterior()

    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_embedding(self, inputs):
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """

        x = inputs
        x = self.resnet(x.unsqueeze(1))
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        x = self.stats(x)

        if self.extracted_embedding == "near":
            xvector = self.fc.affine(x)
        else:
            raise TypeError("Expected near position, but got {}".format(self.extracted_embedding))

        return xvector


    def compute_decay_value(self, start, end, current_postion, T):
        return start - (start - end)/(T-1) * (current_postion%T)


    def step(self, epoch, this_iter, epoch_batchs):
        # heated up for t, s, m
        if self.use_step:
            if self.step_params["record_T"] < self.step_params["T"][epoch]:
                self.current_epoch = epoch*epoch_batchs
                self.T = self.step_params["T"][epoch] * epoch_batchs
                self.step_params["record_T"] = self.step_params["T"][epoch]

            current_postion = self.current_epoch + this_iter

            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(*self.step_params["t_tuple"], current_postion, self.T)
            if self.step_params["s"]:
                self.loss.s = self.compute_decay_value(*self.step_params["s_tuple"], current_postion, self.T)
            if self.step_params["m"]:
                self.loss.m = self.compute_decay_value(*self.step_params["m_tuple"], current_postion, self.T)



