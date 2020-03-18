# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-05)

import torch
import torch.nn.functional as F
import libs.support.utils as utils

from libs.nnet import *

class Xvector(TopVirtualNnet):
    """ A composite x-vector framework """
    
    ## Base parameters - components - loss - training strategy.
    def init(self, inputs_dim, num_targets, extend=False, skip_connection=False,
             aug_dropout=0., context_dropout=0., hidden_dropout=0., dropout_params={},
             SE=False, se_ratio=4,
             tdnn_layer_params={},
             tdnn6=True, tdnn7_params={},
             attentive_pooling=False, attentive_pooling_params={"hidden_size":64},
             LDE_pooling=False, LDE_pooling_params={"c_num":64, "nodes":128},
             focal_loss=False, focal_loss_params={"gamma":2},
             margin_loss=False, margin_loss_params={},
             use_step=False, step_params={},
             transfer_from="softmax_loss",
             training=True, extracted_embedding="far"):

        ## Params.
        default_dropout_params = {
            "type":"default", # default | random
            "start_p":0.,
            "dim":2,
            "method":"uniform", # uniform | normals
            "continuous":False,
            "inplace":True
        }

        default_tdnn_layer_params = {
            "momentum":0.5, "nonlinearity":'relu', "nonlinearity_params":{"inplace":True}
        }

        default_tdnn7_params = {
            "nonlinearity":"default", "bn":True, "bias":True
        }

        default_margin_loss_params = {
            "method":"am", "m":0.2, 
            "feature_normalize":True, "s":30, 
            "mhe_loss":False, "mhe_w":0.01
        }

        default_step_params = {
            "T":None, "record_T":0,
            "s":False, "s_tuple":(30, 12), "s_list":None,
            "t":False, "t_tuple":(0.5, 1.2), 
            "m":False, "lambda_0":0, "lambda_b":1000, "alpha":5, "gamma":1e-4,
            "p":False, "p_tuple":(0.5, 0.1)
        }

        dropout_params = utils.assign_params_dict(default_dropout_params, dropout_params)
        tdnn_layer_params = utils.assign_params_dict(default_tdnn_layer_params, tdnn_layer_params)
        tdnn7_params = utils.assign_params_dict(default_tdnn7_params, tdnn7_params)
        margin_loss_params = utils.assign_params_dict(default_margin_loss_params, margin_loss_params)
        step_params = utils.assign_params_dict(default_step_params, step_params)

        ## Var.
        self.skip_connection = skip_connection
        self.use_step = use_step
        self.step_params = step_params

        self.extracted_embedding = extracted_embedding # For extract.
        
        ## Nnet.
        # Head
        self.aug_dropout = get_dropout_from_wrapper(aug_dropout, dropout_params)
        self.context_dropout = ContextDropout(p=context_dropout) if context_dropout > 0 else None
        self.hidden_dropout = get_dropout_from_wrapper(hidden_dropout, dropout_params)

        # Frame level
        self.tdnn1 = ReluBatchNormTdnnLayer(inputs_dim,512,[-2,-1,0,1,2], **tdnn_layer_params)
        self.se1 = SEBlock(512, se_ratio=se_ratio) if SE else None
        self.ex_tdnn1 = ReluBatchNormTdnnLayer(512,512, **tdnn_layer_params) if extend else None
        self.ex_se1 = SEBlock(512, se_ratio=se_ratio) if SE and extend else None
        self.tdnn2 = ReluBatchNormTdnnLayer(512,512,[-2,0,2], **tdnn_layer_params)
        self.se2 = SEBlock(512, se_ratio=se_ratio) if SE else None
        self.ex_tdnn2 = ReluBatchNormTdnnLayer(512,512, **tdnn_layer_params) if extend else None
        self.ex_se2 = SEBlock(512, se_ratio=se_ratio) if SE and extend else None
        self.tdnn3 = ReluBatchNormTdnnLayer(512,512,[-3,0,3], **tdnn_layer_params)
        self.se3 = SEBlock(512, se_ratio=se_ratio) if SE else None
        self.ex_tdnn3 = ReluBatchNormTdnnLayer(512,512, **tdnn_layer_params) if extend else None
        self.ex_se3 = SEBlock(512, se_ratio=se_ratio) if SE and extend else None
        self.ex_tdnn4 = ReluBatchNormTdnnLayer(512,512,[-4,0,4], **tdnn_layer_params) if extend else None
        self.ex_se4 = SEBlock(512, se_ratio=se_ratio) if SE and extend else None
        self.ex_tdnn5 = ReluBatchNormTdnnLayer(512,512, **tdnn_layer_params) if extend else None
        self.ex_se5 = SEBlock(512, se_ratio=se_ratio) if SE and extend else None
        self.tdnn4 = ReluBatchNormTdnnLayer(512,512, **tdnn_layer_params)
        self.se4 = SEBlock(512, se_ratio=se_ratio) if SE else None

        nodes = LDE_pooling_params["nodes"] if LDE_pooling else 1500

        self.tdnn5 = ReluBatchNormTdnnLayer(512,nodes,**tdnn_layer_params)
        self.se5 = SEBlock(nodes, se_ratio=se_ratio) if SE else None

        # Pooling
        if LDE_pooling:
            self.stats = LDEPooling(nodes, **LDE_pooling_params)
        elif attentive_pooling:
            self.stats = AttentiveStatisticsPooling(nodes, **attentive_pooling_params, stddev=True)
        else:
            self.stats = StatisticsPooling(nodes, stddev=True)

        stats_dim = self.stats.get_output_dim()

        # Segment level
        if tdnn6:
            self.tdnn6 = ReluBatchNormTdnnLayer(stats_dim, 512, **tdnn_layer_params)
            tdnn7_dim = 512
        else:
            self.tdnn6 = None
            tdnn7_dim = stats_dim

        if tdnn7_params["nonlinearity"] == "default":
            tdnn7_params["nonlinearity"] = tdnn_layer_params["nonlinearity"]

        self.tdnn7 = ReluBatchNormTdnnLayer(tdnn7_dim,512, **tdnn7_params, momentum=tdnn_layer_params["momentum"])

        # Loss
        # Do not need when extracting embedding.
        if training :
            if margin_loss:
                self.loss = MarginSoftmaxLoss(512, num_targets, **margin_loss_params)
            elif focal_loss:
                self.loss = FocalLoss(512, num_targets, **focal_loss_params)
            else:
                self.loss = SoftmaxLoss(512, num_targets)

            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["tdnn1","tdnn2","tdnn3","tdnn4","tdnn5","stats","tdnn6","tdnn7",
                                   "ex_tdnn1","ex_tdnn2","ex_tdnn3","ex_tdnn4","ex_tdnn5","scale1",
                                   "se1","se2","se3","se4","se5","ex_se1","ex_se2","ex_se3","ex_se4","ex_se5",
                                   "loss"]

            if margin_loss and transfer_from == "softmax_loss":
                # For softmax_loss to am_softmax_loss
                self.rename_transform_keys = {"loss.affine.weight":"loss.weight"} 

    @utils.for_device_free
    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """

        x = inputs

        x = self.auto(self.aug_dropout, x)
        x = self.auto(self.context_dropout, x)

        x = self.tdnn1(x)
        if self.skip_connection:
            identity = x
        x = self.auto(self.se1, x)
        x = self.auto(self.ex_tdnn1, x)
        x = self.auto(self.ex_se1, x)
        x = self.tdnn2(x)
        x = self.auto(self.se2, x)
        x = self.auto(self.ex_tdnn2, x)
        x = self.auto(self.ex_se2, x)
        x = self.tdnn3(x)
        x = self.auto(self.se3, x)
        x = self.auto(self.ex_tdnn3, x)
        x = self.auto(self.ex_se3, x)
        x = self.auto(self.ex_tdnn4, x)
        x = self.auto(self.ex_se4, x)
        x = self.auto(self.ex_tdnn5, x)
        x = self.auto(self.ex_se5, x)
        x = self.tdnn4(x)
        x = self.auto(self.se4, x)
        if self.skip_connection:
            x += identity
        x = self.tdnn5(x)
        x = self.auto(self.se5, x)
        x = self.auto(self.hidden_dropout, x)
        x = self.stats(x)
        x = self.auto(self.tdnn6, x)
        x = self.tdnn7(x)

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

        x = self.tdnn1(x)
        if self.skip_connection:
            identity = x
        x = self.auto(self.se1, x)
        x = self.auto(self.ex_tdnn1, x)
        x = self.auto(self.ex_se1, x)
        x = self.tdnn2(x)
        x = self.auto(self.se2, x)
        x = self.auto(self.ex_tdnn2, x)
        x = self.auto(self.ex_se2, x)
        x = self.tdnn3(x)
        x = self.auto(self.se3, x)
        x = self.auto(self.ex_tdnn3, x)
        x = self.auto(self.ex_se3, x)
        x = self.auto(self.ex_tdnn4, x)
        x = self.auto(self.ex_se4, x)
        x = self.auto(self.ex_tdnn5, x)
        x = self.auto(self.ex_se5, x)
        x = self.tdnn4(x)
        x = self.auto(self.se4, x)
        if self.skip_connection:
            x += identity
        x = self.tdnn5(x)
        x = self.auto(self.se5, x)
        x = self.stats(x)

        if self.extracted_embedding == "far" :
            assert self.tdnn6 is not None
            xvector = self.tdnn6.affine(x)
        elif self.extracted_embedding == "near":
            x = self.auto(self.tdnn6, x)
            xvector = self.tdnn7.affine(x)

        return xvector


    def compute_decay_value(self, start, end, current_postion, T):
        return start - (start - end)/(T-1) * (current_postion%T)


    def step(self, epoch, this_iter, epoch_batchs):
        # heated up for t, s, m
        if self.use_step:
            if self.step_params["T"] is not None:
                if self.step_params["record_T"] < self.step_params["T"][epoch]:
                    self.current_epoch = epoch*epoch_batchs
                    self.T = self.step_params["T"][epoch] * epoch_batchs
                    self.step_params["record_T"] = self.step_params["T"][epoch]

                current_postion = self.current_epoch + this_iter

            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(*self.step_params["t_tuple"], current_postion, self.T)

            if self.step_params["s"]:
                self.loss.s = self.step_params["s_tuple"][self.step_params["s_list"][epoch]]

            if self.step_params["m"]:
                lambda_factor = max(self.step_params["lambda_0"], 
                                 self.step_params["lambda_b"]*(1+self.step_params["gamma"]*current_postion)**(-self.step_params["alpha"]))
                self.loss.step(lambda_factor)

            if self.step_params["p"]:
                self.aug_dropout.p = self.compute_decay_value(*self.step_params["p_tuple"], current_postion, self.T)
            



