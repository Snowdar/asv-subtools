# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-05)

import torch.nn
import torch.nn.functional as F
import libs.support.utils as utils

from libs.nnet import *


class Xvector(TopVirtualNnet):
    """ A standard x-vector framework """
    
    def init(self, inputs_dim, num_targets, bn_momentum=0.5, nonlinearity="relu", aug_dropout=0., 
             transformer_params = {"embed":False,
                                   "attention_dim":256,
                                   "attention_heads":4,
                                   "linear_units":2048,
                                   "num_blocks":6,
                                   "dropout_rate":0.1,
                                   "positional_dropout_rate":0.1,
                                   "attention_dropout_rate":0.0,
                                   "input_layer":"conv2d",
                                   "normalize_before":True,
                                   "concat_after":False,
                                   "positionwise_layer_type":"linear",
                                   "positionwise_conv_kernel_size":1,
                                   "padding_idx":-1},
             training=True, extracted_embedding="far"):

        # Var
        self.extracted_embedding = extracted_embedding
        
        # Nnet
        self.aug_dropout = torch.nn.Dropout2d(p=aug_dropout) if aug_dropout > 0 else None

        self.tdnn1 = ReluBatchNormTdnnLayer(inputs_dim,transformer_params["attention_dim"],momentum=bn_momentum,nonlinearity=nonlinearity)
        self.transformer_encoder = TransformerEncoder(inputs_dim, **transformer_params)
        self.stats = StatisticsPooling(transformer_params["attention_dim"], stddev=True)
        self.tdnn6 = ReluBatchNormTdnnLayer(self.stats.get_output_dim(),transformer_params["attention_dim"],momentum=bn_momentum,nonlinearity=nonlinearity)
        self.tdnn7 = ReluBatchNormTdnnLayer(transformer_params["attention_dim"],transformer_params["attention_dim"],momentum=bn_momentum,nonlinearity=nonlinearity)

        # Do not need when extracting embedding.
        if training :
            self.loss = SoftmaxLoss(transformer_params["attention_dim"], num_targets)

            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["transformer_encoder","stats","tdnn6","tdnn7"]

    @utils.for_device_free
    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index,  frames-dim-index, frames-index]
        """
        x = inputs
        x = self.auto(self.aug_dropout, x) # This auto function is equal to "x = layer(x) if layer is not None else x" for convenience.

        x = self.tdnn1(x)
        x, _ = self.transformer_encoder(x.transpose(1,2), None)
        x = self.stats(x.transpose(1,2))
        x = self.tdnn6(x)
        outputs = self.tdnn7(x)

        return outputs


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
        x, _ = self.transformer_encoder(x.transpose(1,2), None)
        x = self.stats(x.transpose(1,2))

        if self.extracted_embedding == "far" :
            xvector = self.tdnn6.affine(x)
        elif self.extracted_embedding == "near":
            x = self.tdnn6(x)
            xvector = self.tdnn7.affine(x)

        return xvector