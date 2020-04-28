# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-05)

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "subtools/pytorch")

import libs.support.utils as utils
from libs.nnet import *


class Xvector(TopVirtualNnet):
    """ A standard x-vector framework """
    
    def init(self, inputs_dim, num_targets, nonlinearity="relu", aug_dropout=0.2, training=True, extracted_embedding="far"):

        # Var
        self.extracted_embedding = extracted_embedding
        
        # Nnet
        self.aug_dropout = torch.nn.Dropout2d(p=aug_dropout) if aug_dropout > 0 else None

        self.tdnn1 = ReluBatchNormTdnnLayer(inputs_dim,512,[-2,-1,0,1,2],nonlinearity=nonlinearity)
        self.tdnn2 = ReluBatchNormTdnnLayer(512,512,[-2,0,2],nonlinearity=nonlinearity)
        self.tdnn3 = ReluBatchNormTdnnLayer(512,512,[-3,0,3],nonlinearity=nonlinearity)
        self.tdnn4 = ReluBatchNormTdnnLayer(512,512,nonlinearity=nonlinearity)
        self.tdnn5 = ReluBatchNormTdnnLayer(512,1500,nonlinearity=nonlinearity)
        self.stats = StatisticsPooling(1500, stddev=True)
        self.tdnn6 = ReluBatchNormTdnnLayer(self.stats.get_output_dim(),512,nonlinearity=nonlinearity)
        self.tdnn7 = ReluBatchNormTdnnLayer(512,512,nonlinearity=nonlinearity)

        # Do not need when extracting embedding.
        if training :
            self.loss = SoftmaxLoss(512, num_targets)

            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["tdnn1","tdnn2","tdnn3","tdnn4","tdnn5","stats","tdnn6","tdnn7"]

    @utils.for_device_free
    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index,  frames-dim-index, frames-index]
        """
        x = inputs
        x = self.auto(self.aug_dropout, x) # This auto function is equal to "x = layer(x) if layer is not None else x" for convenience.

        x = self.tdnn1(x)   
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        x = self.stats(x)
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
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        x = self.stats(x)

        if self.extracted_embedding == "far" :
            xvector = self.tdnn6.affine(x)
        elif self.extracted_embedding == "near":
            x = self.tdnn6(x)
            xvector = self.tdnn7.affine(x)

        return xvector

# Test.
if __name__ == "__main__":
    print(Xvector(23, 1211))
