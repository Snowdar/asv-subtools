# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-07-01)

import torch.nn
import libs.support.utils as utils


#### Use 'from libs.nnet import *' in your model.py to use all components and loss functions.

## Function
def for_extract_embedding(maxChunk=10000, isMatrix=True):
    """
    A decorator for extract_embedding class-function to wrap some common process codes like Kaldi's x-vector extractor.
    Used in TopVirtualNnet.
    """
    def wrapper(function):
        def _wrapper(self, input):
            """
            @input: a 3-dimensional tensor with batch-dim=1 or [frames, feature-dim] matrix for 
                    acoustic features only
            @return: an 1-dimensional vector 
            """
            train_status = self.training
            self.eval()

            with torch.no_grad():
                if isMatrix:
                    input = torch.tensor(input)
                    input = torch.unsqueeze(input, dim = 0)
                    input = input.transpose(1,2)

                input = utils.to_device(self, input)
                num_frames = input.shape[2]
                num_split = (num_frames + maxChunk - 1) // maxChunk
                split_size = num_frames // num_split
                
                offset = 0
                embedding_stats = 0.
                for i in range(0, num_split-1):
                    this_embedding = function(self, input[:, :, offset:offset+split_size])
                    offset += split_size
                    embedding_stats += split_size*this_embedding

                last_embedding = function(self, input[:, :, offset:])

                embedding = (embedding_stats + (num_frames-offset) * last_embedding) / num_frames

                if train_status:
                    self.train()

                return torch.squeeze(embedding.transpose(1,2)).cpu()

        return _wrapper
    return wrapper


# Relation: activation -> components -> loss -> framework

## Framework
class TopVirtualNnet(torch.nn.Module):
    """This is a virtual nnet framework at top level and it is applied to the pipline scripts.
    And you should implement four functions after inheriting this object.

    @init(): just like pytorch needed. Note there is 'init' rather than '__init__'.
    @forward(*inputs): just like pytorch needed.
    @get_loss(*inputs, targets) : to support fetching the final loss from multi-loss.
    @get_posterior(): to compute accuracy.
    @extract_embedding(inputs, isMatrix=True) : needed if use pipline/onestep/extract_embeddings.py.
    """
    
    def __init__(self, *args, **kwargs):
        super(TopVirtualNnet, self).__init__()
        params_dict = locals()
        model_name = str(params_dict["self"]).split("()")[0]
        args_str = utils.iterator_to_params_str(params_dict['args'])
        kwargs_str = utils.dict_to_params_str(params_dict['kwargs'])

        self.model_creation = "{0}({1},{2})".format(model_name, args_str, kwargs_str)

        self.loss = None
        self.use_step = False
        self.transform_keys = []
        self.rename_transform_keys = {}
        self.init(*args, **kwargs)


    def init(self, *args, **kwargs):
        raise NotImplementedError

    def get_model_creation(self):
        return self.model_creation

    # You could use this decorator if needed in class function overwriting 
    @utils.for_device_free
    def forward(self, *inputs):
        raise NotImplementedError


    # You could use this decorator if needed in class function overwriting 
    @utils.for_device_free
    def get_loss(self, *inputs, targets):
        """
        @return: return a loss tensor, such as return form torch.nn.CrossEntropyLoss(reduction='mean')
        """
        return self.loss(*inputs, targets)


    def get_posterior(self):
        """
        @return: return posterior
        """
        return self.loss.get_posterior()


    def auto(self, layer, x):
        """It is convenient for forward-computing when layer could be None or not
        """
        return layer(x) if layer is not None else x


    def load_transform_state_dict(self, state_dict):
        """It is used in transform-learning.
        """
        assert isinstance(self.transform_keys, list)
        assert isinstance(self.rename_transform_keys, dict)

        remaining = { utils.key_to_value(self.rename_transform_keys, k, False):v for k,v in state_dict.items() if k.split('.')[0]  \
                     in self.transform_keys or k in self.transform_keys }
        self.load_state_dict(remaining, strict=False)

        return self


    # We could use this decorator if needed when overwriting class function. 
    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_embedding(self, inputs):
        """ If use the decorator, should note:
        @inputs: a 3-dimensional tensor with batch-dim=1 or [frames, feature-dim] matrix for 
                acoustic features only
        @return: an 1-dimensional vector 
        """
        raise NotImplementedError

    @utils.for_device_free
    def predict(self, outputs):
        """
        @outputs: the outputs tensor with [batch-size,n,1] shape comes from affine before computing softmax or 
                  just softmax for n classes
        @return: an 1-dimensional vector including class-id (0-based) for prediction
        """
        with torch.no_grad():
            prediction = torch.squeeze(torch.argmax(outputs, dim=1))

        return prediction


    @utils.for_device_free
    def compute_accuracy(self, outputs, targets):
        """
        @outputs: the outputs tensor with [batch-size,n,1] shape comes from  affine before computing softmax or 
                 just softmax for n classes
        @return: the float accuracy
        """
        assert outputs.shape[0] == len(targets)

        with torch.no_grad():
            prediction = self.predict(outputs)
            num_correct = (targets==prediction).sum()

        return num_correct.item()/len(targets)
    
    def step(self, epoch, this_iter, epoch_batchs):
        pass




