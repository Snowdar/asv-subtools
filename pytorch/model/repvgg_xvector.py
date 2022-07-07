# -*- coding:utf-8 -*-

# Copyright xmuspeech (Leo 2021-12-01)

import sys
import math
import torch
import torch.nn.functional as F
sys.path.insert(0, 'subtools/pytorch')

import libs.support.utils as utils
from libs.nnet import *

class RepVggXvector(TopVirtualNnet):
    """ A repvgg vector framework """

    def init(self, inputs_dim, num_targets, embd_dim=256,aug_dropout=0., tail_dropout=0.,training=True, extracted_embedding="near", 
             deploy=False,repvgg_config={}, pooling="statistics", pooling_params={}, fc1=False, fc1_params={}, fc2_params={}, margin_loss=False, margin_loss_params={},
             use_step=False, step_params={}, adacos=False,transfer_from="softmax_loss"):

        ## Params.
        default_repvgg_config = {
            "auto_model" : False,
            "auto_model_name" : "RepVGG_B0",
            "block": "RepVGG",
                "repvgg_params":{
                    "num_blocks": [4, 6, 16, 1],
                    "strides":[1,1,2,2,2],
                    "base_width": 32,
                    "width_multiplier":[1, 1, 1, 2.5],
                    "norm_layer_params":{"momentum":0.5, "affine":True},
                    "override_groups_map": None,
                    "use_se": False,
                }
        }
        default_pooling_params = {
            "num_head":1,
            "hidden_size":64,
            "share":True,
            "affine_layers":1,
            "context":[0],
            "stddev":True,
            "temperature":False, 
            "fixed":True
        }
        
        default_fc_params = {
            "nonlinearity":'relu', "nonlinearity_params":{"inplace":True},
            "bn-relu":False, 
            "bn":True, 
            "bn_params":{"momentum":0.5, "affine":True, "track_running_stats":True}
            }

        default_margin_loss_params = {
            "method":"am", "m":0.2, "feature_normalize":True, 
            "s":30, "mhe_loss":False, "mhe_w":0.01
            }
        
        default_step_params = {
            "T":None,
            "m":False, "lambda_0":0, "lambda_b":1000, "alpha":5, "gamma":1e-4,
            "s":False, "s_tuple":(30, 12), "s_list":None,
            "t":False, "t_tuple":(0.5, 1.2), 
            "p":False, "p_tuple":(0.5, 0.1)
            }

        repvgg_config = utils.assign_params_dict(default_repvgg_config, repvgg_config)
        if repvgg_config["auto_model"]:
            repvgg_params=auto_model(repvgg_config["auto_model_name"])
        else:
            repvgg_params=repvgg_config["repvgg_params"]
        repvgg_params["deploy"] = deploy
        repvgg_params["block"] = repvgg_config["block"]

        pooling_params = utils.assign_params_dict(default_pooling_params, pooling_params)
        fc1_params = utils.assign_params_dict(default_fc_params, fc1_params)
        fc2_params = utils.assign_params_dict(default_fc_params, fc2_params)
        margin_loss_params = utils.assign_params_dict(default_margin_loss_params, margin_loss_params)
        step_params = utils.assign_params_dict(default_step_params, step_params)

        ## Var.
        self.extracted_embedding = extracted_embedding # only near here.
        self.use_step = use_step
        self.step_params = step_params

        ## Nnet.
        self.aug_dropout = torch.nn.Dropout2d(p=aug_dropout) if aug_dropout > 0 else None

        inplanes = 1
 
        self.repvgg=RepVGG(inplanes,**repvgg_params)
        
                # It is just equal to Ceil function.
        repvgg_output_dim = (inputs_dim + self.repvgg.get_downsample_multiple() - 1) // self.repvgg.get_downsample_multiple() \
                            * self.repvgg.get_output_planes()

       # Pooling
        stddev = pooling_params.pop("stddev")
        if pooling == "lde":
            self.stats = LDEPooling(repvgg_output_dim, c_num=pooling_params["num_head"])
        elif pooling == "attentive":
            self.stats = AttentiveStatisticsPooling(repvgg_output_dim, hidden_size=pooling_params["hidden_size"], 
                                                    context=pooling_params["context"], stddev=stddev)
        elif pooling == "multi-head":
            self.stats = MultiHeadAttentionPooling(repvgg_output_dim, stddev=stddev, **pooling_params)
        elif pooling == "multi-resolution":
            self.stats = MultiResolutionMultiHeadAttentionPooling(repvgg_output_dim, **pooling_params)
        else:
            self.stats = StatisticsPooling(repvgg_output_dim, stddev=stddev)

        self.fc1 = ReluBatchNormTdnnLayer(self.stats.get_output_dim(), embd_dim, **fc1_params) if fc1 else None
        
        if fc1:
            fc2_in_dim = embd_dim
        else:
            fc2_in_dim = self.stats.get_output_dim()

        self.fc2 = ReluBatchNormTdnnLayer(fc2_in_dim, embd_dim, **fc2_params)

        self.tail_dropout = torch.nn.Dropout2d(p=tail_dropout) if tail_dropout > 0 else None

        self.embd_dim=embd_dim
        self.deploy=deploy
        ## Do not need when extracting embedding.
        if training :
            if margin_loss:
                self.loss = MarginSoftmaxLoss(embd_dim, num_targets, **margin_loss_params)
            elif adacos:
                self.loss = AdaCos(embd_dim,num_targets)
            else:
                self.loss = SoftmaxLoss(embd_dim, num_targets)

            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["repvgg", "stats", "fc1", "fc2"]
            # self.transform_keys = ["resnet"]

            if margin_loss and transfer_from == "softmax_loss":
                # For softmax_loss to am_softmax_loss
                self.rename_transform_keys = {"loss.affine.weight":"loss.weight"}

    @torch.jit.unused
    @utils.for_device_free
    def forward(self, x):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        x = self.auto(self.aug_dropout, x) # This auto function is equal to "x = layer(x) if layer is not None else x" for convenience.
        # [samples-index, frames-dim-index, frames-index] -> [samples-index, 1, frames-dim-index, frames-index]
        x = x.unsqueeze(1)
        # print("unsqueeze",x.shape)
        x = self.repvgg(x)

        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])

        x = self.stats(x) 
        x = self.auto(self.fc1, x)
        x = self.fc2(x)
        x = self.auto(self.tail_dropout, x)

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
    def extract_embedding(self, x):
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """


        x = x.unsqueeze(1)
        x = self.repvgg(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        x = self.stats(x)

        if self.extracted_embedding == "far":
            assert self.fc1 is not None
            xvector = self.fc1.affine(x)
        elif self.extracted_embedding == "near_affine":
            x = self.auto(self.fc1, x)
            xvector = self.fc2.affine(x)
        elif self.extracted_embedding == "near":
            x = self.auto(self.fc1, x)
            xvector = self.fc2(x)
            # xvector = F.normalize(xvector)

        else:
            raise TypeError("Expected far or near position, but got {}".format(self.extracted_embedding))

        return xvector

    def extract_embedding_jit(self, x: torch.Tensor, position: str = 'near') -> torch.Tensor:
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """

        # Tensor shape is not modified in libs.nnet.resnet.py for calling free, such as using this framework in cv.
        x = x.unsqueeze(1)
        x = self.repvgg(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        x = self.stats(x)

        if position == "far" and self.fc1 is not None:
            xvector = self.fc1.affine(x)
        elif position == "near_affine":
            if self.fc1 is not None:
                x=self.fc1(x)
            xvector = self.fc2.affine(x)
        elif position == "near":
            if self.fc1 is not None:
                x=self.fc1(x)
            xvector = self.fc2(x)
            # xvector = F.normalize(xvector)

        else:
            raise TypeError("Expected far or near position, but got {}".format(position))

        return xvector

    @torch.jit.export
    def extract_embedding_whole(self,input:torch.Tensor,position:str='near',maxChunk:int=10000,isMatrix:bool=True):
        if isMatrix:
            input=torch.unsqueeze(input,dim=0)
            input=input.transpose(1,2)
        num_frames = input.shape[2]
        num_split = (num_frames + maxChunk - 1) // maxChunk
        split_size = num_frames // num_split
        offset=0
        embedding_stats = torch.zeros(1,self.embd_dim,1)
        for _ in range(0, num_split-1):
            this_embedding = self.extract_embedding_jit(input[:, :, offset:offset+split_size],position)
            offset += split_size
            embedding_stats+=split_size*this_embedding

        last_embedding = self.extract_embedding_jit(input[:, :, offset:],position)
 
        embedding = (embedding_stats + (num_frames-offset) * last_embedding) / num_frames
        return torch.squeeze(embedding.transpose(1,2)).cpu()

    @torch.jit.export
    def embedding_dim(self) -> int:
        """ Export interface for c++ call, return embedding dim of the model
        """

        return self.embd_dim

    def get_warmR_T(self, T_0, T_mult, epoch):
        n = int(math.log(max(0.05, (epoch / T_0 * (T_mult - 1) + 1)), T_mult))
        T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
        T_i = T_0 * T_mult ** (n)
        return T_cur, T_i


    def compute_decay_value(self, start, end, T_cur, T_i):
        # Linear decay in every cycle time.
        return start - (start - end)/(T_i-1) * (T_cur%T_i)


    def step(self, epoch, this_iter, epoch_batchs):
        # Heated up for t and s.
        # Decay for margin and dropout p.
        if self.use_step:
            if self.step_params["m"]:
                current_postion = epoch*epoch_batchs + this_iter
                lambda_factor = max(self.step_params["lambda_0"], 
                                 self.step_params["lambda_b"]*(1+self.step_params["gamma"]*current_postion)**(-self.step_params["alpha"]))
                self.loss.step(lambda_factor)

            if self.step_params["T"] is not None and (self.step_params["t"] or self.step_params["p"]):
                T_cur, T_i = self.get_warmR_T(*self.step_params["T"], epoch)
                T_cur = T_cur*epoch_batchs + this_iter
                T_i = T_i * epoch_batchs

            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(*self.step_params["t_tuple"], T_cur, T_i)

            if self.step_params["p"]:
                self.aug_dropout.p = self.compute_decay_value(*self.step_params["p_tuple"], T_cur, T_i)

            if self.step_params["s"]:
                self.loss.s = self.step_params["s_tuple"][self.step_params["s_list"][epoch]]

    def step_iter(self, epoch, cur_step):
        # For iterabledataset
        if self.use_step:
            if self.step_params["m"]:
                lambda_factor = max(self.step_params["lambda_0"],
                                 self.step_params["lambda_b"]*(1+self.step_params["gamma"]*cur_step)**(-self.step_params["alpha"]))
                self.loss.step(lambda_factor)

            if self.step_params["T"] is not None and (self.step_params["t"] or self.step_params["p"]):
                T_cur, T_i = self.get_warmR_T(*self.step_params["T"], cur_step)


            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(*self.step_params["t_tuple"], T_cur, T_i)

            if self.step_params["p"]:
                self.aug_dropout.p = self.compute_decay_value(*self.step_params["p_tuple"], T_cur, T_i)

            if self.step_params["s"]:
                self.loss.s = self.step_params["s_tuple"][self.step_params["s_list"][epoch]]


def repvgg_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model




def auto_model(model_name):
    optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    g2_map = {l: 2 for l in optional_groupwise_layers}
    g4_map = {l: 4 for l in optional_groupwise_layers}
    model_params_dict={
        'RepVGG_A0':{
            "num_blocks": [2, 4, 14, 1],
            "strides":[1,1,2,2,2],
            "base_width": 64,
            "width_multiplier":[0.75, 0.75, 0.75, 2.5],
            "override_groups_map": None,
            "norm_layer_params":{"momentum":0.5, "affine":True},
        },
        'RepVGG_A1':{
            "num_blocks": [2, 4, 14, 1],
            "strides":[1,1,2,2,2],
            "base_width": 64,
            "width_multiplier":[1, 1, 1, 2.5],
            "norm_layer_params":{"momentum":0.5, "affine":True},
            "override_groups_map": None
        },
        'RepVGG_A2':{
            "num_blocks": [2, 4, 14, 1],
            "strides":[1,1,2,2,2],
            "base_width": 64,
            "width_multiplier":[1.5, 1.5, 1.5, 2.75],
            "norm_layer_params":{"momentum":0.5, "affine":True},
            "override_groups_map": None
        },
        'RepVGG_B0':{
            "num_blocks": [4, 6, 16, 1],
            "strides":[1,1,2,2,2],
            "base_width": 64,
            "width_multiplier":[1, 1, 1, 2.5],
            "norm_layer_params":{"momentum":0.5, "affine":True},
            "override_groups_map": None,
            "use_se": False,
            
        },
        'RepVGG_B1':{
            "num_blocks": [4, 6, 16, 1],
            "strides":[1,1,2,2,2],
            "base_width": 64,
            "width_multiplier":[2, 2, 2, 4],
            "norm_layer_params":{"momentum":0.5, "affine":True},
            "override_groups_map": None
        },

        'RepVGG_B1g2':{
            "num_blocks": [4, 6, 16, 1],
            "strides":[1,1,2,2,2],
            "base_width": 64,
            "width_multiplier":[2, 2, 2, 4],
            "norm_layer_params":{"momentum":0.5, "affine":True},
            "override_groups_map": g2_map
        },
        'RepVGG_B1g4':{
            "num_blocks": [4, 6, 16, 1],
            "strides":[1,1,2,2,2],
            "base_width": 64,
            "width_multiplier":[2, 2, 2, 4],
            "norm_layer_params":{"momentum":0.5, "affine":True},
            "override_groups_map": g4_map
        },
        'RepVGG_B2':{
            "num_blocks": [4, 6, 16, 1],
            "strides":[1,1,2,2,2],
            "base_width": 64,
            "width_multiplier":[2.5, 2.5, 2.5, 5],
            "norm_layer_params":{"momentum":0.5, "affine":True},
            "override_groups_map": None
        },
        "RepVGG_B2g2":{
            "num_blocks": [4, 6, 16, 1],
            "strides":[1,1,2,2,2],
            "base_width": 64,
            "width_multiplier":[2.5, 2.5, 2.5, 5],
            "norm_layer_params":{"momentum":0.5, "affine":True},
            "override_groups_map": g2_map

        },
        "RepVGG_B2g4":{
            "num_blocks": [4, 6, 16, 1],
            "strides":[1,1,2,2,2],
            "base_width": 64,
            "width_multiplier":[2.5, 2.5, 2.5, 5],
            "norm_layer_params":{"momentum":0.5, "affine":True},
            "override_groups_map": g4_map

        },
        "RepVGG_B3":{
            "num_blocks": [4, 6, 16, 1],
            "strides":[1,1,2,2,2],
            "base_width": 64,
            "width_multiplier":[3, 3, 3, 5],
            "norm_layer_params":{"momentum":0.5, "affine":True},
            "override_groups_map": None

        },
        "RepVGG_B3g2":{
            "num_blocks": [4, 6, 16, 1],
            "strides":[1,1,2,2,2],
            "base_width": 64,
            "width_multiplier":[3, 3, 3, 5],
            "norm_layer_params":{"momentum":0.5, "affine":True},
            "override_groups_map": g2_map

        },
        "RepVGG_B3g4":{
            "num_blocks": [4, 6, 16, 1],
            "strides":[1,1,2,2,2],
            "base_width": 64,
            "width_multiplier":[3, 3, 3, 5],
            "norm_layer_params":{"momentum":0.5, "affine":True},
            "override_groups_map": g4_map

        },
        "RepVGG_D2se":{
            "num_blocks": [8, 14, 24, 1],
            "strides":[1,1,2,2,2],
            "base_width": 64,
            "width_multiplier":[2.5, 2.5, 2.5, 5],
            "norm_layer_params":{"momentum":0.5, "affine":True},
            "override_groups_map": None,
            "use_se": True
        },
    }

    model_params=model_params_dict[model_name]
    

    return model_params

# Test.
if __name__ == "__main__":
    # tensor = torch.randn(1, 40, 200)
    import time
    # with torch.no_grad():
    #     model=RepVggXvector(40,1000)
    #     model.eval()
    #     out1=model(tensor)
    #     start=time.time()
    #     out1=model(tensor)
    #     middel=time.time()

    #     for module in model.modules():
    #         if hasattr(module, 'switch_to_deploy'):
    #             module.switch_to_deploy()
    #     out2=model(tensor)
    #     start1= time.time()     
    #     out2=model(tensor)
    #     end=time.time()
        

    #     print('========================== The diff is')
    #     print(((out2 - out1) ** 2).sum())
    #     print(middel-start,end-start1)
    from model.resnet_se_xvector import ResNetXvector


    resnet_params = {
        "aug_dropout":0., "tail_dropout":0.,
        "training":True, "extracted_embedding":"far",
        "cmvn":False,
        "cmvn_params":{
                "mean_norm" : True,
                "std_norm" : False,},
        "resnet_params":{
                "head_conv":True, "head_conv_params":{"kernel_size":3, "stride":1, "padding":1},
                "head_maxpool":False, "head_maxpool_params":{"kernel_size":3, "stride":2, "padding":1},
                "block":"BasicBlock", # BasicBlock, Bottleneck
                "layers":[3, 4, 6, 3],
                "planes":[32, 64, 128, 256],
                "convXd":2,
                "norm_layer_params":{"momentum":0.5, "affine":True},
                "full_pre_activation":False,
                "zero_init_residual":False},

        "pooling":"statistics", # statistics, lde, attentive, multi-head, multi-resolution
        "pooling_params":{"num_head":16,
                        "share":True,
                        "affine_layers":1,
                        "hidden_size":64,
                        "context":[0],
                        "stddev":True,
                        "temperature":False, 
                        "fixed":True
                        },

        "fc1":True,
        "fc1_params":{
                "nonlinearity":'relu', "nonlinearity_params":{"inplace":True},
                "bn-relu":False, 
                "bn":True, 
                "bn_params":{"momentum":0.5, "affine":False, "track_running_stats":True}},

        "fc2_params":{
                "nonlinearity":'relu', "nonlinearity_params":{"inplace":True},
                "bn-relu":False, 
                "bn":True, 
                "bn_params":{"momentum":0.5, "affine":False, "track_running_stats":True}},
    }

    repvgg_params = {
        "aug_dropout":0., "tail_dropout":0.,
        "training":False, "extracted_embedding":"near",
        "deploy": False,
        "repvgg_config":{
                "auto_model" : False,
                "auto_model_name" : "RepVGG_B0",
                "block": "RepVGG",
                "repvgg_params":{
                    "num_blocks": [2, 4, 14, 1],
                    "strides":[1,1,2,2,2],
                    "base_width": 32,
                    "width_multiplier":[1, 1, 1, 2.5],
                    "override_groups_map": None,
                    "use_se": False,
                    "norm_layer_params":{"momentum":0.5, "affine":True},
                }
            
            },

        "pooling":"statistics", # statistics, lde, attentive, multi-head, multi-resolution
        "pooling_params":{"num_head":1,
                        "share":True,
                        "affine_layers":1,
                        "hidden_size":64,
                        "context":[0],
                        "stddev":True,
                        "temperature":False, 
                        "fixed":True
                        },

        "fc1":True,
        "fc1_params":{
                "nonlinearity":'relu', "nonlinearity_params":{"inplace":True},
                "bn-relu":False, 
                "bn":True, 
                "bn_params":{"momentum":0.5, "affine":False, "track_running_stats":True}},

        "fc2_params":{
                "nonlinearity":'relu', "nonlinearity_params":{"inplace":True},
                "bn-relu":False, 
                "bn":True, 
                "bn_params":{"momentum":0.5, "affine":False, "track_running_stats":True}},

    }


    repvggm = RepVggXvector(80,1000,**repvgg_params)
    resnet34 = ResNetXvector(80,1000,training=False)  
 

    a = torch.randn(2000, 80)


    m = torch.jit.script(repvggm)

    m.save("repvgg.pt")
    m2 = torch.jit.load("repvgg.pt")
    m2.eval()
    repvggm.eval()
    resnet34.eval()
    res11 = repvggm.extract_embedding(a)
    res111=resnet34.extract_embedding(a)
    with torch.no_grad():
        ori_start=time.time()
        res1 = repvggm.extract_embedding(a)
        ori_end=time.time()
        res2 = m2.extract_embedding_whole(a)
        resnet_start=time.time()
        resnet_res=resnet34.extract_embedding(a)
        resnet_end=time.time()
        
    print("embedding_dim:{}\n".format(resnet_res.shape))
    for module in repvggm.modules():        
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()

    res33=repvggm.extract_embedding(a)

    with torch.no_grad():
        deploy_start=time.time()
        res3=repvggm.extract_embedding(a)    
        deploy_end=time.time()
    print(res1)
    print(res3)
    deploy_m=torch.jit.script(repvggm)
    deploy_m.save("repvgg_deploy.pt")
    deploy_m2=torch.jit.load("repvgg_deploy.pt")
    deploy_m2.eval()
    with torch.no_grad():
        res4=deploy_m2.extract_embedding_whole(a)
    train_forward_time=ori_end-ori_start
    deploy_forward_time=deploy_end-deploy_start
    resnet_forward_time=resnet_end-resnet_start
    print('========================== The diff between train repvgg and jit train repvgg is')
    print(((res1 - res2) ** 2).sum())
    print('========================== The diff between train repvgg and deploy repvgg is')
    print(((res1 - res3) ** 2).sum())
    print('========================== The diff between deploy repvgg and jit deploy repvgg is')
    print(((res3 - res4) ** 2).sum())
    print("train repvgg forward time:{}\n deploy repvgg forward time:{}\n frac:{}".format(train_forward_time,deploy_forward_time,train_forward_time/deploy_forward_time))
    print('==========================') 
    print("resnet34 forward time:{}\n deploy repvgg forward time:{}\n frac:{}".format(resnet_forward_time,deploy_forward_time,resnet_forward_time/deploy_forward_time))


    
    print("Test done.")