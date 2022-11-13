# Copyright xmuspeech (Author: Leo 2022-07-18)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.insert(0, 'subtools/pytorch')
import libs.support.utils as utils
from libs.nnet import *

def compute_statistics(x, m, dim: int=2, stddev: bool=True,eps: float=1e-5):
    mean = (m * x).sum(dim)


    if stddev:
        # std = torch.sqrt(
        #     (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps) 
        # )
        std = torch.sqrt(
            (torch.sum(m * (x ** 2), dim=dim) - mean ** 2).clamp(eps)
        )
    else:
        std = torch.empty(0)
    return mean, std

class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, hidden_size=128, time_attention=False, stddev=True):
        super().__init__()
        self.stddev = stddev
        self.output_dim = in_dim*2 if self.stddev else in_dim
        self.time_attention = time_attention
        accept_dim = in_dim
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        if time_attention:
            
            accept_dim = in_dim*3 if self.stddev else in_dim*2

        norm =  LayerNorm(hidden_size,dim=1,eps=1e-5)
        self.attention = nn.Sequential(
            nn.Conv1d(accept_dim, hidden_size, kernel_size=1),
            nn.ReLU(),
            norm,
            nn.Tanh(),
            nn.Conv1d(hidden_size, in_dim, kernel_size=1)
        )
 
        # gn_num = 2 if self.stddev else 1
        # self.norm_stats = nn.GroupNorm(gn_num,self.output_dim )
        self.norm_stats = LayerNorm(self.output_dim,dim=1)
        

    def forward(self, x, mask: torch.Tensor = torch.ones((0, 0, 0))):
        B, C ,T = x.shape

        if mask.size(2) == 0 :
            mask = torch.ones((B, 1, T)).to(x.device)

        if self.time_attention:
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = compute_statistics(x, mask / total,stddev = self.stddev)
            mean = mean.unsqueeze(2).repeat(1, 1, T)
            if self.stddev:
                std = std.unsqueeze(2).repeat(1, 1, T)
                x_in = [x,mean,std]
            else:
                x_in = [x,mean]

            x_in = torch.cat(x_in, dim=1)

        else:
            x_in = x

        alpha = self.attention(x_in)

        alpha = alpha.masked_fill(mask == 0, float("-inf"))

        alpha = F.softmax(alpha, dim=2)

        mean, std = compute_statistics(x, alpha,stddev = self.stddev)
        if self.stddev:

            out =  torch.cat([mean, std], dim=1).unsqueeze(2)
        else:
            out = mean.unsqueeze(2)

        return self.norm_stats(out)
    def get_output_dim(self):
        return self.output_dim
        

class TransformerXvector(TopVirtualNnet):
    def init(self, inputs_dim, num_targets, embd_dim=256,training=True,
             extracted_embedding="near", mixup=False, mixup_alpha=1.0, pooling="ecpa-attentive", pooling_params={},
             transformer_type="conformer", transformer_params={},tansformer_out={}, fc1=False, fc1_params={}, fc2_params={},
             margin_loss=True, margin_loss_params={}, lsm_weight=0.0,use_step=False, step_params={}, transfer_from="softmax_loss",wenet_transfer=False):

        default_transformer_params = {
            "attention_dim": 256,
            "att_type": 'multi',                      # [multi, gau] gau attention don't suppport rel_pos.
            "attention_heads": 4,
            "gau_key": 64,                            # gau key dim.
            "gau_units": 512,
            "num_blocks": 6,
            "dropout_rate": 0.1,
            "layer_dropout":0.,
            "positionwise_layer_type": 'linear',      # [linear, conv1d, conv1d-linear, gau, re_conv1d]
            "positional_dropout_rate": 0.1,
            "linear_units": 2048,
            "positionwise_conv_kernel_size": 3,
            "attention_dropout_rate": 0.0,
            "attention_norm_args": {
                "norm_method": "softmax",             # [softmax, relu_plus, softmax_plus]
                "train_len":300.,                     # for softmax_plus.
            },
            "input_layer": "conv2d",                  # [linear, conv2d2, conv2d, re_conv2d, conv2d6, conv2d8]
            "pos_enc_type": "abs_pos",                # [abs_pos, no_pos, rot_pos, rel_pos]
            "cnn_module_kernel": 15,                  # for conformer
            "use_cnn_module": True,                   # for conformer
            "cnn_module_norm": 'layer_norm',          # for conformer ['batch_norm', 'layer_norm']
            "static_chunk_size": 0,
            "left_chunk_size": -1,
            "use_dynamic_chunk": False,
            "use_dynamic_left_chunk": False,
            "combiner_type": "norm",                  #  [norm, mfa, random_frame, random_layer]
            "convfnn_blocks": 0
        }
        default_tansformer_out = {
            "out_dim": 1536,
            "nonlinearity": 'swish', "nonlinearity_params": {"inplace": True},
            "bn-relu": False,
            "bn": True,
            "ln_replace": True, # replace BN with LN
            "bn_params": {"momentum": 0.5, "affine": True, "track_running_stats": True}
        }

        default_pooling_params = {
            "hidden_size": 128,
            "time_attention": False,
            "stddev": True,
        }

        default_fc_params = {
            "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
            "bn-relu": False,
            "bn": True,
            "ln_replace": True, # replace BN with LN
            "bn_params": {"momentum": 0.5, "affine": True, "track_running_stats": True}
        }

        default_margin_loss_params = {
            "method": "am", "m": 0.2,
            "feature_normalize": True, "s": 30,
            "double": False,
            "mhe_loss": False, "mhe_w": 0.01,
            "inter_loss": 0.,
            "ring_loss": 0.,
            "curricular": False
            }

        default_step_params = {
            "margin_warm":False,
            "margin_warm_conf":{"start_epoch":5.,"end_epoch":10.,"offset_margin":-0.2,"init_lambda":0.0},
            "T": None,
            "m": True, "lambda_0": 0, "lambda_b": 1000, "alpha": 5, "gamma": 1e-4,
            "s": False, "s_tuple": (30, 12), "s_list": None,
            "t": False, "t_tuple": (0.5, 1.2),
            "p": False, "p_tuple": (0.5, 0.1)
        }

        self.use_step = use_step
        self.step_params = step_params
        self.extracted_embedding = extracted_embedding
        
        transformer_params = utils.assign_params_dict(
            default_transformer_params, transformer_params,support_unknow=True)
        tansformer_out = utils.assign_params_dict(default_tansformer_out, tansformer_out)
        pooling_params = utils.assign_params_dict(
            default_pooling_params, pooling_params)
        fc1_params = utils.assign_params_dict(default_fc_params, fc1_params)
        fc2_params = utils.assign_params_dict(default_fc_params, fc2_params)
        margin_loss_params = utils.assign_params_dict(
            default_margin_loss_params, margin_loss_params)
        step_params = utils.assign_params_dict(
            default_step_params, step_params)
        self.embd_dim = embd_dim
        self.mixup = Mixup(alpha=mixup_alpha) if mixup else None

        if transformer_type == "transformer":
            transformer_backbone =  TransformerEncoder
        elif transformer_type == "conformer":
            transformer_backbone =  ConformerEncoder
        elif transformer_type == "re_conformer":
            transformer_backbone =  ReConformerEncoder
        else:
            raise ValueError("unknown transformer_type: " + transformer_type)
        self.transformer = transformer_backbone(inputs_dim,**transformer_params)

        self.transform_out = ReluBatchNormTdnnLayer(self.transformer.output_size(),tansformer_out["out_dim"],**tansformer_out)
        # Pooling
        stddev = pooling_params.pop("stddev")

        if pooling == "ecpa-attentive":
            self.stats = AttentiveStatsPool(
                tansformer_out["out_dim"],  stddev=stddev,**pooling_params)

            self.fc1 = ReluBatchNormTdnnLayer(
                self.stats.get_output_dim(), embd_dim, **fc1_params) if fc1 else None
        else:
            raise ValueError("Only supoort asp for conformer now.")



        if fc1:
            fc2_in_dim = embd_dim
        else:
            fc2_in_dim = self.stats.get_output_dim()
        self.fc2 = ReluBatchNormTdnnLayer(fc2_in_dim, embd_dim, **fc2_params)



        # print("num_targets---------------",num_targets)
        # Loss
        # Do not need when extracting embedding.
        if training:
            if margin_loss:
                self.loss = MarginSoftmaxLoss(
                    embd_dim, num_targets, label_smoothing=lsm_weight,**margin_loss_params)
                if self.use_step and self.step_params["margin_warm"]:
                    self.margin_warm = MarginWarm(**step_params["margin_warm_conf"])
        
            else:
                self.loss = SoftmaxLoss(embd_dim, num_targets,label_smoothing=lsm_weight)
                # self.loss = AngleLoss(embd_dim,num_targets)
            self.wrapper_loss = MixupLoss(
                self.loss, self.mixup) if mixup else None
            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["transformer", "transform_out", "stats", "fc1", "fc2", "loss"]

            if margin_loss and transfer_from == "softmax_loss":
                # For softmax_loss to am_softmax_loss
                self.rename_transform_keys = {
                    "loss.affine.weight": "loss.weight"}
            self.wenet_transfer = wenet_transfer
    def load_transform_state_dict(self, state_dict):
        """It is used in transform-learning.
        """
        assert isinstance(self.transform_keys, list)
        assert isinstance(self.rename_transform_keys, dict)
        remaining = {}
        for k,v in state_dict.items():
            
            # if "train_len" in k:
            #     print(k,v)
            if self.wenet_transfer:
                
                k = k.replace("encoder.","transformer.")

                # k = k.replace("embed.","noembed.")
            if k.split('.')[0]  in self.transform_keys or k in self.transform_keys:
                k = utils.key_to_value(self.rename_transform_keys, k, False)
                remaining[k] = v
        # for k in remaining.keys():
        #     print(k)

        # assert 1==0

        self.load_state_dict(remaining, strict=False)
        return self


    @torch.jit.unused
    @utils.for_device_free
    def forward(self, x, x_len,warmup: torch.Tensor=torch.FloatTensor([1.0])):
        # [samples-index, frames-dim-index, frames-index] -> [samples-index, frames-index, frames-dim-index]
        x = x.transpose(1,2)

        x, masks = self.transformer(x,x_len,warmup=float(warmup))

        x = x.transpose(1,2)

        x = self.transform_out(x)

        x = self.stats(x,masks)
        if len(x.shape) != 3:
            x = x.unsqueeze(dim=2)

        with torch.cuda.amp.autocast(enabled=False):
            x = self.auto(self.fc1, x)
            x = self.fc2(x)

        return x

    @utils.for_device_free
    def get_loss(self, inputs, targets):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)

        model.get_loss [custom] -> loss.forward [custom]
          |
          v
        model.get_accuracy [custom] -> loss.get_accuracy [custom] -> loss.compute_accuracy [static] -> loss.predict [static]
        """
        if self.wrapper_loss is not None:
            return self.wrapper_loss(inputs, targets)
        else:
            return self.loss(inputs, targets)

    @utils.for_device_free
    def get_accuracy(self, targets):
        """Should call get_accuracy() after get_loss().
        @return: return accuracy
        """
        if self.wrapper_loss is not None:
            return self.wrapper_loss.get_accuracy(targets)
        else:
            return self.loss.get_accuracy(targets)

    @for_extract_embedding(maxChunk=300, isMatrix=True)
    def extract_embedding(self, x):
        x_lens = torch.LongTensor([x.shape[2]]).to(x.device)

        x = x.transpose(1,2)
        x, _ = self.transformer(x,x_lens)

        x = x.transpose(1,2)
        x = self.transform_out(x)
        x = self.stats(x)

        if len(x.shape) != 3:
            x = x.unsqueeze(dim=2)
        if self.extracted_embedding == "far":
            assert self.fc1 is not None
            xvector = self.fc1.affine(x)
        elif self.extracted_embedding == "near_affine":
            x = self.auto(self.fc1, x)
            xvector = self.fc2.affine(x)
        elif self.extracted_embedding == "near":
            x = self.auto(self.fc1, x)
            xvector = self.fc2(x)
        else:
            raise TypeError("Expected far or near position, but got {}".format(
                self.extracted_embedding))
        return xvector




    def extract_embedding_jit(self, x: torch.Tensor, position: str = 'near') -> torch.Tensor:
        x_lens = torch.tensor([x.shape[2]]).to(x.device)

        x = x.transpose(1,2)
        
        x, _ = self.transformer(x,x_lens)

        x = x.transpose(1,2)

        x = self.transform_out(x)
        x = self.stats(x)
        if len(x.shape) != 3:
            x = x.unsqueeze(dim=2)
        if position == "far" and self.fc1 is not None:
            xvector = self.fc1.affine(x)
        elif position == "near_affine":
            if self.fc1 is not None:
                x = self.fc1(x)
            xvector = self.fc2.affine(x)
        elif position == "near":
            if self.fc1 is not None:
                x = self.fc1(x)
            xvector = self.fc2(x)
        else:
            raise TypeError("Expected far or near position, but got {}".format(
                self.extracted_embedding))
        return xvector

    @torch.jit.export
    def extract_embedding_whole(self, input: torch.Tensor, position: str = 'near', maxChunk: int = 4000, isMatrix: bool = True):
        with torch.no_grad():
            if isMatrix:
                input = torch.unsqueeze(input, dim=0)
                input = input.transpose(1, 2)
            num_frames = input.shape[2]
            num_split = (num_frames + maxChunk - 1) // maxChunk
            split_size = num_frames // num_split
            offset = 0
            embedding_stats = torch.zeros(1, self.embd_dim, 1).to(input.device)
            for _ in range(0, num_split-1):
                this_embedding = self.extract_embedding_jit(
                    input[:, :, offset:offset+split_size], position)
                offset += split_size
                embedding_stats += split_size*this_embedding

            last_embedding = self.extract_embedding_jit(
                input[:, :, offset:], position)

            embedding = (embedding_stats + (num_frames-offset)
                        * last_embedding) / num_frames
            return torch.squeeze(embedding.transpose(1, 2)).cpu()

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
        return start - (start - end)/(T_i-1) * (T_cur % T_i)

    def step(self, epoch, this_iter, epoch_batchs):
        # Heated up for t and s.
        # Decay for margin and dropout p.
        if self.use_step:
            if self.step_params["m"]:
                current_postion = epoch*epoch_batchs + this_iter
                lambda_factor = max(self.step_params["lambda_0"],
                                    self.step_params["lambda_b"]*(1+self.step_params["gamma"]*current_postion)**(-self.step_params["alpha"]))
                lambda_m = 1/(1 + lambda_factor)
                self.loss.step(lambda_m)

            if self.step_params["T"] is not None and (self.step_params["t"] or self.step_params["p"]):
                T_cur, T_i = self.get_warmR_T(*self.step_params["T"], epoch)
                T_cur = T_cur*epoch_batchs + this_iter
                T_i = T_i * epoch_batchs

            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(
                    *self.step_params["t_tuple"], T_cur, T_i)

            if self.step_params["p"]:
                self.aug_dropout.p = self.compute_decay_value(
                    *self.step_params["p_tuple"], T_cur, T_i)

            if self.step_params["s"]:
                self.loss.s = self.step_params["s_tuple"][self.step_params["s_list"][epoch]]

    def step_iter(self, epoch, cur_step):
        # For iterabledataset
        if self.use_step:
            if self.step_params["margin_warm"]:
                offset_margin, lambda_m = self.margin_warm.step(cur_step)
                lambda_m = max(1e-3,lambda_m)
                self.loss.step(lambda_m,offset_margin)
            if self.step_params["m"]:
                lambda_factor = max(self.step_params["lambda_0"],
                                    self.step_params["lambda_b"]*(1+self.step_params["gamma"]*cur_step)**(-self.step_params["alpha"]))
                lambda_m = 1/(1 + lambda_factor)
                self.loss.step(lambda_m)

            if self.step_params["T"] is not None and (self.step_params["t"] or self.step_params["p"]):
                T_cur, T_i = self.get_warmR_T(*self.step_params["T"], cur_step)

            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(
                    *self.step_params["t_tuple"], T_cur, T_i)

            if self.step_params["p"]:
                self.aug_dropout.p = self.compute_decay_value(
                    *self.step_params["p_tuple"], T_cur, T_i)

            if self.step_params["s"]:
                self.loss.s = self.step_params["s_tuple"][self.step_params["s_list"][epoch]]


if __name__ == '__main__':
    # Input size: batch_size * feat_dim * seq_len * 
    timer = utils.Timer()
    x = torch.zeros(1000, 80)


    model = TransformerXvector(inputs_dim=80, num_targets=1211,training=False)
    out = model.extract_embedding(x)
    total = sum(p.numel() for p in model.parameters())
    print(model)
    print(total)
    print(out.shape)  
