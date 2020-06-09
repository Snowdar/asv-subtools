#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-10-17)

import sys
import argparse
import torch

sys.path.insert(0, 'subtools/pytorch')

import libs.support.utils as utils
import libs.support.kaldi_common as kaldi_common
from libs.support.clever_format import clever_format

# Parse
parser = argparse.ArgumentParser(
        description="Print model information.")

parser.add_argument("--input-size", type=str, default="", 
                    help="Give a size of input tensor, such as 1-23-100 to get num params of model.")

parser.add_argument("--exclude", type=str, default="", 
                    help="Exclude some layers when counting the params, such as 'loss'.")

parser.add_argument("nnet_config", metavar="nnet-config", type=str,
                    help="The model used to extract embeddings")

args = parser.parse_args()

# Start
model_blueprint, model_creation = utils.read_nnet_config(args.nnet_config)

model = utils.create_model_from_py(model_blueprint, model_creation)

total_params=0
state_dict=model.state_dict()
for name in state_dict:
    if args.exclude == "" or args.exclude not in name:
        total_params += state_dict[name].numel()

total_learnable_params=0
for name, params in model.named_parameters():
    if args.exclude == "" or args.exclude not in name:
        total_learnable_params += params.numel()

r_total_params, r_total_learnable_params = clever_format([total_params, total_learnable_params])

print(model)
print("\nTotal params: {} ({})\nTotal learnable params: {} ({})".format(total_params, r_total_params, 
                                                total_learnable_params, r_total_learnable_params))

# MACs is not available now.
if args.input_size != "":
    from thop import profile_origin, profile, clever_format
    from libs.nnet.count_rules_for_thop import custom_ops
    input_shape = [ int(x) for x in args.input_size.split('-') ]
    input_tensor = torch.randn(*input_shape)
    macs, params = profile_origin(model, inputs=(input_tensor, ), custom_ops=custom_ops)
    r_macs = clever_format([macs], "%.3f")
    info = "MACs: {} ({})".format(macs, r_macs)
    print(info)