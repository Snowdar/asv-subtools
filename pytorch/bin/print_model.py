# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-10-17)

import sys
import argparse
import torch

sys.path.insert(0, 'subtools/pytorch')

import libs.support.utils as utils
import libs.support.kaldi_common as kaldi_common

# Parse
parser = argparse.ArgumentParser(
        description="Print model information.")

parser.add_argument("--compute-size", type=str, action=kaldi_common.StrToBoolAction,
                    default=False, choices=["true", "false"],
                    help="")

parser.add_argument("--input-size", type=str, default="", 
                    help="")

parser.add_argument("nnet_config", metavar="nnet-config", type=str,
                    help="The model used to extract embeddings")

args = parser.parse_args()

# Start
model_blueprint, model_creation = utils.read_nnet_config(args.nnet_config)

model = utils.create_model_from_py(model_blueprint, model_creation)

print(model)

if args.compute_size:
    from thop import profile, clever_format
    assert args.input_size != ""
    input_shape = [ int(x) for x in args.input_size.split('-') ]
    input_tensor = torch.randn(*input_shape)
    macs, params = profile(model, inputs=(input_tensor, ))
    macs, params = clever_format([macs, params], "%.3f")
    info = "params={}, macs={}".format(params, macs)
    print(info)