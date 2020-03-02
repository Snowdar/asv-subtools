# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-10-17)

import sys
import argparse
import torch

sys.path.insert(0, 'subtools/pytorch')

import libs.support.utils as utils

# Parse
parser = argparse.ArgumentParser(
        description="Print model information.")

parser.add_argument("nnet_config", metavar="nnet-config", type=str,
                    help="The model used to extract embeddings")

args = parser.parse_args()

# Start
model_blueprint, model_creation = utils.read_nnet_config(args.nnet_config)

model = utils.create_model_from_py(model_blueprint, model_creation)

print(model)