# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-06-05)

import sys
import os
import argparse
import traceback
import torch

sys.path.insert(0, 'subtools/pytorch')

import libs.support.utils as utils
import libs.support.kaldi_io as kaldi_io

# Parse
parser = argparse.ArgumentParser(description="Extract embeddings form a piece of feats.scp or pipeline")


parser.add_argument("--nnet-config", type=str, default="",
                        help="This config contains model_blueprint and model_creation.")

parser.add_argument("--model-blueprint", type=str, default=None,
                        help="A *.py which includes the instance of nnet in this training.")

parser.add_argument("--model-creation", type=str, default=None,
                        help="A command to create the model class according to the class \
                        declaration in --model-path, such as using Xvector(40,2) to create \
                        a Xvector nnet.")

parser.add_argument("--use-gpu", type=str, default='true',
                    choices=["true", "false"],
                    help="If true, use GPU to extract embeddings.")

parser.add_argument("--gpu-id", type=str, default="",
                        help="Specify a fixed gpu, or select gpu automatically.")

parser.add_argument("model_path", metavar="model-path", type=str,
                    help="The model used to extract embeddings.")
                
parser.add_argument("feats_rspecifier", metavar="feats-rspecifier",
                    type=str, help="")
                
parser.add_argument("vectors_wspecifier", metavar="vectors-wspecifier",
                    type=str, help="")

print(' '.join(sys.argv))

args = parser.parse_args()

# Start

try:
    if args.nnet_config != "":
        model_blueprint, model_creation = utils.read_nnet_config(args.nnet_config)
    elif args.model_blueprint is not None and args.model_creation is not None:
        model_blueprint = args.model_blueprint
        model_creation = args.model_creation
    else:
        raise ValueError("Expected nnet_config or (model_blueprint, model_creation) to exist.")

    model = utils.create_model_from_py(model_blueprint, model_creation)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=False)

    # Select device
    model = utils.select_model_device(model, args.use_gpu, gpu_id=args.gpu_id)

    model.eval()

    with kaldi_io.open_or_fd(args.feats_rspecifier, "rb") as r, \
        kaldi_io.open_or_fd(args.vectors_wspecifier, 'wb') as w:
        
        while(True):
            key = kaldi_io.read_key(r)
            
            if not key:
                break

            print("Process utterance for key {0}".format(key))

            feats = kaldi_io.read_mat(r)
            embedding = model.extract_embedding(feats)
            kaldi_io.write_vec_flt(w, embedding.numpy(), key=key)

except BaseException as e:
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1) 
        


