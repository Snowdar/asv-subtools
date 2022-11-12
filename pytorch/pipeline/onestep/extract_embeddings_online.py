# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2022-01-15)

import sys
import os
import yaml
import argparse
import traceback
import numpy as np
import torch
import torchaudio
from tqdm.contrib import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, 'subtools/pytorch')

import libs.support.utils as utils
import libs.support.kaldi_io as kaldi_io
import libs.support.kaldi_common as kaldi_common

from libs.egs.egs_online import WavEgsXvector
torchaudio_backend = utils.get_torchaudio_backend()
torchaudio.set_audio_backend(torchaudio_backend)
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

parser.add_argument("--data-type",type=str,default="kaldi",choices=["raw", "shard", "kaldi"],
                help="raw or shard")

parser.add_argument("--de-silence", type=str, action=kaldi_common.StrToBoolAction, default=False, choices=["true", "false"],
                    help="Vad or not")
parser.add_argument("--amp-th", type=int, default=50,
                    help="De_silence threshold (16bit)")

parser.add_argument("--max-chunk", type=int, default=10000,
                    help="Select chun_size of features when extracting xvector") 
                 
parser.add_argument("--feat-config",type=str,default="",help="The config yaml of feat extraction")

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
    position = model.extracted_embedding
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=False)


    # Select device
    model = utils.select_model_device(model, args.use_gpu, gpu_id=args.gpu_id)
    devc = utils.get_device(model)
    model.eval()
    feature_extraction_conf = None
    if args.data_type != "kaldi":
        feat_config=args.feat_config
        with open(feat_config, 'r') as fin:
            feature_extraction_conf = yaml.load(fin, Loader=yaml.FullLoader)
    else:
        pass
    tot_len=0
    with open(args.feats_rspecifier, 'r') as flen:
        for i,_ in enumerate(flen):
            tot_len+=1
            pass
        
    de_sil_conf={}
    de_sil_conf["min_eng"]=args.amp_th
    dataset=WavEgsXvector(args.feats_rspecifier,feat_conf=feature_extraction_conf,data_type=args.data_type,de_silence=args.de_silence,de_sil_conf=de_sil_conf)
    data_loader = DataLoader(dataset, batch_size=None,num_workers=2, prefetch_factor=100)
    timer = utils.Timer()
    with kaldi_io.open_or_fd(args.vectors_wspecifier, 'wb') as w:
        cnt=0
        extract_time=0
        total_dur=0
        pbar=tqdm(total=tot_len, position=0,ascii=True,miniters=tot_len/100,dynamic_ncols=True)
        for idx,sample in enumerate(data_loader):
            key = sample['keys'][0]
            feats = sample['feats'][0].to(devc)

            total_dur += feats.size(0)*0.01

            timer.reset()
            embedding = model.extract_embedding_whole(feats,position=position,maxChunk=args.max_chunk)
            # embedding1 = model.forward_1(feats)
            # print(embedding)
            # print(embedding1)
            # assert 1==0
            extract_time+=timer.elapse()
            if cnt%500==0:
                pbar.update(500)

            cnt += 1
            kaldi_io.write_vec_flt(w, embedding.numpy(), key=key)

        pbar.close()
        print('RTF:{:.7f}'.format(extract_time/total_dur))

 

except BaseException as e:
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1) 
        


