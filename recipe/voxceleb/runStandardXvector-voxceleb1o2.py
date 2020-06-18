# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-12)
# Apache 2.0

import sys, os
import logging
import argparse
import traceback
import time
import math
import numpy as np

import torch

sys.path.insert(0, 'subtools/pytorch')

import libs.egs.egs as egs
import libs.training.optim as optim
import libs.training.lr_scheduler as learn_rate_scheduler
import libs.training.trainer as trainer
import libs.support.kaldi_common as kaldi_common
import libs.support.utils as utils
from  libs.support.logging_stdout import patch_logging_stream

"""A launcher script with python version (Snowdar's launcher to do experiments w.r.t snowdar-xvector.py).
Python version is given rather than Shell is to support more freedom without limitation of parameters transfer from shell to python.

Note, this launcher does not contains dataset preparation, augmentation, extracting acoustic features and back-end scoring etc.
    1.See subtools/recipe/voxceleb/runVoxceleb.sh to get complete stages.
    2.See subtools/newCopyData.sh, subtools/makeFeatures.sh.sh, subtools/computeVad.sh, subtools/augmentDataByNoise.sh and 
          subtools/scoreSets.sh and run these script separately before or after running this launcher.

How to modify and use this launcher:
    1.Prepare your kaldi format dataset and model.py (model blueprint);
    2.Give the path of dataset and model blueprint etc. in main parameters field;
    3.Change the import name of model in 'model = model_py.model_name(...)' a.w.t model.py by yourself;
    4.Modify any training parameters what you want to change (epochs, optimizer and lr_scheduler etc.);
    5.Modify extracting parameters in stage 4 a.w.t your own training config;
    6.Run this launcher.

Conclusion: preprare -> config -> run.

How to run this launcher to training model:
    1.For CPU-based training case. The key option is --use-gpu.
        python3 launcher.py --use-gpu=false
    2.For single-GPU training case (Default).
        python3 launcher.py
    3.For DDP-based multi-GPU training case. Note --nproc_per_node is equal to number of gpu id in --gpu-id.
        python3 -m torch.distributed.launch --nproc_per_node=2 launcher.py --gpu-id=0,1
    4.For Horovod-based multi-GPU training case. Note --np is equal to number of gpu id in --gpu-id.
        horovodrun -np 2 launcher.py --gpu-id=0,1
    5.For all of above, a runLauncher.sh script has been created to launch launcher.py conveniently.
      The key option to use single or multiple GPU is --gpu-id.
      The subtools/runPytorchLauncher.sh is a soft symbolic which is linked to subtools/pytorch/launcher/runLauncher.sh, 
      so just use it.

        [ CPU ]
            subtools/runPytorchLauncher.sh launcher.py --use-gpu=false

        [ Single-GPU ]
        (1) Auto-select GPU device
            subtools/runPytorchLauncher.sh launcher.py
        (2) Specify GPU device
            subtools/runPytorchLauncher.sh launcher.py --gpu-id=2

        [ Multi-GPU ]
        (1) Use DDP solution (Default).
            subtools/runPytorchLauncher.sh launcher.py --gpu-id=2,3 --multi-gpu-solution="ddp"
        (2) Use Horovod solution.
            subtools/runPytorchLauncher.sh launcher.py --gpu-id=2,3 --multi-gpu-solution="horovod"

If you have any other requirements, you could modify the codes in anywhere. 
For more details of multi-GPU devolopment, see subtools/README.md.
"""

# Logger
# Change the logging stream from stderr to stdout to be compatible with horovod.
patch_logging_stream(logging.INFO)

logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ]\n#### %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Parser: add this parser to run launcher with some frequent options (really for conveninece).
parser = argparse.ArgumentParser(
        description="""Train xvector framework with pytorch.""",
        formatter_class=argparse.RawTextHelpFormatter,
        conflict_handler='resolve')

parser.add_argument("--stage", type=int, default=3,
                    help="The stage to control the start of training epoch (default 3).\n"
                         "    stage 0: vad-cmn (preprocess_to_egs.sh).\n"
                         "    stage 1: remove utts (preprocess_to_egs.sh).\n"
                         "    stage 2: get chunk egs (preprocess_to_egs.sh).\n"
                         "    stage 3: training.\n"
                         "    stage 4: extract xvector.")

parser.add_argument("--endstage", type=int, default=4,
                    help="The endstage to control the endstart of training epoch (default 4).")

parser.add_argument("--train-stage", type=int, default=-1,
                    help="The stage to control the start of training epoch (default -1).\n"
                         "    -1 -> creating model_dir.\n"
                         "     0 -> model initialization (e.g. transfer learning).\n"
                         "    >0 -> recovering training.")

parser.add_argument("--force-clear", type=str, action=kaldi_common.StrToBoolAction,
                    default=False, choices=["true", "false"],
                    help="Clear the dir generated by preprocess.")

parser.add_argument("--use-gpu", type=str, action=kaldi_common.StrToBoolAction,
                    default=True, choices=["true", "false"],
                    help="Use GPU or not.")

parser.add_argument("--gpu-id", type=str, default="",
                    help="If NULL, then it will be auto-specified.\n"
                         "If giving multi-gpu like --gpu-id=1,2,3, then use multi-gpu training.")

parser.add_argument("--multi-gpu-solution", type=str, default="ddp",
                    choices=["ddp", "horovod"],
                    help="if number of gpu_id > 1, this option will be valid to init a multi-gpu solution.")

parser.add_argument("--benchmark", type=str, action=kaldi_common.StrToBoolAction,
                    default=True, choices=["true", "false"],
                    help="If true, save training time but require a little more gpu-memory.")

parser.add_argument("--run-lr-finder", type=str, action=kaldi_common.StrToBoolAction,
                    default=False, choices=["true", "false"],
                    help="If true, run lr finder rather than training.")

parser.add_argument("--sleep", type=int, default=0,
                    help="The waiting time to launch a launcher.")

parser.add_argument("--local_rank", type=int, default=0,
                    help="Do not delete it when using DDP-based multi-GPU training.\n"
                         "It is important for torch.distributed.launch.")

parser.add_argument("--port", type=int, default=29500,
                    help="This port is used for DDP solution in multi-GPU training.")

args = parser.parse_args()
##
######################################################### PARAMS ########################################################
##
##--------------------------------------------------##
## Control options
stage = max(0, args.stage)
endstage = min(4, args.endstage)
train_stage = max(-1, args.train_stage)
##--------------------------------------------------##
## Preprocess options
force_clear=args.force_clear
preprocess_nj = 20
cmn = True # Traditional cmn process.

chunk_size = 200
limit_utts = 8

sample_type="speaker_balance" # sequential | speaker_balance
chunk_num=-1 # -1 means using scale, 0 means using max and >0 means itself.
overlap=0.1
scale=1.5 # Get max / num_spks * scale for every speaker.
valid_split_type="--total-spk" # --total-spk or --default
valid_utts = 1024
valid_chunk_num_every_utt = 2
##--------------------------------------------------##
## Training options
use_gpu = args.use_gpu # Default true.
benchmark = args.benchmark # If true, save much training time but require a little more gpu-memory.
gpu_id = args.gpu_id # If NULL, then it will be auto-specified.
run_lr_finder = args.run_lr_finder

egs_params = {
    "aug":None, # None or specaugment. If use aug, you should close the aug_dropout which is in model_params.
    "aug_params":{"frequency":0.2, "frame":0.2, "rows":4, "cols":4, "random_rows":True,"random_rows":True}
}

loader_params = {
    "use_fast_loader":True, # It is a queue loader to prefetch batch and storage.
    "max_prefetch":10,
    "batch_size":512, 
    "shuffle":True, 
    "num_workers":2,
    "pin_memory":False, 
    "drop_last":True,
}

# Difine model_params by model_blueprint w.r.t your model's __init__(model_params).
model_params = {
    "extend":False, "SE":False, "se_ratio":4, "training":True, "extracted_embedding":"far",

    "aug_dropout":0., "hidden_dropout":0., 
    "dropout_params":{"type":"default", "start_p":0., "dim":2, "method":"uniform",
                      "continuous":False, "inplace":True},

    "tdnn_layer_params":{"nonlinearity":'relu', "nonlinearity_params":{"inplace":True},
                         "bn-relu":False, 
                         "bn":True, 
                         "bn_params":{"momentum":0.5, "affine":False, "track_running_stats":True}},

    "pooling":"statistics", # statistics, lde, attentive, multi-head, multi-resolution
    "pooling_params":{"num_nodes":1500,
                      "num_head":1,
                      "share":True,
                      "affine_layers":1,
                      "hidden_size":64,
                      "context":[0],
                      "temperature":False, 
                      "fixed":True
                      },
    "tdnn6":True, 
    "tdnn7_params":{"nonlinearity":"default", "bn":True},

    "margin_loss":False, 
    "margin_loss_params":{"method":"am", "m":0.2, "feature_normalize":True, 
                          "s":30, "mhe_loss":False, "mhe_w":0.01},
    "use_step":False, 
    "step_params":{"T":None,
                   "m":False, "lambda_0":0, "lambda_b":1000, "alpha":5, "gamma":1e-4,
                   "s":False, "s_tuple":(30, 12), "s_list":None,
                   "t":False, "t_tuple":(0.5, 1.2), 
                   "p":False, "p_tuple":(0.5, 0.1)}
}

optimizer_params = {
    "name":"adamW",
    "learn_rate":0.001,
    "beta1":0.9,
    "beta2":0.999,
    "beta3":0.999,
    "weight_decay":1e-1,  # Should be large for decouped weight decay (adamW) and small for L2 regularization (sgd, adam).
    "lookahead.k":5,
    "lookahead.alpha":0.,  # 0 means not using lookahead and if used, suggest to set it as 0.5.
    "gc":False # If true, use gradient centralization.
}

lr_scheduler_params = {
    "name":"warmR",
    "warmR.lr_decay_step":0, # 0 means decay after every epoch and 1 means every iter. 
    "warmR.T_max":5,
    "warmR.T_mult":2,
    "warmR.factor":1.0,  # The max_lr_decay_factor.
    "warmR.eta_min":4e-8,
    "warmR.log_decay":False
}

epochs = 15 # Total epochs to train. It is important.

report_times_every_epoch = None
report_interval_iters = 100 # About validation computation and loss reporting. If report_times_every_epoch is not None, 
                            # then compute report_interval_iters by report_times_every_epoch.
stop_early = False
suffix = "params" # Used in saved model file.
##--------------------------------------------------##
## Other options
exist_model=""  # Use it in transfer learning.
##--------------------------------------------------##
## Main params
traindata="data/mfcc_23_pitch/voxceleb1o2_train_aug"
egs_dir="exp/egs/mfcc_23_pitch_voxceleb1o2_train_aug" + "_" + sample_type

model_blueprint="subtools/pytorch/model/snowdar-xvector.py"
model_dir="exp/standard_voxceleb1o2"
##--------------------------------------------------##
##
######################################################### START #########################################################
##
#### Set seed
utils.set_all_seed(1024)
##
#### Set sleep time for a rest
# Use it to run a launcher with a countdown function when there are no extra GPU memory 
# but you really want to go to bed and know when the GPU memory will be free.
if args.sleep > 0: time.sleep(args.sleep)
##
#### Init environment
# It is used for multi-gpu training if used (number of gpu-id > 1).
# And it will do nothing for single-GPU training.
utils.init_multi_gpu_training(args.gpu_id, args.multi_gpu_solution, args.port)
##
#### Auto-config params
# If multi-GPU used, it will auto-scale learning rate by multiplying number of processes.
optimizer_params["learn_rate"] = utils.auto_scale_lr(optimizer_params["learn_rate"])
# It is used for model.step() defined in model blueprint.
if lr_scheduler_params["name"] == "warmR" and model_params["use_step"]:
    model_params["step_params"]["T"]=(lr_scheduler_params["warmR.T_max"], lr_scheduler_params["warmR.T_mult"])
##
#### Preprocess
if stage <= 2 and endstage >= 0 and utils.is_main_training():
    # Here only give limited options because it is not convenient.
    # Suggest to pre-execute this shell script to make it freedom and then continue to run this launcher.
    kaldi_common.execute_command("sh subtools/pytorch/pipeline/preprocess_to_egs.sh "
                                 "--stage {stage} --endstage {endstage} --valid-split-type {valid_split_type} "
                                 "--nj {nj} --cmn {cmn} --limit-utts {limit_utts} --min-chunk {chunk_size} --overlap {overlap} "
                                 "--sample-type {sample_type} --chunk-num {chunk_num} --scale {scale} --force-clear {force_clear} "
                                 "--valid-num-utts {valid_utts} --valid-chunk-num {valid_chunk_num_every_utt} "
                                 "{traindata} {egs_dir}".format(stage=stage, endstage=endstage, valid_split_type=valid_split_type, 
                                 nj=preprocess_nj, cmn=str(cmn).lower(), limit_utts=limit_utts, chunk_size=chunk_size, overlap=overlap, 
                                 sample_type=sample_type, chunk_num=chunk_num, scale=scale, force_clear=str(force_clear).lower(), 
                                 valid_utts=valid_utts, valid_chunk_num_every_utt=valid_chunk_num_every_utt, traindata=traindata, 
                                 egs_dir=egs_dir))

#### Train model
if stage <= 3 <= endstage:
    if utils.is_main_training(): logger.info("Get model_blueprint from model directory.")
    # Save the raw model_blueprint in model_dir/config and get the copy of model_blueprint path.
    model_blueprint = utils.create_model_dir(model_dir, model_blueprint, stage=train_stage)

    if utils.is_main_training(): logger.info("Load egs to bunch.")
    # The dict [info] contains feat_dim and num_targets.
    bunch, info = egs.BaseBunch.get_bunch_from_egsdir(egs_dir, egs_params, loader_params)

    if utils.is_main_training(): logger.info("Create model from model blueprint.")
    # Another way: import the model.py in this python directly, but it is not friendly to the shell script of extracting and
    # I don't want to change anything about extracting script when the model.py is changed.
    model_py = utils.create_model_from_py(model_blueprint)
    model = model_py.Xvector(info["feat_dim"], info["num_targets"], **model_params)

    # If multi-GPU used, then batchnorm will be converted to synchronized batchnorm, which is important 
    # to make peformance stable. 
    # It will change nothing for single-GPU training.
    model = utils.convert_synchronized_batchnorm(model)

    if utils.is_main_training(): logger.info("Define optimizer and lr_scheduler.")
    optimizer = optim.get_optimizer(model, optimizer_params)
    lr_scheduler = learn_rate_scheduler.LRSchedulerWrapper(optimizer, lr_scheduler_params)

    # Record params to model_dir
    utils.write_list_to_file([egs_params, loader_params, model_params, optimizer_params, 
                              lr_scheduler_params], model_dir+'/config/params.dict')

    if utils.is_main_training(): logger.info("Init a simple trainer.")
    # Package(Elements:dict, Params:dict}. It is a key parameter's package to trainer and model_dir/config/.
    package = ({"data":bunch, "model":model, "optimizer":optimizer, "lr_scheduler":lr_scheduler},
            {"model_dir":model_dir, "model_blueprint":model_blueprint, "exist_model":exist_model, 
            "start_epoch":train_stage, "epochs":epochs, "use_gpu":use_gpu, "gpu_id":gpu_id, 
            "benchmark":benchmark, "suffix":suffix, "report_times_every_epoch":report_times_every_epoch,
            "report_interval_iters":report_interval_iters, "record_file":"train.csv"})

    trainer = trainer.SimpleTrainer(package, stop_early=stop_early)

    if run_lr_finder and utils.is_main_training():
        trainer.run_lr_finder("lr_finder.csv", init_lr=1e-8, final_lr=10., num_iters=2000, beta=0.98)
        endstage = 3 # Do not start extractor.
    else:
        trainer.run()


#### Extract xvector
if stage <= 4 <= endstage and utils.is_main_training():
    # There are some params for xvector extracting.
    data_root = "data" # It contains all dataset just like Kaldi recipe.
    prefix = "mfcc_23_pitch" # For to_extracted_data.

    to_extracted_positions = ["far", "near"] # Define this w.r.t extracted_embedding param of model_blueprint.
    to_extracted_data = ["voxceleb1_train_aug", "voxceleb1_test"] # All dataset should be in data_root/prefix.
    to_extracted_epochs = ["15"] # It is model's name, such as 10.params or final.params (suffix is w.r.t package).

    nj = 10
    force = False
    use_gpu = True
    gpu_id = ""
    sleep_time = 10


    # Run a batch extracting process.
    try:
        for position in to_extracted_positions:
            # Generate the extracting config from nnet config where 
            # which position to extract depends on the 'extracted_embedding' parameter of model_creation (by my design).
            model_blueprint, model_creation = utils.read_nnet_config("{0}/config/nnet.config".format(model_dir))
            model_creation = model_creation.replace("training=True", "training=False") # To save memory without loading some independent components.
            model_creation = model_creation.replace(model_params["extracted_embedding"], position)
            extract_config = "{0}.extract.config".format(position)
            utils.write_nnet_config(model_blueprint, model_creation, "{0}/config/{1}".format(model_dir, extract_config))
            for epoch in to_extracted_epochs:
                model_file = "{0}.{1}".format(epoch, suffix)
                point_name = "{0}_epoch_{1}".format(position, epoch)

                # If run a trainer with background thread (do not be supported now) or run this launcher extrally with stage=4 
                # (it means another process), then this while-listen is useful to start extracting immediately (but require more gpu-memory).
                model_path = "{0}/{1}".format(model_dir, model_file)
                while True:
                    if os.path.exists(model_path):
                        break
                    else:
                        time.sleep(sleep_time)

                for data in to_extracted_data:
                    datadir = "{0}/{1}/{2}".format(data_root, prefix, data)
                    outdir = "{0}/{1}/{2}".format(model_dir, point_name, data)
                    # Use a well-optimized shell script (with multi-processes) to extract xvectors.
                    # Another way: use subtools/splitDataByLength.sh and subtools/pytorch/pipeline/onestep/extract_embeddings.py 
                    # with python's threads to extract xvectors directly, but the shell script is more convenient.
                    kaldi_common.execute_command("sh subtools/pytorch/pipeline/extract_xvectors_for_pytorch.sh "
                                                "--model {model_file} --cmn {cmn} --nj {nj} --use-gpu {use_gpu} --gpu-id '{gpu_id}' "
                                                " --force {force} --nnet-config config/{extract_config} "
                                                "{model_dir} {datadir} {outdir}".format(model_file=model_file, cmn=str(cmn).lower(), nj=nj,
                                                use_gpu=str(use_gpu).lower(), gpu_id=gpu_id, force=str(force).lower(), extract_config=extract_config,
                                                model_dir=model_dir, datadir=datadir, outdir=outdir))
    except BaseException as e:
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)




