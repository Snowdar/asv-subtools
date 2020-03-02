# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-05-29 2020-02-05)

import os
import logging
import pandas as pd

from multiprocessing import Process, Queue

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import libs.support.utils as utils
import libs.support.kaldi_io as kaldi_io

# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Relation: features -> chunk-egs-mapping-file -> chunk-egs -> bunch(dataloader+bunch) => trainer
class ChunkEgs(Dataset):
    """Prepare chunk-egs for time-context-input-nnet (such as xvector which transforms egs form chunk-frames level to 
    single-utterance level and TDNN) or single-channel-2D-input-nnet (such as resnet). Do it by linking to egs path
    temporarily and read them in training time, actually. 
    The acoustic feature based egs are not [frames, feature-dim] matrix format any more and it should be seen as 
    a [feature-dim, frames] tensor after transposing.
    """
    def __init__(self, egs_csv, io_status=True):
        """
        @egs_csv:
            utt-id:str  chunk_feats_path:offset:str chunk-start:int  chunk-end:int  label:int

        Other option
        @io_status: if false, do not read data from disk and return zero, which is useful for saving i/o resource 
        when kipping seed index.
        """
        self.io_status = io_status

        assert egs_csv != "" and egs_csv is not None
        self.data_frame = pd.read_csv(egs_csv, sep=" ").values

    def set_io_status(self, io_status):
        self.io_status = io_status

    def __getitem__(self, index):
        if not self.io_status :
            return 0., 0.

        chunk = [int(self.data_frame[index][2]), int(self.data_frame[index][3])]

        egs = kaldi_io.read_mat(self.data_frame[index][1], chunk=chunk)
        target = self.data_frame[index][4]
        return egs.T, target

    def __len__(self):
        return len(self.data_frame)


class BaseBunch():
    """BaseBunch:(trainset,[valid]).
    """
    def __init__(self, trainset:ChunkEgs, valid:ChunkEgs=None, batch_size=512, shuffle=True, 
                 num_workers=0, pin_memory=False, drop_last=True, queue_loader=False, queue_length=10):

        self.train_loader = DataLoader(trainset, batch_size = batch_size, shuffle=shuffle, num_workers=num_workers, 
                                       pin_memory=pin_memory, drop_last=drop_last)
        self.num_batch_train = len(self.train_loader)

        if pin_memory:
            logger.info("The pin_memory is true, so do not use queue_loader.")
            queue_loader = False

        if queue_loader:
            self.train_loader = QueueLoader(self.train_loader, queue_length=queue_length)

        if valid is not None:
            valid_batch_size = min(batch_size, len(valid)) # To save GPU memory
            self.valid_loader = DataLoader(valid, batch_size = valid_batch_size, shuffle=False, num_workers=0, 
                                           pin_memory=pin_memory, drop_last=False)
            self.num_batch_valid = len(self.valid_loader)
        else:
            self.valid_loader = None
            self.num_batch_valid = 0


    @classmethod
    def get_bunch_from_csv(self, trainset_csv:str, valid_csv:str=None, data_loader_params_dict:dict={}):
        trainset = ChunkEgs(trainset_csv)
        if valid_csv != "" and valid_csv is not None:
            valid = ChunkEgs(valid_csv)
        else:
            valid = None
        return self(trainset, valid, **data_loader_params_dict)

    def get_train_batch_num(self):
        return self.num_batch_train

    def get_valid_batch_num(self):
        return self.num_batch_valid

    def __len__(self):
        # main: train
        return self.num_batch_train

    @classmethod
    def get_bunch_from_egsdir(self, egsdir:str, data_loader_params_dict:dict={}):
        feat_dim, num_targets, train_csv, valid_csv = get_info_from_egsdir(egsdir)
        info = {"feat_dim":feat_dim, "num_targets":num_targets}
        bunch = self.get_bunch_from_csv(train_csv, valid_csv, data_loader_params_dict)
        return bunch, info


## Function
def get_info_from_egsdir(egsdir):
    if os.path.exists(egsdir+"/info"):
        feat_dim = int(utils.read_file_to_list(egsdir+"/info/feat_dim")[0])
        num_targets = int(utils.read_file_to_list(egsdir+"/info/num_targets")[0])
        train_csv = egsdir + "/train.egs.csv"
        valid_csv = egsdir + "/valid.egs.csv"
        if not os.path.exists(valid_csv):
            valid_csv = None
        return feat_dim, num_targets, train_csv, valid_csv
    else:
        raise ValueError("Expected dir {0} to exist.".format(egsdir+"/info"))


