# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Snowdar 2019-05-29 2020-02-05)
# update (Author: Leo 2022-01-11)
# refering https://github.com/wenet-e2e/wenet/blob/main/wenet/dataset/dataset.py
import os
import sys
import copy,logging
import logging
import yaml
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
sys.path.insert(0, "subtools/pytorch")
import libs.support.utils as utils
import libs.egs.processor as processor

# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def change_csv_folder(node,replace):
    if isinstance(node,list):
        for n in node:
            change_csv_folder(n,replace)
    if isinstance(node,dict):
        for k,v in node.items():
            if 'csv' in k:
                if node[k]: # some scenarios don't need file.
                    name=os.path.basename(node[k])
                    r = os.path.join(replace,name)
                    assert os.path.exists(r)
                    node[k]=r
            change_csv_folder(v,replace)

class Processor(IterableDataset):
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def get_data_dur(self):
        return self.source.get_data_dur()

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)
    def __len__(self):
        return len(self.source)

class DistributedSampler:
    def __init__(self, shuffle=True, partition=True):

        self.epoch = -1
        self.main_seed = np.random.get_state()[1][0]
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers,
                    epoch = self.epoch,
                    main_seed = self.main_seed)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data_this_rank= data
        data = data[self.worker_id::self.num_workers]
        return data,data_this_rank

class DataList(IterableDataset):
    def __init__(self, lists, shuffle=True, partition=True):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)
        

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
        lists_copy = copy.deepcopy(self.lists)
        _,self.data_this_rank = self.sampler.sample(lists_copy)
    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes,_ = self.sampler.sample(self.lists)
        worker_seed = sampler_info['main_seed']+ sampler_info['epoch'] +  sampler_info['rank']*sampler_info['num_workers']+sampler_info['worker_id']
        utils.set_all_seed(worker_seed)
        
        for index in indexes:
            # yield dict(src=src)
            data = self.lists[index]
            data.update(sampler_info)
            yield data

    def get_data_dur(self):
        try:
            total_dur=sum([float(self.lists[index]['eg-dur']) for index in self.data_this_rank])/3600
        except (Exception) as e:
            logger.warning('do not support get duration')
            total_dur=None
        num_sample = sum([int(self.lists[index]['eg-num']) for index in self.data_this_rank]) if "eg-num" in self.lists[0] else len(self.data_this_rank)
        # tot_sample = sum([int(self.lists[index]['eg-num']) for index in self.lists]) if "eg-num" in self.lists[0] else len(self.lists)
        return total_dur,num_sample
    def __len__(self):
        return sum([int(self.lists[index]['eg-num']) for index in range(len(self.lists))]) if "eg-num" in self.lists[0] else len(self.lists)

def WavEgs(egs_csv,conf,data_type='raw',partition=True,num_targets=0):
    assert data_type in ['raw', 'shard', 'kaldi']
    lists = utils.csv_to_list(egs_csv)


    shuffle = conf.get('shuffle', True)
    
 
    dataset = DataList(lists, shuffle=shuffle, partition=partition)

    if data_type in ['raw', 'shard']:
        if data_type=='shard':
            dataset = Processor(dataset, processor.url_opener)
            dataset = Processor(dataset, processor.tar_file_and_group)
        else:
            dataset = Processor(dataset, processor.parse_raw)
        filt = conf.get('filter', False)   
        filter_conf = conf.get('filter_conf', {})
        if filt:
            dataset = Processor(dataset, processor.filter, **filter_conf)

        resample = conf.get('resample', False)
        if resample:
            resample_conf = conf.get('resample_conf', {})
            dataset = Processor(dataset, processor.resample, **resample_conf)
 
        
        pre_speed_perturb =  conf.get('pre_speed_perturb', False)
        spkid_aug = 1
        if pre_speed_perturb:
            perturb_conf =  conf.get('perturb_conf',{})       
            sp = processor.PreSpeedPerturb(spk_num=num_targets,**perturb_conf)
            spkid_aug = sp._spkid_aug()
            dataset =  Processor(dataset,sp)

        random_chunk = conf.get('random_chunk',False)
        random_chunk_size = conf.get('random_chunk_size',2.015)
        
                
        if random_chunk:
            dataset = Processor(dataset, processor.random_chunk, random_chunk_size)



        speech_aug = conf.get('speech_aug', False)
        speech_aug_conf_file = conf.get('speech_aug_conf', '')
        if speech_aug and speech_aug_conf_file:
            with open(speech_aug_conf_file, 'r') as fin:
                speech_aug_conf = yaml.load(
                    fin, Loader=yaml.FullLoader)

                csv_aug_folder = conf.get('csv_aug_folder','')
                if csv_aug_folder:change_csv_folder(speech_aug_conf,csv_aug_folder)

            speechaug_pipline = processor.SpeechAugPipline(spk_num=num_targets,**speech_aug_conf)
            spkid_aug_lat = speechaug_pipline.get_spkid_aug()
            if not (spkid_aug==1 or spkid_aug_lat==1):
                raise ValueError("multi speaker id perturb setting, check your speech aug config")
            spkid_aug = spkid_aug_lat*spkid_aug
            dataset =  Processor(dataset, speechaug_pipline)

        feature_extraction_conf = conf.get('feature_extraction_conf',{})
        feature_extraction = processor.KaldiFeature(**feature_extraction_conf)
        dataset =  Processor(dataset, feature_extraction)
    else:
        dataset =  Processor(dataset,processor.offline_feat)

    spec_aug = conf.get('spec_aug', False)
    if spec_aug:
        spec_aug_conf=conf.get('spec_aug_conf',{})
        specaug_pipline = processor.SpecAugPipline(**spec_aug_conf)
        dataset = Processor(dataset,specaug_pipline)
    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = Processor(dataset, processor.shuffle, **shuffle_conf)
    sort = conf.get('sort', False)
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = Processor(dataset, processor.sort, **sort_conf)

    batch_conf = conf.get('batch_conf', {})
    dataset = Processor(dataset, processor.batch, **batch_conf)
    dataset = Processor(dataset, processor.padding)

    return dataset,spkid_aug

def WavEgsXvector(wav_scp,feat_conf={},data_type='kaldi',de_silence=False,de_sil_conf={},partition=False):
    if data_type in ['raw', 'shard']:
        if data_type=='shard':
            lists = utils.csv_to_list(wav_scp)
        else:
            lists = utils.read_wav_list(wav_scp)

        dataset = DataList(lists,shuffle=False)
        if data_type=='shard':
            dataset = Processor(dataset, processor.url_opener)
            dataset = Processor(dataset, processor.tar_file_and_group)
        else:
            dataset = Processor(dataset, processor.parse_raw)
        if de_silence:
            dataset = Processor(dataset,processor.de_sil,**de_sil_conf)
        feature_extraction = processor.KaldiFeature(**feat_conf)
        dataset =  Processor(dataset, feature_extraction)
    elif data_type == "kaldi":
        dataset =  Processor(dataset, processor.offline_feat)
    else:
        raise ValueError("Do not support datatype: {0} now.".format(data_type))
    return dataset



class BaseBunch():
    """BaseBunch:(trainset,[valid]).
    """

    def __init__(self, trainset, valid=None,prefetch_factor=2,num_workers=0, pin_memory=False):

        self.train_loader = DataLoader(trainset, batch_size=None, num_workers=num_workers,
                                        pin_memory=pin_memory,prefetch_factor=prefetch_factor)
        if valid is not None:
            self.valid_loader = DataLoader(valid, batch_size=None, num_workers=0,
                                        pin_memory=pin_memory)


    @classmethod
    def get_bunch_from_csv(self, trainset_csv: str, valid_csv: str = None, egs_params: dict = {},num_targets=-1):

        train_conf = egs_params['dataset_conf']
        valid_conf = copy.deepcopy(train_conf)
        valid_conf['speech_aug'] = False
        valid_conf['pre_speed_perturb'] = False
        valid_conf['spec_aug'] = False
        valid_conf['shuffle'] = False
        data_type = egs_params.get('data_type','raw')
        trainset, num_targets_t = WavEgs(trainset_csv, train_conf, data_type=data_type,partition=True,num_targets=num_targets)
        
        if valid_csv != "" and valid_csv is not None:
            valid,_ = WavEgs(valid_csv, valid_conf,data_type=data_type,partition=False)
        else:
            valid = None
        self.num_targets =num_targets*num_targets_t

        return self(trainset, valid, **egs_params['data_loader_conf'])



    @classmethod
    def get_bunch_from_egsdir(self, egsdir: str, egs_params: dict={}):
        train_csv_name = None
        valid_csv_name = None

        if "train_csv_name" in egs_params.keys():
            train_csv_name = egs_params.pop("train_csv_name")

        if "valid_csv_name" in egs_params.keys():
            valid_csv_name = egs_params.pop("valid_csv_name")

        num_targets, train_csv, valid_csv = get_info_from_egsdir(
            egsdir, train_csv_name=train_csv_name, valid_csv_name=valid_csv_name)
        assert 'feat_dim' in egs_params
        feat_dim = int(egs_params['feat_dim'])

        bunch = self.get_bunch_from_csv(
            train_csv, valid_csv, egs_params,num_targets)
        num_targets = self.num_targets
        tot_samples = len(bunch.train_loader.dataset)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if egs_params['dataset_conf']['batch_conf']['batch_type']=='static':
            epoch_iters = (tot_samples//world_size)//egs_params['dataset_conf']['batch_conf']['batch_size']
        else:
            epoch_iters = None
        info = {"feat_dim": feat_dim, "num_targets": num_targets, "epoch_iters": epoch_iters}
        return bunch, info


def get_info_from_egsdir(egsdir, train_csv_name=None, valid_csv_name=None):
    if os.path.exists(egsdir+"/info"):
        num_targets = int(utils.read_file_to_list(
            egsdir+"/info/num_targets")[0])

        train_csv_name = train_csv_name if train_csv_name is not None else "train.egs.csv"
        valid_csv_name = valid_csv_name if valid_csv_name is not None else "valid.egs.csv"

        train_csv = egsdir + "/" + train_csv_name
        valid_csv = egsdir + "/" + valid_csv_name

        if not os.path.exists(valid_csv):
            valid_csv = None

        return num_targets, train_csv, valid_csv
    else:
        raise ValueError("Expected dir {0} to exist.".format(egsdir+"/info"))


if __name__ == "__main__":
    pass