# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-01-05)

import os, sys
import copy
import logging
import numpy as np

import libs.support.kaldi_io as kaldi_io
# import libs.support.kaldi_common as kaldi_common

# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


### Class
class KaldiDataset():
    """
    Parameters:
        data_dir: datadir in kaldi data/.
        expected_files: use this free config carefully.
        more: if true, ignore expected_files and load all available files which are exist.

    Possible attr:
    == Mapping files ==
        self.utt2spk: dict{str:str}
        self.spk2utt: dict{str:list[str]}
        self.feats_scp: dict{str:str}
        self.utt2num_frames: dict{str:int}
        self.vad_scp: dict{str:str}
        self.wav_scp: dict{str:str}
        self.utt2dur: dict{str:float}
        self.utt2spk_int: dict{str:int}

    == Variables ==
        self.data_dir: str
        self.num_utts, self.num_spks, self.num_frames, self.feat_dim: int
    """
    def __init__(self, data_dir:str="", 
                       expected_files:list=["utt2spk", "spk2utt", "feats.scp", "utt2num_frames"], 
                       more=False):
        # Fixed definition of str-first mapping files.
        # Tuple(Attr:str, FileName:str, Value_type:str, Vector:bool).
        self.utt_first_files = [
            ("wav_scp", "wav.scp", "str", False),
            ("utt2spk", "utt2spk", "str", False),
            ("utt2spk_int", "utt2spk_int", "int", False),
            ("feats_scp", "feats.scp", "str", False),
            ("utt2num_frames", "utt2num_frames", "int", False),
            ("utt2dur", "utt2dur", "float", False),
            ("vad_scp", "vad.scp", "str", False),
            ("text", "text", "str", True)]

        # Should keep spk2utt only now.
        self.spk_first_files = [
            ("spk2utt", "spk2utt", "str", True)]

        # Process parameters
        if data_dir == "": # Here should use "." rather than "" to express current directory.
            self.data_dir = None
        else:
            self.data_dir = data_dir

        self.expected_files = expected_files
        self.more = more

        # Init and Load files
        self.loaded_attr = []

        if self.data_dir is not None:
            self.load_data_()

        self.get_base_attribute_()

    @classmethod
    def load_data_dir(self, data_dir:str, expected_files:list=["utt2spk", "spk2utt", "feats.scp", "utt2num_frames"]):
        return self(data_dir, expected_files)

    def load_data_(self):
        if not os.path.exists(self.data_dir):
            raise ValueError("The datadir {0} is not exist.".format(self.data_dir))

        if self.more:
            logger.info("Load mapping files form {0} as more as possible with more=True".format(self.data_dir))
        else:
            logger.info("Load mapping files form {0} w.r.t expected files {1}".format(self.data_dir, self.expected_files))

        for attr, file_name, value_type, vector in self.utt_first_files + self.spk_first_files:
            file_path = "{0}/{1}".format(self.data_dir, file_name)
            if self.more:
                if os.path.exists(file_path) and attr not in self.loaded_attr:
                    logger.info("Load data from {0} ...".format(file_path))
                    setattr(self, attr, read_str_first_ark(file_path, value_type, vector))
                    self.loaded_attr.append(attr)
            elif file_name in self.expected_files:
                if os.path.exists(file_path) and attr not in self.loaded_attr:
                    logger.info("Load data from {0} ...".format(file_path))
                    setattr(self, attr, read_str_first_ark(file_path, value_type, vector))
                    self.loaded_attr.append(attr)
                else:
                    raise ValueError("The file {0} is not exist.".format(file_path))

    def get_base_attribute_(self):
        ## Base attribute
        # Total utts
        self.num_utts = len(self.utt2spk) if "utt2spk" in self.loaded_attr else None

        # Total spks
        self.num_spks = len(self.spk2utt) if "spk2utt" in self.loaded_attr else None

        # Total frames
        if "utt2num_frames" in self.loaded_attr:
            self.num_frames = 0
            for utt, num_frames in self.utt2num_frames.items():
                self.num_frames += num_frames
        else:
            self.num_frames = None

        # Feature dim
        self.feat_dim = kaldi_io.read_mat(
            self.feats_scp[list(self.feats_scp.keys())[0]]).shape[1] if "feats_scp" in self.loaded_attr else None

    def generate(self, attr:str):
        if attr == "utt2spk_int":
            if attr not in self.loaded_attr:
                spk2int = {}
                self.utt2spk_int = {}
                for index, spk in enumerate(self.spk2utt):
                    spk2int[spk] = index
                for utt, spk in self.utt2spk.items():
                    self.utt2spk_int[utt] =  spk2int[spk]
                self.loaded_attr.append(attr)
            else:
                logger.warn("The utt2spk_int is exist.")
        else:
            raise ValueError("Do not support attr {0} now.".format(attr))

    def filter(self, id_list:set, id_type:str="utt", exclude:bool=False):
        """
        id_list: a id set w.r.t utt-id or spk-id. Could be list.
        id_type: utt or spk.
        exclude: if true, dropout items which in id_list instead of keep them.

        @return: KaldiDataset. Return a new KaldiDataset rather than itself.
        """
        if len(self.loaded_attr) == 0:
            logger.warn("The KaldiDataset has 0 loaded attr.")
            return self

        kaldi_dataset = copy.deepcopy(self)
        kaldi_dataset.data_dir = None

        if not isinstance(id_list, set):
            id_list = set(id_list)

        if id_type == "utt":
            for attr, file_name, value_type, vector in kaldi_dataset.utt_first_files:
                if attr in kaldi_dataset.loaded_attr:
                    this_file_dict = getattr(kaldi_dataset, attr)
                    new_file_dict = {}
                    for k, v in this_file_dict.items():
                        if (k not in id_list and exclude) or (k in id_list and not exclude):
                            new_file_dict[k] = v
                    setattr(kaldi_dataset, attr, new_file_dict)

            for attr, file_name, value_type, vector in kaldi_dataset.spk_first_files:
                if attr in kaldi_dataset.loaded_attr:
                    if attr == "spk2utt":
                        this_file_dict = getattr(kaldi_dataset, attr)
                        if exclude:
                            new_file_dict = { k:list(set(v)-id_list) for k, v in this_file_dict.items() if len(list(set(v)-id_list)) != 0 }
                        else:
                            new_file_dict = { k:list(set(v)&id_list) for k, v in this_file_dict.items() if len(list(set(v)&id_list)) != 0 }
                        setattr(kaldi_dataset, attr, new_file_dict)
                    else:
                        raise ValueError("Do not support file {0} w.r.t spk2utt only.".format(attr))

        elif id_type == "spk":
            for attr, file_name, value_type, vector in kaldi_dataset.spk_first_files:
                if attr in kaldi_dataset.loaded_attr:
                    if attr == "spk2utt":
                        this_file_dict = getattr(kaldi_dataset, attr)
                        if exclude:
                            new_file_dict = { k:v for k, v in this_file_dict.items() if k not in id_list }
                        else:
                            new_file_dict = { k:v for k, v in this_file_dict.items() if k in id_list }
                        setattr(kaldi_dataset, attr, new_file_dict)
                    else:
                        raise ValueError("Do not support file {0} w.r.t spk2utt only.".format(attr))
            
            if len(kaldi_dataset.utt_first_files) > 0:
                if "spk2utt" in kaldi_dataset.loaded_attr:
                    utt_id_list = [] 
                    for spk, utts in kaldi_dataset.spk2utt.items():
                        utt_id_list.append(utts)
                    
                    for attr, file_name, value_type, vector in kaldi_dataset.utt_first_files:
                        if attr in kaldi_dataset.loaded_attr:
                            this_file_dict = getattr(kaldi_dataset, attr)
                            if exclude:
                                new_file_dict = { k:v for k, v in this_file_dict.items() if k not in utt_id_list }
                            else:
                                new_file_dict = { k:v for k, v in this_file_dict.items() if k in utt_id_list }
                            setattr(kaldi_dataset, attr, new_file_dict)
                else:
                    raise ValueError("Expected spk2utt to exist to filter utt_first_files.")
        else:
            raise ValueError("Do not support id_type {0} with utt or spk only.".format(id_type))
        
        kaldi_dataset.get_base_attribute_()

        return kaldi_dataset

    def subset(self, num_utts:int, requirement:str="", drop=True, extra_list:list=[], seed:int=1024):
        """
        requirement: support part of requirements w.r.t subtools/utils/subset_data_dir.sh now.

        @return: KaldiDataset.
        """
        np.random.seed(seed)

        if requirement == "--per-spk":
            logger.info("Subset KaldiDatqaset to {0} utts with --per-spk requirement".format(num_utts*len(self.spk2utt)))
            utt_id_list = []
            for spk, utts in self.spk2utt.items():
                if len(utts) >= num_utts:
                    utt_id_list.extend(list(np.random.choice(utts, num_utts, replace=False)))
                elif not drop: 
                    utt_id_list.extend(utts)
            return self.filter(utt_id_list, id_type="utt")
        elif requirement == "--total-spk":
            # Select num_utts uttetances in total to contain speaker as more as possible. It is useful for valid set.
            logger.info("Subset KaldiDatqaset to {0} utts with --total-spk requirement".format(num_utts))
            if len(self.utt2spk.keys()) < num_utts:
                raise ValueError("The target num_utts {0} is out of total utts {1}".format(num_utts, len(utts)))

            spk2counter = {}
            for spk in self.spk2utt.keys():
                spk2counter[spk] = 0

            if self.num_spks >= num_utts:
                spks = list(np.random.choice(list(self.spk2utt.keys()), num_utts, replace=False))
                for spk in spks:
                    spk2counter[spk] = 1
            else:
                for spk in self.spk2utt.keys():
                    spk2counter[spk] += num_utts//self.num_spks

                if num_utts%self.num_utts > 0:
                    remain_spks = (list(np.random.choice(list(self.spk2utt.keys()), num_utts%self.num_spks, replace=False)))
                    for spk in remain_spks:
                        spk2counter[spk] += 1

            utt_id_list = []
            for spk, utts in self.spk2utt.items():
                utt_id_list.extend(list(np.random.choice(utts, min(spk2counter[spk], len(utts)), replace=False)))

            remain_num_utts = num_utts - len(utt_id_list)
            if remain_num_utts > 0:
                utt_id_list.extend(list(np.random.choice(set(self.utt2spk.keys())-set(utt_id_list)), remain_num_utts, replace=False))
            return self.filter(utt_id_list, id_type="utt")
        elif requirement == "--speakers":
            pass
        elif requirement == "--first":
            pass
        elif requirement == "--last":
            pass
        elif requirement == "--shortest":
            pass
        elif requirement == "--spk-list":
            pass
        elif requirement == "--utt-list":
            pass
        else:
            logger.info("Subset KaldiDatqaset to {0} utts with --default requirement".format(num_utts))
            utts = list(self.utt2spk.keys())
            if len(utts) >= num_utts:
                utt_id_list = list(np.random.choice(utts, num_utts, replace=False))
                return self.filter(utt_id_list, id_type="utt")
            else:
                raise ValueError("The target num_utts {0} is out of total utts {1}".format(num_utts, len(utts)))

    def split(self, num_utts:int, requirement:str="", drop=True, extra_list:list=[], seed:int=1024):
        """
        @return: (KaldiDataset, KaldiDataset).
        """
        split_part = self.subset(num_utts, requirement, drop, extra_list, seed)
        utt_id_list = list(split_part.utt2spk.keys())
        remain_part = self.filter(utt_id_list, id_type="utt", exclude=True)
        return remain_part, split_part

    @classmethod
    def load(self, file_path:str):
        pass

    def save(self, file_path:str):
        pass

    def __len__(self):
        return self.num_utts

    def __str__(self):
        return "<class KaldiDataset>\n[ data_dir = {0}, loaded = {1} ]\n"\
               "[ num_utts = {2}, num_spks = {3}, num_frames = {4}, feat_dim= {5} ]\n"\
               "".format(self.data_dir, self.loaded_attr, self.num_utts, self.num_spks, self.num_frames, self.feat_dim)


### Function
def to(to_type:str, value):
    if to_type == "str" or to_type == "float" or to_type == "int":
        return eval("{0}('{1}')".format(to_type, value))
    else:
        raise ValueError("Do not support {0} to_type".format(to_type))


def read_str_first_ark(file_path:str, value_type="str", vector=False, every_bytes=10000000):
    this_dict = {}

    with open(file_path, 'r') as reader:
            while True :
                lines = reader.readlines(every_bytes)
                if not lines:
                    break
                for line in lines:
                    if vector:
                        # split_line => n
                        split_line = line.split()
                        # split_line => n-1
                        key = split_line.pop(0)
                        value = [ to(value_type, x) for x in split_line ]
                        this_dict[key] = value
                    else:
                        key, value = line.split()
                        this_dict[key] = to(value_type, value)

    return this_dict
