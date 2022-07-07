# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-01-05, JFZhou 2020-01-08)
# Update Multi-task learning. Author: Zheng Li 2020-10
import os
import logging
import random
import numpy as np
import pandas as pd
import torch
import torchaudio

# import libs.support.kaldi_common as kaldi_common # Used to interact with shell
from .kaldi_dataset import KaldiDataset
from libs.support.utils import get_torchaudio_backend
torchaudio_backend = get_torchaudio_backend()
torchaudio.set_audio_backend(torchaudio_backend)
# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ChunkSamples():
    def __init__(self, dataset:KaldiDataset, chunk_size:int, chunk_type='speaker_balance', chunk_num_selection=0, 
                 scale=1.5, overlap=0.1, drop_last=False, seed=1024):
        '''
        Parameters:
            self.dataset: the object which contain the dicts such as utt2spk, utt2spk_int and so on.
            self.chunk_size: the number of frames in a chunk.
            self.chunk_type: which decides how to chunk the feats for training.
            chunk_num_selection: -1->suggestion scale, 0->max, >0->specify.
            self.overlap: the proportion of overlapping for every chunk.
        '''
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.chunk_type = chunk_type
        self.chunk_num_selection = chunk_num_selection
        self.scale = scale
        self.overlap = overlap
        self.drop_last = drop_last

        assert 0<= self.overlap < 1

        np.random.seed(seed)

        # chunk_samples: table [[]]
        # add eg-dur for iter IO (Leo 2022-02-16)
        self.eg_dur = format(self.chunk_size/100,'.2f')
        self.head = ['eg-id', 'ark-path', 'start-position', 'end-position', 'class-label', 'eg-dur']
        self.chunk_samples = self.__sample()

    def __sample(self):
        # JFZhou: speaker_balance and sequential.
        chunk_samples = []

        if self.chunk_type == 'speaker_balance':
            spk2chunks = {}
            total_chunks = 0
            max_chunk_num = 0
            chunk_counter = {}
            for key in self.dataset.spk2utt.keys():
                utt_selected = self.dataset.spk2utt[key]
                spk_chunk_num = 0
                for utt in utt_selected:
                    ark_path = self.dataset.feats_scp[utt]
                    num_frames = self.dataset.utt2num_frames[utt]

                    if num_frames < self.chunk_size:
                        logger.warn('The num frames {0} of {1} is less than chunk size {2}, so skip it.'.format(utt, num_frames, self.chunk_size))
                    else:
                        chunk_counter[utt] = 0
                        offset = 0
                        overlap_size = int(self.overlap * self.chunk_size)
                        while offset + self.chunk_size <= num_frames:
                            chunk = "{0} {1} {2} {3} {4} {5}".format(utt+'-'+str(chunk_counter[utt]),ark_path,offset,offset+self.chunk_size-1,self.dataset.utt2spk_int[utt],self.eg_dur)
                            offset += self.chunk_size - overlap_size

                            if key in spk2chunks.keys():
                                spk2chunks[key].append(chunk)
                            else:
                                spk2chunks[key] = [chunk]

                            chunk_counter[utt] += 1
                            total_chunks += 1
                            spk_chunk_num += 1

                        if not self.drop_last and offset + overlap_size < num_frames:
                            chunk = "{0} {1} {2} {3} {4} {5}".format(utt+'-'+str(chunk_counter[utt]),ark_path,num_frames-self.chunk_size,num_frames-1,self.dataset.utt2spk_int[utt],self.eg_dur)
                            total_chunks += 1
                            spk_chunk_num += 1
                            chunk_counter[utt] += 1

                if spk_chunk_num > max_chunk_num:
                    max_chunk_num = spk_chunk_num

            for key in spk2chunks.keys():
                chunk_selected = spk2chunks[key]
                if self.chunk_num_selection==0:
                    num_chunks_selected = max_chunk_num
                elif self.chunk_num_selection==-1:
                    num_chunks_selected = int(total_chunks//len(self.dataset.spk2utt)*self.scale)
                else:
                    num_chunks_selected = self.chunk_num_selection

                num_chunks = len(chunk_selected)
                if num_chunks < num_chunks_selected:
                    valid_utts = [ utt for utt in self.dataset.spk2utt[key] if self.dataset.utt2num_frames[utt] >= self.chunk_size ]
                    utts = np.random.choice(valid_utts,num_chunks_selected-num_chunks,replace=True)
                    for utt in utts:
                        start = np.random.randint(0, self.dataset.utt2num_frames[utt]-self.chunk_size+1)
                        end = start + self.chunk_size - 1
                        chunk_selected.append("{0} {1} {2} {3} {4} {5}".format(utt+'-'+str(chunk_counter[utt]),self.dataset.feats_scp[utt],start,end,self.dataset.utt2spk_int[utt],self.eg_dur))
                        chunk_counter[utt] += 1
                else:
                    chunk_selected = np.random.choice(spk2chunks[key],num_chunks_selected,replace=False)

                for chunk in chunk_selected:
                    chunk_samples.append(chunk.split())

        elif self.chunk_type == 'sequential':
            for utt in self.dataset.feats_scp.keys():

                ark_path = self.dataset.feats_scp[utt]
                num_frames = self.dataset.utt2num_frames[utt]

                if num_frames < self.chunk_size:
                    logger.warn('The num frames {0} of {1} is less than chunk size {2}, so skip it.'.format(utt, num_frames, self.chunk_size))
                else:
                    chunk_counter = 0
                    offset = 0
                    overlap_size = int(self.overlap * self.chunk_size)
                    while offset + self.chunk_size <= num_frames:
                        chunk_samples.append([utt+'-'+str(chunk_counter),ark_path,offset,offset+self.chunk_size-1,self.dataset.utt2spk_int[utt],self.eg_dur])
                        chunk_counter += 1
                        offset += self.chunk_size - overlap_size

                    if not self.drop_last and offset + overlap_size < num_frames:
                        chunk_samples.append([utt+'-'+str(chunk_counter),ark_path,num_frames-self.chunk_size,num_frames-1,self.dataset.utt2spk_int[utt],self.eg_dur])

        # every_utt for valid
        elif self.chunk_type == "every_utt":
            chunk_selected = []
            for utt in self.dataset.utt2spk.keys():
                ark_path = self.dataset.feats_scp[utt]
                num_frames = self.dataset.utt2num_frames[utt]

                if num_frames < self.chunk_size:
                    logger.warn('The num frames {0} of {1} is less than chunk size {2}, so skip it.'.format(utt, num_frames, self.chunk_size))
                else:
                    for chunk_counter in range(0, self.chunk_num_selection):
                        start = np.random.randint(0, self.dataset.utt2num_frames[utt]-self.chunk_size+1)
                        end = start + self.chunk_size - 1
                        chunk_selected.append("{0} {1} {2} {3} {4} {5}".format(utt+'-'+str(chunk_counter),self.dataset.feats_scp[utt],start,end,self.dataset.utt2spk_int[utt],self.eg_dur))

            for chunk in chunk_selected:
                    chunk_samples.append(chunk.split())

        else:
            raise TypeError("Do not support chunk type {0}.".format(self.chunk_type))

        return chunk_samples

    def save(self, save_path:str, force=True):
        if os.path.exists(save_path) and not force:
            raise ValueError("The path {0} is exist. Please rm it by yourself.".format(save_path))

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data_frame = pd.DataFrame(self.chunk_samples, columns=self.head)
        data_frame.to_csv(save_path, sep=" ", header=True, index=False)


class ChunkSamplesMultiTask():
    def __init__(self, dataset:KaldiDataset, chunk_size:int, chunk_type='speaker_balance', chunk_num_selection=0, 
                 scale=1.5, overlap=0.1, drop_last=False, seed=1024):
        '''
        Parameters:
            self.dataset: the object which contain the dicts such as utt2spk, utt2spk_int and so on.
            self.chunk_size: the number of frames in a chunk.
            self.chunk_type: which decides how to chunk the feats for training.
            chunk_num_selection: -1->suggestion scale, 0->max, >0->specify.
            self.overlap: the proportion of overlapping for every chunk.
        '''
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.chunk_type = chunk_type
        self.chunk_num_selection = chunk_num_selection
        self.scale = scale
        self.overlap = overlap
        self.drop_last = drop_last

        assert 0<= self.overlap < 1

        np.random.seed(seed)

        # chunk_samples: table [[]]
        self.head = ['eg-id', 'ark-path', 'start-position', 'end-position', 'class-label', 'ali-path'] #Zheng Li 2020-10
        self.chunk_samples = self.__sample()

    def __sample(self):
        # JFZhou: speaker_balance and sequential.
        chunk_samples = []

        if self.chunk_type == 'speaker_balance':
            spk2chunks = {}
            total_chunks = 0
            max_chunk_num = 0
            chunk_counter = {}
            for key in self.dataset.spk2utt.keys():
                utt_selected = self.dataset.spk2utt[key]
                spk_chunk_num = 0
                for utt in utt_selected:
                    ark_path = self.dataset.feats_scp[utt]
                    num_frames = self.dataset.utt2num_frames[utt]
                    ali_path = self.dataset.ali_scp[utt]  #Zheng Li 2020-10

                    if num_frames < self.chunk_size:
                        logger.warn('The num frames {0} of {1} is less than chunk size {2}, so skip it.'.format(utt, num_frames, self.chunk_size))
                    else:
                        chunk_counter[utt] = 0
                        offset = 0
                        overlap_size = int(self.overlap * self.chunk_size)
                        while offset + self.chunk_size <= num_frames:
                            chunk = "{0} {1} {2} {3} {4} {5}".format(utt+'-'+str(chunk_counter[utt]),ark_path,offset,offset+self.chunk_size-1,self.dataset.utt2spk_int[utt],ali_path)  #Zheng Li 2020-10
                            offset += self.chunk_size - overlap_size

                            if key in spk2chunks.keys():
                                spk2chunks[key].append(chunk)
                            else:
                                spk2chunks[key] = [chunk]

                            chunk_counter[utt] += 1
                            total_chunks += 1
                            spk_chunk_num += 1

                        if not self.drop_last and offset + overlap_size < num_frames:
                            chunk = "{0} {1} {2} {3} {4} {5}".format(utt+'-'+str(chunk_counter[utt]),ark_path,num_frames-self.chunk_size,num_frames-1,self.dataset.utt2spk_int[utt],ali_path)  #Zheng Li 2020-10
                            total_chunks += 1
                            spk_chunk_num += 1
                            chunk_counter[utt] += 1

                if spk_chunk_num > max_chunk_num:
                    max_chunk_num = spk_chunk_num

            for key in spk2chunks.keys():
                chunk_selected = spk2chunks[key]
                if self.chunk_num_selection==0:
                    num_chunks_selected = max_chunk_num
                elif self.chunk_num_selection==-1:
                    num_chunks_selected = int(total_chunks//len(self.dataset.spk2utt)*self.scale)
                else:
                    num_chunks_selected = self.chunk_num_selection

                num_chunks = len(chunk_selected)
                if num_chunks < num_chunks_selected:
                    valid_utts = [ utt for utt in self.dataset.spk2utt[key] if self.dataset.utt2num_frames[utt] >= self.chunk_size ]
                    utts = np.random.choice(valid_utts,num_chunks_selected-num_chunks,replace=True)
                    for utt in utts:
                        start = np.random.randint(0, self.dataset.utt2num_frames[utt]-self.chunk_size+1)
                        end = start + self.chunk_size - 1
                        chunk_selected.append("{0} {1} {2} {3} {4} {5}".format(utt+'-'+str(chunk_counter[utt]),self.dataset.feats_scp[utt],start,end,self.dataset.utt2spk_int[utt],self.dataset.ali_scp[utt]))    #Zheng Li 2020-10
                        chunk_counter[utt] += 1
                else:
                    chunk_selected = np.random.choice(spk2chunks[key],num_chunks_selected,replace=False)

                for chunk in chunk_selected:
                    chunk_samples.append(chunk.split())

        elif self.chunk_type == 'sequential':
            for utt in self.dataset.feats_scp.keys():

                ark_path = self.dataset.feats_scp[utt]
                num_frames = self.dataset.utt2num_frames[utt]
                ali_path = self.dataset.ali_scp[utt]  #Zheng Li 2020-10

                if num_frames < self.chunk_size:
                    logger.warn('The num frames {0} of {1} is less than chunk size {2}, so skip it.'.format(utt, num_frames, self.chunk_size))
                else:
                    chunk_counter = 0
                    offset = 0
                    overlap_size = int(self.overlap * self.chunk_size)
                    while offset + self.chunk_size <= num_frames:
                        chunk_samples.append([utt+'-'+str(chunk_counter),ark_path,offset,offset+self.chunk_size-1,self.dataset.utt2spk_int[utt],ali_path])  #Zheng Li 2020-10
                        chunk_counter += 1
                        offset += self.chunk_size - overlap_size

                    if not self.drop_last and offset + overlap_size < num_frames:
                        chunk_samples.append([utt+'-'+str(chunk_counter),ark_path,num_frames-self.chunk_size,num_frames-1,self.dataset.utt2spk_int[utt],ali_path])  #Zheng Li 2020-10

        # every_utt for valid
        elif self.chunk_type == "every_utt":
            chunk_selected = []
            for utt in self.dataset.utt2spk.keys():
                ark_path = self.dataset.feats_scp[utt]
                num_frames = self.dataset.utt2num_frames[utt]
                ali_path = self.dataset.ali_scp[utt]  #Zheng Li 2020-10

                if num_frames < self.chunk_size:
                    logger.warn('The num frames {0} of {1} is less than chunk size {2}, so skip it.'.format(utt, num_frames, self.chunk_size))
                else:
                    for chunk_counter in range(0, self.chunk_num_selection):
                        start = np.random.randint(0, self.dataset.utt2num_frames[utt]-self.chunk_size+1)
                        end = start + self.chunk_size - 1
                        chunk_selected.append("{0} {1} {2} {3} {4} {5}".format(utt+'-'+str(chunk_counter),self.dataset.feats_scp[utt],start,end,self.dataset.utt2spk_int[utt],self.dataset.ali_scp[utt]))  #Zheng Li 2020-10

            for chunk in chunk_selected:
                    chunk_samples.append(chunk.split())

        else:
            raise TypeError("Do not support chunk type {0}.".format(self.chunk_type))

        return chunk_samples

    def save(self, save_path:str, force=True):
        if os.path.exists(save_path) and not force:
            raise ValueError("The path {0} is exist. Please rm it by yourself.".format(save_path))

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data_frame = pd.DataFrame(self.chunk_samples, columns=self.head)
        data_frame.to_csv(save_path, sep=" ", header=True, index=False)

# Leo 20210820
class RawWavSample():
    def __init__(self,dataset:KaldiDataset,seed=1024,chunk_num_selection=0,fix_chunk_num=True):
        '''
        Parameters:
            dataset : the object which contain the dicts such as utt2spk, utt2spk_int and so on.
            chunk_num_selection: the chunk numbers of every_utt for valid
            fix_chunk_num: whether append random chunk when the number of utt2chunk is shorter then chunk_num_selection. 
        '''
        self.dataset = dataset

        np.random.seed(seed)

        self.head = ['eg-id','wav-path','sample-rate', 'duration', 'eg-dur', 'start-position', 'end-position',"class-label"]



        self.chunk_num_selection=int(chunk_num_selection)
        self.fix_chunk_num=fix_chunk_num

        self.samples,self.total_sample_dur = self.__sample()


    def __sample(self):
        samples=[]
        total_sample_dur=0
        for utt,path in self.dataset.wav_scp.items():
            duration=self.dataset.utt2dur[utt]
            class_label=self.dataset.utt2spk_int[utt]
            chunk_list=(self.dataset.utt2chunk[utt]).split('#')
            sr = self.dataset.utt2sr[utt]
             # every_utt for valid
            if self.chunk_num_selection > 0:
                if len(chunk_list)>=self.chunk_num_selection:
                    chunk_list=chunk_list[:self.chunk_num_selection]
                else:
                    seg=float(chunk_list[0].split('_')[1])-float(chunk_list[0].split('_')[0])

                    seg_sample = int(sr*seg)
                    duration_sample=int(sr*duration)
                    if duration>seg and self.fix_chunk_num:
                        for _ in range(self.chunk_num_selection-len(chunk_list)):
                            start_sample=random.randint(0,duration_sample - seg_sample)
                            end_sample = start_sample + seg_sample
                            chunk_list.append(str(start_sample/sr)+'_'+str(end_sample/sr))
                    else:
                        pass

            for chunk in chunk_list:
                s,e=float(chunk.split('_')[0]),float(chunk.split('_')[1])
                start_sample = int(float(s) * sr)
                end_sample = int(float(e) * sr)
                sample_dur = e-s
                total_sample_dur+=sample_dur
                sample_dur=format(sample_dur,'.3f')
                eg_id = "{}-{:.3f}_{:.3f}".format(utt,s,e)
                # Composition of the csv_line
                csv_line = [
                eg_id,
                path,
                sr,
                duration,
                sample_dur,
                start_sample,
                end_sample,
                class_label,
                ]
                samples.append(csv_line)

        return samples,total_sample_dur


    def save(self, save_path:str, force=True):
        if os.path.exists(save_path) and not force:
            raise ValueError("The path {0} is exist. Please rm it by yourself.".format(save_path))

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data_frame = pd.DataFrame(self.samples, columns=self.head)
        data_frame.to_csv(save_path, sep=" ", header=True, index=False)
