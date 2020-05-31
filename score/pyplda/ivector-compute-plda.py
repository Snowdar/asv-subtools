# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: JFZhou 2019-12-22)

import scipy
import numpy as np
import math
import os
import sys
from plda_base import PldaStats,PldaEstimation
sys.path.insert(0, 'subtools/pytorch')
import libs.support.kaldi_io as kaldi_io
import logging


# Logger
logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def main():

    if len(sys.argv)!=4:
  
        print("Usage: "+sys.argv[0]+" <spk2utt-rspecifier> <ivector-rspecifier> <plda>\n")
        print("e.g.: "+sys.argv[0]+" spk2utt ivectors.ark plda")
        
        sys.exit() 

    spk2utt = sys.argv[1]
    ivectors_reader = sys.argv[2]
    plda_out = sys.argv[3]


    logger.info('Load vecs and accumulate the stats of vecs.....')
    utt2spk_dict = {}
    with open(spk2utt,'r') as f:
        for line in f:
            temp_list = line.strip().split()
            spk = temp_list[0]
            del temp_list[0]
            for utt in temp_list:
                utt2spk_dict[utt] = spk

    spk2vectors = {}
    for key,vector in kaldi_io.read_vec_flt_auto(ivectors_reader):
        dim = vector.shape[0]
        spk = utt2spk_dict[key]
        try:
            tmp_list = spk2vectors[spk]
            tmp_list.append(vector)
            spk2vectors[spk] = tmp_list
        except KeyError:
            spk2vectors[spk] = [vector]

    plda_stats=PldaStats(dim)
    for key in spk2vectors.keys():
        vectors = np.array(spk2vectors[key], dtype=float)
        weight = 1.0
        plda_stats.add_samples(weight,vectors)

    logger.info('Estimate the parameters of PLDA by EM algorithm...')
    plda_stats.sort()
    plda_estimator=PldaEstimation(plda_stats)
    plda_estimator.estimate()
    logger.info('Save the parameters for the PLDA adaptation...')
    plda_estimator.plda_write(plda_out+'.ori')
    plda_trans = plda_estimator.get_output()
    logger.info('Save the parameters for scoring directly, which is the same with the plda in kaldi...')
    plda_trans.plda_trans_write(plda_out)

if __name__ == "__main__":
    main()