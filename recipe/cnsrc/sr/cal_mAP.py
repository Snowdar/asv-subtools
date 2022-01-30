# -*- coding: utf-8 -*-
"""
Created on 2022-01-29
@author: Tao Jiang
"""
import os
import pandas as pd
import time 
import sys
def cal_mAP(top10_file,test_meta):
    f1=open(top10_file,'r',encoding='utf-8')
    f2=open(test_meta,'r',encoding='utf-8')
    score2allspk=0.0
    spk_num=0
    target_list={}
    for c in f2.readlines():
        c_arr = c.strip('\n').split(' ')
        spk_id = c_arr[1].split('-')[0]
        utt_id = c_arr[0].strip('.wav').split('/')[1]
        if spk_id not in target_list:
            target_list[spk_id]=[utt_id]
        else:
            target_list[spk_id].append(utt_id)
    spk_i = 0
    for c in f1.readlines():
        c_arr = c.strip('\n').split(' ')
        spk_i+=1
        spk_id=c_arr[0]
        spk_score=0.0
        target_num=0
        for i in range(1,len(c_arr)):
            utt_id=c_arr[i]
            if utt_id in target_list[spk_id]:
                target_num+=1
            spk_score+=target_num/i
        spk_score/=10
        score2allspk+=spk_score
    av_score=score2allspk/spk_i
    print("mAP=%.3f"%av_score)

if __name__ == '__main__':
    #import the original score file
    top10_file = sys.argv[1]
    test_meta = sys.argv[2]
    #callulate score file
    cal_mAP(top10_file,test_meta)