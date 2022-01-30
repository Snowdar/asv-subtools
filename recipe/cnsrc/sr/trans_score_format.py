# -*- coding: utf-8 -*-
"""
Created on 2022-01-29
@author: Tao Jiang
"""

import os
import pandas as pd
import time 
import sys

def pd_sort(fileName,outfile):
    header = ['spk-id', 'utt-id', 'scores']
    df = pd.read_csv(fileName,sep='\s+', names=header)
    df = df.groupby('spk-id').apply(lambda x:x.nlargest(10,'scores'))
    #write to file 
    with open(outfile,'w',encoding='utf-8') as f:
        old_spk = ' '
        for index,row in df.iterrows():
            spk = row['spk-id']
            if old_spk == ' ':
                old_spk = spk
                f.write(spk)
            elif old_spk != spk:
                old_spk = spk
                f.write('\n'+spk)
            f.write(' '+row['utt-id'])

if __name__ == '__main__':
    #get the original score file
    fileName = sys.argv[1]
    outfile = sys.argv[2]
    #sort score file and get requred format
    pd_sort(fileName,outfile)
    