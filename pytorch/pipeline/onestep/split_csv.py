# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2021-08-27)


from genericpath import exists
import sys
import os
import math
import logging
import argparse
import traceback
import pandas as pd
"""Get chunk egs for sre and lid ... which use the xvector framework.
"""

def write_splices(data,head,file_name,dir):
    os.makedirs(dir,exist_ok=True)
    file_path = os.path.join(dir,file_name)
    data_frame = pd.DataFrame(data, columns=head)
    data_frame.to_csv(file_path,sep=" ",header=True, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Split csv file.
                       It creates its output in e.g. data/train/split50/{1,2,3,...50}
                    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument('--nj',
                        type=int,
                        default=4,
                        help='num threads for make shards')
    parser.add_argument('--csv-name',
                        type=str,
                        default="wav.csv",
                        help='the csv name to be splited.')


    # Main
    parser.add_argument("eg_dir", type=str,help="wav list dir.")

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
    print(' '.join(sys.argv))

    args = parser.parse_args()
    data_path =os.path.join(args.eg_dir,args.csv_name)
    head = pd.read_csv(data_path,sep=" ",nrows=0).columns
    data = pd.read_csv(data_path, sep=" ").values
    num_data = len(data)

    if num_data < args.nj:
        logging.fatal("the number of nj is larger then data length.")

    num_per_slice = math.ceil(num_data/args.nj)
    cur = 0
    cur_slice = 1
    while cur < num_data:
        end = min(cur+num_per_slice,num_data)
        item=[]
        for i in range(cur,end):
            item.append(data[i])
        cur = end
        split_dir = os.path.join(args.eg_dir,"split{}utt/{}".format(args.nj,cur_slice))
        write_splices(item,head,args.csv_name,split_dir)
        cur_slice+=1





