#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright xmuspeech (Author:Snowdar 2019-01-10)

# It is a little different with Kaldi method.
# By this method, EER is estimated by avaraging the error rates of two points nearby center.

import sys
import argparse

def get_args():
    # Start
    parser = argparse.ArgumentParser(
        description="""Compute EER.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    # Main
    parser.add_argument("trials_path", metavar="trials_path", type=str, help="The path of trials.")
    parser.add_argument("score_path", metavar="score_path", type=str, help="The path of the scores.")

    # End
    print(' '.join(sys.argv))
    args = parser.parse_args()

    return args

def load_data(data_path,n):
    list=[]
    print("Load data from "+data_path+"...")
    with open(data_path,'r') as f:
        content=f.readlines()
        for line in content:
            line=line.strip()
            data_list=line.split()
            if(n!=len(data_list)):
                print('[Error] The %s file has no %s fields'%(data_path,n))
                exit(1)

            list.append(data_list)
    return list

def abs(x):
    if(x<0):
        return -x
    else:
        return x
        
def compute_eer(allScores):
    numP=0
    numN=0
    for x in allScores:
        if(x[1]=="target"):
            x[1]=1
            numP=numP+1
        elif(x[1]=="nontarget"):
            x[1]=0
            numN=numN+1
        else:
            print("[Error in compute_eer()] %s is not target or nontarget in score"%(x[1]))
            exit(1)
            
    allScores=sorted(allScores,reverse=False)
    
    numFA=numN
    numFR=0

    eer=0.0
    threshold=0.0
    memory=[]
    
    for tuple in allScores:
        if(tuple[1]==1):
            numFR=numFR+1
        else:
            numFA=numFA-1
        
        far=numFA*1.0/numN
        frr=numFR*1.0/numP
        
        if(far<=frr):
            lnow=abs(far-frr)
            lmemory=abs(memory[0]-memory[1])
            if(lnow<=lmemory):
                eer=(far+frr)/2
                threshold=tuple[0]
            else:
                eer=(memory[0]+memory[1])/2
                threshold=memory[2]
            return eer, threshold
        else:
            memory=[far,frr,tuple[0]]

def main():
    args = get_args()

    try:
        trials = load_data(args.trials_path, 3)
        scores = load_data(args.score_path, 3)

        allScores = []
        label_dict = {}

        for x in trials:
            label_dict[x[0]+x[1]]=x[2]

        for x in scores:
            allScores.append([float(x[2]),label_dict[x[0]+x[1]]])

        eer, threshold = compute_eer(allScores)

        print("EER% {:.3f} (threshold = {:.5f})".format(eer*100, threshold))
    except BaseException as e:
        # Look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
