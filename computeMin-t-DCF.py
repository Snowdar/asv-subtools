#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright xmuspeech (Author:Snowdar 2019-01-10)

# This metric is for ASVspoof Challenge in 2019.

import sys

# Num of data
N_bona_cm=0
N_spoof_cm=0

# Priors
Pi_tar=0.9405
Pi_non=0.0095
Pi_spoof=0.05

# ASV costs
C_miss_asv=1
C_fa_asv=10

# CM costs
C_miss_cm=1
C_fa_cm=10

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
            return eer,threshold
        else:
            memory=[far,frr,tuple[0]]

            
def t_DCF_min(dcf):
    return min(dcf)

def t_DCF_norm(beta,P_miss_cm,P_fa_cm):
    return beta * P_miss_cm + P_fa_cm

def get_rate(x,y):
    if(y==0):
        return 0
    else:
        return x*1.0/y

def obtain_asv_error_rates(asv_score,asv_threshold):
    N_tar_asv=0
    N_non_asv=0
    N_spoof_asv=0
    
    count_tar=0
    count_non=0
    count_spoof=0
    
    for x in asv_score:
        if(x[1]=="target"):
            N_tar_asv=N_tar_asv+1
            if(float(x[2])<asv_threshold):
                count_tar=count_tar+1
        elif(x[1]=="nontarget"):
            N_non_asv=N_non_asv+1
            if(float(x[2])>=asv_threshold):
                count_non=count_non+1
        elif(x[1]=="spoof"):
            N_spoof_asv=N_spoof_asv+1
            if(float(x[2])<asv_threshold):
                count_spoof=count_spoof+1
        else:
            print("[Error in obtain_asv_error_rates()] %s is not target or nontarget or spoof in score"%(x[1]))
    
    P_miss_asv=get_rate(count_tar,N_tar_asv)
    P_fa_asv=get_rate(count_non,N_non_asv)
    P_miss_spoof_asv=get_rate(count_spoof,N_spoof_asv)
    
    return P_miss_asv,P_fa_asv,P_miss_spoof_asv

def check():
    if(Pi_tar+Pi_non+Pi_spoof!=1):
        print("[Error in check()] Pi_tar+Pi_non+Pi_spoof != 1 ")
        exit(1)

## main ##
if len(sys.argv)-1 != 2 :
    print 'usage: '+sys.argv[0]+' [options] <asv-score> <cm-score>'
    exit(1)
"""
asv-score format with every line:
    attack-way target/nontarget/spoof score
example:
    - target 4.23
    - nontarget 1.24
    VC_1 spoof 2.55
    
cm-score format with every line:
    bonafide/spoof score  
example:
    bonafide 2.34
    spoof -1.2
"""
asv_score_path=sys.argv[1]
cm_score_path=sys.argv[2]

check()

#- start -#
asv_score_file=load_data(asv_score_path,3)
cm_score_file=load_data(cm_score_path,2)

#- asv -#
asv_score_for_eer=[]
for x in asv_score_file:
    if(x[1]=="target" or x[1]=="nontarget"):
        asv_score_for_eer.append([float(x[2]),x[1]])
        
asv_eer,asv_threshold=compute_eer(asv_score_for_eer)
P_miss_asv,P_fa_asv,P_miss_spoof_asv=obtain_asv_error_rates(asv_score_file,asv_threshold)

#- cm -#
cm_score=[]
cm_score_for_eer=[]
for x in cm_score_file:
    if(x[0]=="bonafide"):
        lable=1
        text="target"
        N_bona_cm=N_bona_cm+1
    elif(x[0]=="spoof"):
        lable=0
        text="nontarget"
        N_spoof_cm=N_spoof_cm+1
    else:
        print("[Error in main-cm] the lable %s is not bonafide or spoof"%(x[0]))
        exit(1)

    cm_score.append([float(x[1]),lable])
    cm_score_for_eer.append([float(x[1]),text])
    
cm_eer,cm_threshold=compute_eer(cm_score_for_eer)

#- t-DCF -#
C1=Pi_tar * (C_miss_cm - C_miss_asv * P_miss_asv) - Pi_non * C_fa_asv * P_fa_asv
C2=C_fa_cm * Pi_spoof * (1 - P_miss_spoof_asv)
beta=C1/C2

cm_score=sorted(cm_score,reverse=False)

count_bona=0
count_spoof=N_spoof_cm
dcf=[]

P_miss_cm=count_bona*1.0/N_bona_cm
P_fa_cm=count_spoof*1.0/N_spoof_cm

dcf.append(t_DCF_norm(beta,P_miss_cm,P_fa_cm))

for tuple in cm_score:
        if(tuple[1]==1):
            count_bona=count_bona+1
        else:
            count_spoof=count_spoof-1
        
        P_miss_cm=count_bona*1.0/N_bona_cm
        P_fa_cm=count_spoof*1.0/N_spoof_cm
        dcf.append(t_DCF_norm(beta,P_miss_cm,P_fa_cm))

min_tDCF=t_DCF_min(dcf)

#- print -#
print("\n[Report]")
print("ASV EER=%f%%, threshold=%f"%(asv_eer*100,asv_threshold))
print("ASV Pfa=%f%%, Pmiss=%f%%, 1-Pmiss,spoof=%f%%"%(P_fa_asv*100,P_miss_asv*100,(1-P_miss_spoof_asv)*100))
print("CM EER=%f%%, threshold=%f"%(cm_eer*100,cm_threshold))
print("Final min-tDCF=%f"%(min_tDCF))
