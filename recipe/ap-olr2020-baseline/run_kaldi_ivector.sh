#!/bin/bash

# Copyright 2013   Daniel Povey
#           2014   David Snyder
#           2016   Lantian Li, Yixiang Chen, Zhiyuan Tang, Dong Wang
#           2018   xmuspeech (Author:Snowdar 2018-7-25)
#           2019   xmuspeech (Author:ZHENG LI 2019-10-15)
# Apache 2.0.
# This is a copy-script by run_ivector_lid.sh in ap17-olr kaldi recipe and it 
# just contains train steps but added cmvn options by changing the sub-scripts.

stage=0
endstage=4

# Number of components
cnum=2048 # num of Gaussions
civ=600   # dim of i-vector

train_nj=20 

set -e

train=data/mfcc_20_5.0/train_aug
outputname=kaldi_ivector


exp=exp/$outputname

if [[ $stage -le 0 && 0 -le $endstage ]];then
echo "[step 0] Get subset"
[ -d $exp ] && "[exit] $exp is existed, please rename exp-dir or delete $exp by yourself" && exit 1
# Get the subset to train by adding data gradually
subtools/kaldi/utils/subset_data_dir.sh $train 60000 ${train}_60k
subtools/kaldi/utils/subset_data_dir.sh $train 120000 ${train}_120k
fi

if [[ $stage -le 1 && 1 -le $endstage ]];then
echo "[step 1] Diag UBM training"
subtools/kaldi/sid/train_diag_ubm.sh --nj $train_nj --cmd "run.pl" ${train}_60k ${cnum} $exp/diag_ubm_${cnum}
fi

if [[ $stage -le 2 && 2 -le $endstage ]];then 
echo "[step 2 Full UBM training"
subtools/kaldi/sid/train_full_ubm.sh --nj $train_nj --cmd "run.pl" ${train}_120k $exp/diag_ubm_${cnum} $exp/full_ubm_${cnum} 
fi

if [[ $stage -le 3 && 3 -le $endstage ]];then
echo "[step 3] T-matrix training"
subtools/kaldi/sid/train_ivector_extractor.sh --nj $[$train_nj/4] --cmd "run.pl" \
    --num-iters 6 --ivector_dim $civ $exp/full_ubm_${cnum}/final.ubm $train \
    $exp/extractor_${cnum}_${civ}
fi
echo "Ivector training done."

if [[ $stage -le 4 && 4 -le $endstage ]];then
prefix=mfcc_20_5.0
toEXdata="train task1_test task2_test task3_test task1_enroll task2_enroll task3_enroll task1_dev task2_dev"
nj=5


echo "[step 4] Extract i-vector"
for x in $toEXdata;do
rm -rf $exp/$x
subtools/kaldi/sid/extract_ivectors.sh --cmd "run.pl" --nj $nj \
    $exp/extractor_${cnum}_${civ} data/${prefix}/$x $exp/$x
> $exp/$x/$prefix
echo "data/$prefix/$x extracted done."
done
fi
