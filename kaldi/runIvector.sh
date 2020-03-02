#!/bin/bash

# Copyright 2013   Daniel Povey
#           2014   David Snyder
#           2016   Lantian Li, Yixiang Chen, Zhiyuan Tang, Dong Wang
#           2018   xmuspeech (Author:Snowdar 2018-7-25)
# Apache 2.0.
# This is a copy-script by run_ivector_lid.sh in ap17-olr kaldi recipe.

set -e

stage=0
endstage=4

# Number of components
cnum=2048 # num of Gaussions
civ=400   # dim of i-vector
delta=true # if training by paste phonetic feats,should be false
train_nj=20 
train_stage=-4


train=data/plp_20_5.0/baseTrain_concat_sp
outputname=concat_sp_iv_plp_20_5.0

. subtools/kaldi/utils/parse_options.sh

exp=exp/$outputname

if [[ $stage -le 0 && 0 -le $endstage ]];then
echo "[step 0] Get subset"
[ -d $exp ] && echo "[exit] $exp is existed, please rename exp-dir or delete $exp by yourself" && exit 1
# Get the subset to train by adding data gradually
subtools/kaldi/utils/subset_data_dir.sh $train 18000 ${train}_18k
subtools/kaldi/utils/subset_data_dir.sh $train 36000 ${train}_36k
fi

if [[ $stage -le 1 && 1 -le $endstage ]];then
echo "[step 1] Diag UBM training"
subtools/kaldi/sid/train_diag_ubm.sh --nj $train_nj --cmd "run.pl" --delta $delta ${train}_18k ${cnum} $exp/diag_ubm_${cnum}
fi

if [[ $stage -le 2 && 2 -le $endstage ]];then 
echo "[step 2 Full UBM training"
subtools/kaldi/sid/train_full_ubm.sh --nj $train_nj --cmd "run.pl" --delta $delta ${train}_36k $exp/diag_ubm_${cnum} $exp/full_ubm_${cnum} 
fi

if [[ $stage -le 3 && 3 -le $endstage ]];then
echo "[step 3] T-matrix training"
subtools/kaldi/sid/train_ivector_extractor.sh --stage $train_stage --nj $[$train_nj/4] --cmd "run.pl" \
    --num-iters 6 --delta $delta --ivector_dim $civ $exp/full_ubm_${cnum}/final.ubm $train \
    $exp/extractor_${cnum}_${civ}
fi
echo "Ivector training done."

if [[ $stage -le 4 && 4 -le $endstage ]];then
prefix=plp_20_5.0
toEXdata="baseTrain_volume_sp test_all_3k_interference_concat_sp"
nj=20


echo "[step 4] Extract i-vector"
for x in $toEXdata;do
num=0
[ -f $exp/$x/ivector.scp ] && num=$(grep ERROR $exp/$x/log/extract.*.log | wc -l)
[[ "$force" == "true" || ! -f $exp/$x/ivector.scp || $num -gt 0 ]] && \
subtools/kaldi/sid/extract_ivectors.sh --cmd "run.pl" --nj $nj --delta $delta \
    $exp/extractor_${cnum}_${civ} data/${prefix}/$x $exp/$x
> $exp/$x/$prefix
echo "data/$prefix/$x extracted done."
done
fi
