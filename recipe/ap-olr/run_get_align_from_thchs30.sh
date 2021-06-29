#!/bin/bash

# Copyright 2017 Beijing Shell Shell Tech. Co. Ltd. (Authors: Hui Bu)
#           2017 Jiayu Du
#           2017 Xingyu Na
#           2017 Bengu Wu
#           2017 Hao Zheng
#           2021 Zheng Li
# Apache 2.0



# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# should run this on a machine that has sufficient memory.
. ./path.sh
. ./cmd.sh

# Zheng Li 2021-06
# If you already have a trained ASR model or you want to use a trained open-source ASR model such as THCHS-30, then you run stage 10 to stage 12.
# If you want to train a ASR model based on your ASR datasets, then you run from stage 0 to stage 12.
# Caution: the acoustic features MUST BE EXACTLY SAME between the training set of ASR model and the dataset which wants to get its alignment file.

stage=10
endstage=12


train=data/get_align/ap19_task_1_train # the dataset which wants to get its alignment file
lang=/work/kaldi/egs/thchs30/s5/data/lang # corresponding to the trained ASR model
ali_model_dir=/work/kaldi/egs/thchs30/s5/exp/tri4b  # the trained ASR model
ali_out_dir=/work/kaldi/egs/thchs30/s5/exp/tri4b/ali_thchs30_ap19_task_1_train # the direction for alignment file

if [[ $stage -le 0 && 0 -le $endstage ]];then
# prepare lang
utils/prepare_lang.sh --position-dependent-phones false data/local/dict \
    "<SPOKEN_NOISE>" $lang/tmp $lang || exit 1;
fi

if [[ $stage -le 1 && 1 -le $endstage ]];then
# Now make MFCC plus pitch features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.

mfccdir=exp/feature/mfcc/$train

steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj 20 $train $mfccdir/log $mfccdir || exit 1;
steps/compute_cmvn_stats.sh $train $mfccdir/log $mfccdir || exit 1;
utils/fix_data_dir.sh $train || exit 1;
fi

if [[ $stage -le 2 && 2 -le $endstage ]];then
steps/train_mono.sh --cmd "$train_cmd" --nj 10 \
  $train $lang exp/ASR/mono || exit 1;

# Get alignments from monophone system.
steps/align_si.sh --cmd "$train_cmd" --nj 10 \
  $train $lang exp/ASR/mono exp/ASR/mono_ali || exit 1;
fi

if [[ $stage -le 3 && 3 -le $endstage ]];then
# train tri1 [first triphone pass]
steps/train_deltas.sh --cmd "$train_cmd" \
 2500 20000 $train $lang exp/ASR/mono_ali exp/ASR/tri1 || exit 1;

# align tri1
steps/align_si.sh --cmd "$train_cmd" --nj 10 \
  $train $lang exp/ASR/tri1 exp/ASR/tri1_ali || exit 1;
fi

if [[ $stage -le 4 && 4 -le $endstage ]];then
# train tri2 [delta+delta-deltas]
steps/train_deltas.sh --cmd "$train_cmd" \
 2500 20000 $train $lang exp/ASR/tri1_ali exp/ASR/tri2 || exit 1;

# train and decode tri2b [LDA+MLLT]
steps/align_si.sh --cmd "$train_cmd" --nj 10 \
  $train $lang exp/ASR/tri2 exp/ASR/tri2_ali || exit 1;
fi

if [[ $stage -le 5 && 5 -le $endstage ]];then
# Train tri3a, which is LDA+MLLT,
steps/train_lda_mllt.sh --cmd "$train_cmd" \
 2500 20000 $train $lang exp/ASR/tri2_ali exp/ASR/tri3a || exit 1;

# From now, we start building a more serious system (with SAT), and we'll
# do the alignment with fMLLR.

steps/align_fmllr.sh --cmd "$train_cmd" --nj 10 \
  $train $lang exp/ASR/tri3a exp/ASR/tri3a_ali || exit 1;
fi

if [[ $stage -le 6 && 6 -le $endstage ]];then
steps/train_sat.sh --cmd "$train_cmd" \
  2500 20000 $train $lang exp/ASR/tri3a_ali exp/ASR/tri4a || exit 1;

steps/align_fmllr.sh  --cmd "$train_cmd" --nj 10 \
  $train $lang exp/ASR/tri4a exp/ASR/tri4a_ali
fi

if [[ $stage -le 7 && 7 -le $endstage ]];then
# Building a larger SAT system.

steps/train_sat.sh --cmd "$train_cmd" \
  3500 100000 $train $lang exp/ASR/tri4a_ali exp/ASR/tri5a || exit 1;

steps/align_fmllr.sh --cmd "$train_cmd" --nj 10 \
  $train $lang exp/ASR/tri5a exp/ASR/tri5a_ali || exit 1;
fi

# you better only use GMM based ASR models to get alignment, recommended by Passionlee
if [[ $stage -le 8 && 8 -le $endstage ]];then
# chain
#local/chain/run_tdnn.sh --stage 10 --trainstage 0 exp/ASR $lang `basename $train` exp/ASR/tri5a_ali
fi

if [[ $stage -le 9 && 9 -le $endstage ]];then
# nnet3
#local/nnet3/run_tdnn.sh 
fi

# Zheng Li 2021-06-08
if [[ $stage -le 10 && 10 -le $endstage ]];then
  mfccdir=/data1/lz/mfcc
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 $train exp/make_mfcc/$train $mfccdir
  steps/compute_cmvn_stats.sh $train exp/make_mfcc/$train $mfccdir

  steps/align_fmllr.sh --cmd "$train_cmd" --nj 10 \
    $train $lang $ali_model_dir $ali_model_dir/graph|| exit 1;
  utils/mkgraph.sh ${lang}_test_tgsmall \
    $ali_model_dir $ali_model_dir/graph_tgsmall
  
  steps/decode_fmllr.sh --nj 5 --cmd "$train_cmd" \
    $ali_model_dir/graph_phone $train \
    $ali_out_dir
fi

# Zheng Li 2021-06-08
if [[ $stage -le 11 && 11 -le $endstage ]];then
  num_ali_jobs=$(cat ${ali_out_dir}.si/num_jobs) || exit 1;
  #cmd_string="copy-int-vector ark:-"
  for id in $(seq $num_ali_jobs); do gunzip -c ${ali_out_dir}.si/lat.$id.gz; done | \
  lattice-1best ark:- ark:-| nbest-to-linear ark:- ark,scp:${ali_out_dir}/ali.ark,${ali_out_dir}/ali.scp

fi

# Zheng Li 2021-06-08
if [[ $stage -le 12 && 12 -le $endstage ]];then
  echo "$0: transforing data alignments"
  ali-to-pdf $ali_model_dir/final.mdl scp:${ali_out_dir}/ali.scp ark,scp:${ali_out_dir}/ali.pdf.ark,${ali_out_dir}/ali.pdf.scp
  mv ${ali_out_dir}/ali.pdf.scp ${ali_out_dir}/ali.scp
  rm -rf ${ali_out_dir}/ali.ark  
  phonetic_num_targets=$(tree-info $ali_model_dir/tree | grep num-pdfs | awk '{print $2}') || exit 1
  echo "$phonetic_num_targets" > ${ali_out_dir}/phones
fi

exit 0;
