#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-09-05)

nj=30

stage=0
endstage=0 # If you don't want to use pca, just stop in stage 2. 

. subtools/parse_options.sh
. subtools/path.sh

data=test_1s
srcdir1=data/fbank_40_5.0/$data
srcdir2=data/plp_20_5.0/$data
selection="" # To delete duplications(energy and pitch) and it works to srcdir2, 0-based. If NULL, use all.
# Selection example:
# let selection="1-19"
# fbank_(base_40 + energy_1 + pitch_3) + plp_(energy_1 + base_19 + pitch_3) -> select(1-19) 
# -> final dim : fbank_base_40 + energy_1 + pitch_3 + plp_base_19 = 63

prefixes=fbank_40_paste_plp_20_pitch

datadir=data/$prefixes/$data
mat=data/fbank_40_paste_plp_20_pitch/baseTrain/pca.mat
vadconfig=conf/vad-5.0.conf

! cmp -s $srcdir1/wav.scp $srcdir2/wav.scp && echo "[ exit ] $srcdir1 doesn't match $srcdir2" && exit 1

[ "$selection" == "" ] && selection=0-$(echo "$(feat-to-dim scp:$srcdir2/feats.scp - 2>/dev/null) - 1" | bc)


if [[ $stage -le 0 && 0 -le $endstage ]];then
echo "[stage 0] Prepare datadir"
mkdir -p $datadir
cp $srcdir1/{wav.scp,utt2spk,spk2utt} $datadir

echo $(feat-to-dim scp:$srcdir1/feats.scp -) $(echo $selection | sed 's/-/ /g' | awk '{print $2+1}') > $datadir/num_dims
fi

name=`basename $datadir`
featsname=`echo "${datadir}" | sed 's/\// /g' | awk '{for(i=1;i<=NF-1;i++){printf $i"_";}printf $NF}'`
featsdir=exp/features/paste/$featsname
mkdir -p $featsdir/log

if [[ $stage -le 1 && 1 -le $endstage ]];then
echo "[stage 1] Paste features"

scp1=$srcdir1/feats.scp
scp2=$srcdir2/feats.scp

compress=true

split_scps1=""
split_scps2=""
for n in $(seq $nj); do
    split_scps1="$split_scps1 $featsdir/log/feats_1_${name}.$n.scp"
    split_scps2="$split_scps2 $featsdir/log/feats_2_${name}.$n.scp"
done

subtools/kaldi/utils/split_scp.pl $scp1 $split_scps1 || exit 1;
subtools/kaldi/utils/split_scp.pl $scp2 $split_scps2 || exit 1;

run.pl JOB=1:$nj $featsdir/log/${prefixes}_$name.JOB.log \
    select-feats $selection scp:$featsdir/log/feats_2_${name}.JOB.scp ark:- \| \
    paste-feats scp:$featsdir/log/feats_1_${name}.JOB.scp ark:- \
      ark,scp:$featsdir/${prefixes}_$name.JOB.ark,$featsdir/${prefixes}_$name.JOB.scp

for n in $(seq $nj); do
  cat $featsdir/${prefixes}_$name.$n.scp || exit 1;
done > ${datadir}/feats.scp || exit 1
echo "Succeeded pasting features for ${name}"
fi

if [[ $stage -le 2 && 2 -le $endstage ]];then
echo "[stage 2] Compute VAD"
subtools/computeVad.sh $datadir $vadconfig
fi

if [[ $stage -le 3 && 3 -le $endstage ]];then
echo "[stage 3] Train PCA mat"
num=`wc -l $datadir/feats.scp | awk '{print $1}'`
n=5000
[[ $n > $num ]] && n=$num
subtools/kaldi/utils/shuffle_list.pl $datadir/feats.scp | head -n $n | sort | \
  est-pca scp:- $datadir/pca.mat
fi

if [[ $stage -le 4 && 4 -le $endstage ]];then
echo "[stage 4] Transform feats with PCA mat"
[ -f $datadir/feats.raw.scp ] && cp -f $datadir/feats.raw.scp $datadir/feats.scp
[[ $mat == "" ]] && mat=$datadir/pca.mat
run.pl JOB=1:$nj $featsdir/log/pca_${prefixes}_$name.JOB.log \
    transform-feats $mat scp:$featsdir/${prefixes}_$name.JOB.scp \
    ark,scp:$featsdir/pca_${prefixes}_$name.JOB.ark,$featsdir/pca_${prefixes}_$name.JOB.scp

[ ! -f $datadir/feats.raw.scp ] && mv -f $datadir/feats.scp $datadir/feats.raw.scp

for n in $(seq $nj); do
  cat $featsdir/pca_${prefixes}_$name.$n.scp || exit 1;
done > ${datadir}/feats.scp || exit 1
echo "Transforming done."
fi
echo "All done."
