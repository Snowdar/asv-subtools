#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-09-05)

nj=30

stage=0
endstage=2
volume=false
basefeat=false
vad=true

topdir=data
datasets="test_all"
prefix=mfcc_20_5.0

feat_type=mfcc
feat_conf=conf/sre-mfcc-20.conf
vad_conf=conf/vad-5.0.conf

pitch=true

. subtools/parse_options.sh
. subtools/path.sh

suffix=sp
[ $volume == "true" ] && suffix=volume_$suffix

for data in $datasets ;do
srcdir=$topdir/$prefix/$data

if [[ $stage -le 0 && 0 -le $endstage ]];then
echo "[stage 0] Speed 3way"
subtools/kaldi/utils/data/perturb_data_dir_speed_3way.sh $srcdir ${srcdir}_$suffix
[ $volume == "true" ] && subtools/kaldi/utils/data/perturb_data_dir_volume.sh ${srcdir}_$suffix
subtools/correctSpeakerAfterSp3way.sh ${srcdir}_$suffix
fi

if [[ $stage -le 1 && 1 -le $endstage ]];then
echo "[stage 1] Make features"
[ $basefeat == "true" ] && subtools/makeFeatures.sh --nj $nj --pitch $pitch ${srcdir} $feat_type $feat_conf \
&& [ $vad == "true" ] && subtools/computeVad.sh --nj $nj ${srcdir} $vad_conf
subtools/makeFeatures.sh --nj $nj --pitch $pitch ${srcdir}_$suffix $feat_type $feat_conf
[ $vad == "true" ] && subtools/computeVad.sh --nj $nj ${srcdir}_$suffix $vad_conf
fi

if [[ $stage -le 2 && 2 -le $endstage ]];then
echo "[stage 2] Concatenate features"

rm -rf ${srcdir}_concat_$suffix
mkdir -p ${srcdir}_concat_$suffix
cp ${srcdir}/{wav.scp,utt2spk,spk2utt} ${srcdir}_concat_$suffix

name=`basename $srcdir`
featsname=`echo "${srcdir}_concat_$suffix" | sed 's/\// /g' | awk '{for(i=1;i<=NF-1;i++){printf $i"_";}printf $NF}'`
featsdir=exp/features/$feat_type/$featsname
mkdir -p $featsdir/log

scp=$srcdir/wav.scp
spfeats=${srcdir}_$suffix/feats.scp

compress=true

split_scps=""
for n in $(seq $nj); do
    split_scps="$split_scps $featsdir/log/wav_${name}.$n.scp"
done

subtools/kaldi/utils/split_scp.pl $scp $split_scps || exit 1;

run.pl JOB=1:$nj $featsdir/log/concat_${feat_type}_$name.JOB.log \
	awk 'NR==FNR{a[$1]=$2}NR>FNR{
	printf $1" ";system("concat-feats --binary=false "a[$1"-sp0.9"]" "a[$1]" "a[$1"-sp1.1"]" - 2>/dev/null");
	}' $spfeats $featsdir/log/wav_${name}.JOB.scp \| \
	copy-feats --compress=$compress ark:- \
	  ark,scp:$featsdir/concat_${feat_type}_$name.JOB.ark,$featsdir/concat_${feat_type}_$name.JOB.scp \
	 || exit 1

for n in $(seq $nj); do
  cat $featsdir/concat_${feat_type}_$name.$n.scp || exit 1;
done > ${srcdir}_concat_$suffix/feats.scp || exit 1
[ $vad == "true" ] && subtools/computeVad.sh --nj $nj ${srcdir}_concat_$suffix $vad_conf
echo "Succeeded concatenating speed-perturb-features from ${name}"
fi

done
