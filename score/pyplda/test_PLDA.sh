#!/bin/bash

# Copyright xmuspeech (Author:JFZhou 2020-05-31)

set -e

. subtools/path.sh

if [[ $# != 7 ]];then

	echo "[exit] Num of parameters is not equal to 7"
	echo "usage:$0 <spk2utt> <train-vec> <test-vec> <enroll-vec> <plda> <trials> <score>"
	echo "egs. $0 data/voxceleb1_train/spk2utt ark:egs/vox1/train/train.egs.ark ark:egs/vox1/enroll/train.egs.ark ark:egs/vox1/test/train.egs.ark
			   egs/vox1/train/plda.python data/voxceleb1_test/trials egs/vox1/train/plda.score"
	
exit 1
fi

spk2utt=$1
train_vec=$2
enroll_vec=$3
test_vec=$4
plda=$5
trials=$6
score=$7

# plda
python3 subtools/score/pyplda/ivector-compute-plda.py $spk2utt $train_vec $plda

ivector-plda-scoring --normalize-length=true \
	"ivector-copy-plda --smoothing=0.0 $plda - |" \
	$enroll_vec \
	$test_vec \
	"cat '$trials' | cut -d\  --fields=1,2 |" $score || exit 1;

# # output the metrics
awk '{print $3}' $score | paste - $trials | awk '{print $1, $4}' | compute-eer - \
mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $score $trials 2> /dev/null`
mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $score $trials 2> /dev/null`
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"
