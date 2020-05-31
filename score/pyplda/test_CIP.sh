#!/bin/bash

# Copyright xmuspeech (Author:JFZhou 2020-05-31)

set -e

. subtools/path.sh

if [[ $# != 8 ]];then

	echo "[exit] Num of parameters is not equal to 8"
	echo "usage:$0 <plda-out-domain> <train-vec-adapt> <plda-in-domain> <test-vec> <enroll-vec> <plda-adapt> <trials> <score>"
	echo "egs. $0 egs/vox1/train/plda.python.ori ark:egs/vox1/train-in-domain/train.egs.ark egs/vox1/train-indomain/plda.python.ori ark:egs/vox1/enroll/train.egs.ark ark:egs/vox1/test/train.egs.ark
			   egs/vox1/train/plda.python.adapt data/voxceleb1_test/trials egs/vox1/train/plda.cip.score"
	
exit 1
fi

plda_out_domain=$1
train_vec_adapt=$2
plda_in_domain=$3
enroll_vec=$4
test_vec=$5
plda_adapt=$6
trials=$7
score=$8

# plda
python3 subtools/score/pyplda/ivector-adapt-plda-cip.py $plda_out_domain $train_vec_adapt $plda_in_domain $plda_adapt

ivector-plda-scoring --normalize-length=true \
	"ivector-copy-plda --smoothing=0.0 $plda_adapt - |" \
	$enroll_vec \
	$test_vec \
	"cat '$trials' | cut -d\  --fields=1,2 |" $score || exit 1;

# # output the metrics
awk '{print $3}' $score | paste - $trials | awk '{print $1, $4}' | compute-eer - 
mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $score $trials 2> /dev/null`
mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $score $trials 2> /dev/null`
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"
