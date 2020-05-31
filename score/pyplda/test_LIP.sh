#!/bin/bash

# Copyright xmuspeech (Author:JFZhou 2020-05-31)

set -e

. subtools/path.sh

if [[ $# != 7 ]];then

	echo "[exit] Num of parameters is not equal to 7"
	echo "usage:$0 <plda-out-domain> <plda-in-domain> <test-vec> <enroll-vec> <plda-adapt> <trials> <score>"
	echo "egs. $0 egs/vox1/train/plda.python.ori egs/vox1/train-indomain/plda.python.ori ark:egs/vox1/enroll/train.egs.ark ark:egs/vox1/test/train.egs.ark
			   egs/vox1/train/plda.python.adapt data/voxceleb1_test/trials egs/vox1/train/plda.lip.score"
	
exit 1
fi

plda_out_domain=$1
plda_in_domain=$2
enroll_vec=$3
test_vec=$4
plda_adapt=$5
trials=$6
score=$7

# plda
python3 subtools/score/pyplda/ivector-adapt-plda-lip.py $plda_out_domain $plda_in_domain $plda_adapt

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
