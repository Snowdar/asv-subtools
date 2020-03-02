#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018/02/20)

task=LA # LA or PA
. subtools/parse_options.sh

if [[ $# != 2 ]];then
echo "[exit] Num of parameters is not equal to 2"
echo "usage:$0 <trials> <score>"
exit 1
fi

trials=$1
score=$2

if [[ "$task" != "LA" || "$task" != "PA" ]];then
echo "[exit] The $task is not LA or PA"
exit 1
fi

awk '{print $1,$3}' $score > $score.cm.tmp
out=$(subtools/asvspoof/computeMin-t-DCF.py subtools/asvspoof/2019/$task/ASVspoof2019_${task}_dev_asv_scores_v1.txt $score.cm.tmp)
rm -f $score.cm.tmp
echo "$out" | awk '{if($1=="Final"){split($2,a,"=");print a[2]}}'
