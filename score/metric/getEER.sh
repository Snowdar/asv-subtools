#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2019/10/30)

if [[ $# != 2 ]];then
echo "[exit] Num of parameters is not equal to 2"
echo "usage:$0 <trials> <score>"
exit 1
fi

trials=$1
score=$2


out=$(subtools/computeEER.sh $score 3 $trials 3 2>/dev/null)
echo "$out" | awk '{print $2}'
