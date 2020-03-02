#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-9-4)

weight1=0.5
weight2=0.5

. subtools/parse_options.sh

if [ $# != 3 ];then
echo "usage: $0 [--weight1 0.5] [--weight2 0.5] <score-file1> <score-file2> <output-score>"
exit 1
fi

score1=$1
score2=$2
output=$3

awk -v weight1=$weight1 -v weight2=$weight2 'NR==FNR{a[$1$2]=$3}NR>FNR{print $1,$2,$3*weight1+a[$1$2]*weight2}' $score1 $score2 > $output

echo "Weight score done"

