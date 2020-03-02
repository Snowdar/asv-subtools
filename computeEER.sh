#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-7-27)

write_file=""

. subtools/path.sh
. subtools/parse_options.sh

if [[ $# != 4 ]];then
echo "[exit] Num of parameters is not equal to 4"
echo "usage:$0 [--write-file \"\" | filepath] <score-file> <score-field 1-based> <trials> <target/nontarget-field 1-based>"
echo "[note] You should specify field of score in score-file and field of target/nontarget in trials"
exit 1
fi

score=$1
first=$2
trials=$3
second=$4

workout=`awk -v first=$first '{print $first}' $score | paste - <(awk -v second=$second '{print $second}' $trials ) | \
awk '{if(NF==2){print $0}}' | compute-eer - `

if [ "$write_file" != "" ];then
echo "$workout" > $write_file
fi

echo "EER% $workout"
