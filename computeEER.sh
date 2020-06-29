#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-7-27 2020-06-30)

write_file=""
first=3 # <target/nontarget-field 1-based>
second=3 # <score-field 1-based>

. subtools/path.sh
. subtools/parse_options.sh

if [[ $# != 2 ]];then
echo "[exit] Num of parameters is not equal to 4"
echo "usage:$0 [--write-file \"\" | filepath] <trials> <score-file>"
exit 1
fi

trials=$1
score=$2

workout=`awk -v second=$second '{print $second}' $score | paste - <(awk -v first=$first '{print $first}' $trials ) | \
awk '{if(NF==2){print $0}}' | compute-eer - `

if [ "$write_file" != "" ];then
echo "$workout" > $write_file
fi

echo "EER% $workout"
