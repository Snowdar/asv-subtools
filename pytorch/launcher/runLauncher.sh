#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2020-03-08)

stage=3
endstage=4

. subtools/parse_options.sh
. subtools/path.sh

if [[ $# < 1 ]];then
echo "[exit] Num of parameters is zero, expected a launcher."
echo "$0 <launcher>"
exit 1
fi

launcher=$1
shift

if [[ "$stage" -le 3 && "$endstage" -ge 3 ]];then
    python3 $launcher $@ --stage=$stage --endstage=3 || exit 1 
fi

if [[ "$stage" -le 4 && "$endstage" -ge 4 ]];then
    python3 $launcher --stage=4 || exit 1
fi
