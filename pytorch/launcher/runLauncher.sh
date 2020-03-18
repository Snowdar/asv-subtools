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

stage=$(echo "$@" | awk -v stage=$stage '{for(i=1;i<=NF;i++){
        split($i,a,"=");if(a[1]=="--stage"){change=1;print a[2];}}}END{if(!change){print stage}}')
endstage=$(echo "$@" | awk -v endstage=$endstage '{for(i=1;i<=NF;i++){
        split($i,a,"=");if(a[1]=="--endstage"){change=1;print a[2];}}}END{if(!change){print endstage}}')

launcher_options=$(echo "$@" | awk '{for(i=1;i<=NF;i++){
                                     split($i,a,"=");
                                     if(a[1]=="--stage" || a[1]=="--endstage")
                                     {$i="";}
                                     }}END{print $0;}')

if [[ "$stage" -le 3 && "$endstage" -ge 3 ]];then
    python3 $launcher $launcher_options --stage=$stage --endstage=3 || exit 1 
fi

if [[ "$stage" -le 4 && "$endstage" -ge 4 ]];then
    python3 $launcher $launcher_options --stage=4 || exit 1
fi
