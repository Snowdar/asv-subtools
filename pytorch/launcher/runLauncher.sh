#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2020-03-08)

wait_time=0
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

# Use it to run a lancher with a countdown function when there are no extra GPU memory 
# but you really want to go to bed and know when the GPU memory will be free.
[ $wait_time -gt 0 ] && echo "Run this launcher after ${wait_time}s ..."
sleep $wait_time

# Split this two stage to free GPU memory of model by an exit-python way 
# and use these GPU memory to extract x-vectors.
if [[ "$stage" -le 3 && "$endstage" -ge 3 ]];then
    python3 $launcher $launcher_options --stage=$stage --endstage=3 || exit 1 
fi

if [[ "$stage" -le 4 && "$endstage" -ge 4 ]];then
    python3 $launcher $launcher_options --stage=4 || exit 1
fi

wait
exit 0
