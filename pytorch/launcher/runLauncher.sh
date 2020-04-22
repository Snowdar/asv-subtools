#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2020-03-08)

stage=3
endstage=4
horovod_params="--start-timeout 30"

. subtools/parse_options.sh
. subtools/path.sh

if [[ $# < 1 ]];then
echo "[exit] Num of parameters is zero, expected a launcher."
echo "$0 <launcher>"
echo "e.g. $0 subtools/pytorch/launcher/runSnowdarXvector-multi-GPU.py --gpu-id=1,2"
exit 1
fi

launcher=$1
shift

[ ! -f $launcher ] && echo "Expected $launcher (*.py) to exist." && exit 1

stage=$(echo "$@" | awk -v stage=$stage '{for(i=1;i<=NF;i++){
        split($i,a,"=");if(a[1]=="--stage"){change=1;print a[2];}}}END{if(!change){print stage}}')
endstage=$(echo "$@" | awk -v endstage=$endstage '{for(i=1;i<=NF;i++){
        split($i,a,"=");if(a[1]=="--endstage"){change=1;print a[2];}}}END{if(!change){print endstage}}')

# Should note the " and space char when giving a parameter from shell to python.
launcher_options=""
num_gpu=1
while true;do
    [ $# -eq 0 ] && break

    if [[ $1 == "--gpu-id="* ]];then
        gpu_id_option=$(echo "$1" | sed 's/ /,/g')
        launcher_options="$launcher_options $gpu_id_option"
        num_gpu=$(echo $gpu_id_option | sed 's/=/ /g' | awk '{print $2}' | sed 's/[,-]/\n/g' | sed '/^$/d' | wc -l)
    elif [[ $1 != "--stage="* && $1 != "--endstage="* ]];then
        launcher_options="$launcher_options $1"
    fi
    shift
done


# Add multi-gpu case.
if [ $num_gpu -gt 1 ];then
    sh subtools/pytorch/launcher/multi_gpu/check_horovod.sh || exit 1
    train_cmd="horovodrun -np $num_gpu --log-level INFO $horovod_params python3"
else
    train_cmd="python3"
fi


# Split this two stage to free GPU memory of model by an exit-python way 
# and use these GPU memory to extract x-vectors.
if [[ "$stage" -le 3 && "$endstage" -ge 3 ]];then
    $train_cmd $launcher $launcher_options --stage=$stage --endstage=3 || exit 1 
fi

if [[ "$stage" -le 4 && "$endstage" -ge 4 ]];then
    python3 $launcher $launcher_options --stage=4 || exit 1
fi

exit 0
