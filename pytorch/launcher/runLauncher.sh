#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2020-03-08)

stage=3
endstage=4
multi_gpu_solution="ddp"
omp_num_threads=1
port=0

. subtools/parse_options.sh
. subtools/path.sh

if [[ $# < 1 ]];then
echo "[exit] Num of parameters is zero, expected a launcher."
echo "usage: $0 <launcher> [launcher-options]"
echo "e.g. $0 subtools/pytorch/launcher/runSnowdarXvector-voxceleb1.py --gpu-id=0,1,2"
exit 1
fi

launcher=$1
shift

[ ! -f $launcher ] && echo "Expected $launcher (*.py) to exist." && exit 1

# Should note the " and space char when giving a parameter from shell to python.
launcher_options=""
num_gpu=1
while true;do
    [ $# -eq 0 ] && break

    if [[ $1 == "--gpu-id="* ]];then
        gpu_id_option=$(echo "$1" | sed 's/ /,/g')
        launcher_options="$launcher_options $gpu_id_option"
        num_gpu=$(echo $gpu_id_option | sed 's/=/ /g' | awk '{print $2}' | sed 's/[,-]/\n/g' | sed '/^$/d' | wc -l)
    elif [[ $1 == "--multi-gpu-solution="* ]];then
        multi_gpu_solution=$(echo $1 | sed 's/=/ /g' | awk '{print $2}')
        launcher_options="$launcher_options $1"
    elif [[ $1 == "--port="* ]];then
        port=$(echo $1 | sed 's/=/ /g' | awk '{print $2}')
        launcher_options="$launcher_options $1"
    elif [[ $1 == "--stage="* ]];then
        stage=$(echo $1 | sed 's/=/ /g' | awk '{print $2}')
    elif [[ $1 == "--endstage="* ]];then
        endstage=$(echo $1 | sed 's/=/ /g' | awk '{print $2}')
    else
        launcher_options="$launcher_options $1"
    fi
    shift
done


# Add multi-gpu case.
if [ $num_gpu -gt 1 ];then
    if [ "$multi_gpu_solution" == "horovod" ];then
        sh subtools/pytorch/launcher/multi_gpu/check_horovod.sh || exit 1
        # Ser cache for synchronize batchnorm to avoid WARNING.
        export HOROVOD_CACHE_CAPACITY=0
        train_cmd="horovodrun -np $num_gpu python3"
    elif [ "$multi_gpu_solution" == "ddp" ];then
        export OMP_NUM_THREADS=$omp_num_threads
        if [ "$port" == "0" ];then
            port=$(python3 subtools/pytorch/launcher/multi_gpu/get_free_port.py)
            launcher_options="$launcher_options --port $port"
        fi
        train_cmd="python3 -m torch.distributed.launch --nproc_per_node=$num_gpu"
    else
        echo "[exit] Do not support $multi_gpu_solution solution for multi-GPU training." && exit 1
    fi
else
    train_cmd="python3"
fi

# Split this two stage to free GPU memory of model by an exit-python way 
# and use these GPU memory to extract x-vectors.
if [[ "$stage" -le 3 && "$endstage" -ge 3 ]];then
    $train_cmd $launcher $launcher_options --stage=$stage --endstage=3 || exit 1 
fi

if [[ "$stage" -le 4 && "$endstage" -ge 4 ]];then
    python3 $launcher --stage=4 || exit 1
fi

exit 0
