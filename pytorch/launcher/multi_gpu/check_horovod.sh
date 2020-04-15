#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2020-04-10)


cmd_path=$(command -v horovodrun)

if [ "$cmd_path" == "" ];then
    python3_path=$(command -v python3)
    [ "$python3_path" == "" ] && echo "[exit] No python3 in ($PATH)" && exit 1

    python3_bin_dir=$(dirname $(subtools/linux/decode_symbolic_link.sh --cmd true python3))
    horovodrun_path=$python3_bin_dir/horovodrun
    [ ! -f "$horovodrun_path" ] && echo -e "[exit] No horovodrun in ($PATH) and ($python3_bin_dir).\
    \n[Note] Horovod should be installed by yourself firstly (Using 'pip3 install horovod' with NCCL and see subtools/Readme.txt)." && exit 1

    echo -e "[Warning] The horovod is found in $python3_bin_dir, but it is not in your PATH environment."\
    "\n[Suggestion] Add 'export PTAH=\$PATH:$python3_bin_dir' in /etc/profile or /root/.bashrc by yourself."

    export PTAH=$PATH:$python3_bin_dir
else
    exit 0
fi