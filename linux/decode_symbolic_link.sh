#!/bin/bash  

# Copyright xmuspeech (Author: Snowdar 2020-04-09)

# Find a real file from a symbolic link.

cmd=false # If true, decode symbolic link for cmd.
details=false # If true, print any symbolic link.

. subtools/parse_options.sh

if [[ $# -lt 1 ]];then
echo "[exit] Num of parameters is not more than 1"
echo "usage:$0 <file|cmd>"
exit 1
fi

object=$1

if [ "$cmd" == "true" ];then
    cmd_path=$(command -v $object)
    [ "$cmd_path" == "" ] && echo "[exit] No $object in ($PATH)" && exit 1
    object=$cmd_path
fi

object=$(dirname $object)/$(basename $object)

origin=$object

while true;do
    if [ -L $object ];then
        [ "$details" == "true" ] && echo $object
        next=$(file $object | awk '{print substr($5,2,length($5)-2)}')
        if [ $(dirname $next) == "." ];then
            object=$(dirname $object)/$next
        else
            object=$next
        fi
    elif [[ -f "$object" || -d "$object" ]];then
        echo $object
        exit 0
    else
        if [ "$origin" == "$object" ];then
            echo "[exit] Expected $object is exist."
        else
            echo "[exit] Got $object, but it is not exist. So $origin is a invalid symbolic link."
        fi
        exit 1
    fi
done