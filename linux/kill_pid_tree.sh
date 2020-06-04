#!/bin/bash  

# Copyright xmuspeech (Author: Snowdar 2019-07-13)

# Reference: https://www.linuxidc.com/Linux/2012-05/60315.htm

# Kill a top-pid-start tree (cut a subtree of process) so that we can also kill all 
# bg process when use ctrl + c. And it is usually used with 'trap' command

depth= # If NULL, no limit. 0 means just killing the top pid
show=false # If true, show pids which are killed
signal=9

. subtools/parse_options.sh

if [[ $# -lt 1 ]];then
echo "[exit] Num of parameters is not more than 1"
echo "usage:$0 <pid-1> <pid-2> ..."
exit 1
fi

function find_and_kill(){
    local current_pid=$1  
    local current_depth=$2

    if [[ "$depth" == "" || "$current_depth" -lt "$depth" ]];then
        # Find children's pids
        childs=$(ps -ef | awk -v current_pid=$current_pid 'BEGIN{ ORS=" "; } $3==current_pid{ print $2; }')

        if [ ${#childs[@]} -ne 0 ]; then  
            for child in ${childs[*]};do  
                find_and_kill $child  $[$current_depth+1]
            done  
        fi
    fi

    # Kill the current pid 
    exist=""
    exist=$(ps -ax | awk '{print $1}' | grep -e "^${current_pid}")
    [ "$exist" != "" ] && kill -$signal $current_pid  &&  [ "$show" == "true" ] && echo "$current_pid has been killed"
}

until [ $# -eq 0 ]
do
[ "$show" == "true" ] && echo "kill $1..."
find_and_kill $1 0
shift
done



