#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2020-04-17)

suffixes="reverb noise music babble"
exclude_self=false

force=false

. subtools/parse_options.sh

if [[ $# != 2 ]];then
echo "[exit] Num of parameters is not equal to 2"
echo "usage:$0 <original-scp> <output-scp>"
exit 1
fi

original=$1
output=$2

[ ! -f $original ] && echo "Expected scp file $original to exist" && exit 1

[ "$force" == "true" ] && rm -f $output

[ -f "$output" ] && echo "$output is exist, please delete it by yourself or use '--force true' option." && exit 1

if [ "$exclude_self" == "false" ];then
    cat $original > $output
else
    > $output
fi

for suffix in $suffixes;do
    awk -v suffix=$suffix '{print $1"-"suffix, $2}' $original >> $output
done

echo "Generate done."