
#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2020-03-25)


if [[ $# != 2 ]];then
echo "[exit] Num of parameters is not equal to 2"
echo "usage:$0 <find-dir> <type>
exit 1
fi

dir=$1
type=$2

if [ "$type" == "xvector" ];then
    find $dir -name "xvector_*" | xargs -n 1 rm -f
fi

echo "Clear done."