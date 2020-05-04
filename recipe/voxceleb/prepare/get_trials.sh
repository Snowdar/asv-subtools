#!/bin/bash

# Copyright xmuspeech (Author: Snowdar 2020-05-04)

dir=data/voxceleb1
tasks="voxceleb1-O voxceleb1-O-clean voxceleb1-E voxceleb1-E-clean voxceleb1-H voxceleb1-H-clean"

. subtools/parse_options.sh
. subtools/path.sh

[ "$dir" == "" ] && echo "[exit] Expected a dir to save trials, but got nothing." && exit 1

mkdir -p $dir

for task in $tasks;do
    name=""
    [ "$task" == "voxceleb1-O" ] && name="veri_test.txt"
    [ "$task" == "voxceleb1-O-clean" ] && name="veri_test2.txt"
    [ "$task" == "voxceleb1-H" ] && name="list_test_hard.txt"
    [ "$task" == "voxceleb1-H-clean" ] && name="list_test_hard2.txt"
    [ "$task" == "voxceleb1-E" ] && name="list_test_all.txt"
    [ "$task" == "voxceleb1-E-clean" ] && name="list_test_all2.txt"

    [ "$name" == "" ] && echo "The $task task is invalid here. Please select from voxceleb1-O/E/H[-clean]." && exit 1

    if [ ! -f $dir/$name ];then
        echo "The $dir/$name is not exist, so download it now...(If failed, download the list from" \
             "http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta by yourself.)"
        trap "rm -f $dir/$name && exit 1" INT
        wget -P $dir http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/$name || (rm -f $dir/$name && exit 1)
        trap INT
    fi

    sed 's/\//-/g;s/\.wav//g' $dir/$name | awk '{if($1=="1"){print $2,$3,"target"}else{print $2,$3,"nontarget"}}' > $dir/${task}.trials
    echo "Generate $dir/${task}.trials done."
done