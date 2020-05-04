#!/bin/bash

# Copyright xmuspeech (Author: Snowdar 2020-05-03)


prefix="mfcc_23_pitch"
tasks="voxceleb1-O voxceleb1-E voxceleb1-H voxceleb1-O-clean voxceleb1-E-clean voxceleb1-H-clean"
vectordir=

force=false

. subtools/parse_options.sh
. subtools/path.sh

[ "$vectordir" == "" ] && echo "[exit] Expected a vectordir" && exit 1
[ ! -d "data/$prefix/voxceleb1" ] && echo "[exit] Expected data/$prefix/voxceleb1/ to exist." && exit 1
[ ! -f "$vectordir/voxceleb1/xvector.scp" ] && echo "[exit] Expected $vectordir/voxceleb1/xvector.scp to exit." && exit 1

mkdir -p data/$prefix/voxceleb1/temp
temp=data/$prefix/voxceleb1/temp

for task in $tasks;do
    trials=data/$prefix/voxceleb1/$task.trials
    [ ! -f "$trials" ] && echo "Expected $trials to exist for $task task." && exit 1

    # Change a name with _ replaced by _. It is important for subtools/recipe/voxceleb/gather_results_from_epochs.sh.
    task=$(echo $task | sed 's/-/_/g')

    [ "$force" == "true" ] && rm -rf $temp/$task.enroll.list $temp/$task.test.list data/$prefix/${task}_enroll data/$prefix/${task}_test

    [ ! -f $temp/$task.enroll.list ] && awk '{print $1}' $trials | sort -u > $temp/$task.enroll.list
    [ ! -f $temp/$task.test.list ] && awk '{print $2}' $trials | sort -u > $temp/$task.test.list

    # Datadir
    [[ ! -d data/$prefix/${task}_enroll ]] && subtools/filterDataDir.sh data/$prefix/voxceleb1 \
              $temp/$task.enroll.list data/$prefix/${task}_enroll

    [[ ! -d data/$prefix/${task}_test ]] && subtools/filterDataDir.sh data/$prefix/voxceleb1 \
              $temp/$task.test.list data/$prefix/${task}_test

    [ ! -f data/$prefix/${task}_test/trials ] && cp $trials data/$prefix/${task}_test/trials

    [ "$force" == "true" ] && rm -rf $vectordir/${task}_enroll $vectordir/${task}_test

    # Vectordir
    [[ ! -d $vectordir/${task}_enroll ]]  && subtools/filterVectorDir.sh $vectordir/voxceleb1/xvector.scp \
              $temp/$task.enroll.list $vectordir/${task}_enroll

    [[ ! -d $vectordir/${task}_test ]]  && subtools/filterVectorDir.sh $vectordir/voxceleb1/xvector.scp \
              $temp/$task.test.list $vectordir/${task}_test
done

rm -rf $temp
