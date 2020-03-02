#!/bin/bash  

# Copyright xmuspeech (Author: Snowdar 2019-10-17)

# Source this script and then use this function to do lines task like dealing with scp file

function do_lines_task_parallel(){

    local nj=1

    . subtools/parse_options.sh

    if [[ $# != 3 ]];then
        echo "[exit] Num of parameters is not equal to 3"
        echo "usage:$0 <function> <input-file> <output-file>"
        echo "e.g.: $0 copy_vector data/train/vad.scp data/train/utt2num_frames.nosil"
        echo "note: copy_vector is a function defined in the bash script, such as"
        echo '      function copy_vector(){
                        scp=$1
                        out=$2
                        copy-vector scp:$scp ark,t:$out
                        return 0
                    }'
        exit 1
    fi

    this_function=$1
    input_file=$2
    output_file=$3

    [ ! -f $input ] && echo "No such file $input" && exit 1


    # Split file/scp to files/scps
    name=$(basename $input_file)
    temp_dir=$(dirname $input_file)/_temp
    mkdir -p $temp_dir

    output_name=$(basename $output_file)

    split_scps=
    for i in $(seq $nj);do
    split_scps="$split_scps $temp_dir/$name.$i.input"
    done
    subtools/kaldi/utils/split_scp.pl $input_file $split_scps || exit 1


    # Execute function
    trap "subtools/linux/kill_pid_tree.sh --show true $$ && echo -e '\nAll killed'" INT
    pids=""
    for i in $(seq $nj);do
        $this_function $temp_dir/$name.$i.input $temp_dir/$output_name.$i.output > $temp_dir/$i.log 2>&1 || exit 1 & 
        pids="$pids $!"
    done
    trap "subtools/linux/kill_pid_tree.sh --show true $pids && echo -e '\nAll killed'" INT

    wait

    # Gather output files/scps to file/scp
    > $output_file
    for i in $(seq $nj);do
    cat $temp_dir/$output_name.$i.output
    done >> $output_file

    rm -rf $temp_dir

    return 0
}

# do_lines_task_parallel # A test to get usage