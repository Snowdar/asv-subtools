#!/bin/bash  

# Copyright xmuspeech (Author: Snowdar 2019-10-17)

# Source this script and then use this function to do lines task like dealing with scp file

function parallel_for(){
    # Another name to call this function.
    do_lines_task_parallel $@
    return 0
}

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

    local this_function=$1
    local input_file=$2
    local output_file=$3

    [ ! -f $input ] && echo "No such file $input" && exit 1


    # Split file/scp to files/scps
    _name=$(basename $input_file)
    _temp_dir=$(dirname $input_file)/_temp
    mkdir -p $_temp_dir

    _output_name=$(basename $output_file)

    split_scps=
    for i in $(seq $nj);do
    split_scps="$split_scps $_temp_dir/$_name.$i.input"
    done
    echo "$0: Split $input_file to $nj files."
    subtools/kaldi/utils/split_scp.pl $input_file $split_scps || exit 1

    # Execute function
    trap "subtools/linux/kill_pid_tree.sh --show true $$ && echo -e '\nAll killed'" INT
    pids=""
    for i in $(seq $nj);do
        # $i is input to this_function by $3
        $this_function $_temp_dir/$_name.$i.input $_temp_dir/$_output_name.$i.output $i > $_temp_dir/$i.log 2>&1 && echo \#\# snowdar_ok >> $_temp_dir/$i.log & 
        pids="$pids $!"
    done
    trap "subtools/linux/kill_pid_tree.sh --show true $pids && echo -e '\nAll killed'" INT

    wait

    failed=0
    for i in $(seq $nj);do
        num=$(grep snowdar_ok $_temp_dir/$i.log | wc -l )
        [ "$num" -lt 1 ] && echo "Failed in $_temp_dir/$i.log" && failed=1
    done

    [ "$failed" -ne 0 ] && exit 1

    # Gather output files/scps to file/scp
    > $output_file
    for i in $(seq $nj);do
    cat $_temp_dir/$_output_name.$i.output
    done >> $output_file

    rm -rf $_temp_dir

    return 0
}

# do_lines_task_parallel # A test to get usage

num_params=$#
script=$0

function check_params(){
    local expected_num_params=$1
    local usage=$2

    if [[ "$num_params" != "$expected_num_params" ]];then
        echo "[Exit] Num of parameters is not equal to $expected_num_params"
        [ "$usage" != "" ] && echo "usage:$0 $usage"
        exit 1
    fi
    return 0
}

function force_for(){
    # This function will run cmd if force=true or the specified target files/dirs are not exit.
    # Source subtools/linux/funtions.sh in your script firstly, then call this function.
    # Usage: force_for cmd f:file1 f:file2 d:dir1 ...

    _cmd=$1

    local force=$force # get from upper script, in which the force is defined.
    [ "$force" == "" ] && force=true
    [ "$_cmd" == "" ] && echo "[Exit in force_for] _cmd is empty." && exit 1

    while [ $# -gt 1 ];do
        target=$2
        if [ "$target" != "" ];then
            num=$(echo "$target" | sed 's/:/ /g' | awk '{print NF}')
            [ "$num" -ne 2 ] && echo "Expected a specifier for the target file/dir, but got $target." && exit 1
            type=$(echo "$target" | sed 's/:/ /g' | awk '{print $1}')
            target_path=$(echo "$target" | sed 's/:/ /g' | awk '{print $2}')
            [[ "$type" != "f" && "$type" != "d" ]] && echo "[Exit in force_for] the type of target file $target is not d/dir or f/file." && exit 1
            
            [ "$force" == "true" ] && rm -rf $target_path && break

            if [ "$type" == "d" ];then
                [ ! -d "$target_path" ] && force=true
            else
                [ ! -f "$target_path" ] && force=true
                [ ! -s "$target_path" ] && force=true
            fi
        fi
        shift
    done

    if [ "$force" == "true" ];then
        eval $_cmd && return 0
        exit 1
    else
        return 0
    fi
}

function clean_path(){
    local path=$1
    dir=$(dirname $path)
    if [ "$dir" == "." ];then
        clean_path=$(basename $path)
    else
        clean_path=$dir/$(basename $path)
    fi
    echo $clean_path
    return 0
}

function path2name(){
    local path=$(clean_path $1)
    local name=$(echo "${path}" | sed 's/\// /g' | awk '{for(i=1;i<=NF-1;i++){printf $i"_";}printf $NF}')
    echo $name
    return 0
}

function is_exist(){
    local type=$1
    local target=$2

    [ "$type" == "" ] && echo "Expected the checking type is not empty." && exit 1
    [ "$target" == "" ] && echo "Expected the checking target is not empty." && exit 1

    case $type in
        file) 
            [ ! -f "$target" ] && echo "Expected file $target is exist." && exit 1
            [ ! -s "$target" ] && echo "Expected file $target is not empty." && exit 1
            ;;
        dir)
            [ ! -d "$target" ] && echo "Expected directory $target is exist." && exit 1
            ;;
        link)
            [ ! -L "$target" ] && echo "Expected $target is a link." && exit 1
            ;;
        *) echo && exit 1
    esac
    return 0
}
