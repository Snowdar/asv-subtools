#!/bin/bash

# Copyright xmuspeech (Author: Snowdar 2020-02-27 2019-12-22)

force=false

prefix=fbank_81
epochs="6"
positions="near"
vectordir=

task=vox1-O
trials=
srcdata=voxceleb1
trainset=voxceleb2_dev
enrollset=
testset=

score=cosine
lda=false
clda=256
submean=false
string=

enroll_process="submean-lda-norm"
test_process="submean-lda-norm"
submean_process="getmean"
lda_process="submean-trainlda"
plda_process="submean-lda-norm-trainplda"

score_norm=false # Use as-norm for vox1-O task. Note that, it will be not available if utts of task and cohortset are too many, such as voxceleb1-E and H task.
                 # The algorithm of score normalization could be further optimized with using matrix and multi-threads to acceletate the speed of processing.
                 # But we have not developed this.
score_norm_method=asnorm
top_n=100
cohort_set=voxceleb2_dev # If not NULL, use provided set
score_norm_spk_mean=true

cohort_method="sub" # "sub" | "mean". It works when cohort_set is empty.
cohort_set_from=voxceleb2_dev # Should be a subset of $trainset if use cohort_set_method.
sub_option="" # Could be --per-spk
sub_num=2000

. subtools/parse_options.sh
. subtools/linux/functions.sh
. subtools/path.sh

split=false
if [[ "$enrollset" == "" || "$testset" == "" ]];then
    case $task in 
        vox1-O) task_name=voxceleb1_O; prepare_trials=data/$prefix/$srcdata/voxceleb1-O.trials; split=true;;
        vox1-O-clean) task_name=voxceleb1_O_clean; prepare_trials=data/$prefix/$srcdata/voxceleb1-O-clean.trials; split=true;;
        vox1-E) task_name=voxceleb1_E; prepare_trials=data/$prefix/$srcdata/voxceleb1-E.trials; split=true;;
        vox1-E-clean) task_name=voxceleb1_E_clean; prepare_trials=data/$prefix/$srcdata/voxceleb1-E-clean.trials; split=true;;
        vox1-H) task_name=voxceleb1_H; prepare_trials=data/$prefix/$srcdata/voxceleb1-H.trials; split=true;;
        vox1-H-clean) task_name=voxceleb1_H_clean; prepare_trials=data/$prefix/$srcdata/voxceleb1-H-clean.trials; split=true;;
        *) echo "Do not support $task task" && exit 1;;
    esac
    enrollset=${task_name}_enroll && testset=${task_name}_test
fi

# data/$prefix/$testset/trials will be copied from $prepare_trials if using split mode.
[ "$trials" == "" ] && trials=data/$prefix/$testset/trials

lda_data_config="$trainset[$trainset $enrollset $testset]"
submean_data_config="$trainset[$trainset $enrollset $testset]"

[ "$lda" == "true" ] && lda_string="_lda$clda"
[ "$submean" == "true" ] && submean_string="_submean"

extra_name="$trainset"
[[ "$score" == "cosine" && "$lda" == "false" && "$submean" == "false" ]] && extra_name=""

name="$testset/score/${score}_${enrollset}_${testset}${submean_string}${lda_string}_norm${extra_name:+_$extra_name}"

results="[ $score ] [ lda=$lda clda=$clda submean=$submean trainset=$trainset]"

for position in $positions;do

    results="\n[ $testset ] $results\n\n--- ${position} ---\nepoch\teer%"

    this_name=snorm
    [ "$score_norm_method" == "asnorm" ] && this_name="asnorm\(topn=$top_n\)"
    [ "$score_norm" == "true" ] && results="${results}\t$this_name-eer%"

    for epoch in $epochs;do
        obj_dir=$vectordir/${position}_epoch_${epoch}

        # Prepare task for scoring. Here it is only needed to extract voxceleb1 xvectors and then it will split subsets.
        # voxceleb -> voxceleb1-O/E/H[-clean]_enroll/test
        if [ "$split" == "true" ];then
            subtools/split_enroll_test_by_trials.sh --force $force --outname $task_name --vectordir $obj_dir/$srcdata \
                                      data/$prefix/$srcdata $prepare_trials || exit 1
        fi

        [[ "$force" == "true" || ! -f $obj_dir/$name.eer ]] && \
        subtools/scoreSets.sh  --prefix $prefix --score $score --vectordir $obj_dir --enrollset $enrollset --testset $testset \
            --lda $lda --clda $clda --submean $submean --lda-process $lda_process --trials $trials --extra-name "$extra_name" \
            --enroll-process $enroll_process --test-process $test_process --plda-process $plda_process --submean-process $submean_process \
            --lda-data-config "$lda_data_config" --submean-data-config "$submean_data_config" --plda-trainset $trainset

        # Score Normalization
        if [[ "$score_norm" == "true" && -f $obj_dir/$name.score ]];then
            if [ "$cohort_set" == "" ];then
                if [ "$cohort_method" == "sub" ];then
                    cohort_set=${cohort_set_from}_cohort_sub_${sub_num}$sub_option
                    [[ "$force" == "true" ]] && rm -rf data/$prefix/$cohort_set
                    [ ! -d data/$prefix/$cohort_set ] && subtools/kaldi/utils/subset_data_dir.sh $sub_option \
                    data/$prefix/$cohort_set_from $sub_num data/$prefix/$cohort_set
                elif [ "$cohort_method" == "mean" ];then
                    cohort_set=${cohort_set_from}_cohort_mean
                    [[ "$force" == "true" ]] && rm -rf data/$prefix/$cohort_set
                    [ ! -d data/$prefix/$cohort_set ] && mkdir -p data/$prefix/$cohort_set && \
                    awk '{print $1,$1}' data/$prefix/$cohort_set_from/spk2utt > data/$prefix/$cohort_set/spk2utt && \
                    awk '{print $1,$1}' data/$prefix/$cohort_set_from/spk2utt > data/$prefix/$cohort_set/utt2spk
                fi
            fi

            enroll_cohort_name="$cohort_set/score/${score}_${enrollset}_${cohort_set}${submean_string}${lda_string}_norm${extra_name:+_$extra_name}"
            test_cohort_name="$cohort_set/score/${score}_${testset}_${cohort_set}${submean_string}${lda_string}_norm${extra_name:+_$extra_name}"
            output_name="${name}_snorm_$cohort_set"
            [ "${score_norm_method}" == "asnorm" ] && output_name="${name}_asnorm${top_n}_$cohort_set"

            enroll_key=utt2spk
            num1=$(echo $enroll_process | grep mean | wc -l)
            num2=$(echo $enroll_process | grep submean | wc -l)
            # Means that enroll_process is mean-submean-lda-norm
            [ $num1 -gt $num2 ] && enroll_key=spk2utt

            if [ "$score_norm_spk_mean" == "true" ];then
                cohort_process="$test_process-mean"
                subtools/getTrials.sh 3 data/$prefix/$enrollset/$enroll_key data/$prefix/$cohort_set/spk2utt \
                                                           data/$prefix/$cohort_set/$enrollset.cohort.trials || exit 1
                subtools/getTrials.sh 3 data/$prefix/$testset/utt2spk data/$prefix/$cohort_set/spk2utt \
                                                           data/$prefix/$cohort_set/$testset.cohort.trials || exit 1
            else
                cohort_process=$test_process
                subtools/getTrials.sh 3 data/$prefix/$enrollset/$enroll_key data/$prefix/$cohort_set/utt2spk \
                                                           data/$prefix/$cohort_set/$enrollset.cohort.trials || exit 1
                subtools/getTrials.sh 3 data/$prefix/$testset/utt2spk data/$prefix/$cohort_set/utt2spk \
                                                           data/$prefix/$cohort_set/$testset.cohort.trials || exit 1
            fi

            # enroll -> cohort
            lda_data_config="$trainset[$trainset $enrollset $cohort_set]"
            submean_data_config="$trainset[$trainset $enrollset $cohort_set]"

            subtools/scoreSets.sh  --prefix $prefix --eval true --score $score --vectordir $obj_dir \
                --lda $lda --clda $clda --submean $submean --lda-process $lda_process --extra-name "$extra_name" \
                --enroll-process $enroll_process --test-process $cohort_process --plda-process $plda_process --submean-process $submean_process \
                --lda-data-config "$lda_data_config" --submean-data-config "$submean_data_config" --plda-trainset $trainset \
                --enrollset $enrollset --testset $cohort_set \
                --trials data/$prefix/$cohort_set/$enrollset.cohort.trials $string

            # test -> cohort
            lda_data_config="$trainset[$trainset $testset $cohort_set]"
            submean_data_config="$trainset[$trainset $testset $cohort_set]"

            subtools/scoreSets.sh  --prefix $prefix --eval true --score $score --vectordir $obj_dir \
                --lda $lda --clda $clda --submean $submean --lda-process $lda_process --extra-name "$extra_name" \
                --enroll-process $test_process --test-process $cohort_process --plda-process $plda_process --submean-process $submean_process \
                --lda-data-config "$lda_data_config" --submean-data-config "$submean_data_config" --plda-trainset $trainset \
                --enrollset $testset --testset $cohort_set \
                --trials data/$prefix/$cohort_set/$testset.cohort.trials $string

            # Nomalize scores
            python3 subtools/score/ScoreNormalization.py --top-n=$top_n --method=$score_norm_method $obj_dir/$name.score \
                                                        $obj_dir/$enroll_cohort_name.score $obj_dir/$test_cohort_name.score \
                                                        $obj_dir/$output_name.score || exit 1

            subtools/computeEER.sh --write-file $obj_dir/$output_name.eer $trials $obj_dir/$output_name.score
            
            eer=""
            [ -f "$obj_dir/$output_name.eer" ] && eer=`cat $obj_dir/$output_name.eer`
            results="$results\n$epoch\t`cat $obj_dir/$name.eer`\t$eer"
        else
            eer=""
            [ -f "$obj_dir/$name.eer" ] && eer=`cat $obj_dir/$name.eer`
            results="$results\n$epoch\t$eer"
        fi
    done
done

echo -e "$results\n-----------------------\n" >> $vectordir/${score}_${testset}${lda_string}${submean_string}.results

echo -e $results
