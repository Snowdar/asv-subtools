#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-11-07)

# add a function and add a mark in get_params() and that's all
# this script should be use with subtools/score/process.sh for readconf() and process()

function get_trials(){
	enroll_conf=$1
	test_conf=$2
	trials=$(readconf "trials" $test_conf)
	
	enrolldata=$(readconf "data" $enroll_conf)
	testdata=$(readconf "data" $test_conf)
	testdir=$(readconf "dir" $test_conf)
	input=$(readconf "input" $test_conf)
	
	run.pl $testdir/log/speaker_mean.log \
		ivector-mean ark:$testdata/spk2utt scp:$testdir/$input "ark:/dev/null" "ark,t:/dev/null"
		
	subtools/getTrials.sh --only-pair false 1 $enrolldata/spk2utt $testdata/utt2spk $trials.raw && \
			subtools/getTrials.sh 2 $trials.raw $testdir $trials && echo "Make $testset trials done."
	return 0

}

function get_params_for_score(){
	mark=$1
	enroll_conf=$2
	test_conf=$3
	extra_name=$4
	
	trials=$(readconf "trials" $test_conf)
	enroll_final=$(readconf "dir" $enroll_conf)/$(readconf "final" $enroll_conf)
	test_final=$(readconf "dir" $test_conf)/$(readconf "final" $test_conf)
	
	enroll_data=$(readconf "data" $enroll_conf)
	test_data=$(readconf "data" $test_conf)
	
	enroll_dir=$(readconf "dir" $enroll_conf)
	outdir=$(readconf "dir" $test_conf)/score
	mkdir -p $outdir
	
	enrollname=$(readconf "name" $enroll_conf)
	testname=$(readconf "name" $test_conf)
	
	input=$(readconf "input" $test_conf)
	inputname=${input%.*}
	final_file=$(readconf "final" $test_conf)
	suffix=$(echo ${final_file%.*} | sed 's/^'"$inputname"'//g;'s/spk_xvector_mean//g'')

	extra_name=${extra_name:+_$extra_name}
	
	case $mark in
		cosine)
			out_score=$outdir/${mark}_${enrollname}_${testname}${suffix}${extra_name}
			string="$trials $enroll_final $test_final $out_score.score";;
		plda)
			out_score=$outdir/${mark}_${enrollname}_${testname}${suffix}${extra_name}
			plda=$(get_resource plda $enroll_conf)
			string="$trials $enroll_dir/num_utts.ark $plda $enroll_final $test_final $out_score.score";;
		aplda)
			aplda=$(get_resource aplda $enroll_conf)
			out_score=$outdir/${mark}_${enrollname}_${testname}${suffix}${extra_name}
			string="$trials $enroll_dir/num_utts.ark $aplda $enroll_final $test_final $out_score.score";;
		svm)
			out_score=$outdir/${mark}_${curve}_${enrollname}_${testname}${suffix}${extra_name}
			string="$trials $enroll_data $test_data $enroll_final $test_final $out_score.score";;
		gmm)
			out_score=$outdir/${mark}_${cnum}_${enrollname}_${testname}${suffix}${extra_name}
			string="$trials $enroll_data/utt2spk $enroll_final $test_final $out_score.score";;
		lr)
			out_score=$outdir/${mark}_${enrollname}_${testname}${suffix}${extra_name}
			string="$trials $enroll_data $enroll_final $test_final $out_score.score";;
		*)echo "[exit] Do not support $mark classfier now." && exit 1;;
	esac
	
	echo $out_score $string
	return 0
}

function cosine(){
	the_trials=$1
	final_enroll=$2
	final_test=$3
	out_score=$4
	
	specifier1=ark
	[ ${final_enroll##*.} == "scp" ] && specifier1=scp
	specifier2=ark
	[ ${final_test##*.} == "scp" ] && specifier2=scp
	
	cat $the_trials | awk '{print $1, $2}' | \
        ivector-compute-dot-products - $specifier1:$final_enroll $specifier2:$final_test $out_score || exit 1
		
	return 0
}

function plda(){
	the_trials=$1
	num_utt=$2
	plda=$3
	final_enroll=$4
	final_test=$5
	out_score=$6
	
	specifier1=ark
	[ ${final_enroll##*.} == "scp" ] && specifier1=scp
	specifier2=ark
	[ ${final_test##*.} == "scp" ] && specifier2=scp
	
	num_utt_string=""
	[ -f "$num_utt" ] && num_utt_string="--num-utts=ark:$num_utt"
	
	cat $the_trials | awk '{print $1, $2}' | \
		ivector-plda-scoring --normalize-length=true $num_utt_string \
				"ivector-copy-plda --smoothing=$plda_smoothing $plda - |" \
				$specifier1:$final_enroll $specifier2:$final_test - $out_score || exit 1
				
	return 0
}

function aplda(){
	the_trials=$1
	num_utt=$2
	plda=$3
	final_enroll=$4
	final_test=$5
	out_score=$6
	
	specifier1=ark
	[ ${final_enroll##*.} == "scp" ] && specifier1=scp
	specifier2=ark
	[ ${final_test##*.} == "scp" ] && specifier2=scp
	
	num_utt_string=""
	[ -f "$num_utt" ] && num_utt_string="--num-utts=ark:$num_utt"
	
	cat $the_trials | awk '{print $1, $2}' | \
		ivector-plda-scoring --normalize-length=true $num_utt_string \
				"ivector-copy-plda --smoothing=$aplda_smoothing $plda - |" \
				$specifier1:$final_enroll $specifier2:$final_test - $out_score || exit 1
				
	return 0
}

function svm(){
	the_trials=$1
	enroll_data=$2
	test_data=$3
	final_enroll=$4
	final_test=$5
	out_score=$6
	
	specifier1=ark
	[ ${final_enroll##*.} == "scp" ] && specifier1=scp
	specifier2=ark
	[ ${final_test##*.} == "scp" ] && specifier2=scp
	
	[[ ! -f $final_enroll.svm.data || $score_force_clear == "true" ]] && \
		subtools/score/svm/prepareSVMdata.sh --normalize true $enroll_data $specifier1:$final_enroll  $final_enroll.svm.data
	[[ ! -f $final_test.svm.data || $score_force_clear == "true" ]] && \
		subtools/score/svm/prepareSVMdata.sh --normalize true $test_data $specifier2:$final_test  $final_test.svm.data

	python subtools/score/svm/svm_ratelimit.py $final_enroll.svm.data $final_test.svm.data $out_score.tmp \
	-1 $curve $Cvalue 1>&2 || exit 1

	sed 's/[]\[]//g' $out_score.tmp | sed 's/ /\n/g' | sed '/^$/d' | paste $the_trials - | awk '{print $1,$2,$4}' > $out_score
	rm -f $out_score.tmp
	return 0
}

function gmm(){
	the_trials=$1
	utt2spk=$2
	final_enroll=$3
	final_test=$4
	out_score=$5
	
	specifier1=ark
	[ ${final_enroll##*.} == "scp" ] && specifier1=scp
	specifier2=ark
	[ ${final_test##*.} == "scp" ] && specifier2=scp
	
	subtools/score/gmm/scoreByGMM.sh --mmi $mmi --E $E --num-iters-init $num_iters_init --num-frames $num_frames \
    --min-gaussian-weight $min_gaussian_weight --cnum $cnum --num-iters $num_iters --num-gselect $num_gselect \
	--adapt $adapt --nj $nj --tau $tau --weight-tau $weight_tau --num-frames-den $num_frames_den \
	--smooth-tau $smooth_tau --init-mmi $init_mmi \
	$specifier1:$final_enroll $utt2spk $specifier2:$final_test $the_trials $out_score || exit 1

	return 0

}

function lr(){
	the_trials=$1
	enroll_data=$2
	final_enroll=$3
	final_test=$4
	out_score=$5
	
	lrdir=$(dirname $final_enroll)/lr
	mkdir -p $lrdir
	
	[ ! -f "$enroll_data/utt2lable" ] && awk 'BEGIN{i=0}NR==FNR{a[$1]=i;i++}NR>FNR{print $1,a[$2]}' $enroll_data/spk2utt $enroll_data/utt2spk > $enroll_data/utt2label
	
	awk 'BEGIN{i=0}NR==FNR{a[$1]=i;i++}NR>FNR{$1=a[$1];print $2,$1}' $enroll_data/spk2utt $the_trials > $lrdir/trials.label
	
	specifier1=ark
	[ ${final_enroll##*.} == "scp" ] && specifier1=scp
	specifier2=ark
	[ ${final_test##*.} == "scp" ] && specifier2=scp
	
	scale_string=""
	if [ "$scale" != "" ];then
	echo "$scale" > $lrdir/scale
	scale_string=--scale-priors=$lrdir/scale
	fi
	
	options="--max-steps=$max_steps --mix-up=$mix_up"
	
	logistic-regression-train $options $specifier1:$final_enroll ark:$enroll_data/utt2label $lrdir/lr.raw.model || exit 1
	logistic-regression-copy $scale_string $lrdir/lr.raw.model $lrdir/lr.scale.model || exit 1
	logistic-regression-eval --apply-log=$apply_log $options $lrdir/lr.scale.model ark:$lrdir/trials.label $specifier2:$final_test - | \
	awk 'BEGIN{i=0}NR==FNR{a[i]=$1;i++}NR>FNR{$2=a[$2];print $2,$1,$3}'  $enroll_data/spk2utt - > $out_score || exit 1
	
	return 0
}

