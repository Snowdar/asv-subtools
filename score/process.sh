#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-11-07)

# add a function and add a mark in get_params() and add a config for which needs 'get_resource' in the top script and that's all

function check(){
	varset=$1
	support=$2
	name=$3
	
	for x in $(echo ${varset} | sed 's/-/ /g');do
		lable=0
		for i in $support;do
			[ "$x" == "$i" ] && lable=1 && break
		done
		[ "$lable" == 0 ] && echo "[exit] Don't support [ $x ] in [ $varset ] $name and support [ $support ] only." && exit 1
	done
	
	return 0
}

function readconf(){
	[ $# != 2 ] && echo "[exit] readconf(): num of params != 2" 1>&2 && exit 1
	key=$1
	conf=$2
	
	[ ! -f $conf ] && echo "[exit] $conf is not exist, please check provided dataset" 1>&2 && exit 1
	value=$(awk -v key=$key '{if($1==key) print $2}' $conf)
	[ "$value" == "" ] && echo "[exit] the value of key [$key] in $conf is not exist, please check data config about $key" 1>&2 && exit 1
	echo $value
	return 0
}

function writeconf(){
	[ $# != 3 ] && return 0
	key=$1
	value=$2
	conf=$3
	
	[ -f $conf ] && \
	sed -i '/^'"$key"' /d' $conf
	
	[ -f $conf ] && \
	echo "$key $value" >> $conf
	return 0
}
function findlist(){
	[ $# != 2 ] && echo "[exit] findlist(): num of params != 2" 1>&2 && exit 1
	key=$1
	list=$(cat $2)
	
	num=$(echo -e "$list\n$list\n$key" | sort | uniq -u | wc -l )
	echo $num
	return 0
}

function get_resource(){
	prekey=$1
	conf=$2
	
	tmp_data_conf=$(readconf "${prekey}_data_conf" $conf)
	tmp_string=$(readconf "${prekey}_process" $tmp_data_conf)
	tmp_dir=$(readconf "dir" $tmp_data_conf)
	out=$(process $tmp_data_conf $tmp_string)
	
	echo $tmp_dir/$out
	
	return 0
}

function get_params(){
	mark=$1
	conf=$2
	infile=$3

	data=$(readconf "data" $conf)
	dir=$(readconf "dir" $conf)

	case $mark in
		mean)
			outfile="spk_${infile%.*}_${mark}.ark"
			string="${data}/spk2utt ${dir}/$infile ${dir}/$outfile ${dir}/num_utts.ark";;
		getmean)
			outfile="${infile%.*}.global.vec"
			string="${dir}/$infile ${dir}/$outfile";;
		submean)
			global_mean=$(get_resource submean $conf)
			outfile="${infile%.*}_${mark}.ark"
			string="$global_mean ${dir}/$infile ${dir}/$outfile";;
		norm)
			outfile="${infile%.*}_${mark}.ark"
			string="${dir}/$infile ${dir}/$outfile";;
		lda)
			lda_mat=$(get_resource lda $conf)
			outfile="${infile%.*}_${mark}${clda}.ark"
			string="$lda_mat ${dir}/$infile ${dir}/$outfile";;
		trainlda)
			outfile="transform_$clda.mat"
			string="${dir}/$infile ${data}/utt2spk ${dir}/$outfile";;
		whiten)
			whiten_mat=$(get_resource whiten $conf)
			outfile="${infile%.*}_${mark}.ark"
			string="$whiten_mat ${dir}/$infile ${dir}/$outfile";;
		trainwhiten)
			outfile="zca_whiten.mat"
			string="${dir}/$infile ${dir}/$outfile";;
		trainpcawhiten)
			outfile="pca_whiten.mat"
			string="${dir}/$infile ${dir}/$outfile";;
		trainplda)
			outfile="plda"
			string="$data/spk2utt $dir/$infile $dir/$outfile";;
		trainaplda)
			plda=$(get_resource plda $conf)
			outfile="aplda"
			string="$plda $dir/$infile $dir/$outfile";;
		*)echo "[exit] Do not support $mark process now." 1>&2 && exit 1;;
	esac
	
	echo $outfile $string
	return 0
}

function process(){
	conf=$1
	process_string=$2
	
	current=$(readconf "input" $conf)
	dir=$(readconf "dir" $conf)
	
	for the_process in $(echo ${process_string} | sed 's/-/ /g');do
		doit=1
		[[ "$lda" != "true" && "$the_process" == "lda" ]] && doit=0
		[[ "$submean" != "true" && "$the_process" == "submean" ]] && doit=0
		[[ "$whiten" != "true" && "$the_process" == "whiten" ]] && doit=0
		
		if [ "$doit" == 1 ];then
			tmp=$(get_params $the_process $conf $current)
			current=$(echo "$tmp" | awk '{print $1}')
			params=$(echo "$tmp" | awk '{$1="";print $0}')
			exist=$(findlist "$dir/$current" "$list") # 1 -> do not exist

			[[ ! -f "$dir/$current" || "$process_force_clear" == "true" ]] && [[ "$exist" == 1 ]] && \
			$the_process $params
			echo "$dir/$current" >> $list
		fi
	done
	
	echo $current
	return 0
}


############################################################################
function mean(){
	spk2utt=$1
	infile=$2
	outfile=$3
	num_utts=$4
	
	specifier=ark
	[ ${infile##*.} == "scp" ] && specifier=scp
	
	ivector-mean ark:$spk2utt $specifier:$infile ark:$outfile ark,t:$num_utts || exit 1
	return 0
}

function getmean(){
# compute global mean.vector for substract mean.vector
	infile=$1
	outmean=$2
	
	specifier=ark
	[ ${infile##*.} == "scp" ] && specifier=scp
	
	ivector-mean $specifier:$infile $outmean || exit 1
	return 0
}

function submean(){
# substract global mean.vector
	mean=$1
	infile=$2
	outfile=$3
	
	specifier=ark
	[ ${infile##*.} == "scp" ] && specifier=scp
	
	ivector-subtract-global-mean $mean $specifier:$infile ark:$outfile || exit 1
	return 0
}

function norm(){
	infile=$1
	outfile=$2
	
	specifier=ark
	[ "${infile##*.}" == "scp" ] && specifier=scp
	
	ivector-normalize-length $specifier:$infile ark:$outfile || exit 1
	return 0
}

function transform(){
# an implement of the interface for any matrix to transform data
	mat=$1
	infile=$2
	outfile=$3
	
	specifier=ark
	[ "${infile##*.}" == "scp" ] && specifier=scp
	
	ivector-transform $mat $specifier:$infile ark:$outfile || exit 1
	return 0
}

function trainlda(){
	infile=$1
	utt2spk=$2
	outfile=$3
	
	specifier=ark
	[ "${infile##*.}" == "scp" ] && specifier=scp
	
	ivector-compute-lda --dim=$clda --total-covariance-factor=0.1 $specifier:$infile ark:$utt2spk $outfile || exit 1
	return 0
}

function lda(){
	transform $@
	return 0
}

function trainwhiten(){
# ZCA whitening
	trainfile=$1
	outmat=$2
	
	train_specifier=ark
	[ "${trainfile##*.}" == "scp" ] && train_specifier=scp
	
	[ ! -f $trainfile.txt ] && copy-vector $train_specifier:$trainfile ark,t:$trainfile.txt
	
	# should print information to terminal with 1>&2 when using python or will be error
	python3 subtools/score/whiten/train_ZCA_Whitening.py --ark-format=true $trainfile.txt $outmat 1>&2  || exit 1
	return 0
}

function trainpcawhiten(){
# PCA whitening 
	trainfile=$1
	outmat=$2
	
	train_specifier=ark
	[ "${trainfile##*.}" == "scp" ] && train_specifier=scp
	
	est-pca --read-vectors=true $train_specifier:$trainfile $outmat || exit 1
	return 0
}

function whiten(){
	transform $@
	return 0
}

function trainplda(){
	spk2utt=$1
	infile=$2
	outfile=$3
	
	specifier=ark
	[ "${infile##*.}" == "scp" ] && specifier=scp
	
	ivector-compute-plda ark:$spk2utt $specifier:$infile $outfile || exit 1
	
	return 0
}

function trainaplda(){
	plda=$1
	infile=$2
	outfile=$3
	
	specifier=ark
	[ "${infile##*.}" == "scp" ] && specifier=scp
	
	ivector-adapt-plda --within-covar-scale=$within_covar_scale --between-covar-scale=$between_covar_scale \
		--mean-diff-scale=$mean_diff_scale $plda $specifier:$infile $outfile || exit 1
		
	return 0
}
