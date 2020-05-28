#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-11-20)

# Copy new version of utils and steps from kaldi and run this patch script.
#
# for utils and steps when copy them with updating into subtools
# don't mind the information like "sed: couldn't edit ../steps/nnet3/report: not a regular file"
# where there are some files that are not our target files.

type=$1

[ "$type" == "" ] && type=all

utils=false
steps=false
steps_multitask=false
sid=false

if [ "$type" == "all" ] ;then
utils=true
steps=true
steps_multitask=true
sid=true
else
	case $type in
		utils) utils=true;;
		steps) steps=true;;
		steps_multitask) steps_multitask=true;;
		sid) sid=true;;
		*) echo "do not support type $type" && exit 1;;
	esac
fi

cd subtools/kaldi/patch

if $utils;then
	echo "do utils..."
	cp -rf extra/utils/* ../utils
	find ../utils -name "*" | xargs -n 1 sed -i 's/subtools\/kaldi\///g' 2>/dev/null
	find ../utils -name "*" | xargs -n 1 sed -i 's/utils\//subtools\/kaldi\/utils\//g' 2>/dev/null
	find ../utils -name "*" | xargs -n 1 sed -i "s/path.sh/subtools\/path.sh/g" 2>/dev/null
	find ../utils -name "*" | xargs -n 1 sed -i 's/steps\//subtools\/kaldi\/steps\//g' 2>/dev/null
fi

if $steps;then
	echo "do steps..."
	cp -rf extra/steps/* ../steps
	find ../steps -name "*" | xargs -n 1 sed -i 's/subtools\/kaldi\///g' 2>/dev/null
	find ../steps -name "*" | xargs -n 1 sed -i 's/steps\//subtools\/kaldi\/steps\//g' 2>/dev/null
	find ../steps -name "*" | xargs -n 1 sed -i "s/path.sh/subtools\/path.sh/g" 2>/dev/null
	find ../steps -name "*" | xargs -n 1 sed -i "s/'steps'/'subtools\/kaldi\/steps'/g" 2>/dev/null
	find ../steps -name "*" | xargs -n 1 sed -i 's/utils\//subtools\/kaldi\/utils\//g' 2>/dev/null
fi

if $steps_multitask;then
	if [ -d ../steps_multitask ];then
		echo "do steps_multitask..."
		find ../steps_multitask -name "*" | xargs -n 1 sed -i 's/subtools\/kaldi\///g' 2>/dev/null
		find ../steps_multitask -name "*" | xargs -n 1 sed -i 's/_multitask//g' 2>/dev/null
		find ../steps_multitask -name "*" | xargs -n 1 sed -i 's/steps\/libs/subtools\/kaldi\/steps_multitask\/libs/g' 2>/dev/null
		find ../steps_multitask -name "*" | xargs -n 1 sed -i 's/steps\/nnet3/subtools\/kaldi\/steps_multitask\/nnet3/g' 2>/dev/null
		find ../steps_multitask -name "*" | xargs -n 1 sed -i 's/steps\//subtools\/kaldi\/steps\//g' 2>/dev/null
		find ../steps_multitask -name "*" | xargs -n 1 sed -i 's/utils\//subtools\/kaldi\/utils\//g' 2>/dev/null
		find ../steps_multitask -name "*" | xargs -n 1 sed -i "s/'steps'/'subtools\/kaldi\/steps_multitask'/g" 2>/dev/null
		find ../steps_multitask -name "*" | xargs -n 1 sed -i "s/path.sh/subtools\/path.sh/g" 2>/dev/null
	fi
fi

if $sid;then
	if [ -d ../sid ];then
		echo "do sid..."
		find ../sid -name "*" | xargs -n 1 sed -i 's/subtools\/kaldi\///g' 2>/dev/null
		find ../sid -name "*" | xargs -n 1 sed -i 's/utils\//subtools\/kaldi\/utils\//g' 2>/dev/null
		find ../sid -name "*" | xargs -n 1 sed -i "s/path.sh/subtools\/path.sh/g" 2>/dev/null
		find ../sid -name "*" | xargs -n 1 sed -i 's/steps\//subtools\/kaldi\/steps\//g' 2>/dev/null
		find ../sid -name "*" | xargs -n 1 sed -i 's/sid\//subtools\/kaldi\/sid\//g' 2>/dev/null
	fi
fi

cd -
echo "All done."
