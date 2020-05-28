#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-10-02)

current_dir=$(subtools/linux/decode_symbolic_link.sh $PWD)
kaldi_root_dir=$(dirname $(dirname $(dirname $current_dir)))

cd subtools/kaldi/patch

# compile this files
cfile="src/ivectorbin/select-voiced-ali.cc src/nnet3bin/nnet3-copy-cvector-egs.cc"

[ ! -d ../steps_multitask ] && cp -rf steps ../steps_multitask
cd -

subtools/kaldi/patch/runPatch-base-command.sh --kaldi_root_dir "$kaldi_root_dir" --cfile "$cfile"


echo "Multi-task training is available."

##########################################################################
# Deprecated #
# # if paths of *.so are not right,you could fix them by yourself.
# atlasInclude=$(grep  "ATLASINC.*=" $kaldi_root_dir/src/kaldi.mk | cut -d "=" -f 2 )
# atlasSo=$(grep  "ATLASLIBS.*=" $kaldi_root_dir/src/kaldi.mk | cut -d "=" -f 2 )
# openfstInclude=$(grep  "OPENFSTINC.*=" $kaldi_root_dir/src/kaldi.mk | cut -d "=" -f 2 )
# openfstSo=$(grep  "OPENFSTLIBS.*=" $kaldi_root_dir/src/kaldi.mk | cut -d "=" -f 2 )
# openfstLib=$(grep  "OPENFSTLDFLAGS.*=" $kaldi_root_dir/src/kaldi.mk | cut -d " " -f 3)
# cuda=$(grep  "CUDATKDIR.*=" $kaldi_root_dir/src/kaldi.mk | cut -d "=" -f 2 | sed 's/^ \+//g')

# for x in $cfile;do
# name=`basename ${x%.*}`
# dir=`dirname $x`
# type=`basename $dir`
# [ -f $kaldi_root_dir/$dir/$x ] && mv $kaldi_root_dir/$dir/$x.bk
# cp -f $x $kaldi_root_dir/$dir
# cd  $kaldi_root_dir/$dir
# echo "compile $kaldi_root_dir/$x"
# case $type in
	# ivectorbin)
	# g++ -std=c++11 -I.. -I$openfstInclude -Wno-sign-compare -Wall -Wno-sign-compare -Wno-unused-local-typedefs -Wno-deprecated-declarations -Winit-self -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -I$atlasInclude -msse -msse2 -pthread -g  -DHAVE_CUDA -I$cuda/include -c -o $name.o $name.cc
	
	# g++  $openfstLib  -rdynamic  $name.o ../ivector/kaldi-ivector.a ../hmm/kaldi-hmm.a ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a   $openfstSo $atlasSo -lm -lpthread -ldl -o $name
	# cd -
	# ;;	
	# nnet3bin)
	# g++ -std=c++11 -I.. -I$openfstInclude -Wno-sign-compare -Wall -Wno-sign-compare -Wno-unused-local-typedefs -Wno-deprecated-declarations -Winit-self -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -I$atlasInclude -msse -msse2 -pthread -g  -DHAVE_CUDA -I$cuda/include -c -o $name.o $name.cc
	
	# g++  $openfstLib  -rdynamic -L$cuda/lib64 -Wl,-rpath,$cuda/lib64  $name.o ../nnet3/kaldi-nnet3.a ../chain/kaldi-chain.a ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a   $openfstSo $atlasSo -lm -lpthread -ldl -lcublas -lcusparse -lcudart -lcurand  -o $name
	# cd -
	# ;;
	# *) echo "don't support this type [$type]" && exit 1;;
# esac

# echo "compile $name done."
# echo "start to test this command and if you can see usages of this command later,then it means compiling successfully."
# $kaldi_root_dir/$dir/$name
# done
# echo "All done."
