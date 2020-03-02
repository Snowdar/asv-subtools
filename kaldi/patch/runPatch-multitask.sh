#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-10-02)

kaldidir=/work/kaldi  # your kaldi install-dir => kaldi root dir

# compile this files
cfile="src/ivectorbin/select-voiced-ali.cc src/nnet3bin/nnet3-copy-cvector-egs.cc"

[ ! -d ../steps_multitask ] && cp -rf steps ../steps_multitask

sh runPatch-base-command.sh --kaldidir "$kaldidir" --cfile "$cfile"

echo "Multi-task training is available."

##########################################################################
# Deprecated #
# # if paths of *.so are not right,you could fix them by yourself.
# atlasInclude=$(grep  "ATLASINC.*=" $kaldidir/src/kaldi.mk | cut -d "=" -f 2 )
# atlasSo=$(grep  "ATLASLIBS.*=" $kaldidir/src/kaldi.mk | cut -d "=" -f 2 )
# openfstInclude=$(grep  "OPENFSTINC.*=" $kaldidir/src/kaldi.mk | cut -d "=" -f 2 )
# openfstSo=$(grep  "OPENFSTLIBS.*=" $kaldidir/src/kaldi.mk | cut -d "=" -f 2 )
# openfstLib=$(grep  "OPENFSTLDFLAGS.*=" $kaldidir/src/kaldi.mk | cut -d " " -f 3)
# cuda=$(grep  "CUDATKDIR.*=" $kaldidir/src/kaldi.mk | cut -d "=" -f 2 | sed 's/^ \+//g')

# for x in $cfile;do
# name=`basename ${x%.*}`
# dir=`dirname $x`
# type=`basename $dir`
# [ -f $kaldidir/$dir/$x ] && mv $kaldidir/$dir/$x.bk
# cp -f $x $kaldidir/$dir
# cd  $kaldidir/$dir
# echo "compile $kaldidir/$x"
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
# $kaldidir/$dir/$name
# done
# echo "All done."
