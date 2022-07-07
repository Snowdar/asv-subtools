#!/bin/bash

# Copyright xmuspeech (Author: Leo 2021-07-09)

model_dir=
output_quant_name=
model_file=
output_file_name=
. subtools/path.sh
. subtools/parse_options.sh

if [ ! -z $output_quant_name ];then
output_quant_file=$model_dir/$output_quant_name
fi

python3 subtools/pytorch/pipeline/onestep/export_jit.py \
    --config_dir $model_dir/config \
    --checkpoint $model_dir/$model_file \
    --output_file $model_dir/$output_file_name \
	--output_quant_file $output_quant_file
