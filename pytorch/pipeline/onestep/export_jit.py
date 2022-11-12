# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2021-07-09)

import sys
import os
import argparse
import torch
sys.path.insert(0, "subtools/pytorch")
import libs.support.utils as utils
# from model import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""description='export your script model'.""")
    parser.add_argument('--config_dir', required=True, help='model_dir')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_file', required=True, help='output file')
    parser.add_argument('--output_quant_file',
                        default=None,
                        help='output quantized model file')
    args = parser.parse_args()
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model_blueprint, model_creation = utils.read_nnet_config(
        "{0}/nnet.config".format(args.config_dir))
    model_creation = model_creation.replace("training=True", "training=False")
    model_creation = model_creation.replace(
        "jit_compile=False", "jit_compile=True")

    sys.path.insert(0, os.path.dirname(model_blueprint))
    model_name = os.path.basename(model_blueprint).split('.')[0]
    print(model_creation)
    print(model_name)
    model_module = __import__(model_name)

    # model = eval("{0}.{1}".format(model_name,model_creation))
    model = eval("model_module.{0}".format(model_creation))

    model.load_state_dict(torch.load(
        args.checkpoint, map_location='cpu'), strict=False)
    script_model = torch.jit.script(model)

    script_model.save(args.output_file)
    torch.jit.load(args.output_file)
    print('Export model successfully, see {}'.format(args.output_file))

    # Export quantized jit torch script model
    if args.output_quant_file:
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

        script_quant_model = torch.jit.script(quantized_model)
        script_quant_model.save(args.output_quant_file)
        print('Export quantized model successfully, '
              'see {}'.format(args.output_quant_file))
