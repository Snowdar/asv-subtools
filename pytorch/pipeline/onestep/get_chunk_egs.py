# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-01-05)


import sys
import os
import logging
import argparse
import traceback

sys.path.insert(0, 'subtools/pytorch')

import libs.support.kaldi_common as kaldi_common
from libs.egs.kaldi_dataset import KaldiDataset
from libs.egs.samples import ChunkSamples

"""Get chunk egs for sre and lid ... which use the xvector framework.
"""

logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ]\n#### %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_args():
    # Start
    parser = argparse.ArgumentParser(
        description="""Split data to chunk-egs.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    # Options
    parser.add_argument("--chunk-size", type=int, default=200,
                    help="A fixed chunk size.")

    parser.add_argument("--valid-sample", type=str, action=kaldi_common.StrToBoolAction,
                    default=True, choices=["true", "false"],
                    help="Get the valid samples or not.")

    parser.add_argument("--valid-split-type", type=str, default='--total-spk',
                    choices=["--default", "--per-spk", "--total-spk"],
                    help="Get the valid samples or not.")

    parser.add_argument("--valid-num-utts", type=int, default=1024,
                    help="The num utts to split for valid. 1024 for --total-spk.")

    parser.add_argument("--sample-type", type=str, default='speaker_balance',
                    choices=["speaker_balance", "sequential"],
                    help="The sample type for trainset.")

    parser.add_argument("--chunk-num", type=int, default=-1,
                    help="Define the avg chunk num. -1->suggestion（max / num_spks * scale）, 0->max, int->int")

    parser.add_argument("--scale", type=float, default=1.5,
                    help="The scale for --chunk-num:-1.")

    parser.add_argument("--overlap", type=float, default=0.1,
                    help="The scale of overlab to generate chunks.")

    parser.add_argument("--drop-last", type=str, action=kaldi_common.StrToBoolAction,
                    default=False, choices=["true", "false"],
                    help="Drop the last sample of every utterance or not.")

    parser.add_argument("--valid-sample-type", type=str, default='every_utt',
                    choices=["speaker_balance", "sequential", "every_utt"],
                    help="The sample type for valid set.")

    parser.add_argument("--valid-chunk-num", type=int, default=2,
                    help="Define the avg chunk num. -1->suggestion（max / num_spks * scale）, 0->max, int->int")

    parser.add_argument("--valid-scale", type=float, default=1.5,
                    help="The scale for --valid-chunk-num:-1.")

    # Main
    parser.add_argument("data_dir", metavar="data-dir", type=str, help="A kaldi datadir.")
    parser.add_argument("save_dir", metavar="save-dir", type=str, help="The save dir of mapping file of chunk-egs.")

    # End
    print(' '.join(sys.argv))
    args = parser.parse_args()

    return args


def get_chunk_egs(args):
    logger.info("Load kaldi datadir {0}".format(args.data_dir))
    dataset = KaldiDataset.load_data_dir(args.data_dir)
    dataset.generate("utt2spk_int")

    if args.valid_sample:
        logger.info("Split valid dataset from {0}".format(args.data_dir))
        if args.valid_num_utts > len(dataset)//10:
            logger.info("Warning: the --valid-num-utts ({0}) of valid set is out of 1/10 * num of original dataset ({1}). Suggest to be less.".format(args.valid_num_utts, len(dataset)))
        trainset, valid = dataset.split(args.valid_num_utts, args.valid_split_type)
    else:
        trainset = dataset

    logger.info("Generate chunk egs with chunk-size={0}.".format(args.chunk_size))
    trainset_samples = ChunkSamples(trainset, args.chunk_size, chunk_type=args.sample_type,
                            chunk_num_selection=args.chunk_num, scale=args.scale, overlap=args.overlap, drop_last=args.drop_last)

    if args.valid_sample:
        valid_sample = ChunkSamples(valid, args.chunk_size, chunk_type=args.valid_sample_type, 
                            chunk_num_selection=args.valid_chunk_num, scale=args.valid_scale, overlap=args.overlap, drop_last=args.drop_last)

    logger.info("Save mapping file of chunk egs to {0}".format(args.save_dir))
    if not os.path.exists("{0}/info".format(args.save_dir)):
        os.makedirs("{0}/info".format(args.save_dir))

    trainset_samples.save("{0}/train.egs.csv".format(args.save_dir))

    if args.valid_sample:
        valid_sample.save("{0}/valid.egs.csv".format(args.save_dir))

    with open("{0}/info/num_frames".format(args.save_dir),'w') as writer:
        writer.write(str(trainset.num_frames))

    with open("{0}/info/feat_dim".format(args.save_dir),'w') as writer:
        writer.write(str(trainset.feat_dim))

    with open("{0}/info/num_targets".format(args.save_dir),'w') as writer:
        writer.write(str(trainset.num_spks))

    logger.info("Generate egs from {0} done.".format(args.data_dir))

def main():
    args = get_args()

    try:
        get_chunk_egs(args)
    except BaseException as e:
        # Look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

