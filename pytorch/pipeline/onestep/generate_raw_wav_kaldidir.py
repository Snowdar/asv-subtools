# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2021-08-27)


import sys
import os
import shutil
import glob
import time
import logging
import argparse
import traceback
import random
import numpy as np
import torch
import torchaudio
from tqdm.contrib import tqdm
from pathlib import Path
sys.path.insert(0, 'subtools/pytorch')
from libs.egs.kaldi_dataset import KaldiDataset, KaldiDatasetMultiTask
import libs.support.kaldi_common as kaldi_common
from libs.egs.signal_processing import de_silence
from libs.support.utils import get_torchaudio_backend

torchaudio_backend = get_torchaudio_backend()
torchaudio.set_audio_backend(torchaudio_backend)
AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])
logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [ %(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ]\n#### %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    # Start
    parser = argparse.ArgumentParser(
        description="""Generate raw wav kaldidir which contains utt2chunk and utt2dur.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    # Options
    parser.add_argument("--expected-files", type=str, default='utt2spk:spk2utt:wav.scp',
                        help="Files of original kaldi dir")

    parser.add_argument("--whole-utt", type=str, action=kaldi_common.StrToBoolAction, default=False, choices=["true", "false"],
                        help="whole wav or segment")
    parser.add_argument("--seg-dur", type=float, default=2.015,
                        help="Segment duration of a chunk in seconds.")
    parser.add_argument("--de-silence", type=str, action=kaldi_common.StrToBoolAction, default=False, choices=["true", "false"],
                        help="Vad or not")
    parser.add_argument("--vad-save-dir", type=str, default='',
                    help="Save vad wavs.")
    parser.add_argument("--vad-win-len", type=float, default=0.1,
                        help="win duration of a chunk in seconds.")
    parser.add_argument("--amp-th", type=int, default=100,
                        help="Remove segments whose average amplitude is below the given threshold (16bit)")
    parser.add_argument("--random-segment", type=str, action=kaldi_common.StrToBoolAction, default=False, choices=["true", "false"],
                        help="Read random segments")
    # Main
    parser.add_argument("data_dir", metavar="data-dir",
                        type=str, help="A kaldi datadir.")
    parser.add_argument("dst_dir", metavar="data-dir",
                        type=str, help="Dst datadir.")
    # End
    print(' '.join(sys.argv))
    args = parser.parse_args()

    return args




def generate_raw_wav_kaldidir(args):
    expected_files = (args.expected_files).strip().split(':')

    dataset = KaldiDataset.load_data_dir(args.data_dir, expected_files)

    dst_dir = args.dst_dir
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.mkdir(dst_dir)

    f1 = open(dst_dir + "/wav.scp", "w")
    f2 = open(dst_dir + "/utt2spk", "w")
    f3 = open(dst_dir + "/utt2dur", "w")
    f4 = open(dst_dir + "/utt2chunk", "w")
    f5 = open(dst_dir + "/utt2sr", "w")

    time.sleep(3)
    if args.de_silence:
        vad_save_dir = args.vad_save_dir
        if not vad_save_dir:            
            logging.warning('Specify your dir to save vad wavs.'.format(vad_wav_dir))
            sys.exit(1)

        os.makedirs(vad_save_dir, exist_ok=True)
    pbar = tqdm(total=len(dataset), position=0,
                ascii=True, miniters=len(dataset)/100)
    torch.set_num_threads(1)
    ori_duration=0
    desilence_duration=0
    cnt=0
    for utt, spk in dataset.utt2spk.items():
        cnt+=1
        wav_path = dataset.wav_scp[utt]

        signal, sr = torchaudio.load(wav_path)
        ori_duration+=signal.shape[1]/sr
        if cnt%100==0:
            pbar.update(100)
        if args.de_silence:
            signal,lens = de_silence(signal,sr,win_len=args.vad_win_len,min_eng=args.amp_th)
            
            if lens==0:
                continue
            pos = utt.rfind('.')
            if pos>0:
                prefix, postfix = utt[:pos], utt[pos + 1:]
                assert postfix in AUDIO_FORMAT_SETS
            else:
                prefix, postfix = utt,'.wav'
            file_name = prefix+postfix
            wav_path = os.path.join(vad_save_dir,file_name)
            torchaudio.save(wav_path, signal, sr)
        duration_sample = len(signal[0])
        audio_duration = duration_sample / sr
        desilence_duration+=audio_duration
        seg_dur = args.seg_dur
        if args.whole_utt:
            remain_chunk = str(0) + "_" + str(audio_duration)

            f1.write('{} {}\n'.format(utt, wav_path))
            f2.write('{} {}\n'.format(utt, spk))
            f3.write('{} {}\n'.format(utt, audio_duration))
            f4.write('{} {}\n'.format(utt, remain_chunk))
            f5.write('{} {}\n'.format(utt, sr))
        else:
            uniq_chunks_list = get_chunks(seg_dur, utt, audio_duration)
            res_chunks_list = []
            for chunk in uniq_chunks_list:

                if args.random_segment:
                    seg_sample = int(seg_dur * sr)
                    start_sample = random.randint(
                        0, duration_sample - seg_sample)
                    end_sample = start_sample + seg_sample
                    s = round(start_sample / sr, 3)
                    e = round(end_sample / sr, 3)
                else:
                    s, e = chunk.split("_")[-2:]
                    start_sample = int(float(s) * sr)
                    end_sample = int(float(e) * sr)

                #  Avoid chunks with very small energy
                mean_sig = torch.mean(np.abs(signal[0,start_sample:end_sample]))
                if mean_sig < (args.amp_th / (1<<15)):
                    continue
                res_chunks_list.append(str(s)+'_'+str(e))
            if res_chunks_list:
                remain_chunk = '#'.join(res_chunks_list)
                f1.write('{} {}\n'.format(utt, wav_path))
                f2.write('{} {}\n'.format(utt, spk))
                f3.write('{} {}\n'.format(utt, audio_duration))
                f4.write('{} {}\n'.format(utt, remain_chunk))
                f5.write('{} {}\n'.format(utt, sr))

    pbar.close()
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    with open(dst_dir + "/ori_dur", "w") as f6:
        f6.write(format(ori_duration/3600,".2f")+'\n')
    with open(dst_dir + "/desil_dur", "w") as f7:
        f7.write(format(desilence_duration/3600,".2f")+'\n')


def get_chunks(seg_dur, audio_id, audio_duration):
    """
    Returns list of chunks.
    """
    floor_num = int(audio_duration / seg_dur)
    mod = audio_duration / seg_dur - floor_num

    chunk_lst = [
        audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
        for i in range(floor_num)
    ]
    if chunk_lst and mod/seg_dur > 0.5:
        chunk_lst.append(audio_id + "_" + str(audio_duration-seg_dur) + "_" + str(audio_duration))

    return chunk_lst


def main():
    args = get_args()

    try:
        generate_raw_wav_kaldidir(args)
    except BaseException as e:
        # Look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
