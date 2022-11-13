#!/usr/bin/env python3

# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modify from https://github.com/wenet-e2e/wenet/blob/main/tools/make_shard_list.py
#  xmuspeech  (Leo 2022-01-21)

import argparse
import io
import logging
import os,sys
import random
import tarfile
import pandas as pd
import time
import multiprocessing
import shutil
import torch
import torchaudio
sys.path.insert(0, "subtools/pytorch")
import libs.support.kaldi_common as kaldi_common
import libs.support.utils as utils
torchaudio_backend = utils.get_torchaudio_backend()
torchaudio.set_audio_backend(torchaudio_backend)


logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [ %(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ]\n#### %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def write_tar_file(data_list,
                   result_list,
                   tar_file,
                   index=0,
                   total=1):
    logging.info('Processing {} {}/{}'.format(tar_file, index, total))
    read_time = 0.0
    save_time = 0.0
    write_time = 0.0
    total_dur = 0.0
    eg_num=0
    with tarfile.open(tar_file, "w") as tar:

        for item in data_list:
            try:
                key=item['eg-id']
                if 'start-position' in item:
                    start=int(item['start-position'])
                    end=int(item['end-position'])
                    num_frames = end-start
                label = str(item['class-label'])
                wav = item['wav-path']
                suffix = wav.split('.')[-1]
                assert suffix in AUDIO_FORMAT_SETS
                if 'start-position' in item:

                    ts = time.time()
                    waveforms, sample_rate = torchaudio.load(wav,num_frames=num_frames, frame_offset=start, normalize=False)
                    read_time += (time.time() - ts)
                    audio = waveforms[:1,:]
                    ts = time.time()
                    f = io.BytesIO()
                    torchaudio.save(f, audio, sample_rate, format="wav", bits_per_sample=16)
                    suffix = "wav"
                    f.seek(0)
                    data = f.read()
                    save_time += (time.time() - ts)
                else:
                    ts = time.time()
                    with open(wav, 'rb') as fin:
                        data = fin.read()
                    read_time += (time.time() - ts)

                ts = time.time()
                label_file = key + '.txt'
                label = label.encode('utf8')
                label_data = io.BytesIO(label)
                label_info = tarfile.TarInfo(label_file)
                label_info.size = len(label)
                tar.addfile(label_info, label_data)
                wav_file = key + '.' + suffix
                wav_data = io.BytesIO(data)
                wav_info = tarfile.TarInfo(wav_file)
                wav_info.size = len(data)
                tar.addfile(wav_info, wav_data)
                if 'eg-dur' in item:
                    total_dur+=float(item['eg-dur'])
                eg_num+=1
                write_time += (time.time() - ts)
            except (Exception) as e:
                print(e)
                logging.warning('proceccing eg {} error, pass it.'.format(key))
                pass
        try:
            total_dur = format(total_dur,".2f") if total_dur > 0 else 'None'
            result_list.append([tar_file,total_dur,eg_num])
        except (Exception) as e:
            logging.error('debug manager list,append {} failure'.format(tar_file))
        logging.info('read {} save {} write {}'.format(read_time, save_time,
                                                       write_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""make shard wav tar.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument('--num-utts-per-shard',
                        type=int,
                        default=2000,
                        help='num utts per shard')
    parser.add_argument('--nj',
                        type=int,
                        default=16,
                        help='num threads for make shards')
    parser.add_argument('--prefix',
                        default='shards',
                        help='prefix of shards tar file')

    parser.add_argument('--eg-type',
                        default='csv',
                        choices=['csv','scp'],
                        help='wav.scp or csv list')
    parser.add_argument('--shuffle',
                        type=str,
                        action=kaldi_common.StrToBoolAction,
                        default=True,
                        choices=["true","false"],
                        help='shuffle the list before tar')

    parser.add_argument('eg_file', help='wav list file')
    parser.add_argument('shards_dir', help='output shards dir')
    parser.add_argument('shards_list', help='output shards list file')
    print(' '.join(sys.argv))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')


    torch.set_num_threads(1)

    random.seed(1024)
    if os.path.exists(args.shards_list):
        logging.warning('{} exists, check or remove it.'.format(args.shards_list))
        sys.exit(1)
    if args.eg_type == 'csv':
        head = pd.read_csv(args.eg_file, sep=" ", nrows=0).columns
        expected_head = ['eg-id', 'eg-dur', 'wav-path',
                        'start-position', 'end-position', 'class-label']
        for col in expected_head:
            assert col in head
        lists = utils.csv_to_list(args.eg_file)
    else:
        lists = utils.read_wav_list(args.eg_file)

    if args.shuffle:
        random.shuffle(lists)

    num = args.num_utts_per_shard
    chunks = [lists[i:i + num] for i in range(0, len(lists), num)]
    os.makedirs(args.shards_dir, exist_ok=True)

    # Using thread pool to speedup

    pool = multiprocessing.Pool(processes=args.nj)
    # shards_list = []
    num_chunks = len(chunks)
    shards_list=multiprocessing.Manager().list()
    for i, chunk in enumerate(chunks):
        tar_file = os.path.join(args.shards_dir,
                                '{}_{:09d}.tar'.format(args.prefix, i))
        pool.apply_async(
            write_tar_file,
            (chunk,shards_list, tar_file, i, num_chunks))

    pool.close()
    pool.join()

    head=['eg-path','eg-dur','eg-num']

    data_frame = pd.DataFrame(list(shards_list), columns=head)
    data_frame.to_csv(args.shards_list, sep=" ", header=True, index=False)
    shutil.copy(args.shards_list,args.shards_dir)

