# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2021-09-20)
import os
import sys
import logging
import shutil
import glob
import argparse
import torchaudio
from tqdm.contrib import tqdm
import pandas as pd
sys.path.insert(0, "subtools/pytorch")
import libs.support.kaldi_common as kaldi_common
from libs.support.utils import get_torchaudio_backend
torchaudio_backend = get_torchaudio_backend()
torchaudio.set_audio_backend(torchaudio_backend)


logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [ %(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ]\n#### %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def prepare_speech_aug(openrir_folder, musan_folder,csv_folder='exp/aug_csv',savewav_folder='data/speech_aug2', max_noise_len=2.015, force_clear=False):
    """Prepare the openrir and musan dataset for adding reverb and noises.

    Arguments
    ---------
    openrir_folder,musan_folder : str
        The location of the folder containing the dataset.
    csv_folder : str
        csv file save dir.
    savewav_folder : str
        The processed noise wav save dir.
    max_noise_len : float
        The maximum noise length in seconds. Noises longer
        than this will be cut into pieces.
    force_clear : bool
        whether clear the old dir.
    """

    if not os.path.isdir(os.path.join(openrir_folder, "RIRS_NOISES")):
        raise OSError("{} is not exist, please download it.".format(
            os.path.join(openrir_folder, "RIRS_NOISES")))
    if not os.path.isdir(os.path.join(musan_folder, "musan")):
        raise OSError("{} is not exist, please download it.".format(
            os.path.join(musan_folder, "musan")))

    if force_clear:
        if os.path.isdir(csv_folder):
            shutil.rmtree(csv_folder)
        if os.path.isdir(savewav_folder):
            shutil.rmtree(savewav_folder)


    if not os.path.isdir(csv_folder):
        os.makedirs(csv_folder)
    if not os.path.isdir(savewav_folder):
        os.makedirs(savewav_folder)


    musan_speech_files = glob.glob(os.path.join(
        musan_folder, 'musan/speech/*/*.wav'))
    musan_music_files = glob.glob(os.path.join(
        musan_folder, 'musan/music/*/*.wav'))
    musan_noise_files = glob.glob(os.path.join(
        musan_folder, 'musan/noise/*/*.wav'))

    musan_speech_item = []
    musan_music_item = []
    musan_noise_item = []
    for file in musan_speech_files:
        new_filename = os.path.join(savewav_folder, '/'.join(file.split('/')[-4:]))
        musan_speech_item.append((file, new_filename))
    for file in musan_music_files:
        new_filename = os.path.join(savewav_folder, '/'.join(file.split('/')[-4:]))
        musan_music_item.append((file, new_filename))
    for file in musan_noise_files:
        new_filename = os.path.join(savewav_folder, '/'.join(file.split('/')[-4:]))
        musan_noise_item.append((file, new_filename))

    rir_point_noise_filelist = os.path.join(
        openrir_folder, "RIRS_NOISES", "pointsource_noises", "noise_list"
    )
    rir_point_noise_item=[]
    for line in open(rir_point_noise_filelist):
        file=line.split()[-1]
        file_name=os.path.join(openrir_folder, file)
        new_filename= os.path.join(savewav_folder, file)
        rir_point_noise_item.append((file_name, new_filename))

    real_rir_filelist = os.path.join(
        openrir_folder, "RIRS_NOISES", "real_rirs_isotropic_noises", "rir_list"
    )
    real_rir_rev_item=[]
    for line in open(real_rir_filelist):
        file=line.split()[-1]
        file_name=os.path.join(openrir_folder, file)
        new_filename= os.path.join(savewav_folder, file)
        real_rir_rev_item.append((file_name, new_filename))

    sim_medium_rir_filelist = os.path.join(
        openrir_folder, "RIRS_NOISES", "simulated_rirs", "mediumroom","rir_list"
    )
    sim_medium_rir_rev_item=[]
    for line in open(sim_medium_rir_filelist):
        file=line.split()[-1]
        file_name=os.path.join(openrir_folder, file)
        file_dir,base_file=file.rsplit('/',1)
        new_base_file = "medium_"+base_file
        new_filename= os.path.join(savewav_folder,file_dir,new_base_file)
        sim_medium_rir_rev_item.append((file_name, new_filename))

    sim_small_rir_filelist = os.path.join(
        openrir_folder, "RIRS_NOISES",  "simulated_rirs", "smallroom","rir_list"
    )
    sim_small_rir_rev_item=[]
    for line in open(sim_small_rir_filelist):
        file=line.split()[-1]
        file_name=os.path.join(openrir_folder, file)
        file_dir,base_file=file.rsplit('/',1)
        new_base_file = "small_"+base_file
        new_filename= os.path.join(savewav_folder,file_dir,new_base_file)
        sim_small_rir_rev_item.append((file_name, new_filename))
    sim_large_rir_filelist = os.path.join(
        openrir_folder, "RIRS_NOISES",  "simulated_rirs", "largeroom","rir_list"
    )
    sim_large_rir_rev_item=[]
    for line in open(sim_large_rir_filelist):
        file=line.split()[-1]
        file_name=os.path.join(openrir_folder, file)
        file_dir,base_file=file.rsplit('/',1)
        new_base_file = "large_"+base_file
        new_filename= os.path.join(savewav_folder,file_dir,new_base_file)
        sim_large_rir_rev_item.append((file_name, new_filename))

    noise_items=musan_noise_item
    csv_dct={}
    reverb_csv = os.path.join(csv_folder, "real_reverb.csv")
    csv_dct[reverb_csv]=real_rir_rev_item
    sim_small_csv = os.path.join(csv_folder, "sim_small_reverb.csv")
    csv_dct[sim_small_csv]=sim_small_rir_rev_item
    sim_medium_csv = os.path.join(csv_folder, "sim_medium_reverb.csv")
    csv_dct[sim_medium_csv]=sim_medium_rir_rev_item
    sim_large_csv = os.path.join(csv_folder, "sim_large_reverb.csv")
    csv_dct[sim_large_csv]=sim_large_rir_rev_item
    noise_csv = os.path.join(csv_folder, "musan_noise.csv")
    pointsrc_noises_csv = os.path.join(csv_folder, "pointsrc_noise.csv")
    csv_dct[pointsrc_noises_csv] = rir_point_noise_item
    noise_csv = os.path.join(csv_folder, "musan_noise.csv")
    csv_dct[noise_csv]=noise_items
    bg_music_csv = os.path.join(csv_folder, "musan_music.csv")
    csv_dct[bg_music_csv]=musan_music_item
    speech_csv = os.path.join(csv_folder, "musan_speech.csv")
    csv_dct[speech_csv]=musan_speech_item

    # Prepare csv if necessary
    for csv_file,items in csv_dct.items():

        if not os.path.isfile(csv_file):
            if csv_file in [noise_csv,bg_music_csv,pointsrc_noises_csv]:
                prepare_aug_csv(items,csv_file,max_noise_len)
            else:
                prepare_aug_csv(items,csv_file,max_length=None)
# ---------------------------------------------------------------------------------------
# concate csv
    combine_music_noise_csv = os.path.join(csv_folder, "combine_music_noise.csv")
    combine_sim_small_medium_rev_csv = os.path.join(csv_folder, "combine_sim_small_medium_rev.csv")
    combine_sim_rev_csv = os.path.join(csv_folder, "combine_sim_rev.csv")
    concat_csv(combine_music_noise_csv,bg_music_csv,noise_csv)

    concat_csv(combine_sim_small_medium_rev_csv,sim_small_csv,sim_medium_csv)
    concat_csv(combine_sim_rev_csv,sim_small_csv,sim_medium_csv,sim_large_csv)

    print("Prepare the speech augment dataset Done, csv files is in {}, wavs in {}.\n".format(csv_folder,savewav_folder))




def prepare_aug_csv(items, csv_file, max_length=None):
    """Iterate a set of wavs and write the corresponding csv file.

    Arguments
    ---------
    folder : str
        The folder relative to which the files in the list are listed.

    filelist : str
        The location of a file listing the files to be used.
    csvfile : str
        The location to use for writing the csv file.
    max_length : float
        The maximum length in seconds. Waveforms longer
        than this will be cut into pieces.
    """

    with open(csv_file, "w") as w:
        w.write("ID duration wav sr tot_frame wav_format\n\n")
        # copy ordinary wav.
        for item in tqdm(items, dynamic_ncols=True):
            if not os.path.isdir(os.path.dirname(item[1])):
                os.makedirs(os.path.dirname(item[1]))
            shutil.copyfile(item[0], item[1])
            filename = item[1]
            # Read file for duration/channel info
            signal, rate = torchaudio.load(filename)
  
            # Ensure only one channel
            if signal.shape[0] > 1:
                signal = signal[0].unsqueeze(0)
                torchaudio.save(filename, signal, rate)

            ID, ext = os.path.basename(filename).split(".")
            duration = signal.shape[1] / rate

            # Handle long waveforms
            if max_length is not None and duration > max_length:
                # Delete old file
                os.remove(filename)
                for i in range(int(duration / max_length)):
                    start = int(max_length * i * rate)
                    stop = int(
                        min(max_length * (i + 1), duration) * rate
                    )
                    new_filename = (
                        filename[: -len(f".{ext}")] + f"_{i}.{ext}"
                    )
                    torchaudio.save(
                        new_filename, signal[:, start:stop], rate
                    )
                    csv_row = (
                        f"{ID}_{i}",
                        str((stop - start) / rate),
                        new_filename,
                        str(rate),
                        str(stop - start),
                        ext,
                    )
                    w.write(" ".join(csv_row)+'\n')
            else:
                w.write(
                    " ".join((ID, str(duration), filename,str(rate),str(signal.shape[1]), ext))+'\n'
                )

def concat_csv(out_file,*csv_files):
    pd_list = []
    for f in csv_files:
        pd_list.append(pd.read_csv(f, sep=" ",header=0))
    out = pd.concat(pd_list)
    out.to_csv(out_file, sep=" ", header=True, index=False)

if __name__ == '__main__':
    
    # Start
    parser = argparse.ArgumentParser(
        description=""" Prepare speech augmention csv files.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    # Options
    parser.add_argument("--openrir-folder", type=str, default='/tsdata/ASR',
                    help="where has openslr rir.")

    parser.add_argument("--musan-folder", type=str, default='/tsdata/ASR',
                    help="where has openslr musan.")
    parser.add_argument("--savewav-folder", type=str, default='/work1/ldx/speech_aug_2_new',
                    help="noise clips for online speechaug, set it in SSD.")
    parser.add_argument("--force-clear", type=str, action=kaldi_common.StrToBoolAction,
                    default=True, choices=["true", "false"],
                    help="force clear")
    parser.add_argument("--max-noise-len", type=float, default=2.015,
                    help="the maximum noise length in seconds. Noises longer than this will be cut into pieces")
    parser.add_argument("csv_aug_folder", type=str, help="csv file folder.")


    # End
    print(' '.join(sys.argv))
    args = parser.parse_args()
    assert args.max_noise_len > 0.4

    prepare_speech_aug(args.openrir_folder, args.musan_folder, \
                        csv_folder=args.csv_aug_folder, \
                        savewav_folder=args.savewav_folder, \
                        max_noise_len=args.max_noise_len, \
                        force_clear=args.force_clear)
