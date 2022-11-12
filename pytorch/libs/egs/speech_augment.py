# Copyright xmuspeech (Author: Leo 2021-09-06)
# We borrowed the code from speechbrain.

# Importing libraries
import math
from typing import Optional,List
import numpy as np
import random
import sklearn
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import (
    RandomSampler,
    Dataset,
    DataLoader,
)
from .signal_processing import (
    compute_amplitude,
    dB_to_amplitude,
    convolve1d,
    notch_filter,
    reverberate,
)
from libs.support.utils import batch_pad_right
import torch.distributed as dist

class NoiseDataset(Dataset):
    def __init__(self, csv_file, sorting="original", max_len=None, filt_min=None):

        head = pd.read_csv(csv_file, sep=" ", nrows=0).columns

        assert "ID" in head
        assert "wav" in head
        assert "duration" in head   
        assert sorting in ["original", "decending", "ascending","random"]

        data = pd.read_csv(csv_file, sep=" ",header=0)


        if filt_min:
            data =data[data['duration']>filt_min]

        
        if sorting == "decending":
            data = data.sort_values(by=['duration'], ascending=False)
        elif sorting == "ascending":
            data = data.sort_values(by=['duration'], ascending=True)
        elif sorting == "random":
            data = sklearn.utils.shuffle(data)
        else:
            pass
 
        self.path = data['wav'].values.astype(np.string_)


        if max_len:
            assert max_len > 0.0
            self.lens=data['tot_frame'].values
            self.sr=data['sr'].values

        self.max_len = max_len
        del data
 

    def __getitem__(self, index):
        wav=str(self.path[index], encoding='utf-8')
         
        if self.max_len:
            audio_len = self.lens[index]
            max_frame = min(int(self.sr[index] * self.max_len),audio_len)
            start = max(int(random.random()*(audio_len - max_frame)),0)
            wavforms,fs = torchaudio.load(wav,num_frames=max_frame, frame_offset=start)
        else:
            wavforms, fs = torchaudio.load(wav)


        return wavforms[0]

    def __len__(self):
        return len(self.path)



class ReproducibleRandomSampler(RandomSampler):
    """A modification of RandomSampler which always returns the same values.

    Also look at `torch.utils.data.RandomSampler`. This has mostly
    the same behaviour and arguments, except for adding 'seed' and 'epoch' and
    not supporting 'generator'.

    Note
    ----
    Call `set_epoch` before every epoch. Otherwise, the sampler will produce the
    same sequence of indices every epoch.

    Arguments
    ---------
    data_source : Dataset
        The data source to sample indices for.
    seed : int
        The base seed to use for the random number generator. It is recommended
        to use a value which has a good mix of 0 and 1 bits.
    epoch : int
        The epoch to start at.


    """

    def __init__(self, data_source, seed=563375142, epoch=0, **kwargs):
        if "generator" in kwargs:
            MSG = (
                "Cannot give a separate generator when using "
                + "ReproducibleRandomSampler"
            )
            raise ValueError(MSG)
        super().__init__(data_source, **kwargs)
        self.seed = int(seed)
        self.epoch = epoch
        self.generator = torch.Generator()

    def set_epoch(self, epoch):
        """
        You can also just access self.epoch, but we maintain this interface
        to mirror torch.utils.data.distributed.DistributedSampler
        """
        self.epoch = epoch

    def __iter__(self):
        self.generator.manual_seed(self.seed + self.epoch)
        return super().__iter__()

def make_dataloader(dataset,  **loader_kwargs):
    """Makes a basic DataLoader.

    Arguments
    ---------
    dataset : Dataset
        The dataset to make a DataLoader.
    **loader_kwargs : dict
        Keyword args to DataLoader, see PyTorch DataLoader for
        options.

    Returns
    -------
    DataLoader
    """

    # Reproducible random sampling
    if loader_kwargs.get("shuffle", False):
        if loader_kwargs.get("sampler") is not None:
            raise ValueError(
                "Cannot specify both shuffle=True and a "
                "sampler in loader_kwargs"
            )
        sampler = ReproducibleRandomSampler(dataset)
        loader_kwargs["sampler"] = sampler
        # Should delete shuffle because you can't set both Sampler and
        # shuffle
        # NOTE: the dict of loader options may get used elsewhere!
        # However, this del doesn't touch those because loader_kwargs comes
        # from a **kwargs dict.
        del loader_kwargs["shuffle"]

    dataloader = DataLoader(dataset, **loader_kwargs)

    return dataloader


class AddNoise(torch.nn.Module):
    """This class additively combines a noise signal to the input signal.

    Arguments
    ---------
    csv_file : str
        The name of a csv file containing the location of the
        noise audio files. If none is provided, white noise will be used.
    sorting : str
        The order to iterate the csv file, from one of the
        following options: random, original, ascending, and descending.
    num_workers : int
        Number of workers in the DataLoader (See PyTorch DataLoader docs).
    snr_low : int
        The low end of the mixing ratios, in decibels.
    snr_high : int
        The high end of the mixing ratios, in decibels.
    pad_noise : bool
        If True, copy noise signals that are shorter than
        their corresponding clean signals so as to cover the whole clean
        signal. Otherwise, leave the noise un-padded.
    mix_prob : float
        The probability that a batch of signals will be mixed
        with a noise signal. By default, every batch is mixed with noise.
    start_index : int
        The index in the noise waveforms to start from. By default, chooses
        a random index in [0, len(noise) - len(waveforms)].
    normalize : bool
        If True, output noisy signals that exceed [-1,1] will be
        normalized to [-1,1].

    """

    def __init__(
        self,
        csv_file=None,
        add_filt_min=None,
        sorting="random",
        num_workers=0,
        snr_low=0,
        snr_high=0,
        pad_noise=False,
        mix_prob=1.0,
        start_index=None,
        normalize=False,
    ):
        super().__init__()

        self.csv_file = csv_file
        self.add_filt_min=add_filt_min
        self.sorting = sorting
        self.num_workers = num_workers
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.pad_noise = pad_noise
        self.mix_prob = mix_prob
        self.start_index = start_index
        self.normalize = normalize

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Copy clean waveform to initialize noisy waveform
        noisy_waveform = waveforms.clone()
        lengths = (lengths * waveforms.shape[1]).unsqueeze(1)

        # Don't add noise (return early) 1-`mix_prob` portion of the batches
        if torch.rand(1) > self.mix_prob:
            return noisy_waveform

        # Compute the average amplitude of the clean waveforms
        clean_amplitude = compute_amplitude(waveforms, lengths)

        # Pick an SNR and use it to compute the mixture amplitude factors
        SNR = torch.rand(len(waveforms), 1, device=waveforms.device)
        SNR = SNR * (self.snr_high - self.snr_low) + self.snr_low
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        # Scale clean signal appropriately
        noisy_waveform *= 1 - noise_amplitude_factor

        # Loop through clean samples and create mixture
        if self.csv_file is None:
            white_noise = torch.randn_like(waveforms)
            noisy_waveform += new_noise_amplitude * white_noise
        else:
            tensor_length = waveforms.shape[1]
            noise_waveform, noise_length = self._load_noise(
                lengths, tensor_length,
            )

            # Rescale and add
            noise_amplitude = compute_amplitude(noise_waveform, noise_length)
            noise_waveform *= new_noise_amplitude / (noise_amplitude + 1e-14)
            noisy_waveform += noise_waveform

        # Normalizing to prevent clipping
        if self.normalize:
            abs_max, _ = torch.max(
                torch.abs(noisy_waveform), dim=1, keepdim=True
            )
            noisy_waveform = noisy_waveform / abs_max.clamp(min=1.0)

        return noisy_waveform

    def _load_noise(self, lengths, max_length):
        """Load a batch of noises"""
        lengths = lengths.long().squeeze(1)
        batch_size = len(lengths)

        # Load a noise batch
        if not hasattr(self, "data_loader"):
            # Set parameters based on input
            self.device = lengths.device

            # Create a data loader for the noise wavforms
            if self.csv_file is not None:
                dataset = NoiseDataset(
                    self.csv_file,
                    filt_min = self.add_filt_min,
                    sorting=self.sorting)
                shuffle = (self.sorting == "random")
                if torch.distributed.is_initialized():
                    sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset, shuffle=shuffle)
                    shuffle = False
                else:
                    sampler = None
                self.data_loader = make_dataloader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=self.num_workers,
                    shuffle=shuffle,
                    sampler=sampler,
                    collate_fn=batch_pad_right
                )

                self.noise_data = iter(self.data_loader)

        # Load noise to correct device
        noise_batch, noise_len = self._load_noise_batch_of_size(batch_size)
        noise_batch = noise_batch.to(lengths.device)
        noise_len = noise_len.to(lengths.device)

        # Convert relative length to an index
        noise_len = (noise_len * noise_batch.shape[1]).long()

        # Ensure shortest wav can cover speech signal
        # WARNING: THIS COULD BE SLOW IF THERE ARE VERY SHORT NOISES
        if self.pad_noise:
            while torch.any(noise_len < lengths):
                min_len = torch.min(noise_len)
                prepend = noise_batch[:, :min_len]
                noise_batch = torch.cat((prepend, noise_batch), axis=1)
                noise_len += min_len

        # Ensure noise batch is long enough
        elif noise_batch.size(1) < max_length:
            pad = max_length - noise_batch.size(1)
            left_padding = torch.randint(high = pad+1, size=(1,))[0]
            padding = (left_padding,pad-left_padding)
            noise_batch = torch.nn.functional.pad(noise_batch, padding)

        # Select a random starting location in the waveform
        start_index = self.start_index
        if self.start_index is None:
            start_index = 0
            max_chop = (noise_len - lengths).min().clamp(min=1)
            start_index = torch.randint(
                high=max_chop, size=(1,), device=lengths.device
            )

        # Truncate noise_batch to max_length
        noise_batch = noise_batch[:, start_index: start_index + max_length]
        noise_len = (
            noise_len - start_index).clamp(max=max_length).unsqueeze(1)
        return noise_batch, noise_len

    def _load_noise_batch_of_size(self, batch_size):
        """Concatenate noise batches, then chop to correct size"""

        noise_batch, noise_lens = self._load_noise_batch()

        # Expand
        while len(noise_batch) < batch_size:
            added_noise, added_lens = self._load_noise_batch()
            noise_batch, noise_lens = AddNoise._concat_batch(
                noise_batch, noise_lens, added_noise, added_lens
            )

        # Contract
        if len(noise_batch) > batch_size:
            noise_batch = noise_batch[:batch_size]
            noise_lens = noise_lens[:batch_size]

        return noise_batch, noise_lens

    @ staticmethod
    def _concat_batch(noise_batch, noise_lens, added_noise, added_lens):
        """Concatenate two noise batches of potentially different lengths"""

        # pad shorter batch to correct length
        noise_tensor_len = noise_batch.shape[1]
        added_tensor_len = added_noise.shape[1]
        pad = (0, abs(noise_tensor_len - added_tensor_len))
        if noise_tensor_len > added_tensor_len:
            added_noise = torch.nn.functional.pad(added_noise, pad)
            added_lens = added_lens * added_tensor_len / noise_tensor_len
        else:
            noise_batch = torch.nn.functional.pad(noise_batch, pad)
            noise_lens = noise_lens * noise_tensor_len / added_tensor_len

        noise_batch = torch.cat((noise_batch, added_noise))
        noise_lens = torch.cat((noise_lens, added_lens))

        return noise_batch, noise_lens

    def _load_noise_batch(self):
        """Load a batch of noises, restarting iteration if necessary."""

        try:
            # Don't necessarily know the key
            noises, lens = next(self.noise_data)

        except StopIteration:
            self.noise_data = iter(self.data_loader)
            noises, lens = next(self.noise_data)
        return noises, lens


class AddReverb(torch.nn.Module):
    """This class convolves an audio signal with an impulse response.

    Arguments
    ---------
    csv_file : str
        The name of a csv file containing the location of the
        impulse response files.
    sorting : str
        The order to iterate the csv file, from one of
        the following options: random, original, ascending, and descending.
    reverb_prob : float
        The chance that the audio signal will be reverbed.
        By default, every batch is reverbed.
    rir_scale_factor: float
        It compresses or dilates the given impulse response.
        If 0 < scale_factor < 1, the impulse response is compressed
        (less reverb), while if scale_factor > 1 it is dilated
        (more reverb).

    """

    def __init__(
        self,
        csv_file,
        sorting="random",
        reverb_prob=1.0,
        rir_scale_factor=1.0,

    ):
        super().__init__()
        self.csv_file = csv_file
        self.sorting = sorting
        self.reverb_prob = reverb_prob
        self.rir_scale_factor = rir_scale_factor

        # Create a data loader for the RIR waveforms
        dataset = NoiseDataset(
            self.csv_file, 
            sorting=self.sorting)
        shuffle = (self.sorting == "random")
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=shuffle)
            shuffle = False
        else:
            sampler = None
        self.data_loader = make_dataloader(
            dataset, shuffle=shuffle,
             sampler=sampler
        )
        self.rir_data = iter(self.data_loader)

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Don't add reverb (return early) 1-`reverb_prob` portion of the time
        if torch.rand(1) > self.reverb_prob:
            return waveforms.clone()

        # Add channels dimension if necessary
        channel_added = False
        if len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(-1)
            channel_added = True

        # Convert length from ratio to number of indices
        # lengths = (lengths * waveforms.shape[1])[:, None, None]

        # Load and prepare RIR
        rir_waveform = self._load_rir(waveforms)

        # Compress or dilate RIR
        if self.rir_scale_factor != 1:
            rir_waveform = F.interpolate(
                rir_waveform.transpose(1, -1),
                scale_factor=self.rir_scale_factor,
                mode="linear",
                align_corners=False,
            )
            rir_waveform = rir_waveform.transpose(1, -1)

        rev_waveform = reverberate(waveforms, rir_waveform, rescale_amp="avg")

        # Remove channels dimension if added
        if channel_added:
            return rev_waveform.squeeze(-1)

        return rev_waveform

    def _load_rir(self, waveforms):
        try:
            rir_waveform = next(self.rir_data)
        except StopIteration:
            self.rir_data = iter(self.data_loader)
            rir_waveform = next(self.rir_data)
        # Make sure RIR has correct channels
        if len(rir_waveform.shape) == 2:
            rir_waveform = rir_waveform.unsqueeze(-1)

        # Make sure RIR has correct type and device
        rir_waveform = rir_waveform.type(waveforms.dtype)
        return rir_waveform.to(waveforms.device)


class AddBabble(torch.nn.Module):
    """Add babble noise.

    Arguments
    ---------
    csv_file : str
        The name of a csv file containing the location of the
        noise audio files. If none is provided, white noise will be used.
    sorting : str
        The order to iterate the csv file, from one of the
        following options: random, original, ascending, and descending.
    num_workers : int
        Number of workers in the DataLoader (See PyTorch DataLoader docs).
    snr_low : int
        The low end of the mixing ratios, in decibels.
    snr_high : int
        The high end of the mixing ratios, in decibels.
    pad_noise : bool
        If True, copy noise signals that are shorter than
        their corresponding clean signals so as to cover the whole clean
        signal. Otherwise, leave the noise un-padded.
    mix_prob : float
        The probability that the batch of signals will be
        mixed with babble noise. By default, every signal is mixed.
    speaker_count : int
        The number of signals to mix with the original signal.  
    start_index : int
        The index in the noise waveforms to start from. By default, chooses
        a random index in [0, len(noise) - len(waveforms)].
    babble_noise_max_len : float
        Limit the length when reading the babble noise. 
    """

    def __init__(
        self,
        csv_file=None,
        add_filt_min=None,
        sorting="random",
        num_workers=0,
        snr_low=0,
        snr_high=0,
        pad_noise=False,
        mix_prob=1.0,
        speaker_count=3,
        start_index=None,
        babble_noise_max_len=2.0
    ):
        super().__init__()

        self.csv_file = csv_file
        self.add_filt_min = add_filt_min
        self.sorting = sorting
        self.num_workers = num_workers
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.pad_noise = pad_noise
        self.mix_prob = mix_prob
        self.speaker_count = speaker_count
        self.start_index = start_index
        self.babble_noise_max_len = babble_noise_max_len

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Copy clean waveform to initialize noisy waveform
        babbled_waveform = waveforms.clone()
        lengths = (lengths * waveforms.shape[1]).unsqueeze(1)

        # Don't add noise (return early) 1-`mix_prob` portion of the batches
        if torch.rand(1) > self.mix_prob:
            return babbled_waveform

        # Compute the average amplitude of the clean waveforms
        clean_amplitude = compute_amplitude(waveforms, lengths)

        # Pick an SNR and use it to compute the mixture amplitude factors
        SNR = torch.rand(len(waveforms), 1, device=waveforms.device)
        SNR = SNR * (self.snr_high - self.snr_low) + self.snr_low
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        # Scale clean signal appropriately
        babbled_waveform *= 1 - noise_amplitude_factor

        # Simulate babble noise by mixing the signals in a batch.
        if self.csv_file is None:
            babble_waveform = waveforms.roll((1,), dims=0)
            babble_len = lengths.roll((1,), dims=0)
            for i in range(1, self.speaker_count):
                babble_waveform += waveforms.roll((1 + i,), dims=0)
                babble_len = torch.max(
                    babble_len, babble_len.roll((1,), dims=0))
        # Add babble noise from wavs.
        else:
            tensor_length = waveforms.shape[1]
            original_babble_waveform, babble_len = self._load_noise(
                lengths, tensor_length)


            babble_waveform = original_babble_waveform.clone()
            for i in range(1, self.speaker_count):
                babble_waveform += original_babble_waveform.roll((i,), dims=0)
                babble_len = torch.max(
                    babble_len, babble_len.roll((1,), dims=0))

            babble_waveform = babble_waveform[:len(lengths)]
            babble_len = babble_len[:len(lengths)]

        # Rescale and add to mixture
        babble_amplitude = compute_amplitude(babble_waveform, babble_len)
        babble_waveform *= new_noise_amplitude / (babble_amplitude + 1e-14)
        babbled_waveform += babble_waveform

        return babbled_waveform

    def _load_noise(self, lengths, max_length):
        """Load a batch of noises"""

        lengths = torch.cat(
            (lengths, torch.zeros(self.speaker_count).unsqueeze(1)))
        lengths = lengths.long().squeeze(1)

        batch_size = len(lengths)

        # Load a noise batch
        if not hasattr(self, "data_loader"):
            # Set parameters based on input
            self.device = lengths.device

            # Create a data loader for the noise wavforms
            if self.csv_file is not None:
                dataset = NoiseDataset(
                    self.csv_file,
                    filt_min=self.add_filt_min,
                    sorting=self.sorting, 
                    max_len=self.babble_noise_max_len)
                shuffle = (self.sorting == "random")
                if torch.distributed.is_initialized():
                    sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset, shuffle=shuffle)
                    shuffle = False
                else:
                    sampler = None
                self.data_loader = make_dataloader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=self.num_workers,
                    shuffle=shuffle,
                    sampler=sampler,
                    collate_fn=batch_pad_right
                )

                self.noise_data = iter(self.data_loader)

        # Load noise to correct device
        noise_batch, noise_len = self._load_noise_batch_of_size(batch_size)
        noise_batch = noise_batch.to(lengths.device)
        noise_len = noise_len.to(lengths.device)

        # Convert relative length to an index
        noise_len = (noise_len * noise_batch.shape[1]).long()

        # Ensure shortest wav can cover speech signal
        # WARNING: THIS COULD BE SLOW IF THERE ARE VERY SHORT NOISES
        if self.pad_noise:
            while torch.any(noise_len < lengths):
                min_len = torch.min(noise_len)
                prepend = noise_batch[:, :min_len]
                noise_batch = torch.cat((prepend, noise_batch), axis=1)
                noise_len += min_len

        # Ensure noise batch is long enough
        elif noise_batch.size(1) < max_length:
            pad = max_length - noise_batch.size(1)
            left_padding = torch.randint(high = pad+1, size=(1,))[0]
            padding = (left_padding,pad-left_padding)
            noise_batch = torch.nn.functional.pad(noise_batch, padding)

        # Select a random starting location in the waveform
        start_index = self.start_index
        if self.start_index is None:
            start_index = 0
            max_chop = (noise_len - lengths).min().clamp(min=1)
            start_index = torch.randint(
                high=max_chop, size=(1,), device=lengths.device
            )

        # Truncate noise_batch to max_length
        noise_batch = noise_batch[:, start_index: start_index + max_length]
        noise_len = (
            noise_len - start_index).clamp(max=max_length).unsqueeze(1)

        return noise_batch, noise_len


    def _load_noise_batch_of_size(self, batch_size):
        """Concatenate noise batches, then chop to correct size"""

        noise_batch, noise_lens = self._load_noise_batch()

        # Expand
        while len(noise_batch) < batch_size:
            added_noise, added_lens = self._load_noise_batch()
            noise_batch, noise_lens = AddNoise._concat_batch(
                noise_batch, noise_lens, added_noise, added_lens
            )

        # Contract
        if len(noise_batch) > batch_size:
            noise_batch = noise_batch[:batch_size]
            noise_lens = noise_lens[:batch_size]

        return noise_batch, noise_lens


    def _load_noise_batch(self):
        """Load a batch of noises, restarting iteration if necessary."""

        try:
            # Don't necessarily know the key
            noises, lens = next(self.noise_data)
        except StopIteration:
            self.noise_data = iter(self.data_loader)
            noises, lens = next(self.noise_data)
        return noises, lens


class DropFreq(torch.nn.Module):
    """This class drops a random frequency from the signal.

    The purpose of this class is to teach models to learn to rely on all parts
    of the signal, not just a few frequency bands.

    Arguments
    ---------
    drop_freq_low : float
        The low end of frequencies that can be dropped,
        as a fraction of the sampling rate / 2.
    drop_freq_high : float
        The high end of frequencies that can be
        dropped, as a fraction of the sampling rate / 2.
    drop_count_low : int
        The low end of number of frequencies that could be dropped.
    drop_count_high : int
        The high end of number of frequencies that could be dropped.
    drop_width : float
        The width of the frequency band to drop, as
        a fraction of the sampling_rate / 2.
    drop_prob : float
        The probability that the batch of signals will  have a frequency
        dropped. By default, every batch has frequencies dropped.
    """

    def __init__(
        self,
        drop_freq_low=1e-14,
        drop_freq_high=1,
        drop_count_low=1,
        drop_count_high=2,
        drop_width=0.05,
        drop_prob=1,
    ):
        super().__init__()
        self.drop_freq_low = drop_freq_low
        self.drop_freq_high = drop_freq_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_width = drop_width
        self.drop_prob = drop_prob

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        dropped_waveform = waveforms.clone()
        if torch.rand(1) > self.drop_prob:
            return dropped_waveform

        # Add channels dimension
        if len(waveforms.shape) == 2:
            dropped_waveform = dropped_waveform.unsqueeze(-1)

        # Pick number of frequencies to drop
        drop_count = torch.randint(
            low=self.drop_count_low, high=self.drop_count_high + 1, size=(1,),
        )

        # Pick a frequency to drop
        drop_range = self.drop_freq_high - self.drop_freq_low
        drop_frequency = (
            torch.rand(drop_count) * drop_range + self.drop_freq_low
        )

        # Filter parameters
        filter_length = 101
        pad = filter_length // 2

        # Start with delta function
        drop_filter = torch.zeros(1, filter_length, 1, device=waveforms.device)
        drop_filter[0, pad, 0] = 1

        # Subtract each frequency
        for frequency in drop_frequency:
            notch_kernel = notch_filter(
                frequency, filter_length, self.drop_width,
            ).to(waveforms.device)
            drop_filter = convolve1d(drop_filter, notch_kernel, pad)

        # Apply filter
        dropped_waveform = convolve1d(dropped_waveform, drop_filter, pad)

        # Remove channels dimension if added
        return dropped_waveform.squeeze(-1)


class DropChunk(torch.nn.Module):
    """This class drops portions of the input signal.

    Using `DropChunk` as an augmentation strategy helps a models learn to rely
    on all parts of the signal, since it can't expect a given part to be
    present.

    Arguments
    ---------
    drop_length_low : int
        The low end of lengths for which to set the
        signal to zero, in samples.
    drop_length_high : int
        The high end of lengths for which to set the
        signal to zero, in samples.
    drop_count_low : int
        The low end of number of times that the signal
        can be dropped to zero.
    drop_count_high : int
        The high end of number of times that the signal
        can be dropped to zero.
    drop_start : int
        The first index for which dropping will be allowed.
    drop_end : int
        The last index for which dropping will be allowed.
    drop_prob : float
        The probability that the batch of signals will
        have a portion dropped. By default, every batch
        has portions dropped.
    noise_factor : float
        The factor relative to average amplitude of an utterance
        to use for scaling the white noise inserted. 1 keeps
        the average amplitude the same, while 0 inserts all 0's.


    """

    def __init__(
        self,
        drop_length_low=100,
        drop_length_high=1000,
        drop_count_low=1,
        drop_count_high=10,
        drop_start=0,
        drop_end=None,
        drop_prob=1,
        noise_factor=0.0,
    ):
        super().__init__()
        self.drop_length_low = drop_length_low
        self.drop_length_high = drop_length_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_start = drop_start
        self.drop_end = drop_end
        self.drop_prob = drop_prob
        self.noise_factor = noise_factor

        # Validate low < high
        if drop_length_low > drop_length_high:
            raise ValueError("Low limit must not be more than high limit")
        if drop_count_low > drop_count_high:
            raise ValueError("Low limit must not be more than high limit")

        # Make sure the length doesn't exceed end - start
        if drop_end is not None and drop_end >= 0:
            if drop_start > drop_end:
                raise ValueError("Low limit must not be more than high limit")

            drop_range = drop_end - drop_start
            self.drop_length_low = min(drop_length_low, drop_range)
            self.drop_length_high = min(drop_length_high, drop_range)

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or
            `[batch, time, channels]`
        """

        # Reading input list
        lengths = (lengths * waveforms.size(1)).long()
        batch_size = waveforms.size(0)
        dropped_waveform = waveforms.clone()

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        if torch.rand(1) > self.drop_prob:
            return dropped_waveform

        # Store original amplitude for computing white noise amplitude
        clean_amplitude = compute_amplitude(waveforms, lengths.unsqueeze(1))

        # Pick a number of times to drop
        drop_times = torch.randint(
            low=self.drop_count_low,
            high=self.drop_count_high + 1,
            size=(batch_size,),
        )

        # Iterate batch to set mask
        for i in range(batch_size):
            if drop_times[i] == 0:
                continue

            # Pick lengths
            length = torch.randint(
                low=self.drop_length_low,
                high=self.drop_length_high + 1,
                size=(drop_times[i],),
            )

            # Compute range of starting locations
            start_min = self.drop_start
            if start_min < 0:
                start_min += lengths[i]
            start_max = self.drop_end
            if start_max is None:
                start_max = lengths[i]
            if start_max < 0:
                start_max += lengths[i]
            start_max = max(0, start_max - length.max())

            # Pick starting locations
            start = torch.randint(
                low=start_min, high=start_max + 1, size=(drop_times[i],),
            )

            end = start + length

            # Update waveform
            if not self.noise_factor:
                for j in range(drop_times[i]):
                    dropped_waveform[i, start[j]: end[j]] = 0.0
            else:
                # Uniform distribution of -2 to +2 * avg amplitude should
                # preserve the average for normalization
                noise_max = 2 * clean_amplitude[i] * self.noise_factor
                for j in range(drop_times[i]):
                    # zero-center the noise distribution
                    noise_vec = torch.rand(length[j], device=waveforms.device)
                    noise_vec = 2 * noise_max * noise_vec - noise_max
                    dropped_waveform[i, start[j]: end[j]] = noise_vec

        return dropped_waveform

class RandomChunk(torch.nn.Module):
    """Get segment.
    Arguments
    ---------
    chunk_len : float
        Get segment of utts, in senconds (s).
    sample_rate : int
        the sampling frequency of the input signal.
    """
    def __init__(
        self, 
        random_chunk=False,
        chunk_len=2.015, 
        sample_rate=16000,
    ):
        super().__init__()
        self.random_chunk=random_chunk
        self.lens = int(chunk_len*sample_rate)
    def forward(self, waveforms,lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.
        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """
        if not self.random_chunk:
            return waveforms,lengths
        lengths=(lengths * waveforms.shape[1])  # [B]
        shape = list(waveforms.shape)
        shape[1] = self.lens
        chunk_sig = torch.zeros(shape,device=lengths.device)
        for i in range(shape[0]):
            if lengths[i] > self.lens:
                max_chop = (lengths[i] - self.lens).long()
                start_index = torch.randint(
                    high=max_chop, size=(1,))
                chunk_sig[i] = waveforms[i,start_index: start_index + self.lens]  
            else:
                repeat_num = math.ceil(self.lens/lengths[i])
                chunk_sig[i:i+1] = waveforms[i:i+1,: ].repeat(1,repeat_num)[:,:self.lens]             
        lengths = torch.ones(shape[0])
        if chunk_sig.shape!=48240:
            print(chunk_sig.shape)
        return chunk_sig, lengths


class DoClip(torch.nn.Module):
    """This function mimics audio clipping by clamping the input tensor.

    Arguments
    ---------
    clip_low : float
        The low end of amplitudes for which to clip the signal.
    clip_high : float
        The high end of amplitudes for which to clip the signal.
    clip_prob : float
        The probability that the batch of signals will have a portion clipped.
        By default, every batch has portions clipped.


    """

    def __init__(
        self, clip_low=0.5, clip_high=1, clip_prob=1,
    ):
        super().__init__()
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.clip_prob = clip_prob

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """

        # Don't clip (return early) 1-`clip_prob` portion of the batches
        if torch.rand(1) > self.clip_prob:
            return waveforms.clone()

        # Randomly select clip value
        clipping_range = self.clip_high - self.clip_low
        clip_value = torch.rand(1,)[0] * clipping_range + self.clip_low

        # Apply clipping
        clipped_waveform = waveforms.clamp(-clip_value, clip_value)

        return clipped_waveform

class SoxEffectTransform(torch.nn.Module):
        effects: List[List[str]]

        def __init__(self, effects: List[List[str]],sample_rate:int):
            
            super().__init__()
            self.effects = effects
            self.sample_rate = sample_rate
        def forward(self, waveforms: torch.Tensor):
            """
            Arguments
            ---------
            waveforms : tensor
                Shape should be `[batch, time]` or `[batch, time, channels]`.
            Returns
            -------
            Tensor of shape `[batch, time]` or `[batch, time, channels]`.
            """

            wavs = []
            if self.effects == [[]]:
                return waveforms
            unsqueezed = False
            if len(waveforms.shape)==2:
                # add channel
                waveforms=waveforms.unsqueeze(1)
                unsqueezed = True
            else:
                waveforms = waveforms.transpose(1, 2) 
            for i,wav in enumerate(waveforms):

                wav,_ = torchaudio.sox_effects.apply_effects_tensor(wav, self.sample_rate, self.effects)
                
                wavs.append(wav.unsqueeze(0))
            wavs = torch.cat(wavs,dim=0)


            if unsqueezed:
                wavs=wavs.squeeze(1)
            else:
                wavs=wavs.transpose(1,2)
            return wavs

class SpeedPerturb(torch.nn.Module):
    """Slightly speed up or slow down an audio signal.

    Resample the audio signal at a rate that is similar to the original rate,
    to achieve a slightly slower or slightly faster signal. This technique is
    outlined in the paper: "Audio Augmentation for Speech Recognition"

    Arguments
    ---------
    orig_freq : int
        The frequency of the original signal.
    speeds : list
        A set of different speeds to use to perturb each batch. larger -> slower.
    perturb_prob : float
        The chance that the batch will be speed-
        perturbed. By default, every batch is perturbed.


    """

    def __init__(
        self, orig_freq, speeds=[95, 100, 105], perturb_prob=1.0, keep_shape=True, perturb_type='resample', change_spk=False,spk_num=0
    ):
        super().__init__()
        assert perturb_type in ['resample','sox_speed','sox_tempo']

        self.orig_freq = orig_freq
        self.speeds = speeds
        self.perturb_prob = perturb_prob
        self.keep_shape = keep_shape
        self.change_spk = change_spk

        if change_spk:
            assert spk_num>0, "change_spk need total spk number."

            self.aug_spks = self._speed_to_speaker(speeds)
            self.aug_spks = [spk_num*aug_spk for aug_spk in self.aug_spks]

        # Initialize resamplers
        self.speeders = []
        for speed in self.speeds:
            if perturb_type == 'resample':

                config = {
                    "orig_freq": self.orig_freq,
                    "new_freq": self.orig_freq * speed // 100,
                }
                self.speeders.append(Resample(**config))
            else:
                
                if perturb_type == 'sox_speed':
                    if speed==100:
                        effects = [[]]
                    else:
                        speed = round(100/speed,2)
                        effects = [['speed',str(speed)],['rate',str(orig_freq)]]

                elif perturb_type == 'sox_tempo':
                    if speed==100:
                        effects = [[]]
                    else:
                        speed = round(100/speed,2)
                        effects = [['tempo', str(speed)]]
                else:
                    raise ValueError("unsupport perturb_type: {}".format(perturb_type))
                self.speeders.append(SoxEffectTransform(effects,orig_freq))
    def forward(self, waveform: torch.Tensor, spk_id: torch.Tensor=torch.ones((0), dtype=torch.long)):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.
        spk_id: tensor
            Shape should be a single dimension, `[batch]`

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Don't perturb (return early) 1-`perturb_prob` portion of the batches
        if torch.rand(1) > self.perturb_prob:
            return waveform.clone(),spk_id

        # Perform a random perturbation
        speed_index = torch.randint(len(self.speeds), (1,))[0]
        perturbed_waveform = self.speeders[speed_index](waveform)
        if self.change_spk:
            spk_id = self.aug_spks[speed_index]+ spk_id

        if self.keep_shape:
            # Managing speed change
            if perturbed_waveform.shape[1] > waveform.shape[1]:
                perturbed_waveform = perturbed_waveform[:,
                                                        0: waveform.shape[1]]
            else:
                zero_sig = torch.zeros_like(waveform)
                zero_sig[:, 0: perturbed_waveform.shape[1]
                         ] = perturbed_waveform
                perturbed_waveform = zero_sig
        return perturbed_waveform,spk_id

    def get_spkid_aug(self):
        sp_aug,spkid_aug =1,1
        if self.perturb_prob>0:
            sp_aug = len(set(self.speeds))
            if self.change_spk:
                spkid_aug=len(set(self.speeds))
        return spkid_aug,sp_aug

    def _speed_to_speaker(self,speeds):
        assert 100 in speeds, "speed perturb with speaker aug need origin speed."
        t = {}
        spk_cont = 0
        for s in sorted(set(speeds),key = speeds.index):
            if s ==100:
                t[s]=0
            else:
                spk_cont+=1
                t[s]=spk_cont
        return [t[sp] for sp in speeds]


class Resample(torch.nn.Module):
    """This class resamples an audio signal using sinc-based interpolation.

    It is a modification of the `resample` function from torchaudio
    (https://pytorch.org/audio/transforms.html#resample)

    Arguments
    ---------
    orig_freq : int
        the sampling frequency of the input signal.
    new_freq : int
        the new sampling frequency after this operation is performed.
    lowpass_filter_width : int
        Controls the sharpness of the filter, larger numbers result in a
        sharper filter, but they are less efficient. Values from 4 to 10 are
        allowed.

    Example
    -------

    >>> signal = torch.randn(52173)
    >>> signal = signal.unsqueeze(0) # [batch, time, channels]
    >>> resampler = Resample(orig_freq=16000, new_freq=8000)
    >>> resampled = resampler(signal)
    >>> signal.shape
    torch.Size([1, 52173])
    >>> resampled.shape
    torch.Size([1, 26087])
    """

    def __init__(
        self, orig_freq=16000, new_freq=16000, lowpass_filter_width=6,
    ):
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.lowpass_filter_width = lowpass_filter_width

        # Compute rate for striding
        self._compute_strides()
        assert self.orig_freq % self.conv_stride == 0
        assert self.new_freq % self.conv_transpose_stride == 0

    def _compute_strides(self):
        """Compute the phases in polyphase filter.

        (almost directly from torchaudio.compliance.kaldi)
        """

        # Compute new unit based on ratio of in/out frequencies
        base_freq = math.gcd(self.orig_freq, self.new_freq)
        input_samples_in_unit = self.orig_freq // base_freq
        self.output_samples = self.new_freq // base_freq

        # Store the appropriate stride based on the new units
        self.conv_stride = input_samples_in_unit
        self.conv_transpose_stride = self.output_samples

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        if not hasattr(self, "first_indices"):
            self._indices_and_weights(waveforms)

        # Don't do anything if the frequencies are the same
        if self.orig_freq == self.new_freq:
            return waveforms

        unsqueezed = False
        if len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(1)
            unsqueezed = True
        elif len(waveforms.shape) == 3:
            waveforms = waveforms.transpose(1, 2)
        else:
            raise ValueError("Input must be 2 or 3 dimensions")

        # Do resampling
        resampled_waveform = self._perform_resample(waveforms)

        if unsqueezed:
            resampled_waveform = resampled_waveform.squeeze(1)
        else:
            resampled_waveform = resampled_waveform.transpose(1, 2)

        return resampled_waveform

    def _perform_resample(self, waveforms):
        """Resamples the waveform at the new frequency.

        This matches Kaldi's OfflineFeatureTpl ResampleWaveform which uses a
        LinearResample (resample a signal at linearly spaced intervals to
        up/downsample a signal). LinearResample (LR) means that the output
        signal is at linearly spaced intervals (i.e the output signal has a
        frequency of `new_freq`). It uses sinc/bandlimited interpolation to
        upsample/downsample the signal.

        (almost directly from torchaudio.compliance.kaldi)

        https://ccrma.stanford.edu/~jos/resample/
        Theory_Ideal_Bandlimited_Interpolation.html

        https://github.com/kaldi-asr/kaldi/blob/master/src/feat/resample.h#L56

        Arguments
        ---------
        waveforms : tensor
            The batch of audio signals to resample.

        Returns
        -------
        The waveforms at the new frequency.
        """

        # Compute output size and initialize
        batch_size, num_channels, wave_len = waveforms.size()
        window_size = self.weights.size(1)
        tot_output_samp = self._output_samples(wave_len)
        resampled_waveform = torch.zeros(
            (batch_size, num_channels, tot_output_samp),
            device=waveforms.device,
        )
        self.weights = self.weights.to(waveforms.device)

        # Check weights are on correct device
        if waveforms.device != self.weights.device:
            self.weights = self.weights.to(waveforms.device)

        # eye size: (num_channels, num_channels, 1)
        eye = torch.eye(num_channels, device=waveforms.device).unsqueeze(2)

        # Iterate over the phases in the polyphase filter
        for i in range(self.first_indices.size(0)):
            wave_to_conv = waveforms
            first_index = int(self.first_indices[i].item())
            if first_index >= 0:
                # trim the signal as the filter will not be applied
                # before the first_index
                wave_to_conv = wave_to_conv[..., first_index:]

            # pad the right of the signal to allow partial convolutions
            # meaning compute values for partial windows (e.g. end of the
            # window is outside the signal length)
            max_index = (tot_output_samp - 1) // self.output_samples
            end_index = max_index * self.conv_stride + window_size
            current_wave_len = wave_len - first_index
            right_padding = max(0, end_index + 1 - current_wave_len)
            left_padding = max(0, -first_index)
            wave_to_conv = torch.nn.functional.pad(
                wave_to_conv, (left_padding, right_padding)
            )
            conv_wave = torch.nn.functional.conv1d(
                input=wave_to_conv,
                weight=self.weights[i].repeat(num_channels, 1, 1),
                stride=self.conv_stride,
                groups=num_channels,
            )

            # we want conv_wave[:, i] to be at
            # output[:, i + n*conv_transpose_stride]
            dilated_conv_wave = torch.nn.functional.conv_transpose1d(
                conv_wave, eye, stride=self.conv_transpose_stride
            )

            # pad dilated_conv_wave so it reaches the output length if needed.
            left_padding = i
            previous_padding = left_padding + dilated_conv_wave.size(-1)
            right_padding = max(0, tot_output_samp - previous_padding)
            dilated_conv_wave = torch.nn.functional.pad(
                dilated_conv_wave, (left_padding, right_padding)
            )
            dilated_conv_wave = dilated_conv_wave[..., :tot_output_samp]

            resampled_waveform += dilated_conv_wave

        return resampled_waveform

    def _output_samples(self, input_num_samp):
        """Based on LinearResample::GetNumOutputSamples.

        LinearResample (LR) means that the output signal is at
        linearly spaced intervals (i.e the output signal has a
        frequency of ``new_freq``). It uses sinc/bandlimited
        interpolation to upsample/downsample the signal.

        (almost directly from torchaudio.compliance.kaldi)

        Arguments
        ---------
        input_num_samp : int
            The number of samples in each example in the batch.

        Returns
        -------
        Number of samples in the output waveform.
        """

        # For exact computation, we measure time in "ticks" of 1.0 / tick_freq,
        # where tick_freq is the least common multiple of samp_in and
        # samp_out.
        samp_in = int(self.orig_freq)
        samp_out = int(self.new_freq)

        tick_freq = abs(samp_in * samp_out) // math.gcd(samp_in, samp_out)
        ticks_per_input_period = tick_freq // samp_in

        # work out the number of ticks in the time interval
        # [ 0, input_num_samp/samp_in ).
        interval_length = input_num_samp * ticks_per_input_period
        if interval_length <= 0:
            return 0
        ticks_per_output_period = tick_freq // samp_out

        # Get the last output-sample in the closed interval,
        # i.e. replacing [ ) with [ ]. Note: integer division rounds down.
        # See http://en.wikipedia.org/wiki/Interval_(mathematics) for an
        # explanation of the notation.
        last_output_samp = interval_length // ticks_per_output_period

        # We need the last output-sample in the open interval, so if it
        # takes us to the end of the interval exactly, subtract one.
        if last_output_samp * ticks_per_output_period == interval_length:
            last_output_samp -= 1

        # First output-sample index is zero, so the number of output samples
        # is the last output-sample plus one.
        num_output_samp = last_output_samp + 1

        return num_output_samp

    def _indices_and_weights(self, waveforms):
        """Based on LinearResample::SetIndexesAndWeights

        Retrieves the weights for resampling as well as the indices in which
        they are valid. LinearResample (LR) means that the output signal is at
        linearly spaced intervals (i.e the output signal has a frequency
        of ``new_freq``). It uses sinc/bandlimited interpolation to
        upsample/downsample the signal.

        Returns
        -------
        - the place where each filter should start being applied
        - the filters to be applied to the signal for resampling
        """

        # Lowpass filter frequency depends on smaller of two frequencies
        min_freq = min(self.orig_freq, self.new_freq)
        lowpass_cutoff = 0.99 * 0.5 * min_freq

        assert lowpass_cutoff * 2 <= min_freq
        window_width = self.lowpass_filter_width / (2.0 * lowpass_cutoff)

        assert lowpass_cutoff < min(self.orig_freq, self.new_freq) / 2
        output_t = torch.arange(
            start=0.0, end=self.output_samples, device=waveforms.device,
        )
        output_t /= self.new_freq
        min_t = output_t - window_width
        max_t = output_t + window_width

        min_input_index = torch.ceil(min_t * self.orig_freq)
        max_input_index = torch.floor(max_t * self.orig_freq)
        num_indices = max_input_index - min_input_index + 1

        max_weight_width = num_indices.max()
        j = torch.arange(max_weight_width, device=waveforms.device)
        input_index = min_input_index.unsqueeze(1) + j.unsqueeze(0)
        delta_t = (input_index / self.orig_freq) - output_t.unsqueeze(1)

        weights = torch.zeros_like(delta_t)
        inside_window_indices = delta_t.abs().lt(window_width)

        # raised-cosine (Hanning) window with width `window_width`
        weights[inside_window_indices] = 0.5 * (
            1
            + torch.cos(
                2
                * math.pi
                * lowpass_cutoff
                / self.lowpass_filter_width
                * delta_t[inside_window_indices]
            )
        )

        t_eq_zero_indices = delta_t.eq(0.0)
        t_not_eq_zero_indices = ~t_eq_zero_indices

        # sinc filter function
        weights[t_not_eq_zero_indices] *= torch.sin(
            2 * math.pi * lowpass_cutoff * delta_t[t_not_eq_zero_indices]
        ) / (math.pi * delta_t[t_not_eq_zero_indices])

        # limit of the function at t = 0
        weights[t_eq_zero_indices] *= 2 * lowpass_cutoff

        # size (output_samples, max_weight_width)
        weights /= self.orig_freq

        self.first_indices = min_input_index
        self.weights = weights


class EnvCorrupt(torch.nn.Module):
    """Speech augment for speech signals: noise, reverb, babble.

    Arguments
    ---------
    reverb_prob : float from 0 to 1
        The probability that each batch will have reverberation applied.
    noise_prob : float from 0 to 1
        The probability that each batch will have noise added.
    babble_prob : float from 0 to 1
        The probability that each batch will have babble noise added.
    reverb_csv : str
        A prepared csv file for loading room impulse responses.
    noise_csv : str
        A prepared csv file for loading noise data, if None, means white noise.
    babble_csv : str
        A prepared csv file for loading babble data, if None, means simulated babble noise.
    add_filt_min : float
        Filt the short noises in when loading noises and babble from csv.
    noise_num_workers : int
        Number of workers to use for loading noises.   
    babble_speaker_count : int
        Number of speakers to use for babble. Must be less than batch size if not use babble_csv but simulate babble noise by the batch data itself.
    babble_snr_low : int
        Lowest generated SNR of reverbed signal to babble.
    babble_snr_high : int
        Highest generated SNR of reverbed signal to babble.
    noise_snr_low : int
        Lowest generated SNR of babbled signal to noise.
    noise_snr_high : int
        Highest generated SNR of babbled signal to noise.
    rir_scale_factor : float
        It compresses or dilates the given impulse response.
        If ``0 < rir_scale_factor < 1``, the impulse response is compressed
        (less reverb), while if ``rir_scale_factor > 1`` it is dilated
        (more reverb).

    Example
    -------
    >>> inputs = torch.randn([10, 16000])
    >>> corrupter = EnvCorrupt(babble_speaker_count=9)
    >>> feats = corrupter(inputs, torch.ones(10))
    """

    def __init__(
        self,
        reverb_prob=1.0,
        noise_prob=1.0,
        babble_prob=1.0,
        reverb_csv=None,
        noise_csv=None,
        babble_csv=None,
        add_filt_min = None,
        babble_noise_max_len=2.0,
        noise_num_workers=0,
        babble_speaker_count=0,
        babble_snr_low=13,
        babble_snr_high=20,
        noise_snr_low=0,
        noise_snr_high=15,
        pad_noise = False,
        rir_scale_factor=1.0,        
        **ops
    ):
        super().__init__()

        # Initialize corrupters
        if reverb_csv is not None and reverb_prob > 0.0:
            self.add_reverb = AddReverb(
                reverb_prob=reverb_prob,
                csv_file=reverb_csv,
                rir_scale_factor=rir_scale_factor,
            )

        if babble_speaker_count > 0 and babble_prob > 0.0:
            self.add_babble = AddBabble(
                mix_prob=babble_prob,
                csv_file=babble_csv,
                add_filt_min = add_filt_min,
                num_workers=noise_num_workers,
                speaker_count=babble_speaker_count,
                snr_low=babble_snr_low,
                snr_high=babble_snr_high,
                babble_noise_max_len=babble_noise_max_len,
                pad_noise=pad_noise,
            )

        if noise_prob > 0.0:
            self.add_noise = AddNoise(
                mix_prob=noise_prob,
                csv_file=noise_csv,
                add_filt_min = add_filt_min,
                num_workers=noise_num_workers,
                snr_low=noise_snr_low,
                snr_high=noise_snr_high,
                pad_noise=pad_noise,
            )

    def forward(self, waveforms, lengths,spk_id:torch.ones((0),dtype=torch.long)):
        """Returns the distorted waveforms.

        Arguments
        ---------
        waveforms : torch.Tensor
            The waveforms to distort.
        """
        # Augmentation
        with torch.no_grad():
            if hasattr(self, "add_reverb"):
                try:
                    waveforms = self.add_reverb(waveforms, lengths)

                except Exception:
                    pass
            if hasattr(self, "add_babble"):
                waveforms = self.add_babble(waveforms, lengths)
            if hasattr(self, "add_noise"):
                waveforms = self.add_noise(waveforms, lengths)

        return waveforms,lengths, spk_id


class TimeDomainSpecAugment(torch.nn.Module):
    """A time-domain approximation of the SpecAugment algorithm.

    This augmentation module implements three augmentations in
    the time-domain.

     1. Drop chunks of the audio (zero amplitude or white noise)
     2. RandomChunk selection.
     3. Drop frequency bands (with band-drop filters)
     4. Speed peturbation (via resampling to slightly different rate)

    Arguments
    ---------
    perturb_prob : float from 0 to 1
        The probability that a batch will have speed perturbation applied.
    drop_freq_prob : float from 0 to 1
        The probability that a batch will have frequencies dropped.
    drop_chunk_prob : float from 0 to 1
        The probability that a batch will have chunks dropped.
    speeds : list of ints
        A set of different speeds to use to perturb each batch. larger -> slower. 
    spk_num: int
        The total speker num, for aug spkid if needed.
    perturb_type: str
        ['resample','sox_speed','sox_tempo']
    change_spk: bool
        Whether aug spkid
    keep_shape: bool
        keep time length after speed perturb.
    random_chunk: bool
        random select chunks after speed perturb.
    ramddom_chunsize:
        random chunks length in seconds (s).        
    sample_rate : int
        Sampling rate of the input waveforms.
    drop_freq_count_low : int
        Lowest number of frequencies that could be dropped.
    drop_freq_count_high : int
        Highest number of frequencies that could be dropped.
    drop_chunk_count_low : int
        Lowest number of chunks that could be dropped.
    drop_chunk_count_high : int
        Highest number of chunks that could be dropped.
    drop_chunk_length_low : int
        Lowest length of chunks that could be dropped.
    drop_chunk_length_high : int
        Highest length of chunks that could be dropped.
    drop_chunk_noise_factor : float
        The noise factor used to scale the white noise inserted, relative to
        the average amplitude of the utterance. Default 0 (no noise inserted).

    Example
    -------
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = TimeDomainSpecAugment(speeds=[80])
    >>> feats = feature_maker(inputs, torch.ones(10))
    >>> feats.shape
    torch.Size([10, 12800])
    """

    def __init__(
        self,
        perturb_prob=1.0,
        drop_freq_prob=0.0,
        drop_chunk_prob=0.0,
        speeds=[95, 100, 110],
        spk_num=0,
        perturb_type='resample', 
        change_spk=False,
        keep_shape=True,
        random_chunk=False,
        ramddom_chunsize=2.015,
        sample_rate=16000,
        drop_freq_count_low=0,
        drop_freq_count_high=3,
        drop_chunk_count_low=0,
        drop_chunk_count_high=5,
        drop_chunk_length_low=1000,
        drop_chunk_length_high=2000,
        drop_chunk_noise_factor=0,
        **ops
    ):
        super().__init__()
        self.speed_perturb = SpeedPerturb(
            perturb_prob=perturb_prob, 
            orig_freq=sample_rate, 
            speeds=speeds, 
            spk_num=spk_num,
            perturb_type=perturb_type, 
            change_spk=change_spk,
            keep_shape=keep_shape
        )
        self.random_chunk = RandomChunk(
            random_chunk = random_chunk,
            chunk_len = ramddom_chunsize,
            sample_rate = sample_rate
        )
        self.drop_freq = DropFreq(
            drop_prob=drop_freq_prob,
            drop_count_low=drop_freq_count_low,
            drop_count_high=drop_freq_count_high,
        )
        self.drop_chunk = DropChunk(
            drop_prob=drop_chunk_prob,
            drop_count_low=drop_chunk_count_low,
            drop_count_high=drop_chunk_count_high,
            drop_length_low=drop_chunk_length_low,
            drop_length_high=drop_chunk_length_high,
            noise_factor=drop_chunk_noise_factor,
        )

    def forward(self, waveforms, lengths, spk_id:torch.ones((0),dtype=torch.long)):
        """Returns the distorted waveforms.

        Arguments
        ---------
        waveforms : torch.Tensor
            The waveforms to distort
        """
        # Augmentation
        with torch.no_grad():
            waveforms,spk_id = self.speed_perturb(waveforms,spk_id)
            waveforms,lengths = self.random_chunk(waveforms,lengths)
            waveforms = self.drop_freq(waveforms)
            waveforms = self.drop_chunk(waveforms, lengths)

        return waveforms,lengths,spk_id

    def get_spkid_aug(self):

        return self.speed_perturb.get_spkid_aug()




class SpeechAug(torch.nn.Module):
    """This class implement three types of speech augment (chain,random,concat).

    Arguments
    --------- 
    aug_classes: list
        A list of aug_classes which contains its config information and sequence.      
    mod: str
        Speech augment pipline type, random means random select one augment from the list (NOTE: contains a clean wav automaticaly),
                                     concat means concat all the augment classes,
                                     chain means sequencely apply augment subject to the aug_classes list. 


    Example
    -------
    aug_classes=[{'aug_name': 'augment_speed', 'aug_type': 'Time', 
                  'random_mod_weight': 1, 'perturb_prob': 1.0, 
                  'drop_freq_prob': 0.0, 'drop_chunk_prob': 0.0, 
                  'sample_rate': 16000, 'speeds': [95, 100, 105], 
                  'keep_shape': True},
                 {'aug_name': 'augment_wavedrop', 'aug_type': 'Time', 
                  'random_mod_weight': 1, 'perturb_prob': 0.0, 
                  'drop_freq_prob': 1.0, 'drop_chunk_prob': 1.0, 
                  'sample_rate': 16000, 'speeds': [100]}]
    speech_aug = SpeechAug(aug_classes)
    signal = torch.randn(52173).unsqueeze(0)
    signal, lens = speech_aug(signal,torch.ones(1))
    """

    def __init__(self, spk_num=0,aug_classes=[], mod="random"):
        super().__init__()
        assert mod in ["random", "concat", "chain"]
        self.mod = mod
        self.augment = []
        self.augment_name = []
        self.spk_num=spk_num
        # define a weight of clean wav type
        random_weights = [1] if self.mod == 'random' else []
        self.spkid_aug,self.sp_aug = 1,1
        spt_num=0
        for aug_class in aug_classes:

            assert 'aug_type' in aug_class
            assert aug_class['aug_type'] in ['Env', 'Time']
            aug_type = aug_class['aug_type']
            del aug_class['aug_type']
            if 'aug_name' in aug_class:
                self.augment_name.append(aug_class['aug_name'])
                del aug_class['aug_name']
            else:
                self.augment_name.append('Anonymousness_aug')

            if self.mod == 'random':
                random_weight = 1.0
                if "random_mod_weight" in aug_class:
                    random_weight = float(aug_class['random_mod_weight'])
                    del aug_class['random_mod_weight']
                random_weights.append(random_weight)
            else:
                aug_class.pop('random_mod_weight', None)

            if aug_type == 'Env':
                self.augment.append(EnvCorrupt(**aug_class))
            if aug_type == 'Time':
                td_aug = TimeDomainSpecAugment(spk_num=spk_num,**aug_class)
                spkid_aug,sp_aug = td_aug.get_spkid_aug()
                spt_num+=(int(spkid_aug>1))
                
                if spt_num > 1:
                    raise ValueError("multi speaker id perturb setting, check your speech aug config")
                if spkid_aug>1:
                    self.spkid_aug = spkid_aug
                self.augment.append(td_aug)
                

        self.random_weight = torch.tensor(
            random_weights, dtype=torch.float) if random_weights else None
        self.print_augment()


    def forward(self, waveforms, lengths, spkid: torch.Tensor=torch.ones((0),dtype=torch.long)):
        if not self.augment:
            return waveforms, lengths, spkid
            
        if self.mod == 'random':
            return self._random_forward(waveforms, lengths,spkid)
        elif self.mod == 'chain':
            return self._chain_forward(waveforms, lengths,spkid)
        else:
            return self._concat_forward(waveforms, lengths,spkid)

    def _random_forward(self, waveforms, lengths,spkid):
        aug_idx = torch.multinomial(self.random_weight, 1)[0]

        if aug_idx == 0:
            return waveforms, lengths,spkid
        else:
            waves,lengths,spkid=self.augment[aug_idx-1](waveforms, lengths,spkid)


            if(torch.any((torch.isnan(waves)))):
                raise ValueError('random_1:{},type:{},typename:{}'.format(waveforms,self.augment[aug_idx-1],self.augment_name[aug_idx-1]))
            return waves, lengths, spkid

    def _concat_forward(self, waveforms, lengths,spkid):
        wavs_aug_tot = []
        spkids=[]
        lens = []
        wavs_aug_tot.append(waveforms.clone())
        spkids.append(spkid)
        lens.append(lengths)
        for count, augment in enumerate(self.augment):
            wavs_aug,len,spkid_a = augment(waveforms, lengths,spkid)

            if(torch.any((torch.isnan(wavs_aug)))):
                raise ValueError('concat:{},type:{},typename:{}'.format(waveforms,self.augment[count],self.augment_name[count]))
            wavs_aug_tot.append(wavs_aug)
            spkids.append(spkid_a)
            lens.append(len)
        waveforms = torch.cat(wavs_aug_tot, dim=0)
        lens = torch.cat(lens)
        spkids = torch.cat(spkids, dim=0)
        return waveforms, lens, spkids

    def _chain_forward(self, waveforms, lengths,spkid):

        for count, augment in enumerate(self.augment):
            waveforms,lengths,spkid = augment(waveforms, lengths,spkid)


            if(torch.any((torch.isnan(waveforms)))):
                raise ValueError('chian:{},type:{},typename:{}'.format(waveforms,self.augment[count],self.augment_name[count]))

        return waveforms, lengths,spkid

    
    def get_num_concat(self):
        if self.mod=='concat':
            return len(self.augment_name)+1
        else:
            return 1

    def get_spkid_aug(self):

        return self.spkid_aug
        


    def print_augment(self):
        if self.augment:
            print('speech augment type is {}.'.format(self.mod))
            aug_dict=dict(zip(self.augment_name,self.augment))
            for i,k in enumerate(aug_dict.items()):
                print('({}) {}:  {}'.format(i,k[0],k[1]))
        else:
            print('no speech augment applied')
