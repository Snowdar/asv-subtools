# Copyright xmuspeech (Author: Leo 2021-09-06)

# Importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from .augmentation import *



class InputSequenceNormalization(object):
    """Performs mean and variance normalization of the input tensor.

    Arguments
    ---------
    mean_norm : True
         If True, the mean will be normalized.
    std_norm : True
         If True, the standard deviation will be normalized.

    Example
    -------
    >>> import torch
    >>> norm = InputSequenceNormalization()
    >>> input = torch.randn([101, 20])
    >>> feature = norm(inputs)
    """


    def __init__(
        self,
        mean_norm=True,
        std_norm=False,
    ):
        super().__init__()
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.eps = 1e-10
    def __call__(self, x):
        """Returns the tensor with the surrounding context.

        Arguments
        ---------
        x : tensor
            A tensor `[t,f]`.

        
        """
        if self.mean_norm:
            mean = torch.mean(x, dim=0).detach().data
        else:
            mean = torch.tensor([0.0], device=x.device)

        # Compute current std
        if self.std_norm:
            std = torch.std(x, dim=0).detach().data
        else:
            std = torch.tensor([1.0], device=x.device)

        # Improving numerical stability of std
        std = torch.max(
            std, self.eps * torch.ones_like(std)
        )
        x = (x - mean.data) / std.data


        return x



class KaldiFeature(object):
    """ This class extract features as kaldi's compute-mfcc-feats.

    Arguments
    ---------
    feat_type: str (fbank or mfcc).
    feature_extraction_conf: dict
        The config according to the kaldi's feature config.   
    mean_var_conf: dict
        Mean std norm config `{'mean_norm':True, 'std_norm':False}`.
       
    """

    def __init__(self,feature_type='mfcc',kaldi_featset={},mean_var_conf={}):
        super().__init__()
        assert feature_type in ['mfcc','fbank']
        self.feat_type=feature_type

        self.kaldi_featset=kaldi_featset
        if self.feat_type=='mfcc':
            self.extract=kaldi.mfcc
        else:
            self.extract=kaldi.fbank
        if mean_var_conf:
            self.mean_var=InputSequenceNormalization(**mean_var_conf)                      
        else:
            self.mean_var=nn.Identity()

    def __call__(self,waveforms,lengths=None):
        """Return the list of feature mat.
        Arguments
        ---------
        waveforms : torch.Tensor
            Should be `[batch, time]` or `[batch, time, channel]`
        lens: torch.Tensor
            Should be `[batch]`,  relative value lengths.

        Returns:
        ---------
            A list of feats mat
        """
        feats=[]

        with torch.no_grad():
            if lengths is not None:
                lens=lengths*waveforms.shape[1]
                
            if(torch.any((torch.isnan(waveforms)))):
                raise ValueError('feats:{}'.format(waveforms))
            for i,wav in enumerate(waveforms):

                if len(wav.shape)==1:
                    # add channel
                    wav=wav.unsqueeze(0)
                else:
                    wav=wav.transpose(0,1)

                if lengths is not None:                
                    wav= wav[:,:lens[i].long()]

                feat=self.extract(wav,**self.kaldi_featset)
                feat = feat.detach()
                feat=self.mean_var(feat)
                feats.append(feat)
        return feats

