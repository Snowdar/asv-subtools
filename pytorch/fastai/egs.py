import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from fastai.torch_core import *
from fastai.basic_data import *

import libs.support.utils as utils
import libs.support.kaldi_io as kaldi_io

class AcousticFeatures(ItemBase):
    """Fbank, MFCC and PLP etc.
    All of them are matrix[frames, feature].
    """
    def __init__(self, features:Tensor):
        self._features = features

    def read(self, fn:PathOrStr, chunk:Tuple)->AcousticFeatures:
        """Create an AcousticFeatures by reading from file.
        chunk: (start-frames-index, end-frames-index) , index is 0-based
        """
        features = kaldi_io.read_mat(fn, chunk=chunk)
        return self(torch.tensor(features))

    @property
    def shape(self)->Tuple[int,int]: return self._features.shape
    @property
    def size(self)->Tuple[int,int]: return self.shape
    @property
    def device(self)->torch.device: return self._features.device

    def __repr__(self): return f'{self.__class__.__name__} {tuple(self.shape)}'

    @property
    def features(self)->TensorAcousticFeatures:
        "Get the acoustic features."
        return self._features
    @features.setter
    def features(self, v:TensorAcousticFeatures)->None:
        "Set the acoustic features to `v`."
        self._features=v

    @property
    def data(self)->TensorAcousticFeatures:
        "Return this features as a tensor."
        return self.features


class AcousticFeaturesList(ItemList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self, i):
        fn, chunk = super().get(i)
        res = AcousticFeatures.read(fn, chunk)
        self.sizes[i] = res.size
        return res
