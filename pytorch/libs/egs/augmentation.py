# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-03-18)

# Augmentation: segment augmentation (before extracting acoustics), 
#               features augmentation (before gathered in batch)
#               dropout augmentation (in forward).
#
# There are some features augmentation methods which are used in ChunkEgs to avoid too hard implementation
# for batch when we want to make different egs have different augmentation. It is really not simple
# comparing with torch.nn.Dropout and it is very efficient that augmenting featues before gathered-in-batch.

import torch
import numpy as np 

import libs.support.utils as utils

### Augmentation
class SpecAugment():
    """Implement specaugment for acoustics features' augmentation but without time wraping.
    Reference: Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). 
               Specaugment: A simple data augmentation method for automatic speech recognition. arXiv 
               preprint arXiv:1904.08779.

    Likes in Compute Vision: 
           [1] DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks 
               with cutout. arXiv preprint arXiv:1708.04552.

           [2] Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2017). Random erasing data augmentation. 
               arXiv preprint arXiv:1708.04896. 
    """
    def __init__(self, frequency=0.2, frame=0.2, rows=1, cols=1, random_rows=False, random_cols=False):
        assert 0. <= frequency < 1.
        assert 0. <= frame < 1. # a.k.a time axis.

        self.p_f = frequency
        self.p_t = frame

        # Multi-mask.
        self.rows = rows # Mask rows times for frequency.
        self.cols = cols # Mask cols times for frame.

        self.random_rows = random_rows
        self.random_cols = random_cols

        self.init = False

    def __call__(self, inputs):
        """
        @inputs: a 2-dimensional tensor (a matrix), including [frenquency, time]
        """
        if self.p_f > 0. or self.p_t > 0.:
            if isinstance(inputs, np.ndarray):
                    numpy_tensor = True
            elif isinstance(inputs, torch.Tensor):
                    numpy_tensor = False
            else:
                raise TypeError("Expected np.ndarray or torch.Tensor, but got {}".format(type(inputs).__name__))

            if not self.init:
                input_size = inputs.shape
                assert len(input_size) == 2
                if self.p_f > 0.:
                    self.num_f = input_size[0] # Total channels.
                    self.F = int(self.num_f * self.p_f) # Max channels to drop.
                if self.p_t > 0.:
                    self.num_t = input_size[1] # Total frames. It requires all egs with the same frames.
                    self.T = int(self.num_t * self.p_t) # Max frames to drop.
                self.init = True

            if self.p_f > 0.:
                if self.random_rows:
                    multi = np.random.randint(1, self.rows+1)
                else:
                    multi = self.rows

                for i in range(multi):
                    f = np.random.randint(0, self.F + 1)
                    f_0 = np.random.randint(0, self.num_f - f + 1)

                    inverted_factor = self.num_f / (self.num_f - f)
                    if numpy_tensor:
                        inputs[f_0:f_0+f,:].fill(0.)
                        inputs = torch.from_numpy(inputs).mul_(inverted_factor).numpy()
                    else:
                        inputs[f_0:f_0+f,:].fill_(0.)
                        inputs.mul_(inverted_factor)


            if self.p_t > 0.:
                if self.random_cols:
                    multi = np.random.randint(1, self.cols+1)
                else:
                    multi = self.cols

                for i in range(multi):
                    t = np.random.randint(0, self.T + 1)
                    t_0 = np.random.randint(0, self.num_t - t + 1)

                    if numpy_tensor:
                        inputs[:,t_0:t_0+t].fill(0.)
                    else:
                        inputs[:,t_0:t_0+t].fill_(0.)

        return inputs


# To do.
class Cutout():
    """Cutout for CNN training like CV. 
    It is different to SpecAugment for it does not mask whole time or frequency axis instead of a rectangle area.
    Suggest to use it for fbank or spectrogram features.
    Reference: 
           [1] DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks 
               with cutout. arXiv preprint arXiv:1708.04552.

           [2] Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2017). Random erasing data augmentation. 
               arXiv preprint arXiv:1708.04896.
    """
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        pass

### Wrapper
def get_augmentation(aug=None, aug_params={}):
    default_aug_params = {
        "frequency":0.2,
        "frame":0.,
        "rows":1, 
        "cols":0,
        "random_rows":False, 
        "random_cols":False
    }

    aug_params = utils.assign_params_dict(default_aug_params, aug_params)

    if aug is None or aug == "" or aug == False:
        return None
    elif aug == "specaugment":
        return SpecAugment(frequency=aug_params["frequency"], frame=aug_params["frame"], 
                                rows=aug_params["rows"], cols=aug_params["cols"],
                                random_rows=aug_params["random_rows"], random_cols=aug_params["random_cols"])
    elif aug == "cutout":
        raise NotImplementedError
    else:
        raise TypeError("Do not support {} augmentation.".format(aug))


# Test.
if __name__ == "__main__":
    print("Test aug frenquency only with numpy array...")
    np_array = np.random.randn(8,4)
    aug_frenquency = SpecAugment(frequency=0.5, frame=0., rows=1, cols=1)
    print(aug_frenquency(np_array),"\n")

    print("Test aug time only with torch tensor...")
    tensor = torch.randn(4,8)
    aug_time = SpecAugment(frequency=0., frame=0.5, rows=1, cols=1)
    print(aug_time(tensor),"\n")

    print("Test aug frenquency and time with torch tensor...")
    tensor = torch.randn(8,8)
    aug_all =SpecAugment(frequency=0.5, frame=0.5, rows=2, cols=2)
    print(aug_all(tensor))

    print("Test done.")