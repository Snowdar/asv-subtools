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
    def __init__(self, frequency=0.2, frame=0.2, rows=1, cols=1, continuous=False, std=0.2):
        assert 0. <= frequency < 1.
        assert 0. <= frame < 1. # a.k.a time axis.

        self.p_f = frequency
        self.p_t = frame

        # Multi-mask.
        self.rows = rows # Mask rows times for frequency.
        self.cols = cols # Mask cols times for frame.

        self.continuous = continuous
        self.std = std

        self.init = False

    def __call__(self, inputs):
        """
        @inputs: a 2-dimensional tensor (a batch), including [frenquency, time]
        """
        if self.p_f > 0. or self.p_t > 0.:
            if isinstance(inputs, np.ndarray):
                    numpy_tensor = True
                    # To avoid "ValueError: assignment destination is read-only".
                    # Do not use inputs.flags.writeable = True when the version of numpy >= 1.17.
                    inputs = np.require(inputs, requirements=['O', 'W']) 
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
                for i in range(self.rows):
                    f = np.random.randint(0, self.F + 1)
                    f_0 = np.random.randint(0, self.num_f - f + 1)

                    if self.continuous:
                        scale_size = (f, self.num_t)
                        scale = torch.tensor(np.random.normal(0., self.std, size=scale_size).clip(0., 1.))
                        if numpy_tensor:
                            # Use torch.from_numpy to transforms numpy.ndarray to tensor without a deep copy
                            # and use tensor inplace funtion torch.Tensor.mul_().
                            inputs = torch.from_numpy(inputs)
                            inputs[f_0:f_0+f,:].mul_(scale)
                            inputs = inputs.numpy()
                        else:
                            inputs[f_0:f_0+f,:].mul_(scale)
                    else:
                        inverted_factor = self.num_f / (self.num_f - f)
                        if numpy_tensor:
                            inputs[f_0:f_0+f,:].fill(0.)
                            inputs = torch.from_numpy(inputs).mul_(inverted_factor).numpy()
                        else:
                            inputs[f_0:f_0+f,:].fill_(0.)
                            inputs.mul_(inverted_factor)


            if self.p_t > 0.:
                for i in range(self.cols):
                    t = np.random.randint(0, self.T + 1)
                    t_0 = np.random.randint(0, self.num_t - t + 1)

                    if self.continuous:
                        scale_size = (self.num_f, t)
                        scale = torch.tensor(np.random.normal(0., self.std, size=scale_size).clip(0., 1.))
                        if numpy_tensor:
                            inputs = torch.from_numpy(inputs)
                            inputs[:,t_0:t_0+t].mul_(scale)
                            inputs = inputs.numpy()
                        else:
                            inputs[:,t_0:t_0+t].mul_(scale)
                    else:
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
