# ASV-Subtools: An Open Source Tools for Speaker Recognition
> Copyright: [XMU Speech Lab](https://speech.xmu.edu.cn/) (Xiamen University, China)
> Apache 2.0
>
> Author   : Miao Zhao (**Snowdar**), Jianfeng Zhou, Zheng Li, Hao Lu
> Co-author: Lin Li, Qingyang Hong

[TOC]

## Introduction  
ASV-Subtools is developed based on Pytorch and Kaldi for speaker recognition and language identification etc.. 
The basic training framework is provided here and the relation between every part is very clear. So you could change anything you want to obtain a custom ASV-Subtools.

### Support List
- Multi-GPU Training Solution
  + [x] [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/nn.html#distributeddataparallel) [Built-in function of Pytorch]
  + [x] [Horovod](https://github.com/horovod/horovod)


- Front-end
  + [x] [Convenient Augmentation of Reverb, Noise, Music and Babble](https://github.com/Snowdar/asv-subtools/augmentDataByNoise.sh)
  + [x] Inverted [Specaugment](https://arxiv.org/pdf/1904.08779.pdf)

- Model
  + [x] [Standard X-vector](http://www.danielpovey.com/files/2017_interspeech_embeddings.pdf)
  + [x] [Extended X-vector](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8683760)
  + [x] Resnet1d
  + [x] [Resnet2d](http://www.danielpovey.com/files/2019_interspeech_nist_sre18.pdf)
  + [ ] [F-TDNN X-vector](http://www.danielpovey.com/files/2019_interspeech_nist_sre18.pdf)

- Components
  + [x] [Attentive Statistics Pooling](https://arxiv.org/pdf/1803.10963v1.pdf)
  + [x] [ Learnable Dictionary Encoding (LDE) Pooling](https://arxiv.org/pdf/1804.00385.pdf)
  + [x] [Sequeze and Excitation (SE)](https://arxiv.org/pdf/1709.01507.pdf) [An [example](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1704.pdf) of speaker recognition based on Resnet1d by Jianfeng Zhou]
  + [ ] Multi-head Attention Pooling

- Loss Functions
  + [x] Softmax Loss (Affine + Softmax + Cross-Entropy)
  + [x] AM-Softmax Loss
  + [x] AAM-Softmax Loss
  + [x] Double AM-Softmax Loss
  + [x] Ring Loss

- Optimizer [Out of Pytorch built-in functions]
  + [x] Lookahead [A wrapper optimizer]
  + [x] RAdam
  + [x] Ralamb
  + [x] Novograd
  + [x] Gradient Centralization [Extra bound to optimizer]

- Training Stratagies
  + [x] [AdamW](https://arxiv.org/pdf/1711.05101v1.pdf) + [WarmRestarts](https://arxiv.org/pdf/1608.03983v4.pdf)
  + [ ] SGD + [ReduceLROnPlateau](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
  + [x] Training with Magin Decay Stratagy
  + [x] Heated Up Stratagy
  + [x] [Multi-task Training with Phonetic Information](http://yiliu.org.cn/papers/Speaker_Embedding_Extraction_with_Phonetic_Information.pdf) (Kaldi) [[Source codes](https://github.com/mycrazycracy/speaker-embedding-with-phonetic-information) was provided by [Yi Liu](http://yiliu.org.cn/). Thanks.]
  + [ ] Multi-task Training with Phonetic Information (Pytorch)
  + [ ] GAN

- Back-End
  + [x] LDA, Submean, Whiten (ZCA), Vector Length Normalization
  + [x] Cosine Similarity
  + [x] Classifiers: SVM, GMM, Logistic Regression (LR), PLDA, APLDA, CORAL, CORAL+, LIP, CIP
  + [x] Score Normalization: S-Norm, AS-Norm
  + [ ] Calibration
  + [x] Metric: EER, Cavg, minDCF

- Others
  + [x] [Learning Rate Finder](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html)
  + [ ] Use **matplotlib** to Plot DET Curve a.w.t the Format of DETware (Matlab Version) of [NIST's Tools](https://www.nist.gov/itl/iad/mig/tools)

### Project Structure  
![Project-Structure.png](https://github.com/Snowdar/asv-subtools/tree/master/doc/ASV-Subtools-project-structure.png)
### Training Framework  
![!img](https://github.com/Snowdar/asv-subtools/tree/master/doc/pytorch-training-framework.png)

### Data Pipeline  
![Project-Structure.png](https://github.com/Snowdar/asv-subtools/tree/master/doc/pytorch-data-pipeline.png)

## Ready to Start  
### 1\. Install Kaldi  
The Pytorch-training has less relation to Kaldi, but we have not provided other interfaces to concatenate acoustic features and training now. So if you don't want to use Kaldi, it is easy to change the **libs.egs.egs.ChunkEgs** class for the features are given to Pytorch only by [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset). Of course, you should also change the interface of extracting x-vector after training done. And most of scripts which requires Kaldi could be not available, such as subtools/makeFeatures.sh and subtools/augmentDataByNoise.sh etc..

**If you prefer to use Kaldi, then install Kaldi firstly w.r.t http://www.kaldi-asr.org/doc/install.html.**

```
# Download Kaldi
git clone https://github.com/kaldi-asr/kaldi.git kaldi --origin upstream
cd kaldi

# You could check the INSTALL file of current directory to install for more details
cat INSTALL

# Compile tools firstly
cd tools
sh extras/check_dependencies.sh
make -j 4

# Config src before compiling
cd ../src
./configure --shared

# Check depend and compile
make depend -j 4
make -j 4
cd ..
```

### 2\. Create Project  
Create your project with **4-level name** relative to Kaldi root directory (1-level), such as **kaldi/egs/xmuspeech/sre**. It is important to environment. For more details, see [subtools/path.sh](https://github.com/Snowdar/asv-subtools/path.sh).

```
# Suppose current directory is kaldi root directory
mkdir -p kaldi/egs/xmuspeech/sre
```

### 3\. Clone ASV-Subtools  
ASV-Subtools could be saw as a set of tools like utils/steps of Kaldi, so there are only two extra stages to complete the installation:
+ Clone ASV-Subtools to your project.
+ Install the requirements of python (**Python3 is recommended**).

```
# Clone asv-subtools from github
cd kaldi/egs/xmuspeech/sre
git clone https://github.com/Snowdar/asv-subtools/.git
```

### 4\. Install Python Requirements  
+ Pytorch>=1.2: ```pip3 install torch```
+ Other requirements: numpy, thop, pandas, progressbar2, matplotlib, scipy (option), sklearn (option)
  ```pip3 install -r subtools/requirements.txt```

### 5\. Support Multi-GPU Training  
ASV-Subtools provides both **DDP (recommended)** and Horovod solutions to support multi-GPU training.

**Some answers about how to use multi-GPU taining, see [subtools/pytorch/launcher/runSnowdarXvector.py](https://github.com/Snowdar/asv-subtools/tree/master/pytorch/launcher/runSnowdarXvector.py). It is very convenient and easy now.**

Requirements List:  
+ DDP: Pytorch, NCCL  
+ Horovod: Pytorch, NCCL, Openmpi, Horovod  

#### An Example of Install NCCL Based on Linux-Centos-7 and CUDA-10.2  
Reference: https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html.  
```
# For a simple way, there are only three stages.
# [1] Download rpm package of nvidia
wget https://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm

# [2] Add nvidia repo to yum (NOKEY could be ignored)
sudo rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm

# [3] Install NCCL by yum
sudo yum install libnccl-2.6.4-1+cuda10.2 libnccl-devel-2.6.4-1+cuda10.2 libnccl-static-2.6.4-1+cuda10.2
```

These yum-clean commands could be very useful when you get some troubles when using yum.

```
# Install yum-utils firstly
yum -y install yum-utils

#
yum clean all

#
yum-complete-transaction --cleanup-only

#
package-cleanup --cleandupes
```

If you want to install Openmpi and Horovod, see https://github.com/horovod/horovod for more details.

### 6\. Extra Installation (Option)

## Recipe
### Voxceleb Recipe [Speaker Recognition]
There are two recipes of Voxceleb:

(1) see subtools/recipe/voxceleb/runVoxceleb.sh.

(2) see subtools/recipe/voxcelebSRC/runVoxceleb.sh


### AP-OLR 2020 Baseline Recipe [Language Identification]
see http://cslt.riit.tsinghua.edu.cn/mediawiki/index.php/ASR-events-AP16-details.

Kaldi baseline:

Pytorch baseline:

## Feedback
+ If you find bugs or have some questions, please create an issue in issues of github to let everyone know it so that a good solution could be provided.
+ If you have any questions to ask me, you could also send e-mail to snowdar@stu.xmu.edu.cn and I will reply this in my free time.

## Acknowledgement
+ Thanks to Kaldi, Pytorch, kaldi_io
+ Thanks to everyone that contribute their time and ideas to ASV-Subtools.
+ Thanks to myself also (\^_^).