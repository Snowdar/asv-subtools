# ASV-Subtools: An Open Source Tools for Speaker Recognition
> Copyright: [XMU Speech Lab](https://speech.xmu.edu.cn/) (Xiamen University, China)  
> Apache 2.0
>
> Author   : Miao Zhao (Snowdar), Jianfeng Zhou, Zheng Li, Hao Lu  
> Co-author: Lin Li, Qingyang Hong

---
- **Content**
  * [Introduction](#introduction)
    + [Project Structure](#project-structure)
    + [Training Framework](#training-framework)
    + [Data Pipeline](#data-pipeline)
    + [Support List](#support-list)
  * [Ready to Start](#ready-to-start)
    + [1. Install Kaldi](#1-install-kaldi)
    + [2. Create Project](#2-create-project)
    + [3. Clone ASV-Subtools](#3-clone-asv-subtools)
    + [4. Install Python Requirements](#4-install-python-requirements)
    + [5. Support Multi-GPU Training](#5-support-multi-gpu-training)
    + [6. Extra Installation (Option)](#6-extra-installation-option)
      - [Train A Multi-Task Learning Model with Kaldi](#train-a-multi-task-learning-model-with-kaldi)
      - [Accelerate X-vector Extractor of Kaldi](#accelerate-x-vector-extractor-of-kaldi)
      - [Add A MMI-GMM Classifier for The Back-End](#add-a-mmi-gmm-classifier-in-the-back-end)
  * [Recipe](#recipe)
    + [[1] Voxceleb Recipe [Speaker Recognition]](#1-voxceleb-recipe-speaker-recognition)
    + [[2] AP-OLR 2020 Baseline Recipe [Language Identification]](#2-ap-olr-2020-baseline-recipe-language-identification)
  * [Feedback](#feedback)
  * [Acknowledgement](#acknowledgement)

<!--Table of contents generated with markdown-toc, see http://ecotrust-canada.github.io/markdown-toc-->
---

## Introduction  
ASV-Subtools is developed based on [Pytorch](https://pytorch.org/) and [Kaldi](http://www.kaldi-asr.org/) for speaker recognition and language identification etc..  

On the one hand, [Kaldi](http://www.kaldi-asr.org/) is used to extract acoustic features and scoring in the back-end. On the other hand, [Pytorch](https://pytorch.org/) is used to build a model freely and train it with a custom style.


### Project Structure  
ASV-Subtools contains **three main branches**:
+ Basic Shell Scripts: data processing, back-end scoring (most are based on Kaldi)
+ Kaldi: training of basic model (i-vector, TDNN, F-TDNN and multi-task x-vector)
+ Pytorch: training of custom model (less limitation)

</br>
<center><img src="./doc/ASV-Subtools-project-structure.png" width="600"/></center>
</br>

For pytorch branch, there are **two important concepts**:
+ **Model Blueprint**: the path of ```your_model.py```
+ **Model Creation** : the code to init a model class, such as ```resnet(40, 1211, loss="AM")```

In ASV-Subtools, the model is individual. This means that we should know the the path of ```model.py``` and how to init this model class at least when using this model in training or testing module. This structure is designed to avoid modifying codes of other module frequently. For example, if the embedding extractor is wrote down as a called program and we use a fixed method ```from my_model_py import my_model``` to import this model, then it will be in trouble for ```model_2.py``` and ```model_3.py``` etc..

**Note that**, all model ([torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)) shoud inherit [libs.nnet.framework.TopVirtualNnet](./pytorch/libs/nnet/framework.py) class to get some default functions, such as **auto-saving model creation and blueprint**, extracting emdedding of whole utterance, step-training and computing accuracy etc.. By inheriting, it is very convenient to transform your pytorch model to ASV-Subtools model.

### Training Framework  
The basic training framework is provided here and the relation between every module is very clear. So it will be not complex if you want to change anything you want to have a custom ASV-Subtools.  

**Note that**, [libs/support/utils.py](./pytorch/libs/support/utils.py) has many common functions, so it is imported in most of ```*.py```.

</br>
<center><img src="./doc/pytorch-training-framework.png" width="600"/></center>
</br>

### Data Pipeline  
Here, a data pipeline is given to show the relation between Kaldi and Pytorch. There are only two interfaces, **reading acoustic features** and **writing x-vectors**, and both of them are implemented by [kaldi_io](https://github.com/vesis84/kaldi-io-for-python).

Of course, this data pipeline could be also followed to know the basic principle of xvector-based speaker recognition.  

</br>
<center><img src="./doc/pytorch-data-pipeline.png" width="600"/></center>
</br>

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
  + [x] [Sequeze and Excitation (SE)](https://arxiv.org/pdf/1709.01507.pdf) [An [example](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1704.pdf) of speaker recognition based on Resnet1d by Jianfeng Zhou.]
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
  + [x] [Multi-task Learning with Phonetic Information](http://yiliu.org.cn/papers/Speaker_Embedding_Extraction_with_Phonetic_Information.pdf) (Kaldi) [[Source codes](https://github.com/mycrazycracy/speaker-embedding-with-phonetic-information) was contributed by [Yi Liu](http://yiliu.org.cn/). Thanks.]
  + [ ] Multi-task Learning with Phonetic Information (Pytorch)
  + [ ] GAN

- Back-End
  + [x] LDA, Submean, Whiten (ZCA), Vector Length Normalization
  + [x] Cosine Similarity
  + [x] Basic Classifiers: SVM, GMM, Logistic Regression (LR)
  + [x] PLDA Classifiers: [PLDA](https://ravisoji.com/assets/papers/ioffe2006probabilistic.pdf), APLDA, [CORAL](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12443/11842), [CORAL+](https://arxiv.org/pdf/1812.10260), [LIP](http://150.162.46.34:8080/icassp2014/papers/p4075-garcia-romero.pdf), [CIP](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9054113) [[Python versions](./score/pyplda) was contributed by Jianfeng Zhou. For more details, see the <a href="./score/pyplda/Domain-Adaptation-of-PLDA-in-Speaker-Recognition.pdf" target="_blank">note</a>.]
  + [x] Score Normalization: [S-Norm](http://www.crim.ca/perso/patrick.kenny/kenny_Odyssey2010_presentation.pdf), [AS-Norm](https://www.researchgate.net/profile/Daniele_Colibro/publication/221480280_Comparison_of_Speaker_Recognition_Approaches_for_Real_Applications/links/545e4f6e0cf295b561602c42/Comparison-of-Speaker-Recognition-Approaches-for-Real-Applications.pdf)
  + [ ] Calibration
  + [x] Metric: EER, Cavg, minDCF

- Others
  + [x] [Learning Rate Finder](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html)
  + [ ] Plot DET Curve withs ```matplotlib``` w.r.t the Format of DETware (Matlab Version) of [NIST's Tools](https://www.nist.gov/itl/iad/mig/tools)
  + [ ] Accumulate Total MACs and Flops of Model Based on ```thop```

## Ready to Start  
### 1. Install Kaldi  
The Pytorch-training has less relation to Kaldi, but we have not provided other interfaces to concatenate acoustic features and training now. So if you don't want to use Kaldi, it is easy to change the [libs.egs.egs.ChunkEgs](./pytorch/libs/egs/egs.py) class where the features are given to Pytorch only by [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset). Of course, you should also change the interface of extracting x-vector after training done. And most of scripts which requires Kaldi could be not available, such as subtools/makeFeatures.sh and subtools/augmentDataByNoise.sh etc..

**If you prefer to use Kaldi, then install Kaldi firstly w.r.t http://www.kaldi-asr.org/doc/install.html.**

Here are conclusive stages:

```shell
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

### 2. Create Project  
Create your project with **4-level name** relative to Kaldi root directory (1-level), such as **kaldi/egs/xmuspeech/sre**. It is important to environment. For more details, see [subtools/path.sh](./path.sh).

```shell
# Suppose current directory is kaldi root directory
mkdir -p kaldi/egs/xmuspeech/sre
```

### 3. Clone ASV-Subtools  
ASV-Subtools could be saw as a set of tools like utils/steps of Kaldi, so there are only two extra stages to complete the installation:
+ Clone ASV-Subtools to your project.
+ Install the requirements of python (**Python3 is recommended**).

```shell
# Clone asv-subtools from github
cd kaldi/egs/xmuspeech/sre
git clone https://github.com/Snowdar/asv-subtools/.git
```

### 4. Install Python Requirements  
+ Pytorch>=1.2: ```pip3 install torch```
+ Other requirements: numpy, thop, pandas, progressbar2, matplotlib, scipy (option), sklearn (option)  
  ```pip3 install -r requirements.txt```

### 5. Support Multi-GPU Training  
ASV-Subtools provides both **DDP (recommended)** and Horovod solutions to support multi-GPU training.

**Some answers about how to use multi-GPU taining, see [subtools/pytorch/launcher/runSnowdarXvector.py](./pytorch/launcher/runSnowdarXvector.py). It is very convenient and easy now.**

Requirements List:  
+ DDP: Pytorch, NCCL  
+ Horovod: Pytorch, NCCL, Openmpi, Horovod  

**An Example of Installing NCCL Based on Linux-Centos-7 and CUDA-10.2**  
Reference: https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html.  

```shell
# For a simple way, there are only three stages.
# [1] Download rpm package of nvidia
wget https://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm

# [2] Add nvidia repo to yum (NOKEY could be ignored)
sudo rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm

# [3] Install NCCL by yum
sudo yum install libnccl-2.6.4-1+cuda10.2 libnccl-devel-2.6.4-1+cuda10.2 libnccl-static-2.6.4-1+cuda10.2
```

These yum-clean commands could be very useful when you get some troubles when using yum.

```shell
# Install yum-utils firstly
yum -y install yum-utils

# Stop unfinished transactions
yum-complete-transaction --cleanup-only

# Clean duplicate and conflict
package-cleanup --cleandupes

# Clean cached headers and packages
yum clean all
```

If you want to install Openmpi and Horovod, see https://github.com/horovod/horovod for more details.

### 6. Extra Installation (Option)
There are some extra installations for some special applications.

#### Train A Multi-Task Learning Model with Kaldi
```shell
# Enter your project, such as kaldi/egs/xmuspeech/sre and make sure ASV-Subtools is cloned here
# Just run this patch to compile some extra C++ commands with Kaldi's format
cd kaldi/egs/xmuspeech/sre
sh subtools/kaldi/patch/runPatch-multitask.sh
```

#### Accelerate X-vector Extractor of Kaldi
It spends so much time to compile model for different chunk utterances when extracting x-vectors . ASV-Subtools provides a **offine** modification (MOD) in [subtools/kaldi/sid/nnet3/xvector/extract_xvectors.sh](./kaldi/sid/nnet3/xvector/extract_xvectors.sh) to accelerate extracting. This MOD requires two extra commands, **nnet3-compile-xvector-net** and **nnet3-offline-xvector-compute**. When extracting x-vectors, all model with different input chunk-size will be compiled firstly. Then the utterances which have the same frames could share a compiled nnet3 network. It saves much time by avoiding a lot of duplicate dynamic compilations.

Besides, the ```scp``` spliting type w.r.t length of utterances ([subtools/splitDataByLength.sh](./splitDataByLength.sh)) is adopted to balance the frames of different ```nj``` when multi-processes is used.

```shell
# Enter your project, such as kaldi/egs/xmuspeech/sre and make sure ASV-Subtools is cloned here
# Just run this patch to compile some extra C++ commands with Kaldi's format

# Target *.cc:
#     src/nnet3bin/nnet3-compile-xvector-net.cc
#     src/nnet3bin/nnet3-offline-xvector-compute.cc

cd kaldi/egs/xmuspeech/sre
sh subtools/kaldi/patch/runPatch-base-command.sh
```

#### Add A MMI-GMM Classifier for The Back-End
If you have run [subtools/kaldi/patch/runPatch-base-command.sh](./kaldi/patch/runPatch-base-command.sh), then it dosen't need to run again.

```shell
# Enter your project, such as kaldi/egs/xmuspeech/sre and make sure ASV-Subtools is cloned here
# Just run this patch to compile some extra C++ commands with Kaldi's format

# Target *.cc:
#    src/gmmbin/gmm-global-init-from-feats-mmi.cc
#    src/gmmbin/gmm-global-est-gaussians-ebw.cc
#    src/gmmbin/gmm-global-est-map.cc
#    src/gmmbin/gmm-global-est-weights-ebw.cc

cd kaldi/egs/xmuspeech/sre
sh subtools/kaldi/patch/runPatch-base-command.sh
```

## Recipe
### [1] Voxceleb Recipe [Speaker Recognition]
[Voxceleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/index.html#about) is a popular dataset in speaker recognition field. It has two part now, [Voxceleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and [Voxceleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html).

There are **two recipes for Voxceleb**:

**i. Test Voxceleb1-O only**

It means trainset could come from Voxceleb1.dev and Voxceleb2 with a fixed training condition. The training script is available in [subtools/recipe/voxceleb/runVoxceleb.sh](./recipe/voxceleb/runVoxceleb.sh).

**Results of Voxceleb1-O with Voxceleb1.dev Training only**

Index|Features|Model|InSpecAug|AM-Softmax (m=0.2)|Back-End|EER%
:-:|:-:|:-:|:-:|:-:|:-:|:-:|
1|mfcc23&pitch|x-vector|no|no|PLDA|3.362
2|mfcc23&pitch|x-vector|yes|no|PLDA|2.778
3|mfcc23&pitch|x-vector|no|yes|PLDA|3.240
4|mfcc23&pitch|x-vector|yes|yes|PLDA|2.635
5|mfcc23&pitch|extended x-vector|no|no|PLDA|3.112
6|mfcc23&pitch|extended x-vector|yes|no|PLDA|2.598
7|mfcc23&pitch|extended x-vector|no|yes|PLDA|3.293
8|mfcc23&pitch|extended x-vector|yes|yes|PLDA|2.444

***... information updating ...***

**ii. Test Voxceleb1-O/E/H**

It means trainset could come from Voxceleb2 only with a fixed training condition. The training script is available in [subtools/recipe/voxcelebSRC/runVoxcelebSRC.sh](./recipe/voxcelebSRC/runVoxcelebSRC.sh).

***... information updating ...***

### [2] AP-OLR 2020 Baseline Recipe [Language Identification]

***... information updating ...***

see http://olr.cslt.org for more details.

The training script is available in [subtools/recipe/ap-olr2020-baseline/run.sh](./recipe/ap-olr2020-baseline/run.sh)

## Feedback
+ If you find bugs or have some questions, please create a  github issue in this project to let everyone know it so that a good solution could be contributed.
+ If you want to question me, you could also send e-mail to snowdar@stu.xmu.edu.cn and I will reply this in my free time.

## Acknowledgement
+ Thanks to everyone who contribute their time, ideas and codes to ASV-Subtools.
+ Thanks to [XMU Speech Lab](https://speech.xmu.edu.cn/) providing machine and GPU.
+ Thanks to the excelent projects: [Kaldi](http://www.kaldi-asr.org/), [Pytorch](https://pytorch.org/), [Kaldi I/O](https://github.com/vesis84/kaldi-io-for-python), [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Horovod](https://github.com/horovod/horovod), [Progressbar2](https://github.com/WoLpH/python-progressbar), [Matplotlib](https://matplotlib.org/index.html), [Prefetch Generator](https://github.com/justheuristic/prefetch_generator), [Thop](https://github.com/Lyken17/pytorch-OpCounter), [GPU Manager](https://github.com/QuantumLiu/tf_gpu_manager) etc..