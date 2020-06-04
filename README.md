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
      - [Train A Multi-Task Learning Model Based on Kaldi](#train-a-multi-task-learning-model-based-on-kaldi)
      - [Accelerate X-vector Extractor of Kaldi](#accelerate-x-vector-extractor-of-kaldi)
      - [Add A MMI-GMM Classifier for The Back-End](#add-a-mmi-gmm-classifier-in-the-back-end)
  * [Training Model](#training-model)
  * [Recipe](#recipe)
    + [[1] Voxceleb Recipe [Speaker Recognition]](#1-voxceleb-recipe-speaker-recognition)
    + [[2] AP-OLR Challenge 2020 Baseline Recipe [Language Identification]](#2-ap-olr-challenge-2020-baseline-recipe-language-identification)
  * [Feedback](#feedback)
  * [Acknowledgement](#acknowledgement)

<!--Table of contents generated with markdown-toc, see http://ecotrust-canada.github.io/markdown-toc-->
---

## Introduction  
ASV-Subtools is developed based on [Pytorch](https://pytorch.org/) and [Kaldi](http://www.kaldi-asr.org/) for the task of speaker recognition, language identification, etc.. The 'sub' of 'subtools' means that there are many modular tools and the parts constitute the whole. 

In ASV-Subtools, [Kaldi](http://www.kaldi-asr.org/) is used to extract acoustic features and scoring in the back-end. And [Pytorch](https://pytorch.org/) is used to build a model freely and train it with a custom style.


### Project Structure  
ASV-Subtools contains **three main branches**:
+ Basic Shell Scripts: data processing, back-end scoring (most are based on Kaldi)
+ Kaldi: training of basic model (i-vector, TDNN, F-TDNN and multi-task learning x-vector)
+ Pytorch: training of custom model (less limitation)

</br>
<center><img src="./doc/ASV-Subtools-project-structure.png" width="600"/></center>
</br>

For pytorch branch, there are **two important concepts**:
+ **Model Blueprint**: the path of ```your_model.py```
+ **Model Creation** : the code to init a model class, such as ```resnet(40, 1211, loss="AM")```

In ASV-Subtools, the model is individual. This means that we should know the the path of ```model.py``` and how to init this model class at least when using this model in training or testing module. This structure is designed to avoid modifying codes of static modules frequently. For example, if the embedding extractor is wrote down as a called program and we use a inline method ```from my_model_py import my_model``` to import a fixed model from a fixed ```model.py``` , then it will be not free for ```model_2.py```, ```model_3.py``` and so on.

**Note that**, all model ([torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)) shoud inherit [libs.nnet.framework.TopVirtualNnet](./pytorch/libs/nnet/framework.py) class to get some default functions, such as **auto-saving model creation and blueprint**, extracting emdedding of whole utterance, step-training, computing accuracy, etc.. It is easy to transform the original model of Pytorch to ASV-Subtools model by inheriting. Just modify your ```model.py``` w.r.t this [x-vector example](./pytorch/model/xvector.py).

### Training Framework  
The basic training framework is provided here and the relations between every module are very clear. So it will be not complex if you want to change anything you want to have a custom ASV-Subtools.  

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
  + [x] [Convenient Augmentation of Reverb, Noise, Music and Babble](./augmentDataByNoise.sh)
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
  + [x] PLDA Classifiers: [PLDA](https://ravisoji.com/assets/papers/ioffe2006probabilistic.pdf), APLDA, [CORAL](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12443/11842), [CORAL+](https://arxiv.org/pdf/1812.10260), [LIP](http://150.162.46.34:8080/icassp2014/papers/p4075-garcia-romero.pdf), [CIP](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9054113) [[Python versions](./score/pyplda) was contributed by Jianfeng Zhou. For more details, see the [note](./score/pyplda/Domain-Adaptation-of-PLDA-in-Speaker-Recognition.pdf).]
  + [x] Score Normalization: [S-Norm](http://www.crim.ca/perso/patrick.kenny/kenny_Odyssey2010_presentation.pdf), [AS-Norm](https://www.researchgate.net/profile/Daniele_Colibro/publication/221480280_Comparison_of_Speaker_Recognition_Approaches_for_Real_Applications/links/545e4f6e0cf295b561602c42/Comparison-of-Speaker-Recognition-Approaches-for-Real-Applications.pdf)
  + [ ] Calibration
  + [x] Metric: EER, Cavg, minDCF

- Others
  + [x] [Learning Rate Finder](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html)
  + [ ] Plot DET Curve with ```matplotlib``` w.r.t the Format of DETware (Matlab Version) of [NIST's Tools](https://www.nist.gov/itl/iad/mig/tools)
  + [ ] Accumulate Total MACs and Flops of Model Based on ```thop```

## Ready to Start  
### 1. Install Kaldi  
The Pytorch-training has less relation to Kaldi, but we have not provided other interfaces to concatenate acoustic features and training now. So if you don't want to use Kaldi, you could change the [libs.egs.egs.ChunkEgs](./pytorch/libs/egs/egs.py) class where the features are given to Pytorch only by [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset). Besides, you should also change the interface of extracting x-vector after training done. Note that, most of scripts which require Kaldi could be not available in this case, such as subtools/makeFeatures.sh and subtools/augmentDataByNoise.sh.

**If you prefer to use Kaldi, then install Kaldi firstly w.r.t http://www.kaldi-asr.org/doc/install.html.**

Here are conclusive stages:

```shell
# Download Kaldi
git clone https://github.com/kaldi-asr/kaldi.git kaldi --origin upstream
cd kaldi

# You could check the INSTALL file of current directory for more details of installation
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
Create your project with **4-level name** relative to Kaldi root directory (1-level), such as **kaldi/egs/xmuspeech/sre**. It is important for the project environment. For more details, see [subtools/path.sh](./path.sh).

```shell
# Suppose current directory is kaldi root directory
mkdir -p kaldi/egs/xmuspeech/sre
```

### 3. Clone ASV-Subtools  
ASV-Subtools could be seen as a set of tools like 'utils' or 'steps' of Kaldi, so there are only two extra stages to complete the final installation:
+ Clone ASV-Subtools to your project.
+ Install the requirements of python (**Python3 is recommended**).

Here is the method cloning ASV-Subtools from Github:

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
ASV-Subtools provide both **DDP (recommended)** and Horovod solutions to support multi-GPU training.

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

#### Train A Multi-Task Learning Model Based on Kaldi
Use [subtools/kaldi/runMultiTaskXvector.sh](./kaldi/runMultiTaskXvector.sh) to train a model with multi-task learning,  but it requires some extra codes.
```shell
# Enter your project, such as kaldi/egs/xmuspeech/sre and make sure ASV-Subtools is cloned here
# Just run this patch to compile some extra C++ commands with Kaldi's format
cd kaldi/egs/xmuspeech/sre
sh subtools/kaldi/patch/runPatch-multitask.sh
```

#### Accelerate X-vector Extractor of Kaldi
It spends so much time to compile models for the utterances with different frames when extracting x-vectors based on Kaldi nnet3. ASV-Subtools provide an **offine** modification (MOD) in [subtools/kaldi/sid/nnet3/xvector/extract_xvectors.sh](./kaldi/sid/nnet3/xvector/extract_xvectors.sh) to accelerate extracting. This MOD requires two extra commands, **nnet3-compile-xvector-net** and **nnet3-offline-xvector-compute**. When extracting x-vectors, all model with different input chunk-size will be compiled firstly. Then the utterances which have the same frames could share a compiled nnet3 network. It saves much time by avoiding a lot of duplicate dynamic compilations.

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
## Training Model
If you have completed the [Ready to Start](#ready-to-start) stage, then you could try to train a model with ASV-Subtools.

For kaldi trainig, some launcher scripts named ```run*.sh``` could be found in [subtoos/Kaldi/](./kaldi).

For pytorch training, some launcher scripts named ```run*.py``` could be found in [subtools/pytorch/launcher/](./pytorch/launcher/). And some models named ```*.py``` could be found in [subtools/pytorch/model/](./pytorch/model).  Note that, model will be called in ```launcher.py```.

Here is a pytorch training example, but you should follow a [recipe](#recipe) to prepare your data and features before training. The part of data preprocessing is not complex and it is same with Kaldi. 
```shell
# Suppose you have followed the recipe and prepare your data and faetures, then the training could be run by follows.
# Enter your project, such as kaldi/egs/xmuspeech/sre and make sure ASV-Subtools is cloned here

# Firsty, copy a launcher to your project
cp subtools/pytorch/launcher/runSnowdarXvector.py ./

# Modify this launcher and run
# In most of time, there are only two files, model.py and launcher.py, will be changed.
subtools/runLauncher.sh runSnowdarXvector.py --gpu-id=0,1,2,3 --stage=0
```

## Recipe
### [1] Voxceleb Recipe [Speaker Recognition]
[Voxceleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/index.html#about) is a popular dataset for the task of speaker recognition. It has two part now, [Voxceleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and [Voxceleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html).

There are **two recipes for Voxceleb**:

**i. Test Voxceleb1-O only**

It means the trainset could be sampled from both Voxceleb1.dev and Voxceleb2 with a fixed training condition. The training script is available in [subtools/recipe/voxceleb/runVoxceleb.sh](./recipe/voxceleb/runVoxceleb.sh).

**Results of Voxceleb1-O with Voxceleb1.dev.aug1:1 Training only**

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

**Results of Voxceleb1-O with Voxceleb1&2.dev.aug1:1 Training**
Index|Features|Model|InSpecAug|AM-Softmax (m=0.2)|Back-End|EER%
:-:|:-:|:-:|:-:|:-:|:-:|:-:|
9|mfcc23&pitch|x-vector|no|no|PLDA|2.020
10|mfcc23&pitch|x-vector|yes|no|PLDA|1.967
11|mfcc23&pitch|x-vector|no|yes|PLDA|2.375
12|mfcc23&pitch|x-vector|yes|yes|PLDA|2.349
13|mfcc23&pitch|extended x-vector|no|no|PLDA|1.972
14|mfcc23&pitch|extended x-vector|yes|no|PLDA|2.169
15|mfcc23&pitch|extended x-vector|no|yes|PLDA|1.771
||||||Cosine->+AS-Norm|2.163->2.025
16|mfcc23&pitch|extended x-vector|yes|yes|PLDA|1.888
||||||Cosine->+AS-Norm|1.967->1.729

Note, 2000 utterances were selected from no-aug-trainset as the cohort set of AS-Norm, the same below.

***... information updating ...***

---

**ii. Test Voxceleb1-O/E/H**

It means the trainset could only be sampled from Voxceleb2 with a fixed training condition. The training script is available in [subtools/recipe/voxcelebSRC/runVoxcelebSRC.sh](./recipe/voxcelebSRC/runVoxcelebSRC.sh).

**Results of Voxceleb1-O/E/H with Voxceleb2.dev.aug1:4 Training (EER%)**

Index|Features|Model|InSpecAug|AM-Softmax (m=0.2)|Back-End|voxceleb1-O*|voxceleb1-O|voxceleb1-E|voxceleb1-H
:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
1|mfcc23&pitch|extended x-vector|no|no|PLDA|1.622|2.089|2.221|3.842
2|fbank40&pitch|resnet34-2d|no|no|PLDA|1.909|3.065|2.392|3.912
||||||Cosine->+AS-Norm|2.158->-|2.423->2.344|2.215->2.01|4.873->3.734
3|fbank40&pitch|resnet34-2d|no|yes|PLDA|1.622|1.893|1.962|3.546
||||||Cosine->+AS-Norm|1.612->1.543|1.713->1.591|1.817->1.747|3.269->3.119
4|fbank40&pitch|resnet34-2d|yes|yes|PLDA|1.495|1.813|1.920|3.465
||||||Cosine->+AS-Norm|1.601->1.559|1.676->1.601|1.817->1.742|3.233->3.097
5|fbank80|resnet34-2d|no|yes|PLDA|1.511|1.808|1.847|3.251
||||||Cosine->+AS-Norm|1.538->-|1.628->1.538|1.767->1.705|3.111->2.985

Note, Voxceleb1.dev was used as a trainset in the back-end for Voxceleb1-O* and Voxceleb2.dev for others. 


 > **These basic models performs good but the results are not the state-of-the-art yet**. I found that training strategies could have an important influence to the final performance, such as the number of epoch, the value of weight decay, the selection of optimizer, and so on. Unfortunately, I have not enough time and GPU to fine-tune so many models, especially training model with big dataset like Voxceleb2 whose duration is more than 2300h (In this case, it will spend 1~2 days if to train one fbank80-based Resnet2d model for 6 epochs with 4 V100 GPUs).
 >
 > ------Snowdar---2020-06-02------

***... information updating ...***

---

### [2] AP-OLR Challenge 2020 Baseline Recipe [Language Identification]

***... information updating ...***

**Information**

AP-OLR Challenge is opened now, welcome to register. 

Home Page: http://cslt.riit.tsinghua.edu.cn/mediawiki/index.php/OLR_Challenge_2020.

Plan:
Important Dates:

For previous challenges (2016-2019), see http://olr.cslt.org.

**Baseline**

The baseline training script is available in [subtools/recipe/ap-olr2020-baseline/run.sh](./recipe/ap-olr2020-baseline/run.sh).

---

## Feedback
+ If you find bugs or have some questions, please create a github issue in this project to let everyone know it, so that a good solution could be contributed.
+ If you have questions to me, you can also send e-mail to snowdar@stu.xmu.edu.cn and I will reply in my free time.

## Acknowledgement
+ Thanks to everyone who contribute their time, ideas and codes to ASV-Subtools.
+ Thanks to [XMU Speech Lab](https://speech.xmu.edu.cn/) providing machine and GPU.
+ Thanks to the excelent projects: [Kaldi](http://www.kaldi-asr.org/), [Pytorch](https://pytorch.org/), [Kaldi I/O](https://github.com/vesis84/kaldi-io-for-python), [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Horovod](https://github.com/horovod/horovod), [Progressbar2](https://github.com/WoLpH/python-progressbar), [Matplotlib](https://matplotlib.org/index.html), [Prefetch Generator](https://github.com/justheuristic/prefetch_generator), [Thop](https://github.com/Lyken17/pytorch-OpCounter), [GPU Manager](https://github.com/QuantumLiu/tf_gpu_manager), etc.