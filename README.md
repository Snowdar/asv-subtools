# ASV-Subtools: An Open Source Tools for Speaker Recognition

ASV-Subtools is developed based on [Pytorch](https://pytorch.org/) and [Kaldi](http://www.kaldi-asr.org/) for the task of speaker recognition, language identification, etc.  
The 'sub' of 'subtools' means that there are many modular tools and the parts constitute the whole. 

> Copyright: [TalentedSoft-XMU Speech Lab] [XMU Speech Lab](https://speech.xmu.edu.cn/) (Xiamen University, China) [TalentedSoft](http://www.talentedsoft.com/) (TalentedSoft, China)
> Apache 2.0
>
> Author   : Miao Zhao (Email: snowdar@stu.xmu.edu.cn), Jianfeng Zhou, Zheng Li, Hao Lu, Fuchuan Tong, Dexin Liao, Tao Jiang  
> Current Maintainer: Tao Jiang (Email: sssyousen@163.com)  
> Co-author: Lin Li, Qingyang Hong


Citation: 

```
@inproceedings{tong2021asv,
  title={{ASV-Subtools}: {Open} Source Toolkit for Automatic Speaker Verification},
  author={Tong, Fuchuan and Zhao, Miao and Zhou, Jianfeng and Lu, Hao and Li, Zheng and Li, Lin and Hong, Qingyang},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6184--6188},
  year={2021},
  organization={IEEE}
}
```


---
- **Content**
  * [Introduction](#introduction)
    + [Project Structure](#project-structure)
    + [Training Framework](#training-framework)
    + [Data Pipeline](#data-pipeline)
    + [Update Pipeline](#update-pipeline)
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
      - [Add A MMI-GMM Classifier for The Back-End](#add-a-mmi-gmm-classifier-for-the-back-end)
  * [Training Model](#training-model)
  * [Recipe](#recipe)
    + [[1] Voxceleb Recipe [Speaker Recognition]](#1-voxceleb-recipe-speaker-recognition)
    + [[2] OLR Challenge 2020 Baseline Recipe [Language Identification]](#2-olr-challenge-2020-baseline-recipe-language-identification)
    + [[3] OLR Challenge 2021 Baseline Recipe [Language Identification]](#3-olr-challenge-2021-baseline-recipe-language-identification)
    + [[4] CNSRC 2022 Baseline Recipe [Speaker Recognitiopn]](#4-cnsrc-2022-baseline-recipe-speaker-recognition)
  * [Feedback](#feedback)
  * [Acknowledgement](#acknowledgement)

<!--Table of contents generated with markdown-toc, see http://ecotrust-canada.github.io/markdown-toc-->
---

## Introduction  

In ASV-Subtools, [Kaldi](http://www.kaldi-asr.org/) is used to extract acoustic features and scoring in the back-end and [Pytorch](https://pytorch.org/) is used to build a model freely and train it with a custom style.

The project structure, training framework and data pipeline shown as follows could help you to have some insights into ASV-Subtools.

> By the way, **if you can not see the pictures in Github**, maybe you should try to check the DNS of your network or use a VPN agent. If you are a student of XMU, then the VPN of campus network could be very helpful for these types of problems (see [https://vpn.xmu.edu.cn](https://vpn.xmu.edu.cn) for a configuration). Of course, **at least the last way is to clone ASV-Subtools to your local notebook.**

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

In ASV-Subtools, the model is individual, which means that we should know the path of ```model.py``` and how to initialize this model class at least when using this model in training or testing module. This structure is designed to avoid modifying codes of static modules frequently. For example, if the embedding extractor is wrote down as a called program and we use an inline method ```from my_model_py import my_model``` to import a fixed model from a fixed ```model.py``` , then it will be not free for ```model_2.py```, ```model_3.py``` and so on.

**Note that**, all models ([torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)) shoud inherit [libs.nnet.framework.TopVirtualNnet](./pytorch/libs/nnet/framework.py) class to get some default functions, such as **auto-saving model creation and blueprint**, extracting emdedding of whole utterance, step-training, computing accuracy, etc.. It is easy to transform the original model of Pytorch to ASV-Subtools model by inheriting. Just modify your ```model.py``` w.r.t this [x-vector example](./pytorch/model/xvector.py).

### Training Framework  
The basic training framework is provided here and the relations between every module are very clear. So it will be not complex if you want to change anything when you want to have a custom ASV-Subtools.  

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

### Update Pipeline
- **20220707**
  + Online Datasets is implemented (Including online feature extracting, online VAD, online augmentation and online x-vector extracting)
  + Supporting mixed precision training.
  + Runtime module for exporting jit model.
  + Updating some models.
### Support List

- **Multi-GPU Training Solution**
  + [x] [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/nn.html#distributeddataparallel) [Built-in function of Pytorch]
  + [x] [Horovod](https://github.com/horovod/horovod)

- **Front-end**
  + [x] [Convenient Augmentation of Reverb, Noise, Music and Babble](./augmentDataByNoise.sh)
  + [x] Inverted [Specaugment](https://arxiv.org/pdf/1904.08779.pdf) [Note, it is still not available with multi-gpu and you will not get a better result if do it.]
  + [x] [Mixup](https://arxiv.org/pdf/1710.09412.pdf) [For speaker recognition, see this [paper](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2250.pdf).]
  + [x] [Online Datasets] [Including online feature extracting, online VAD, online augmentation and online xvector extracting, developed by Dexin Liao] 

- **Model**
  + [x] [Standard X-vector](http://www.danielpovey.com/files/2017_interspeech_embeddings.pdf)
  + [x] [Extended X-vector](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8683760)
  + [x] Resnet1d
  + [x] [Resnet2d](http://www.danielpovey.com/files/2019_interspeech_nist_sre18.pdf)
  + [x] [F-TDNN X-vector](http://www.danielpovey.com/files/2019_interspeech_nist_sre18.pdf)
  + [x] [ECAPA X-vector](https://arxiv.org/abs/2005.07143) [[Source codes](https://github.com/lawlict/ECAPA-TDNN) ]
  + [x] [RepVGG]() 

- **Component**
  + [x] [Attentive Statistics Pooling](https://arxiv.org/pdf/1803.10963v1.pdf)
  + [x] [Learnable Dictionary Encoding (LDE) Pooling](https://arxiv.org/pdf/1804.00385.pdf)
  + [x] [Multi-Head Attention Pooling](https://upcommons.upc.edu/bitstream/handle/2117/178623/2616.pdf?sequence=1&isAllowed=y) [The codes could be found [here](./pytorch/libs/nnet/pooling.py), by Snowdar.]
  + [x] [Global Multi-Head Attention Pooling](https://www.researchgate.net/publication/341085045_Multi-Resolution_Multi-Head_Attention_in_Deep_Speaker_Embedding)
  + [x] [Multi-Resolution Multi-Head Attention Pooling](https://www.researchgate.net/publication/341085045_Multi-Resolution_Multi-Head_Attention_in_Deep_Speaker_Embedding)
  + [x] [Sequeze and Excitation (SE)](https://arxiv.org/pdf/1709.01507.pdf) [A resnet1d-based SE example of speaker recognition could be found in this [paper](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1704.pdf), by Jianfeng Zhou.] [Updating resnet2d-based SE]
  + [x] [Xi-vector embedding](https://ieeexplore.ieee.org/document/9463712), [by [Dr. Kong Aik Lee](https://ieeexplore.ieee.org/author/37293718000).]

- **Loss Function**
  + [x] Softmax Loss (Affine + Softmax + Cross-Entropy)
  + [x] [AM-Softmax Loss](https://arxiv.org/pdf/1801.05599.pdf)
  + [x] [AAM-Softmax Loss](https://arxiv.org/pdf/1801.07698v1.pdf)
  + [x] [Ring Loss](https://arxiv.org/pdf/1803.00130.pdf)

  <!--+ [x] [Curricular Margin Softmax Loss](https://arxiv.org/pdf/2004.00288.pdf)-->
  <!--It does not work in my experiments-->

- **Optimizer** [Out of Pytorch built-in functions]
  + [x] [Lookahead](https://arxiv.org/pdf/1907.08610.pdf) [A wrapper optimizer]
  + [x] [RAdam](https://arxiv.org/pdf/1908.03265v1.pdf)
  + [x] Ralamb [RAdam + [Layer-wise Adaptive Rate Scaling](https://openreview.net/pdf?id=rJ4uaX2aW) (LARS)]
  + [x] [Novograd](https://arxiv.org/pdf/1905.11286.pdf)
  + [x] [Gradient Centralization](https://arxiv.org/pdf/2004.01461.pdf) [Extra bound to optimizer]

- **Training Strategy**
  + [x] [AdamW](https://arxiv.org/pdf/1711.05101v1.pdf) + [WarmRestarts](https://arxiv.org/pdf/1608.03983v4.pdf)
  + [x] SGD + [ReduceLROnPlateau](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
  + [x] [Training with Magin Warmup Strategy](https://arxiv.org/pdf/1904.03479.pdf)
  + [x] [Heated Up Strategy](https://arxiv.org/pdf/1809.04157.pdf)
  + [x] [Multi-task Learning with Phonetic Information](http://yiliu.org.cn/papers/Speaker_Embedding_Extraction_with_Phonetic_Information.pdf) (Kaldi) [[Source codes](https://github.com/mycrazycracy/speaker-embedding-with-phonetic-information) was contributed by [Yi Liu](http://yiliu.org.cn/). Thanks.]
  + [x] [Multi-task Learning with Phonetic Information (Pytorch)](./recipe/ap-olr/runMultiTaskXvector.py) [developed by Zheng Li]
  + [x] [Feature Decomposition and Cosine Similar Adversarial Learning (FD-AL)](./pytorch/launcher/runEtdnn-FD-AL-trainer.py) [[Reference] (https://doi.org/10.48550/arXiv.2205.14294)] [developed by Fuchuan Tong] 

- **Back-End**
  + [x] LDA, Submean, Whiten (ZCA), Vector Length Normalization
  + [x] Cosine Similarity
  + [x] Basic Classifiers: SVM, GMM, Logistic Regression (LR)
  + [x] PLDA Classifiers: [PLDA](https://ravisoji.com/assets/papers/ioffe2006probabilistic.pdf), APLDA, [CORAL](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12443/11842), [CORAL+](https://arxiv.org/pdf/1812.10260), [LIP](http://150.162.46.34:8080/icassp2014/papers/p4075-garcia-romero.pdf), [CIP](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9054113) [[Python versions](./score/pyplda) was contributed by Jianfeng Zhou. For more details, see the [note](./score/pyplda/Domain-Adaptation-of-PLDA-in-Speaker-Recognition.pdf).]
  + [x] Score Normalization: [S-Norm](http://www.crim.ca/perso/patrick.kenny/kenny_Odyssey2010_presentation.pdf), [AS-Norm](https://www.researchgate.net/profile/Daniele_Colibro/publication/221480280_Comparison_of_Speaker_Recognition_Approaches_for_Real_Applications/links/545e4f6e0cf295b561602c42/Comparison-of-Speaker-Recognition-Approaches-for-Real-Applications.pdf)
  + [ ] Calibration
  + [x] Metric: EER, Cavg, minDCF

- **Runtime**
  + [x] export jit model.

- **Others**
  + [x] [Learning Rate Finder](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html)
  + [x] Support [TensorboardX](https://tensorflow.google.cn/tensorboard) in Log System ==*new*==
  + [ ] Plot DET Curve with ```matplotlib``` w.r.t the Format of DETware (Matlab Version) of [NIST's Tools](https://www.nist.gov/itl/iad/mig/tools)
  + [ ] Accumulate Total MACs and Flops of Model Based on ```thop```
  + [x] Training with AMP (apex or torch1.9)

## Ready to Start  
### 1. Install Kaldi  
Pytorch-training is not much related to Kaldi, but we have not provided other interfaces to concatenate acoustic feature and training module now. So if you don't want to use Kaldi, you could change the [libs.egs.egs.ChunkEgs](./pytorch/libs/egs/egs.py) class where the features are given to Pytorch only by [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset). Besides, you should also change the interface of extracting x-vector after training. Note that, most of scripts which require Kaldi could be not available in this case, such as subtools/makeFeatures.sh and subtools/augmentDataByNoise.sh.

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
bash extras/check_dependencies.sh
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
git clone https://github.com/Snowdar/asv-subtools.git subtools
```

### 4. Install Python Requirements  
+ Pytorch>=1.2: ```pip3 install torch```
+ Other requirements: numpy, thop, pandas, progressbar2, matplotlib, scipy (option), sklearn (option)  
  ```pip3 install -r requirements.txt```

### 5. Support Multi-GPU Training  
ASV-Subtools provide both **DDP (recommended)** and Horovod solutions to support multi-GPU training.

**Some answers about how to use multi-GPU training, see [subtools/pytorch/launcher/runSnowdarXvector.py](./pytorch/launcher/runSnowdarXvector.py). It is very convenient and easy now.**

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
bash subtools/kaldi/patch/runPatch-multitask.sh
```

#### Accelerate X-vector Extractor of Kaldi
It will spend so much time to compile nnet3 models for the utterances with different frames when extracting x-vectors based on Kaldi. To optimize this problem, ASV-Subtools provides an **offine** modification (MOD) in [subtools/kaldi/sid/nnet3/xvector/extract_xvectors.sh](./kaldi/sid/nnet3/xvector/extract_xvectors.sh) to accelerate extracting. This MOD requires two extra commands, **nnet3-compile-xvector-net** and **nnet3-offline-xvector-compute**. When extracting x-vectors, all models with different input chunk-size will be compiled firstly. Then the utterances which have the same frames could share a compiled nnet3 network. It saves much time by avoiding a lot of duplicate dynamic compilations.

Besides, the ```scp``` spliting type w.r.t length of utterances ([subtools/splitDataByLength.sh](./splitDataByLength.sh)) is adopted to balance the frames of different ```nj``` when multi-processes is used.

```shell
# Enter your project, such as kaldi/egs/xmuspeech/sre and make sure ASV-Subtools is cloned here
# Just run this patch to compile some extra C++ commands with Kaldi's format

# Target *.cc:
#     src/nnet3bin/nnet3-compile-xvector-net.cc
#     src/nnet3bin/nnet3-offline-xvector-compute.cc

cd kaldi/egs/xmuspeech/sre
bash subtools/kaldi/patch/runPatch-base-command.sh
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
bash subtools/kaldi/patch/runPatch-base-command.sh
```
## Training Model
If you have completed the [Ready to Start](#ready-to-start) stage, then you could try to train a model with ASV-Subtools.

For kaldi training, some launcher scripts named ```run*.sh``` could be found in [subtoos/Kaldi/](./kaldi).

For pytorch training, some launcher scripts named ```run*.py``` could be found in [subtools/pytorch/launcher/](./pytorch/launcher/). And some models named ```*.py``` could be found in [subtools/pytorch/model/](./pytorch/model).  Note that, model will be called in ```launcher.py```.

Here is a pytorch training example, but you should follow a [pipeline](./recipe/voxceleb/runVoxceleb.sh) of [recipe](#recipe) to prepare your data and features before training. The part of data preprocessing is not complex and it is the same as Kaldi. 

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
[Voxceleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/index.html#about) is a popular dataset for the task of speaker recognition. It has two parts now, [Voxceleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and [Voxceleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html).

There are **two recipes for Voxceleb**:

**i. Test Voxceleb1-O only**

It means the trainset could be sampled from both Voxceleb1.dev and Voxceleb2 with a fixed training condition. The training script is available in [subtools/recipe/voxceleb/runVoxceleb.sh](./recipe/voxceleb/runVoxceleb.sh).

The voxceleb1 recipe with mfcc23&pitch features is available:  
**Link**: https://pan.baidu.com/s/1nMXaAXiOnFGRhahzVyrQmg  
**Password**: 24sg

```shell
# Download this recipe to kaldi/egs/xmuspeech directory
cd kaldi/egs/xmuspeech
tar xzf voxceleb1_recipe.tar.gz
cd voxceleb1

# Clone ASV-Subtools (Suppose the configuration of related environment has been done)
git clone https://github.com/Snowdar/asv-subtools.git subtools

# Train an extended x-vector model (Do not use multi-GPU training for it is not stable for specaugment.)
subtools/runPytorchLauncher.sh runSnowdarXvector-extended-spec-am.py --stage=0

# Score (EER = 2.444% for voxceleb1.test)
subtools/recipe/voxceleb/gather_results_from_epochs.sh --vectordir exp/extended_spec_am --epochs 21 --score plda
```

**Results of Voxceleb1-O with Voxceleb1.dev.aug1:1 Training only**

![results-1.png](./recipe/voxceleb/results-1.png)

<!--
<table>
<tr style="white-space: nowrap;text-align:left;">
<th>Index</th>
<th>Features</th>
<th>Model</th>
<th>InSpecAug</th>
<th>AM-Softmax (m=0.2)</th>
<th>Back-End</th>
<th>EER%</th>
</tr>
<tr style="white-space: nowrap;text-align:left;">
<td>1</td>
<td>mfcc23&pitch</td>
<td>x-vector</td>
<td>no</td>
<td>no</td>
<td>PLDA</td>
<td>3.362</td>
</tr>
<tr style="white-space: nowrap;text-align:left;">
<td>2</td>
<td>mfcc23&pitch</td>
<td>x-vector</td>
<td>yes</td>
<td>no</td>
<td>PLDA</td>
<td>2.778</td>
</tr>
<tr style="white-space: nowrap;text-align:left;">
<td>3</td>
<td>mfcc23&pitch</td>
<td>x-vector</td>
<td>no</td>
<td>yes</td>
<td>PLDA</td>
<td>3.240</td>
</tr>
<tr style="white-space: nowrap;text-align:left;">
<td>4</td>
<td>mfcc23&pitch</td>
<td>x-vector</td>
<td>yes</td>
<td>yes</td>
<td>PLDA</td>
<td>2.635</td>
</tr>
<tr style="white-space: nowrap;text-align:left;">
<td>5</td>
<td>mfcc23&pitch</td>
<td>extended x-vector</td>
<td>no</td>
<td>no</td>
<td>PLDA</td>
<td>3.112</td>
</tr>
<tr style="white-space: nowrap;text-align:left;">
<td>6</td>
<td>mfcc23&pitch</td>
<td>extended x-vector</td>
<td>yes</td>
<td>no</td>
<td>PLDA</td>
<td>2.598</td>
</tr>
<tr style="white-space: nowrap;text-align:left;">
<td>7</td>
<td>mfcc23&pitch</td>
<td>extended x-vector</td>
<td>no</td>
<td>yes</td>
<td>PLDA</td>
<td>3.293</td>
</tr>
<tr style="white-space: nowrap;text-align:left;">
<td>8</td>
<td>mfcc23&pitch</td>
<td>extended x-vector</td>
<td>yes</td>
<td>yes</td>
<td>PLDA</td>
<td>2.444</td>
</tr>
</table>
-->

<!--HTML codes of table is generated by subtools/linux/generate_html_table_for_markdown.sh-->

**Results of Voxceleb1-O with Voxceleb1&2.dev.aug1:1 Training**

![results-2.png](./recipe/voxceleb/results-2.png)

Note, 2000 utterances are selected from no-aug-trainset as the cohort set of AS-Norm, the same below.

---

**ii. Test Voxceleb1-O/E/H**

It means the trainset could only be sampled from Voxceleb2 with a fixed training condition.

**Old Results of Voxceleb1-O/E/H with Voxceleb2.dev.aug1:4 Training (EER%)**

![results-3.png](./recipe/voxcelebSRC/results-adam.png)

These models are trained by adam + warmRestarts and they are old (so related scripts was removed).
Note, Voxceleb1.dev is used as the trainset of back-end for the Voxceleb1-O* task and Voxceleb2.dev for others. 

 > **These basic models performs good but the results are not the state-of-the-art yet**. I found that training strategies could have an important influence on the final performance, such as the number of epoch, the value of weight decay, the selection of optimizer, and so on. Unfortunately, I have not enough time and GPU to fine-tune so many models, especially training model with a large dataset like Voxceleb2 whose duration is more than 2300h (In this case, it will spend 1~2 days to train one fbank81-based Resnet2d model for 6 epochs with 4 V100 GPUs).
 >
 > --#--Snowdar--2020-06-02--#--

**New Results of Voxceleb1-O/E/H with Voxceleb2.dev.aug1:4 Training (EER%)**

Here, this is a resnet34 benchmark model. And the training script is available in [subtools/recipe/voxcelebSRC/runVoxcelebSRC.sh](./recipe/voxcelebSRC/runVoxcelebSRC.sh). For more details, see it also. (by Snowdar)

|EER%|vox1-O|vox1-O-clean|vox1-E|vox1-E-clean|vox1-H|vox1-H-clean|
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|Baseline|1.304|1.159|1.35|1.223|2.357|2.238|
|Submean|1.262|1.096|1.338|1.206|2.355|2.223|
|AS-Norm|1.161|1.026|-|-|-|-|
---
**New Results of Voxceleb1-O/E/H with Voxceleb2.dev.aug.speed1:4:2 Training (EER%)**
Here, this is an ECAPA benchmark model. And the training script is available in [subtools/pytorch/launcher/runEcapaXvector.py](./pytorch/launcher/runEcapaXvector.py). For more details, see it also. (by Fuchuan Tong) ==new==

|EER%|vox1-O|vox1-O-clean|vox1-E|vox1-E-clean|vox1-H|vox1-H-clean|
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|Baseline|1.506|1.393|1.583|1.462|2.811|2.683|
|Submean|1.225|1.112|1.515|1.394|2.781|2.652|
|AS-Norm|1.140|0.963|-|-|-|-|
---
**New Results of Voxceleb1-O/E/H with original Voxceleb2.dev (without data augmentation) Training (EER%)**
Here, this is an statistical pooling and Xi-vector embedding benchmark model (implement on TDNN). And the training script is available in [subtools/pytorch/launcher/runSnowdar_Xivector.py](./pytorch/launcher/runSnowdar_Xivector.py). We would like to thank Dr. Kong Aik Lee for providing codes and useful discussion. (experiments conducted by Fuchuan Tong) ==2021-10-30==
|EER%|vox1-O|vox1-E|vox1-H|
| :--: | :--: | :--: | :--: |
|Statistics Pooling|1.85|2.01|3.57|
|Multi-head|1.76|2.00|3.54|
|Xi-Vector(∅,𝜎)|1.59|1.90|3.38|
---

**New Results of Voxceleb1-O/E/H with Voxceleb2.dev (online random augmentation) Training(EER%)**
Here, this is a resnet34 benchmark model. And the training script is available in [subtools/pytorch/launcher/runResnetXvector_online.py](./pytorch/launcher/runResnetXvector_online.py). For more details, see it also. (experiments conducted by Dexin Liao) ==2022-07-07==
|EER%|vox1-O|vox1-O-clean|vox1-E|vox1-E-clean|vox1-H|vox1-H-clean|
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|Submean|1.071|0.920|1.257|1.135|2.205|2.072|
|AS-Norm|0.970|0.819|-|-|-|-|

Here, this is a ECAPA benchmark model. And the training script is available in [subtools/pytorch/launcher/runEcapaXvector_online.py](./pytorch/launcher/runEcapaXvector_online.py). For more details, see it also. (experiments conducted by Dexin Liao) ==2022-07-07==
|EER%|vox1-O|vox1-O-clean|vox1-E|vox1-E-clean|vox1-H|vox1-H-clean|
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|Submean|1.045|0.904|1.330|1.211|2.430|2.303|
|AS-Norm|0.991|0.856|-|-|-|-|
---
### [2] OLR Challenge 2020 Baseline Recipe [Language Identification]

OLR Challenge 2020 is closed now.

**Baseline**: [subtools/recipe/ap-olr2020-baseline](./recipe/ap-olr2020-baseline).  
> The **top training script of baseline** is available in [subtools/recipe/ap-olr2020-baseline/run.sh](./recipe/ap-olr2020-baseline/run.sh). And the baseline results could be seen in [subtools/recipe/ap-olr2020-baseline/results.txt](./recipe/ap-olr2020-baseline/results.txt).

**Plan**: Zheng Li, Miao Zhao, Qingyang Hong, Lin Li, Zhiyuan Tang, Dong Wang, Liming Song and Cheng Yang: [AP20-OLR Challenge: Three Tasks and Their Baselines](https://arxiv.org/pdf/2006.03473.pdf), submitted to APSIPA ASC 2020.

### [3] OLR Challenge 2021 Baseline Recipe [Language Identification]

**Baseline**: [subtools/recipe/olr2021-baseline](./recipe/olr2021-baseline).  
> The **top training script of baseline** is available in [subtools/recipe/olr2021-baseline/run.sh](./recipe/olr2021-baseline/run.sh). 

**Plan**: Binling Wang, Wenxuan Hu, Jing Li, Yiming Zhi, Zheng Li, Qingyang Hong, Lin Li, Dong Wang, Liming Song and Cheng Yang: [OLR 2021 Challenge: Datasets, Rules and Baselines](http://cslt.riit.tsinghua.edu.cn/mediawiki/images/a/a8/OLR_2021_Plan.pdf), submitted to APSIPA ASC 2021.

For previous challenges (2016-2020), see http://olr.cslt.org.

### [4] CNSRC 2022 Baseline Recipe [Speaker Recognition]

**Baseline**: [subtools/recipe/cnsrc](./recipe/cnsrc).  
> The **top training script of baseline** is available in [subtools/recipe/cnsrc/sv/run-cnsrc_sv.sh](./recipe/cnsrc/sv/run-cnsrc_sv.sh) and [subtools/recipe/cnsrc/sr/run-cnsrc_sr.sh](./recipe/cnsrc/sr/run-cnsrc_sr.sh).

**Plan**: Dong Wang, Qingyang Hong, Liantian Li, Hui Bu: [CNSRC 2022 Evaluation Plan](http://aishell-jiaofu.oss-cn-hangzhou.aliyuncs.com/cnsrc.pdf).

For more informations, see http://cnceleb.org.
For any Challenge questions please contact lilt@cslt.org and for any baseline questions contact sssyousen@163.com.

---

## Feedback
+ If you find bugs or have some questions, please create a github issue in this repository to let everyone knows it, so that a good solution could be contributed.
+ If you want to ask some questions, just send e-mail to sssyousen@163.com (Tao Jiang) or snowdar@stu.xmu.edu.cn (Snowdar) for SRE answers and xmulizheng@stu.xmu.edu.cn for LID answers. In general, we will reply you in our free time.
+ If you want to join the WeChat group of asv-subtools, you can scan the QR code on the left to follow XMUSPEECH and reply "join group" + your institution/university + your name. In addtion, you can also scan the QR code on the right and the guy will invite you to the chat group.
</br>
<img src="./doc/xmuspeech.jpg" width="300"/><img src="./doc/sssyousen_wechat_qr.jpg" width="300"/>
</br>


## Acknowledgement
+ Thanks to everyone who contribute their time, ideas and codes to ASV-Subtools.
+ Thanks to [XMU Speech Lab](https://speech.xmu.edu.cn/) providing machine and GPU.
+ Thanks to the excelent projects: [Kaldi](http://www.kaldi-asr.org/), [Pytorch](https://pytorch.org/), [Kaldi I/O](https://github.com/vesis84/kaldi-io-for-python), [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Horovod](https://github.com/horovod/horovod), [Progressbar2](https://github.com/WoLpH/python-progressbar), [Matplotlib](https://matplotlib.org/index.html), [Prefetch Generator](https://github.com/justheuristic/prefetch_generator), [Thop](https://github.com/Lyken17/pytorch-OpCounter), [GPU Manager](https://github.com/QuantumLiu/tf_gpu_manager), etc.

