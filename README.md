# ASV-Subtools
    Copyright xmuspeech (Author: Snowdar 2020-02-27)

    Subtools is a set of tools which is based on Pytorch + Kaldi for speaker recognition etc..

## 声明
本项目工程在开源之前不可外传，仅供内部使用

## 用途
ASV-Subtools包含一整套声纹识别流程，每个模块均有大量优化脚本和算法，且上层有良好的封装，可快速构建实验工程，开展实验  
另外，神经网络训练使用pytorch框架，网络结构相关idea的实现清晰易做，可促进论文发表工作  
ASV-Subtools争取跟进state-of-the-art的算法框架，可用于比赛和项目的实验

## 声纹识别 Recipe
一个基于voxceleb的标准pipeline，以供参考学习，详见脚本：

    subtools/recipe/voxceleb/runVoxceleb.sh

## 克隆
在工程目录下，如kaldi/egs/xmuspeech/sre, 克隆subtools：

    git clone https://github.com/Snowdar/subtools/.git

## 更新
第一次克隆之后，之后更新进入到subtools目录并使用更新命令:

    cd subtools
    git pull

## 依赖库安装

yum安装出问题残留清理

    yum -y install yum-utils
    yum clean all
    yum-complete-transaction --cleanup-only
    package-cleanup --cleandupes

[1] 基本依赖包

    pip3 install torch numpy pandas progressbar2

[2] 多GPU训练依赖包 <方案 = Horovod：https://github.com/horovod/horovod#id10>
    
    + NCCL安装 <方案 = 从yum网络库安装：https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html>

      # Navidia yum库下载（Centos7，cuda10.2：https://developer.nvidia.com/nccl/nccl-download）
      wget https://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm

      # Navidia yum库安装(NOKEY警告可忽略)
      sudo rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm

      # 安装NCCL(nccl2.6.4+cuda10.2)
      sudo yum install libnccl-2.6.4-1+cuda10.2 libnccl-devel-2.6.4-1+cuda10.2 libnccl-static-2.6.4-1+cuda10.2
    
    + Openmpi安装（高性能通信包） <方案 = 下载编译安装：https://www.open-mpi.org/faq/?category=building#easy-build>

      # 源代码下载（3.1.2版本正常，高版本可能异常）
      wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.2.tar.gz

      # 解压
      tar zxf openmpi-3.1.2.tar.gz

      # 配置检查与编译安装
      cd openmpi-3.1.2

      ./configure --prefix=/usr/local

      make -j 4

      make install

    + Horovod安装

      # GCC版本问题 < 方案 = 临时使用高版本GCC-6.3：https://www.vpser.net/manage/centos-6-upgrade-gcc.html>
          # 更新yum源并安装GCC-6.3
          yum -y install centos-release-scl
          yum -y install devtoolset-6-gcc devtoolset-6-gcc-c++ devtoolset-6-binutils
          
          # 临时启用GCC-6.3（仅当前终端生效）
          scl enable devtoolset-6 bash 或 source /opt/rh/devtoolset-6/enable

     # 若上述方法安装后 import horovod.torch时，出现 "/lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found" 问题 < 方案 = 编译安装: https://blog.csdn.net/Yanci_/article/details/80016097>
          # 下载源码包
          wget http://mirrors.concertpass.com/gcc/releases/gcc-6.3.0/gcc-6.3.0.tar.gz
          tar xzf gcc-6.3.0.tar.gz
          cd gcc-6.3.0
          ./contrib/download_prerequisites
          mkdir gcc-build-6.3.0
          cd gcc-build-6.3.0
          ../configure --enable-checking=release --enable-languages=c,c++ --disable-multilib
          make -j 4
          make install

          在 /root/.bashrc 中添加环境变量 export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

      # 安装GPU支持版本（基于NCCL依赖）
      HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL pip3 install horovod

      # 环境变量
      在 /etc/profile 或 /root/.bashrc 中添加

        export PATH=$PATH:/usr/local/python3/bin/

## 问题反馈
本项目定位为开源项目，若有相关问题请联系作者Snowdar [snowdar@stu.xmu.edu.cn]  
欢迎报告存在的缺陷和bug，并协助进行相关的修复工作  
欢迎提供新的idea并贡献代码