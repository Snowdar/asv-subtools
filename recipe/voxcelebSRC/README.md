## Reports
### Results of ResNet34
* Egs = Voxceleb2_dev(online random aug) + sequential sampling(2s)  
* Optimization = [SGD (lr = 0.04) + ReduceLROnPlateau] x 4 GPUs (total batch-size=512)
* ResNet34 (channels = 32, 64, 128, 256) + Stats-Pooling + FC-BN + AM-Softmax (margin = 0.2) + AMP training
* Back-end = near + Cosine

| EER% | vox1-O | vox1-O-clean | vox1-E | vox1-E-clean | vox1-H | vox1-H-clean |
|:-----|:------:|:------------:|:------:|:------------:|:------:|:------------:|
|  Submean | 1.071 |  0.920 | 1.257 | 1.135 | 2.205 | 2.072 |
|  AS-Norm | 0.970 |  0.819 |   -   |   -   |   -   |   -   |
<br/>

### Results of ECAPA-TDNN
* Egs = Voxceleb2_dev(online random aug) + random chunk(2s)  
* Optimization = [adamW (lr = 1e-8 - 1e-3) + cyclic for 3 cycle with triangular2 strategy] x 4 GPUs (total batch-size=512)
* ECAPA-TDNN (channels = 1024) + FC-BN + AAM-Softmax (margin = 0.2)
* Back-end = near + Cosine

| EER% | vox1-O | vox1-O-clean | vox1-E | vox1-E-clean | vox1-H | vox1-H-clean |
|:-----|:------:|:------------:|:------:|:------------:|:------:|:------------:|
|  Submean | 1.045 |  0.904 | 1.330 | 1.211 | 2.430 | 2.303 |
|  AS-Norm | 0.991 |  0.856 |   -   |   -   |   -   |   -   |
<br/>


### Results of Conformer
* Egs = Voxceleb2_dev(online random aug) + random chunk(3s) 
* Optimization = [adamW (lr = 1e-6 - 1e-3) + 1cycle] x 4 GPUs (total batch-size=512)
* Conformer + FC-Swish-LN + ASP + FC-LN + AAM-Softmax (margin = 0.2))
* Back-end = near + Cosine
* LM: Large-Margin Fine-tune (margin: 0.2 --> 0.5, chunk: 6s)

| Config                       |        | vox1-O | vox1-O-clean | vox1-E | vox1-E-clean | vox1-H | vox1-H-clean |
|:---------------------------- |:------:|:------:|:------------:|:------:|:------------:|:------:|:------------:|
| 6L-256D-4H-4Sub (50 epochs)  |  Cosine  | 1.204 |  1.074 | 1.386 | 1.267 | 2.416 | 2.294 |
|                              |  AS-Norm | 1.092 |  0.952 |   -   |   -   |   -   |   -   |         
| $\quad+$ SAM training        |  Cosine  | 1.103 |  0.984 | 1.350 | 1.234 | 2.380 | 2.257 |
|                              |  LM      | 1.034 |  0.899 | 1.181 | 1.060 | 2.079 | 1.953 |
|                              |  AS-Norm | 0.943 |  0.792 |   -   |   -   |   -   |   -   |
| 6L-256D-4H-2Sub (30 epochs)  |  Cosine  | 1.066 |  0.915 | 1.298 | 1.177 | 2.167 | 2.034 |
|                              |  LM      | 1.029 |  0.888 | 1.160 | 1.043 | 1.923 | 1.792 |
|                              |  AS-Norm | 0.949 |  0.792 |   -   |   -   |   -   |   -   |      
<br/>

### Results of RTF
* RTF is evaluated on LibTorch-based runtime, see `subtools/runtime`
* One thread is used for CPU threading and TorchScript inference. 
* CPU: Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz.

| Model | Config | Params | RTF | 
|:-----|:------  |:------:|:---:|
|  ResNet34  | base32 |  6.80M  | 0.090 |
|  ECAPA     | C1024  |  16.0M  | 0.071 |
|            | C512   |  6.53M  | 0.030 |
|  Conformer | 6L-256D-4H-4Sub  |  18.8M |   0.025   |  
|            | 6L-256D-4H-2Sub  |  22.5M |   0.070   |   