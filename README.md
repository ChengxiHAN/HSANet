# HSANET：https://chengxihan.github.io/

“[HSANET: A HYBRID SELF-CROSS ATTENTION NETWORK FOR REMOTE SENSING CHANGE DETECTION](https://ieeexplore.ieee.org/document/10283341), IGARSS 2025, Chengxi Han, Xiaoyu Su, Zhiqiang Wei, Meiqi Hu, Yichu Xu*, :yum::yum::yum:


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hcgmnet-a-hierarchical-change-guiding-map/change-detection-on-googlegz-cd)](https://paperswithcode.com/sota/change-detection-on-googlegz-cd?p=hcgmnet-a-hierarchical-change-guiding-map)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hcgmnet-a-hierarchical-change-guiding-map/change-detection-on-levir)](https://paperswithcode.com/sota/change-detection-on-levir?p=hcgmnet-a-hierarchical-change-guiding-map)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hcgmnet-a-hierarchical-change-guiding-map/change-detection-on-sysu-cd)](https://paperswithcode.com/sota/change-detection-on-sysu-cd?p=hcgmnet-a-hierarchical-change-guiding-map)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hcgmnet-a-hierarchical-change-guiding-map/change-detection-on-whu-cd)](https://paperswithcode.com/sota/change-detection-on-whu-cd?p=hcgmnet-a-hierarchical-change-guiding-map)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hcgmnet-a-hierarchical-change-guiding-map/change-detection-on-s2looking)](https://paperswithcode.com/sota/change-detection-on-s2looking?p=hcgmnet-a-hierarchical-change-guiding-map)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hcgmnet-a-hierarchical-change-guiding-map/change-detection-on-dsifn-cd)](https://paperswithcode.com/sota/change-detection-on-dsifn-cd?p=hcgmnet-a-hierarchical-change-guiding-map)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hcgmnet-a-hierarchical-change-guiding-map/change-detection-on-levir-cd)](https://paperswithcode.com/sota/change-detection-on-levir-cd?p=hcgmnet-a-hierarchical-change-guiding-map)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hcgmnet-a-hierarchical-change-guiding-map/change-detection-on-cdd-dataset-season-1)](https://paperswithcode.com/sota/change-detection-on-cdd-dataset-season-1?p=hcgmnet-a-hierarchical-change-guiding-map)


[21st Apir. 2023] Release the first version of the HSANet
![image-20250421](/network/HSANet.png)

### Requirement  
```bash
-Pytorch 1.8.0  
-torchvision 0.9.0  
-python 3.8  
-opencv-python  4.5.3.56  
-tensorboardx 2.4  
-Cuda 11.3.1  
-Cudnn 11.3  
```
## Training, Test and Visualization Process   

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --epoch 50 --batchsize 8  --data_name 'WHU' --model_name 'HSANet'
CUDA_VISIBLE_DEVICES=0 python train.py --epoch 50 --batchsize 8  --data_name 'LEVIR' --model_name 'HSANet'
CUDA_VISIBLE_DEVICES=0 python train.py --epoch 50 --batchsize 8  --data_name 'SYSU' --model_name 'HSANet'
CUDA_VISIBLE_DEVICES=0 python train.py --epoch 50 --batchsize 8  --data_name 'S2Looking' --model_name 'HSANet'

CUDA_VISIBLE_DEVICES=0 python test.py  --data_name 'WHU' --model_name 'HSANet'
CUDA_VISIBLE_DEVICES=0 python test.py  --data_name 'LEVIR' --model_name 'HSANet'
CUDA_VISIBLE_DEVICES=0 python test.py  --data_name 'SYSU' --model_name 'HSANet'
CUDA_VISIBLE_DEVICES=0 python test.py  --data_name 'S2Looking' --model_name 'HSANet'

```
You can change data_name for different datasets like "LEVIR", "WHU", "SYSU", "S2Looking", "CDD", and "DSIFN".
## Test our trained model result 
You can directly test our model by our provided HCGMNet weights in  `output/WHU, LEVIR, SYSU, S2Looking, CDD, and DSIFN `. Download in  [Baidu Disk](https://pan.baidu.com/s/1oZBgohoo0hGstPVU-qC5gw?pwd=2023),pwd:2023 :yum::yum::yum:

And also we provide all test results of our CGNet in the HCGMNetTestResult!!!! Download in HCGMNetTestResult or [Baidu Disk](https://pan.baidu.com/s/1ehrRLAsxgjUMQR4qALSI1g?pwd=2023 ),pwd:2023 :yum::yum::yum:
## Dataset Download   
LEVIR-CD：https://justchenhao.github.io/LEVIR/  , our paper split in [Baidu Disk](https://pan.baidu.com/s/1VVry18KFl2MSWS6_IOlYRA?pwd=2023),pwd:2023 

WHU-CD：http://gpcv.whu.edu.cn/data/building_dataset.html ,our paper split in [Baidu Disk](https://pan.baidu.com/s/1ZLmIyWvHnwyzhyl4xt-GwQ?pwd=2023),pwd:2023

SYSU-CD: Our split in [Baidu Disk](https://pan.baidu.com/s/1p0QfogZm4BM0dd1a0LTBBw?pwd=2023),pwd:2023

S2Looking-CD: Our split in [Baidu Disk](https://pan.baidu.com/s/1wAXPHhCLJTqPX0pC2RBMsg?pwd=2023),pwd:2023

CDD-CD: Our split in [Baidu Disk](https://pan.baidu.com/s/1cwJ0mEhcrbCWOJn5n-N5Jw?pwd=2023),pwd:2023

DSIFN-CD: Our split in [Baidu Disk]( https://pan.baidu.com/s/1-GD3z_eMoQglSJoi9P-6gw?pwd=2023),pwd:2023

Note: We crop all datasets to a slice of 256×256 before training with it.

## Dataset Path Setting
```
 LEVIR-CD or WHU-CD 
     |—train  
          |   |—A  
          |   |—B  
          |   |—label  
     |—val  
          |   |—A  
          |   |—B  
          |   |—label  
     |—test  
          |   |—A  
          |   |—B  
          |   |—label
  ```        
 Where A contains images of the first temporal image, B contains images of the second temporal image, and label contains ground truth maps.  
![image-20230415](/network/HSANet-2.png)



## Acknowledgments
 
Thanks to all my co-authors [Yichu Xu](https://scholar.google.com/citations?user=CxKy4lEAAAAJ&hl=en),[Meiqi Hu](https://meiqihu.github.io/)Thanks  for their great work!!  


## Citation 

 If you use this code for your research, please cite our papers.  

```


```

## Reference  
[1] C. HAN, C. WU, H. GUO, M. HU, J.Li, AND H. CHEN, 
“[Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery](https://ieeexplore.ieee.org/document/10234560?denied=),” IEEE J. SEL. TOP. APPL.EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI:10.1109/JSTARS.2023.3310208 .

[2] C. HAN, C. WU, H. GUO, M. HU, AND H. CHEN, 
“[HANet: A hierarchical attention network for change detection with bi-temporal very-high-resolution remote sensing images](https://ieeexplore.ieee.org/abstract/document/10093022),” IEEE J. SEL. TOP. APPL.EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3264802.


[3] [HCGMNET: A Hierarchical Change Guiding Map Network For Change Detection](https://doi.org/10.48550/arXiv.2302.10420).



