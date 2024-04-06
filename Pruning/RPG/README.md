# [NeurIPS 2023] Towards Higher Ranks via Adversarial Weight Pruning 

<p align="left">
<a href="https://arxiv.org/abs/2311.17493" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2311.17493-b31b1b.svg?style=flat" /></a>
<a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/040ace837dd270a87055bb10dd7c0392-Abstract-Conference.html" alt="arXiv">
    <img src="https://img.shields.io/badge/Proceedings-NeurIPS2023-orange.svg?style=flat" /></a>
    <img src="https://img.shields.io/badge/Weights-Available-green.svg?style=flat" /></a>
</p>

*Yuchuan Tian, Hanting Chen, Tianyu Guo, Chao Xu, Yunhe Wang\**

BibTex Formatted Citation:

```
@inproceedings{NEURIPS2023_040ace83,
 author = {Tian, Yuchuan and Chen, Hanting and Guo, Tianyu and Xu, Chao and Wang, Yunhe},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {1189--1207},
 publisher = {Curran Associates, Inc.},
 title = {Towards Higher Ranks via Adversarial Weight Pruning},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/040ace837dd270a87055bb10dd7c0392-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}
```

## Pruned Model Weights

**ResNet-50**

| Sparsity | ImageNet Top1 Acc. | Google Drive                                                 | Baidu Netdisk                                                |
| -------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 80       | 76.66              | [Link](https://drive.google.com/file/d/1-x3f_PIcSZkmhv7-X9vBpXN-zSp7W9F3/view?usp=drive_link) | [Link (PIN:1234)](https://pan.baidu.com/s/1AUqzU4uA7RW9gQRCssXOeg) |
| 90       | 75.80              | [Link](https://drive.google.com/file/d/10-nz5vYoE-qXp0nhAJ_9CYeHLJEr2CXI/view?usp=drive_link) | [Link (PIN:1234)](https://pan.baidu.com/s/11rwVrwtc-mnL87tnShbaBw) |
| 95       | 74.05              | [Link](https://drive.google.com/file/d/107NmBo_DP_Niit6QxAF0qhdkcnXi6hDC/view?usp=drive_link) | [Link (PIN:1234)](https://pan.baidu.com/s/1w-ykBeGa1ZNw04rzbLWC-A) |
| 98       | 69.57              | [Link](https://drive.google.com/file/d/1-mXDm0qyCANxD2-Y-oqf-Obi-omWq1UT/view?usp=drive_link) | [Link (PIN:1234)](https://pan.baidu.com/s/1XGG75o5tineDiJ6YIFkvow) |

**ResNet-50 Pruned from Scratch**

| Sparsity | ImageNet Top1 Acc. | Google Drive                                                 | Baidu Netdisk                                                |
| -------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 90       | 75.35              | [Link](https://drive.google.com/file/d/103wYuFpmJj3Bo3InXj-iKmgKULyY3ECH/view?usp=drive_link) | [Link (PIN:1234)](https://pan.baidu.com/s/1ZpF_Cf7jupBhnwR8u7uEZg) |
| 95       | 73.62              | [Link](https://drive.google.com/file/d/1-lUETG6EZu_GpQMz1OriKxHf2OOeKJ3_/view?usp=drive_link) | [Link (PIN:1234)](https://pan.baidu.com/s/1jV0nYHhHEqcUZMa2P0tShA) |

## Running

### Supporting Package Installation
We recommend torch version of 1.8.
You will have to install the following packages using pip:

```
pip install -r requirements.txt
```

You might need to manually install NVIDIA packages from GitHub rather than using pip：

```
nvidia-dali
apex
dllogger
```

### Training

Code running need to be done on a server with 8 GPUs. Each GPU should have a minimum GPU capacity of 16GB. Please refer to ```train.sh``` for training details.

### Inference

Single-GPU inference:

```
python validation.py --data-backend dali-cpu --data <path_to_imagenet> --batch-size 25 --num-classes 1000 --workers 8 --dataset imagenet --arch resnet50 --pretrained-weights <path_to_weight>
```

### Acknowledgement

Our codes are modified from the following GitHub repos: 

https://github.com/nollied/rigl-torch

https://github.com/boone891214/nv_rigl_imagenet

https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets

We sincerely thank their authors!

### References

```
@inproceedings{evci2020rigging,
  title={Rigging the lottery: Making all tickets winners},
  author={Evci, Utku and Gale, Trevor and Menick, Jacob and Castro, Pablo Samuel and Elsen, Erich},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={2943--2952},
  year={2020},
  organization={PMLR}
}

@inproceedings{ma2022effective,
    title={Effective Model Sparsification by Scheduled Grow-and-Prune Methods},
    author={Xiaolong Ma and Minghai Qin and Fei Sun and Zejiang Hou and Kun Yuan and Yi Xu and Yanzhi Wang and Yen-Kuang Chen and Rong Jin and Yuan Xie},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2022},
    url={https://openreview.net/forum?id=xa6otUDdP2W}
}
```


