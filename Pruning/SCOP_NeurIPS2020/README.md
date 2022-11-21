# SCOP: Scientific Control for Reliable Neural Network Pruning

Code for our NeurIPS 2020 paper, [SCOP: Scientific Control for Reliable Neural Network Pruning](https://arxiv.org/abs/2010.10732).

This paper proposes a reliable neural network pruning algorithm by setting up a scientific control. Existing pruning methods have developed various hypotheses to approximate the importance of filters to the network and then execute filter pruning accordingly. To increase the reliability of the results, we prefer to have a more rigorous research design by including a scientific control group as an essential part to minimize the effect of all factors except the association between the filter and expected network output. Acting as a control group, knockoff feature is generated to mimic the feature map produced by the network filter, but they are conditionally independent of the example label given the real feature map. We theoretically suggest that the knockoff condition can be approximately preserved given the information propagation of network layers. Besides the real feature map on an intermediate layer, the corresponding knockoff feature is brought in as another auxiliary input signal for the subsequent layers. Redundant filters can be discovered in the adversarial process of different features. Through experiments, we demonstrate the superiority of the proposed algorithm over state-of-the-art methods. For example, our method can reduce 57.8% parameters and 60.2% FLOPs of ResNet-101 with only 0.01% top-1 accuracy loss on ImageNet.

<p align="center">
<img src="fig/framework.PNG" width="800">
</p>



## Requirements

- python 3
- pytorch >= 1.3.0
- torchvision

## Usage


Run  `SCOP_NeurIPS2020/train.py` to  prune networks. For example,  you can run the following code to prune a ResNet-50 on ImageNet dataset. 

```shell
python SCOP_NeurIPS2020/train.py --ngpu=8 --arch=resnet50  --dataset=imagenet --prune_rate=0.45 --pretrain_path='SCOP_NeurIPS2020/pretrain_path/' --data_path='...'
```
- `--ngpu` The number of GPUs.
- `--arch` The architecture of nueral network.
- `--dataset` The dataset for training models
- `prune_rate` The desired pruning rate.
- `pretrain_path` Path of the pre-trained network.
- `data_path` Path of the dataset.

The checkpoints of pre-trained networks can downloaded from the official [PyTorch model zoo](https://pytorch.org/docs/stable/model_zoo.html). The pre-trained generators for generating knockoff features are available in [Google drive](https://drive.google.com/drive/folders/1sNcB08JbSW4RLgg5sc2j5os_vPX6KtE8?usp=sharing) or [Baidu cloud](https://pan.baidu.com/s/1jAMfvrMgr9D4lb6ciEULWg)(access code:d5cv), which should be downloaded to  `SCOP_NeurIPS2020/pretrain_path/` beforing running codes.

Note: File `train.py`  is directly converted from `train.ipynb`  by Jupyter notebook, and`train.ipynb` is also provided for readability.


## Results
Comparison of the pruned networks with different methods on ImageNet.




<p align="center">
<img src="fig/imagenet.PNG" width="600">
</p>

## Citation
    @article{tang2020scop,
      title={SCOP: Scientific Control for Reliable Neural Network Pruning},
      author={Tang, Yehui and Wang, Yunhe and Xu, Yixing and Tao, Dacheng and Xu, Chunjing and Xu, Chao and Xu, Chang},
      journal={Advances in Neural Information Processing Systems},
      volume={33},
      year={2020}
    }