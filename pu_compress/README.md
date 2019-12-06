# Positive-Unlabeled Compression on the Cloud
This code is the Pytorch implementation of NeurIPS 2019 paper [Positive-Unlabeled Compression on the Cloud](https://arxiv.org/pdf/1909.09757.pdf).

We propose a novel framework for training efficient deep neural networks with little training data by using positive-unlabeled (PU) learning method and robust knowledge distillation (RKD) method. To be specific, PU learning method enlarge the training dataset by selecting positive data from unlabeled set, and then the enlarged training set is used to compress the teacher network using RKD method.

<p align="center">
<img src="fig/1.png" width="800">
</p>


## Requirements
- python 3
- pytorch >= 1.0.0
- torchvision

## Run the demo

This is a demo by using 2% of the CIFAR-10 dataset to compress network. ImageNet dataset is regarded as unlabeled dataset. The teacher network is a pre-trained ResNet-34, and the student network is ResNet-18.

In order to run the demo code, you first need to download CIFAR-10 dataset as the positive dataset and ImageNet dataset as the unlabeled dataset. Besides, a pre-trained ResNet-34 model is used as the teacher model with can be downloaded at:  

https://pan.baidu.com/s/1qw4136eq-kiC8tmtnDwUTA 
code: mvxp

The datasets and the pre-trained model should be placed in the '/cache/' folder. And then run:

```shell
python main.py --pos_num 100 --prior 0.21
```
in which
```
pos_num: Number of positive data used in each class of CIFAR-10 dataset.
prior: The estimated class prior in unlabeled dataset.
```


## Results
<p align="center">
<img src="fig/result.png" width="600">
</p>

You should get at least 93.75% accuracy on CIFAR-10 dataset when using the default parameter.


## Citation
	@inproceedings{xu2019positive,
	  title={Positive-Unlabeled Compression on the Cloud},
	  author={Xu, Yixing and Wang, Yunhe and Zeng, Jia and Han, Kai and Chunjing, XU and Tao, Dacheng and Xu, Chang},
	  booktitle={Advances in Neural Information Processing Systems},
	  pages={2561--2570},
	  year={2019}
	}

## Contributing
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions or extensions to the core, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR, because we might be taking the core in a different direction than you might be aware of.
