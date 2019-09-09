# DAFL: Data-Free Learning of Student Networks
This code is the Pytorch implementation of ICCV 2019 paper [DAFL: Data-Free Learning of Student Networks](https://arxiv.org/pdf/1904.01186.pdf)

We propose a novel framework for training efficient deep neural networks by exploiting generative adversarial networks (GANs). To be specific, the pre-trained teacher networks are regarded as a fixed discriminator and the generator is utilized for derivating training samples which can obtain the maximum response on the discriminator. Then, an efficient network with smaller model size and computational complexity is trained using the generated data and the teacher network, simultaneously. 

<p align="center">
<img src="figure/figure.jpg" width="800">
</p>


## Requirements
- python 3
- pytorch >= 1.0.0
- torchvision

## Run the demo
```shell
python teacher-train.py
```
First, you should train a teacher network.
```shell
python DAFL-train.py
```
Then, you can use the DAFL to train a student network without training data on the MNIST dataset.

To run DAFL on the CIFAR-10 dataset
```shell
python teacher-train.py --dataset cifar10
python DAFL-train.py --dataset cifar10 --channels 3 --n_epochs 2000 --batch_size 1024 --lr_G 0.02 --lr_S 0.1 --latent_dim 1000  
```

To run DAFL on the CIFAR-100 dataset
```shell
python teacher-train.py --dataset cifar100
python DAFL-train.py --dataset cifar100 --channels 3 --n_epochs 2000 --batch_size 1024 --lr_G 0.02 --lr_S 0.1 --latent_dim 1000 --oh 0.5
```

## Results
<img src="figure/Table1.jpg" width="600">
</p>

<img src="figure/Table2.jpg" width="600">
</p>


## Citation
	@inproceedings{DAFL,
		title={DAFL: Data-Free Learning of Student Networks},
		author={Chen, Hanting and Wang, Yunhe and Xu, Chang and Yang, Zhaohui and Liu, Chuanjian and Shi, Boxin and Xu, Chunjing and Xu, Chao and Tian, Qi},
		booktitle={ICCV},
		year={2019}
	}
	
## Contributing
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions or extensions to the core, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR, because we might be taking the core in a different direction than you might be aware of.
