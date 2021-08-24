# Data-Free Learning of Student Networks
This code is the Pytorch implementation of CVPR 2021 paper [Learning Student Networks in the Wild](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Learning_Student_Networks_in_the_Wild_CVPR_2021_paper.pdf)

We present to utilize the large amount of unlabeled data in the wild to address the data-free knowledge distillation problem. Instead of generating images from the teacher network with a series of priori, images most relevant to the given pre-trained network and tasks will be identified from a large unlabeled dataset (e.g., Flickr) to conduct the knowledge distillation task.

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
python DFND-train.py
```
Then, you can use the DAFL to train a student network without training data on the MNIST dataset.

To run DAFL on the CIFAR-10 dataset
```shell
python teacher-train.py --dataset cifar10
python DAFL-train.py --dataset cifar10 
```

To run DAFL on the CIFAR-100 dataset
```shell
python teacher-train.py --dataset cifar100
python DAFL-train.py --dataset cifar100 
```

## Results
<img src="figure/Table1.jpg" width="600">
</p>



## Citation
	@inproceedings{DFND,
    title={Learning Student Networks in the Wild},
    author={Chen, Hanting and Guo, Tianyu and Xu, Chang and Li, Wenshuo and Xu, Chunjing and Xu, Chao and Wang, Yunhe},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={6428--6437},
    year={2021}
  }
	
## Contributing
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions or extensions to the core, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR, because we might be taking the core in a different direction than you might be aware of.

