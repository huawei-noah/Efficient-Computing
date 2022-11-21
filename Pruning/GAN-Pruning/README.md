### GAN-pruning
Code for our ICCV 2019 paper, [Co-Evolutionary Compression for unpaired image Translation](https://arxiv.org/abs/1907.10804)

This paper proposes a co-evolutionary approach for reducing memory usage and FLOPs of generators on image-to-image transfer task simultaneously while maintains their performances.

<p align="center">
<img src="fig/framework.PNG" width="600">
</p>

### Description
- GAN pruning search/finetune/test code for image to image translation task.

### Files description
Requirements: Python3.6, PyTorch0.4

- `search.py` is the search script ultilizing Genetic Algorithem for GAN pruning.
- `finetune.py` is the script for finetuning searched pruned architectures.
- `test.py` is the script for testing pruned architectures.
- `models.py` defines original architecture of generators and discriminators.
- `models_prune.py` defines searched pruned architecture with binary channel mask.
- `GA.py` defines evolutionary operations .

### Dataset
Image to image translation dataset, like horse2zebra, summer2winter_yosemite, cityscapes.  

### Performance
Performance on cityscapes compared with conventional pruning method:
<img src="fig/FCN.PNG" width="600">
</p>

### Citation
	@inproceedings{GAN pruning,
		title={Co-Evolutionary Compression for Unpaired Image Translation},
		author={Shu, Han and Wang, Yunhe and Jia, Xu and Han, Kai and Chen, Hanting and Xu, Chunjing and Tian, Qi and Xu, Chang},
		booktitle={ICCV},
		year={2019}
	}
