# VanillaKD: Revisit the Power of Vanilla Knowledge Distillation from Small Scale to Large Scale 
<p align="left">
<a href="https://arxiv.org/abs/2305.15781" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2305.15781-%23b31b1b" /></a>
</p>

Official PyTorch implementation of **VanillaKD**, from the following paper: \
[VanillaKD: Revisit the Power of Vanilla Knowledge Distillation from Small Scale to Large Scale](https://arxiv.org/abs/2305.15781) \
Zhiwei Hao, Jianyuan Guo, Kai Han, Han Hu, Chang Xu, Yunhe Wang


This paper emphasizes the importance of scale in achieving superior results. It reveals that previous KD methods designed solely based on small-scale datasets has underestimated the effectiveness of vanilla KD on large-scale datasets, which is referred as to **small data pitfall**. By incorporating **stronger data augmentation** and **larger datasets**, the performance gap between vanilla KD and other approaches is narrowed:

<img src="fig\\data_scale_bar.png" width="500px"/>

Without bells and whistles, state-of-the-art results are achieved for ResNet-50, ViT-S, and ConvNeXtV2-T models on ImageNet, showcasing the vanilla KD is elegantly simple but astonishingly effective in large-scale scenarios.

If you find this project useful in your research, please cite:

```
@article{hao2023vanillakd,
  title={VanillaKD: Revisit the Power of Vanilla Knowledge Distillation from Small Scale to Large Scale },
  author={Hao, Zhiwei and Guo, Jianyuan and Han, Kai and Hu, Han and Xu, Chang and Wang, Yunhe},
  journal={arXiv preprint arXiv:2305.15781},
  year={2023}
}
```

## Model Zoo

We provide models trained by vanilla KD on ImageNet. 

| name | acc@1 | acc@5 | model |
|:---:|:---:|:---:|:---:|
|resnet50|83.08|96.35|[model](https://github.com/Hao840/vanillaKD/releases/download/checkpoint/resnet50-83.078.pth)|
|vit_tiny_patch16_224|78.11|94.26|[model](https://github.com/Hao840/vanillaKD/releases/download/checkpoint/vit_tiny_patch16_224-78.106.pth)|
|vit_small_patch16_224|84.33|97.09|[model](https://github.com/Hao840/vanillaKD/releases/download/checkpoint/vit_small_patch16_224-84.328.pth)|
|convnextv2_tiny|85.03|97.44|[model](https://github.com/Hao840/vanillaKD/releases/download/checkpoint/convnextv2_tiny-85.030.pth)|


## Usage
First, clone the repository locally:

```
git clone https://github.com/Hao840/vanillaKD.git
```

Then, install PyTorch and [timm 0.6.5](https://github.com/huggingface/pytorch-image-models/tree/v0.6.5)

```
conda install -c pytorch pytorch torchvision
pip install timm==0.6.5
```

Our results are produced with `torch==1.10.2+cu113 torchvision==0.11.3+cu113 timm==0.6.5`. Other versions might also work.

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/. The directory structure is:

```
│path/to/imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Evaluation

To evaluate a distilled model on ImageNet val with a single GPU, run:

```
python validate.py /path/to/imagenet --model <model name> --checkpoint /path/to/checkpoint
```


### Training

To train a ResNet50 student using BEiTv2-B teacher on ImageNet on a single node with 8 GPUs, run:

Strategy A2:

```
python -m torch.distributed.launch --nproc_per_node=8 train-kd.py /path/to/imagenet --model resnet50 --teacher beitv2_base_patch16_224 --teacher-pretrained /path/to/teacher_checkpoint --kd-loss kd --amp --epochs 300 --batch-size 256 --lr 5e-3 --opt lamb --sched cosine --weight-decay 0.02 --warmup-epochs 5 --warmup-lr 1e-6 --smoothing 0.0 --drop 0 --drop-path 0.05 --aug-repeats 3 --aa rand-m7-mstd0.5 --mixup 0.1 --cutmix 1.0 --color-jitter 0 --crop-pct 0.95 --bce-loss 1
```

Strategy A1:

```
python -m torch.distributed.launch --nproc_per_node=8 train-kd.py /path/to/imagenet --model resnet50 --teacher beitv2_base_patch16_224 --teacher-pretrained /path/to/teacher_checkpoint --kd-loss kd --amp --epochs 600 --batch-size 256 --lr 5e-3 --opt lamb --sched cosine --weight-decay 0.01 --warmup-epochs 5 --warmup-lr 1e-6 --smoothing 0.1 --drop 0 --drop-path 0.05 --aug-repeats 3 --aa rand-m7-mstd0.5 --mixup 0.2 --cutmix 1.0 --color-jitter 0 --crop-pct 0.95 --bce-loss 1
```



Commands for reproducing baseline results:

<details>
<summary>
DKD
</summary>
Training with ResNet50 student, BEiTv2-B teacher, and strategy A2 for 300 epochs

```
python -m torch.distributed.launch --nproc_per_node=8 train-kd.py /path/to/imagenet --model resnet50 --teacher beitv2_base_patch16_224 --teacher-pretrained /path/to/teacher_checkpoint --kd-loss dkd --amp --epochs 300 --batch-size 256 --lr 5e-3 --opt lamb --sched cosine --weight-decay 0.02 --warmup-epochs 5 --warmup-lr 1e-6 --smoothing 0.0 --drop 0 --drop-path 0.05 --aug-repeats 3 --aa rand-m7-mstd0.5 --mixup 0.1 --cutmix 1.0 --color-jitter 0 --crop-pct 0.95 --bce-loss 1
```
</details>



<details>
<summary>
DIST
</summary>
Training with ResNet50 student, BEiTv2-B teacher, and strategy A2 for 300 epochs

```
python -m torch.distributed.launch --nproc_per_node=8 train-kd.py /path/to/imagenet --model resnet50 --teacher beitv2_base_patch16_224 --teacher-pretrained /path/to/teacher_checkpoint --kd-loss dist --amp --epochs 300 --batch-size 256 --lr 5e-3 --opt lamb --sched cosine --weight-decay 0.02 --warmup-epochs 5 --warmup-lr 1e-6 --smoothing 0.0 --drop 0 --drop-path 0.05 --aug-repeats 3 --aa rand-m7-mstd0.5 --mixup 0.1 --cutmix 1.0 --color-jitter 0 --crop-pct 0.95 --bce-loss 1
```
</details>



<details>
<summary>
Correlation
</summary>
Training with ResNet50 student, ResNet152 teacher, and strategy A2 for 300 epochs

```
python -m torch.distributed.launch --nproc_per_node=8 train-fd.py /path/to/imagenet --model resnet50 --teacher resnet152 --teacher-pretrained /path/to/teacher_checkpoint --kd-loss correlation --amp --epochs 300 --batch-size 256 --lr 5e-3 --opt lamb --sched cosine --weight-decay 0.02 --warmup-epochs 5 --warmup-lr 1e-6 --smoothing 0.0 --drop 0 --drop-path 0.05 --aug-repeats 3 --aa rand-m7-mstd0.5 --mixup 0.1 --cutmix 1.0 --color-jitter 0 --crop-pct 0.95 --bce-loss 0
```
</details>



<details>
<summary>
RKD
</summary>
Training with ResNet50 student, ResNet152 teacher, and strategy A2 for 300 epochs

```
python -m torch.distributed.launch --nproc_per_node=8 train-fd.py /path/to/imagenet --model resnet50 --teacher resnet152 --teacher-pretrained /path/to/teacher_checkpoint --kd-loss rkd --amp --epochs 300 --batch-size 256 --lr 5e-3 --opt lamb --sched cosine --weight-decay 0.02 --warmup-epochs 5 --warmup-lr 1e-6 --smoothing 0.0 --drop 0 --drop-path 0.05 --aug-repeats 3 --aa rand-m7-mstd0.5 --mixup 0.1 --cutmix 1.0 --color-jitter 0 --crop-pct 0.95 --bce-loss 0
```
</details>



<details>
<summary>
ReviewKD
</summary>
Training with ResNet50 student, ResNet152 teacher, and strategy A2 for 300 epochs

```
python -m torch.distributed.launch --nproc_per_node=8 train-fd.py /path/to/imagenet --model resnet50 --teacher resnet152 --teacher-pretrained /path/to/teacher_checkpoint --kd-loss review --amp --epochs 300 --batch-size 256 --lr 5e-3 --opt lamb --sched cosine --weight-decay 0.02 --warmup-epochs 5 --warmup-lr 1e-6 --smoothing 0.0 --drop 0 --drop-path 0.05 --aug-repeats 3 --aa rand-m7-mstd0.5 --mixup 0.1 --cutmix 1.0 --color-jitter 0 --crop-pct 0.95 --bce-loss 0
```
</details>



<details>
<summary>
CRD
</summary>
Training with ResNet50 student, ResNet152 teacher, and strategy A2 for 300 epochs

```
python -m torch.distributed.launch --nproc_per_node=8 train-crd.py /path/to/imagenet --model resnet50 --teacher resnet152 --teacher-pretrained /path/to/teacher_checkpoint --kd-loss crd --amp --epochs 300 --batch-size 256 --lr 5e-3 --opt lamb --sched cosine --weight-decay 0.02 --warmup-epochs 5 --warmup-lr 1e-6 --smoothing 0.0 --drop 0 --drop-path 0.05 --aug-repeats 3 --aa rand-m7-mstd0.5 --mixup 0.1 --cutmix 1.0 --color-jitter 0 --crop-pct 0.95 --bce-loss 0

```
</details>

## Acknowledgement

This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, [DKD](https://github.com/megvii-research/mdistiller), [DIST](https://github.com/hunto/DIST_KD), [DeiT](https://github.com/facebookresearch/deit), [BEiT v2](https://github.com/microsoft/unilm/tree/master/beit2), and [ConvNeXt v2](https://github.com/facebookresearch/ConvNeXt-V2) repositories.
