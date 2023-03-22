
# Network Expansion For Practical Training Acceleration

PyTorch implementation for CVPR 2023 paper "Network Expansion For Practical Training Acceleration".


## Requirements

```
python == 3.7
torch==1.7.0
torchvision==0.8.1
timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet
  ├── train
  │     ├── class1
  │     │      ├── img1.jpeg
  │     ├── class2
  │            ├──img2.jpeg
  │
  └─── val
        ├── class1
        │      ├──img3.jpeg
        ├── class2
               ├──img4.jpeg
```

## Training
Use depth expansion to accelerate the training of Deit-base, run:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
  --data-path=/path/to/imagenet --output_dir=./outputs \
  --batch-size=128 --dist-eval --expand=deit_base_depth_6_12
```
This command runs on a machine with 8 V100(32Gb) GPUs, and shall give the top1 accuracy ~81.49% .

To run baseline training for [original Deit-base](https://github.com/facebookresearch/deit/blob/main/README_deit.md), just remove the argument```--expand=deit_base_depth_6_12```.

## Citation
If you find this project useful in your research, please cite:
```
@inproceedings{ding2023expansion,
  title={Network Expansion For Practical Training Acceleration},
  author={Ding, Ning and Tang, Yehui and Han, Kai and Xu, Chao and Wang, Yunhe},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```