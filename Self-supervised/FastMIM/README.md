
## Implementation of  "[FastMIM: Expediting Masked Image Modeling Pre-training for Vision](https://arxiv.org/pdf/2212.06593.pdf)".


<p align="center">
  <img src="figs/fastmim.png" >
</p>
<p align="center">
</p>
###### Comparison among the MAE [22], SimMIM [48] and our FastMIM framework. MAE randomly masks and discards the input patches. Although there is only small amount of encoder patches, MAE can only be used to pre-train the isotropic ViT which generates single-scale intermediate features. SimMIM preserves input resolution and can serve as a generic framework for all kinds of vision backbones, but it needs to tackle with large amount of patches. Our FastMIM simply reduces the input resolution and replaces the pixel target with HOG target. These modifications are simple yet effective. FastMIM (i) pre-train faster; (ii) has a lighter memory consumption; (iii) can serve as a generic framework for all kinds of architectures; and (iv) achieves comparable and even better performances compared to previous methods.


#### Set up
```
- python==3.x
- cuda==10.x
- torch==1.7.0+
- mmcv-full-1.4.4+

# other pytorch/cuda/timm version can also work

# To pip your environment
sh requirement_pip_install.sh

# build your apex (optional)
cd /your_path_to/apex-master/;
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

#### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is:

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

#### Pre-training on ImageNet-1K
To train Swin-B on ImageNet-1K on a single node with 8 gpus:

```
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --model mim_swin_base --data_path /your_path_to/data/imagenet/ --epochs 400 --warmup_epochs 10 --blr 1.5e-4 --weight_decay 0.05 --output_dir /your_path_to/fastmim_pretrain_output/ --batch_size 256 --save_ckpt_freq 50 --num_workers 10 --mask_ratio 0.75 --norm_pix_loss --input_size 128 --rrc_scale 0.2 1.0 --window_size 4 --decoder_embed_dim 256 --decoder_depth 4 --mim_loss HOG --block_size 32
```

#### Finetuning on ImageNet-1K
To fine-tune Swin-B on ImageNet-1K on a single node with 8 gpus:

```
python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py --model swin_base_patch4_window7_224 --data_path /your_path_to/data/imagenet/ --batch_size 128 --epochs 100 --blr 1.0e-3 --layer_decay 0.80 --weight_decay 0.05 --drop_path 0.1 --dist_eval --finetune /your_path_to_ckpt/checkpoint-399.pth --output_dir /your_path_to/fastmim_finetune_output/
```


### Results and Models

#### Classification on ImageNet-1K (ViT-B/Swin-B/PVTv2-b2/CMT-S)

| Model | #Params | PT Res. | PT Epoch | PT log/ckpt | FT Res. | FT log/ckpt | Top-1 (%) |
| :------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ViT-B | 86M | 128x128 | 800 | [log](https://github.com/ggjy/FastMIM.pytorch/releases/download/release-cls/fastmim_vit_base_hog_800e_pretrain.txt)/[ckpt](https://github.com/ggjy/FastMIM.pytorch/releases/download/release-cls/vit_base_fastmim_hog_800e_pretrain.pth) | 224x224 | [log](https://github.com/ggjy/FastMIM.pytorch/releases/download/release-cls/fastmim_vit_base_hog_800e_finetune_100e.txt)/[ckpt](https://github.com/ggjy/FastMIM.pytorch/releases/download/release-cls/vit_base_fastmim_hog_800e_finetune_100e.pth) | 83.8 |
| Swin-B | 88M | 128x128 | 400 | [log](https://github.com/ggjy/FastMIM.pytorch/releases/download/release-cls/fastmim_swin_base_hog_400e_pretrain.txt)/[ckpt](https://github.com/ggjy/FastMIM.pytorch/releases/download/release-cls/swin_base_fastmim_hog_400e_pretrain.pth) | 224x224 | [log](https://github.com/ggjy/FastMIM.pytorch/releases/download/release-cls/fastmim_swin_base_hog_400e_finetune_100e.txt)/[ckpt](https://github.com/ggjy/FastMIM.pytorch/releases/download/release-cls/swin_base_fastmim_hog_400e_finetune_100e.pth) | 84.1 |


### Citation

If you find this project useful in your research, please consider cite:

```bibtex
@article{guo2022fastmim,
  title={FastMIM: Expediting Masked Image Modeling Pre-training for Vision},
  author={Guo, Jianyuan and Han, Kai and Wu, Han and Tang, Yehui and Wang, Yunhe and Xu, Chang},
  journal={arXiv preprint arXiv:2212.06593},
  year={2022}
}
```


### Acknowledgement

The classification task in this repo is based on [MAE](https://github.com/facebookresearch/mae), [SimMIM](https://github.com/microsoft/SimMIM), [SlowFast](https://github.com/facebookresearch/SlowFast) and [timm](https://github.com/rwightman/pytorch-image-models).