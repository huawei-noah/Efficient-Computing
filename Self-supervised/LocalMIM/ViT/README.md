## LocalMIM for ViT
All our experiments can be implemented on a single node with 8 Tesla V100-32G GPUs.

### Data preparation

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

### Pre-Training
For 100-epoch pre-training, we set `warmup_epochs=10`.
#### To pre-train ViT-B:
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_pretrain.py --batch_size 256 --model MIM_vit_base_patch16 --hog_nbins 9 --mask_ratio 0.75 --epochs 1600 --warmup_epochs 40 --blr 2e-4 --weight_decay 0.05 --data_path /path/to/imagenet/ --output_dir /output_dir/
```
#### To pre-train ViT-L:
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_pretrain.py --batch_size 128 --accum_iter 4 --model MIM_vit_large_patch16 --hog_nbins 18 --hog_bias --mask_ratio 0.75 --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --data_path /path/to/imagenet/ --output_dir /output_dir/
```

### Fine-tuning
#### To fine-tune ViT-B:
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_finetune.py --batch_size 128 --model vit_base_patch16 --finetune /path/to/checkpoint.pth --epochs 100 --warmup_epochs 20 --lr 2e-3 --min_lr 1e-5 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --data_path /path/to/imagenet/ --output_dir /output_dir/
```
This is for 1600-epoch pre-trained model. For 100-epoch pre-trained model, we set `lr=4e-3`, `layer_decay=0.75` and `min_lr=1e-6`.

#### To fine-tune ViT-L:
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_finetune.py --batch_size 64 --accum_iter 2 --model vit_large_patch16 --finetune /path/to/checkpoint.pth --epochs 50 --warmup_epochs 5 --lr 3e-3 --layer_decay 0.75 --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --data_path /path/to/imagenet/ --output_dir /output_dir/
```
