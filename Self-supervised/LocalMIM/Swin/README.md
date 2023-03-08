## LocalMIM for Swin
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
#### To pre-train Swin-B with HOG target:
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_pretrain.py --output_dir /output_dir/ --batch_size 256 --model mim_swin_base_patch4_win7 --target HOG --hog_nbins 18 --hog_bias --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 --blr 1e-4 --weight_decay 0.05 --data_path /path/to/imagenet/
```
#### To pre-train Swin-B with Pixel target:
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_pretrain.py --output_dir /output_dir/ --batch_size 256 --model mim_swin_base_patch4_win7 --target Pixel --norm_pix_loss --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 --blr 1e-4 --weight_decay 0.05 --data_path /path/to/imagenet/
```
#### To pre-train Swin-L with HOG target:
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_pretrain.py --output_dir /output_dir/ --batch_size 128 --accum_iter 4 --blr 1e-4 --model mim_swin_large_patch4_win14 --target HOG --hog_nbins 18 --hog_bias --mask_ratio 0.75 --epochs 800 --warmup_epochs 40 --weight_decay 0.05 --data_path /path/to/imagenet/
```

### Fine-tuning
#### To fine-tune Swin-B:
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_finetune.py --batch_size 128 --model swin_base_patch4_win7 --finetune /path/to/checkpoint.pth --epochs 100 --lr 4e-3 --layer_decay 0.9 --weight_decay 0.05 --drop_path 0.1 --dist_eval --data_path /path/to/imagenet/ --output_dir /output_dir/
```
This is for 400-epoch pre-trained model with Pixel target. For 400-epoch pre-trained model with HOG target, we set `accum_iter=2`. For 100-epoch pre-trained model, we set `lr=5e-3`.

#### To fine-tune Swin-L:
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_finetune.py --batch_size 32 --accum_iter 8 --model swin_large_patch4_win14 --finetune /path/to/checkpoint.pth --epochs 100 --warmup_epochs 20 --lr 2e-3 --layer_decay 0.8 --weight_decay 0.05 --drop_path 0.3 --dist_eval --data_path /path/to/imagenet/ --output_dir /output_dir/
```
