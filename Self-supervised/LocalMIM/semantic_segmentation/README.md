# Semantic segmentation on ADE20k

## Getting started 

1. Install the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library and some required packages.

```bash
pip install mmcv-full==1.3.0 mmsegmentation==0.11.0
pip install scipy timm==0.3.2
```

2. Install [apex](https://github.com/NVIDIA/apex) for mixed-precision training

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

3. Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k) to prepare the ADE20k dataset.


## Fine-tuning

For ViT-B
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=12332 train.py --config ./configs/upernet_vit_base_12_512_slide_160k_ade20k.py --deterministic --options model.pretrained=/path/to/checkpoint.pth --work-dir /output_dir/
```

For Swin-B
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=12332 train.py --config ./configs/upernet_swin_base_12_512_slide_160k_ade20k.py --deterministic --options model.pretrained=/path/to/checkpoint.pth --work-dir /output_dir/
```

---

## Acknowledgment 

This code is built based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit/semantic_segmentation).
