# COCO Object detection

## Getting started 

We add VanillaNet model and config files based on [mmdetection-2.x](https://github.com/open-mmlab/mmdetection/tree/2.x). Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/2.x/docs/en/get_started.md) for mmdetection installation and dataset preparation instructions.

## Results and Fine-tuned Models

|     Framework     |   Backbone   | LR Schedule | AP<sup>b</sup> | AP<sup>m</sup> |
| :---------------: | :----------: | :---------: | :------------: | :------------: |
|     Mask RCNN     |   ResNet50   |     1x      |      41.8      |      37.7      |
|                   |   ResNet50   |     2x      |      42.1      |      38.0      |
|     Mask RCNN     | ConvNeXtV2-T |     1x      |      45.7      |      42.0      |
|                   | ConvNeXtV2-T |     3x      |      47.9      |      43.3      |
| Cascade Mask RCNN | ConvNeXtV2-T |     1x      |      50.6      |      44.3      |
|                   | ConvNeXtV2-T |     3x      |      52.1      |      45.4      |


### Training

To train a model with 8 gpus, run:
```
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py <CONFIG_FILE> --gpus 8 --launcher pytorch --work-dir <WORK_DIR>
```


## Acknowledgment 

This code is built based on [mmdetection](https://github.com/open-mmlab/mmdetection), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) repositories.