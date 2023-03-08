# 2022.3.3-Changed for building LocalMIM
#          Huawei Technologies Co., Ltd. <foss@huawei.com>
#
# Modified by Haoqing Wang
# Based on timm, mmseg, deit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/microsoft/Swin-Transformer
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

_base_ = ['./_base_/models/upernet_swin.py', './_base_/datasets/ade20k_512x512.py', './_base_/default_runtime.py', './_base_/schedules/schedule_160k.py']
crop_size = (512, 512)

model = dict(
    backbone=dict(type='SWIN', embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=7,
                  drop_path_rate=0.3, mlp_ratio=4, qkv_bias=True, patch_norm=True),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=150),
    auxiliary_head=dict(in_channels=512, num_classes=150))

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.), 'relative_position_bias_table': dict(decay_mult=0.), 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly', warmup='linear', warmup_iters=1500, warmup_ratio=1e-6, power=1.0,
                 min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)