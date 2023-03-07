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

_base_ = ['./_base_/models/upernet_vit.py', './_base_/datasets/ade20k_512x512.py', './_base_/default_runtime.py', './_base_/schedules/schedule_160k.py']
crop_size = (512, 512)

model = dict(
    backbone=dict(type='ViT', img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                  use_abs_pos_emb=True, use_rel_pos_bias=True, init_values=1., drop_path_rate=0.1, out_indices=[1, 3, 9, 11]),
    decode_head=dict(in_channels=[768, 768, 768, 768], num_classes=150, channels=768),
    auxiliary_head=dict(in_channels=768, num_classes=150),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

optimizer = dict(_delete_=True, type='AdamW', lr=4e-4, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor', paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65))

lr_config = dict(_delete_=True, policy='poly', warmup='linear', warmup_iters=1500,
                 warmup_ratio=1e-6, power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)

runner = dict(type='IterBasedRunnerAmp')

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True)
