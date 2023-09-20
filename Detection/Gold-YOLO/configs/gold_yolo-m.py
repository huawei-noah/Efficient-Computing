# 2023.09.18-Changed for config of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# GoldYOLO-m model

use_checkpoint = False

model = dict(
        type='GoldYOLO-m',
        pretrained=None,
        depth_multiple=0.60,
        width_multiple=0.75,
        backbone=dict(
                type='CSPBepBackbone',
                num_repeats=[1, 6, 12, 18, 6],
                out_channels=[64, 128, 256, 512, 1024],
                csp_e=float(2) / 3,
                fuse_P2=True
        ),
        neck=dict(
                type='GDNeck',
                num_repeats=[12, 12, 12, 12],
                out_channels=[256, 128, 128, 256, 256, 512],
                csp_e=float(2) / 3,
                extra_cfg=dict(
                        norm_cfg=dict(type='SyncBN', requires_grad=True),
                        depths=2,
                        fusion_in=1440,
                        fusion_act=dict(type='ReLU'),
                        fuse_block_num=3,
                        embed_dim_p=192,
                        embed_dim_n=1056,
                        key_dim=8,
                        num_heads=4,
                        mlp_ratios=1,
                        attn_ratios=2,
                        c2t_stride=2,
                        drop_path_rate=0.1,
                        trans_channels=[192, 96, 192, 384],
                        pool_mode='torch'
                )
        ),
        head=dict(
                type='EffiDeHead',
                in_channels=[128, 256, 512],
                num_layers=3,
                begin_indices=24,
                anchors=3,
                anchors_init=[[10, 13, 19, 19, 33, 23],
                              [30, 61, 59, 59, 59, 119],
                              [116, 90, 185, 185, 373, 326]],
                out_indices=[17, 20, 23],
                strides=[8, 16, 32],
                atss_warmup_epoch=0,
                iou_type='giou',
                use_dfl=True,
                reg_max=16,  # if use_dfl is False, please set reg_max to 0
                distill_weight={
                    'class': 0.8,
                    'dfl'  : 1.0,
                },
                # extra_cfg=dict(use_checkpoint=use_checkpoint)
        )
)

solver = dict(
        optim='SGD',
        lr_scheduler='Cosine',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1
)

data_aug = dict(
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.9,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
)
