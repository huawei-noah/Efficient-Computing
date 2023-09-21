# 2023.09.18-Changed for Neck implementation of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
from torch import nn

from yolov6.layers.common import RepVGGBlock, BottleRep, BepC3, RepBlock, SimConv

from .layers import Conv
from .common import AdvPoolFusion, SimFusion_3in, SimFusion_4in
from .transformer import PyramidPoolAgg, TopBasicLayer, InjectionMultiSum_Auto_pool


class RepGDNeck(nn.Module):
    def __init__(
            self,
            channels_list=None,
            num_repeats=None,
            block=RepVGGBlock,
            extra_cfg=None
    ):
        super().__init__()
        
        assert channels_list is not None
        assert num_repeats is not None
        
        self.low_FAM = SimFusion_4in()
        self.low_IFM = nn.Sequential(
                Conv(extra_cfg.fusion_in, extra_cfg.embed_dim_p, kernel_size=1, stride=1, padding=0),
                *[block(extra_cfg.embed_dim_p, extra_cfg.embed_dim_p) for _ in range(extra_cfg.fuse_block_num)],
                Conv(extra_cfg.embed_dim_p, sum(extra_cfg.trans_channels[0:2]), kernel_size=1, stride=1, padding=0),
        )
        
        self.reduce_layer_c5 = SimConv(
                in_channels=channels_list[4],  # 1024
                out_channels=channels_list[5],  # 512
                kernel_size=1,
                stride=1
        )
        self.LAF_p4 = SimFusion_3in(
                in_channel_list=[channels_list[3], channels_list[3]],  # 512, 256
                out_channels=channels_list[5],  # 256
        )
        self.Inject_p4 = InjectionMultiSum_Auto_pool(channels_list[5], channels_list[5], norm_cfg=extra_cfg.norm_cfg,
                                                     activations=nn.ReLU6)
        self.Rep_p4 = RepBlock(
                in_channels=channels_list[5],  # 256
                out_channels=channels_list[5],  # 256
                n=num_repeats[5],
                block=block
        )
        
        self.reduce_layer_p4 = SimConv(
                in_channels=channels_list[5],  # 256
                out_channels=channels_list[6],  # 128
                kernel_size=1,
                stride=1
        )
        self.LAF_p3 = SimFusion_3in(
                in_channel_list=[channels_list[5], channels_list[5]],  # 512, 256
                out_channels=channels_list[6],  # 256
        )
        self.Inject_p3 = InjectionMultiSum_Auto_pool(channels_list[6], channels_list[6], norm_cfg=extra_cfg.norm_cfg,
                                                     activations=nn.ReLU6)
        self.Rep_p3 = RepBlock(
                in_channels=channels_list[6],  # 128
                out_channels=channels_list[6],  # 128
                n=num_repeats[6],
                block=block
        )
        
        self.high_FAM = PyramidPoolAgg(stride=extra_cfg.c2t_stride, pool_mode=extra_cfg.pool_mode)
        dpr = [x.item() for x in torch.linspace(0, extra_cfg.drop_path_rate, extra_cfg.depths)]
        self.high_IFM = TopBasicLayer(
                block_num=extra_cfg.depths,
                embedding_dim=extra_cfg.embed_dim_n,
                key_dim=extra_cfg.key_dim,
                num_heads=extra_cfg.num_heads,
                mlp_ratio=extra_cfg.mlp_ratios,
                attn_ratio=extra_cfg.attn_ratios,
                drop=0, attn_drop=0,
                drop_path=dpr,
                norm_cfg=extra_cfg.norm_cfg
        )
        self.conv_1x1_n = nn.Conv2d(extra_cfg.embed_dim_n, sum(extra_cfg.trans_channels[2:4]), 1, 1, 0)
        
        self.LAF_n4 = AdvPoolFusion()
        self.Inject_n4 = InjectionMultiSum_Auto_pool(channels_list[8], channels_list[8],
                                                     norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6)
        self.Rep_n4 = RepBlock(
                in_channels=channels_list[6] + channels_list[7],  # 128 + 128
                out_channels=channels_list[8],  # 256
                n=num_repeats[7],
                block=block
        )
        
        self.LAF_n5 = AdvPoolFusion()
        self.Inject_n5 = InjectionMultiSum_Auto_pool(channels_list[10], channels_list[10],
                                                     norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6)
        self.Rep_n5 = RepBlock(
                in_channels=channels_list[5] + channels_list[9],  # 256 + 256
                out_channels=channels_list[10],  # 512
                n=num_repeats[8],
                block=block
        )
        
        self.trans_channels = extra_cfg.trans_channels
    
    def forward(self, input):
        (c2, c3, c4, c5) = input
        
        # Low-GD
        ## use conv fusion global info
        low_align_feat = self.low_FAM(input)
        low_fuse_feat = self.low_IFM(low_align_feat)
        low_global_info = low_fuse_feat.split(self.trans_channels[0:2], dim=1)
        
        ## inject low-level global info to p4
        c5_half = self.reduce_layer_c5(c5)
        p4_adjacent_info = self.LAF_p4([c3, c4, c5_half])
        p4 = self.Inject_p4(p4_adjacent_info, low_global_info[0])
        p4 = self.Rep_p4(p4)
        
        ## inject low-level global info to p3
        p4_half = self.reduce_layer_p4(p4)
        p3_adjacent_info = self.LAF_p3([c2, c3, p4_half])
        p3 = self.Inject_p3(p3_adjacent_info, low_global_info[1])
        p3 = self.Rep_p3(p3)
        
        # High-GD
        ## use transformer fusion global info
        high_align_feat = self.high_FAM([p3, p4, c5])
        high_fuse_feat = self.high_IFM(high_align_feat)
        high_fuse_feat = self.conv_1x1_n(high_fuse_feat)
        high_global_info = high_fuse_feat.split(self.trans_channels[2:4], dim=1)
        
        ## inject low-level global info to n4
        n4_adjacent_info = self.LAF_n4(p3, p4_half)
        n4 = self.Inject_n4(n4_adjacent_info, high_global_info[0])
        n4 = self.Rep_n4(n4)
        
        ## inject low-level global info to n5
        n5_adjacent_info = self.LAF_n5(n4, c5_half)
        n5 = self.Inject_n5(n5_adjacent_info, high_global_info[1])
        n5 = self.Rep_n5(n5)
        
        outputs = [p3, n4, n5]
        
        return outputs


class GDNeck(nn.Module):
    def __init__(
            self,
            channels_list=None,
            num_repeats=None,
            block=BottleRep,
            csp_e=float(1) / 2,
            extra_cfg=None
    ):
        super().__init__()
        
        assert channels_list is not None
        assert num_repeats is not None
        
        inj_block = InjectionMultiSum_Auto_pool
        
        self.low_FAM = SimFusion_4in()
        self.low_IFM = nn.Sequential(
                Conv(extra_cfg.fusion_in, extra_cfg.embed_dim_p, kernel_size=1, stride=1, padding=0),
                *[block(extra_cfg.embed_dim_p, extra_cfg.embed_dim_p) for _ in range(extra_cfg.fuse_block_num)],
                Conv(extra_cfg.embed_dim_p, sum(extra_cfg.trans_channels[0:2]), kernel_size=1, stride=1, padding=0),
        )
        
        self.reduce_layer_c5 = SimConv(
                in_channels=channels_list[4],  # 1024
                out_channels=channels_list[5],  # 512
                kernel_size=1,
                stride=1
        )
        self.LAF_p4 = SimFusion_3in(
                in_channel_list=[channels_list[3], channels_list[3]],  # 512, 256
                out_channels=channels_list[5],  # 256
        )
        self.Inject_p4 = inj_block(channels_list[5], channels_list[5], norm_cfg=extra_cfg.norm_cfg,
                                   activations=nn.ReLU6)
        self.Rep_p4 = BepC3(
                in_channels=channels_list[5],  # 256
                out_channels=channels_list[5],  # 256
                n=num_repeats[5],
                e=csp_e,
                block=block
        )
        
        self.reduce_layer_p4 = SimConv(
                in_channels=channels_list[5],  # 256
                out_channels=channels_list[6],  # 128
                kernel_size=1,
                stride=1
        )
        self.LAF_p3 = SimFusion_3in(
                in_channel_list=[channels_list[5], channels_list[5]],  # 512, 256
                out_channels=channels_list[6],  # 256
        )
        self.Inject_p3 = inj_block(channels_list[6], channels_list[6], norm_cfg=extra_cfg.norm_cfg,
                                   activations=nn.ReLU6)
        self.Rep_p3 = BepC3(
                in_channels=channels_list[6],  # 128
                out_channels=channels_list[6],  # 128
                n=num_repeats[6],
                e=csp_e,
                block=block
        )
        
        self.high_FAM = PyramidPoolAgg(stride=extra_cfg.c2t_stride, pool_mode=extra_cfg.pool_mode)
        dpr = [x.item() for x in torch.linspace(0, extra_cfg.drop_path_rate, extra_cfg.depths)]
        self.high_IFM = TopBasicLayer(
                block_num=extra_cfg.depths,
                embedding_dim=extra_cfg.embed_dim_n,
                key_dim=extra_cfg.key_dim,
                num_heads=extra_cfg.num_heads,
                mlp_ratio=extra_cfg.mlp_ratios,
                attn_ratio=extra_cfg.attn_ratios,
                drop=0, attn_drop=0,
                drop_path=dpr,
                norm_cfg=extra_cfg.norm_cfg
        )
        self.conv_1x1_n = nn.Conv2d(extra_cfg.embed_dim_n, sum(extra_cfg.trans_channels[2:4]), 1, 1, 0)
        
        self.LAF_n4 = AdvPoolFusion()
        self.Inject_n4 = inj_block(channels_list[8], channels_list[8], norm_cfg=extra_cfg.norm_cfg,
                                   activations=nn.ReLU6)
        self.Rep_n4 = BepC3(
                in_channels=channels_list[6] + channels_list[7],  # 128 + 128
                out_channels=channels_list[8],  # 256
                n=num_repeats[7],
                e=csp_e,
                block=block
        )
        
        self.LAF_n5 = AdvPoolFusion()
        self.Inject_n5 = inj_block(channels_list[10], channels_list[10], norm_cfg=extra_cfg.norm_cfg,
                                   activations=nn.ReLU6)
        self.Rep_n5 = BepC3(
                in_channels=channels_list[5] + channels_list[9],  # 256 + 256
                out_channels=channels_list[10],  # 512
                n=num_repeats[8],
                e=csp_e,
                block=block
        )
        
        self.trans_channels = extra_cfg.trans_channels
    
    def forward(self, input):
        (c2, c3, c4, c5) = input
        
        # Low-GD
        ## use conv fusion global info
        low_align_feat = self.low_FAM(input)
        low_fuse_feat = self.low_IFM(low_align_feat)
        low_global_info = low_fuse_feat.split(self.trans_channels[0:2], dim=1)
        
        ## inject low-level global info to p4
        c5_half = self.reduce_layer_c5(c5)
        p4_adjacent_info = self.LAF_p4([c3, c4, c5_half])
        p4 = self.Inject_p4(p4_adjacent_info, low_global_info[0])
        p4 = self.Rep_p4(p4)
        
        ## inject low-level global info to p3
        p4_half = self.reduce_layer_p4(p4)
        p3_adjacent_info = self.LAF_p3([c2, c3, p4_half])
        p3 = self.Inject_p3(p3_adjacent_info, low_global_info[1])
        p3 = self.Rep_p3(p3)
        
        # High-GD
        ## use transformer fusion global info
        high_align_feat = self.high_FAM([p3, p4, c5])
        high_fuse_feat = self.high_IFM(high_align_feat)
        high_fuse_feat = self.conv_1x1_n(high_fuse_feat)
        high_global_info = high_fuse_feat.split(self.trans_channels[2:4], dim=1)
        
        ## inject low-level global info to n4
        n4_adjacent_info = self.LAF_n4(p3, p4_half)
        n4 = self.Inject_n4(n4_adjacent_info, high_global_info[0])
        n4 = self.Rep_n4(n4)
        
        ## inject low-level global info to n5
        n5_adjacent_info = self.LAF_n5(n4, c5_half)
        n5 = self.Inject_n5(n5_adjacent_info, high_global_info[1])
        n5 = self.Rep_n5(n5)
        
        outputs = [p3, n4, n5]
        
        return outputs


class GDNeck2(
        GDNeck):
    def forward(self, input):
        (c2, c3, c4, c5) = input
        
        # Low-GD
        ## use conv fusion global info
        low_align_feat = self.low_FAM(input)
        low_fuse_feat = self.low_IFM(low_align_feat)
        low_global_info = low_fuse_feat.split(self.trans_channels[0:2], dim=1)
        
        ## inject low-level global info to p4
        c5_half = self.reduce_layer_c5(c5)
        p4_adjacent_info = self.LAF_p4([c3, c4, c5_half])
        p4 = self.Inject_p4(p4_adjacent_info, low_global_info[0])
        p4 = self.Rep_p4(p4)
        
        ## inject low-level global info to p3
        p4_half = self.reduce_layer_p4(p4)
        p3_adjacent_info = self.LAF_p3([c2, c3, p4_half])
        p3 = self.Inject_p3(p3_adjacent_info, low_global_info[1])
        p3 = self.Rep_p3(p3)
        
        # High-GD
        ## use transformer fusion global info
        high_align_feat = self.high_FAM([p3, p4, c5])
        high_fuse_feat = self.high_IFM(high_align_feat)
        high_fuse_feat = self.conv_1x1_n(high_fuse_feat)
        high_global_info = high_fuse_feat.split(self.trans_channels[2:4], dim=1)
        
        ## inject low-level global info to n4
        n4_adjacent_info = self.LAF_n4(p3, p4_half)
        n4 = self.Inject_n4(n4_adjacent_info, high_global_info[0])
        n4 = self.Rep_n4(n4)
        
        ## inject low-level global info to n5
        n5_adjacent_info = self.LAF_n5(p4, c5_half)
        n5 = self.Inject_n5(n5_adjacent_info, high_global_info[1])
        n5 = self.Rep_n5(n5)
        
        outputs = [p3, n4, n5]
        
        return outputs
