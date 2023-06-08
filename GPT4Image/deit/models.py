# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# 2023.6.5-Changed for building GPT4Image
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from utils import build_mlp


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 
    'deit_base_patch16_224', 'deit_base_patch16_384'
]


class VisionTransformer(TimmVisionTransformer):
    def __init__(self, *args, proj_type=None, proj_dim=768, **kwargs):
        super(VisionTransformer, self).__init__(*args, **kwargs)

        if proj_type is None:
            print('no emb projection')
            self.proj_layer = nn.Identity()
        elif proj_type == 'linear':
            print('building linear projection')
            self.proj_layer = nn.Linear(self.embed_dim, proj_dim, bias=False)
            trunc_normal_(self.proj_layer.weight, std=.02)
        elif proj_type == 'mlp':
            print('building mlp')
            self.proj_layer = build_mlp(in_dim=self.embed_dim, hidden_dim=int(2*proj_dim), out_dim=proj_dim, bn=False)
            trunc_normal_(self.proj_layer[0].weight, std=.02)
            trunc_normal_(self.proj_layer[-1].weight, std=.02)
        else:
            raise NotImplementedError

    def forward(self, x, get_feat=False):
        feat = self.forward_features(x)
        pred = self.head(feat)
        if get_feat:
            return self.proj_layer(feat), pred
        else:
            return pred

                                                                                                                    
@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
