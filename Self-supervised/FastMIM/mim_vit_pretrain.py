# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
#
# 2022.12.14-Changed for building FastMIM
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from util.hog_layer import HOGLayerC


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, block_size=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mim_loss='l2', **kwargs):
        super().__init__()
        
        decoder_num_heads = int(decoder_embed_dim / 32)

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True) if decoder_depth > 0 else None

        self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if mim_loss == "HOG":
            num_class = (patch_size//8)**2 * 9 * 3
        else:
            num_class = patch_size**2 * in_chans
        self.decoder_pred = nn.Linear(decoder_embed_dim, num_class, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.img_size = img_size
        self.block_size = block_size
        self.patch_size = patch_size
        self.mim_loss = mim_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p

        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->npqchw', x)
        x = x.reshape(shape=(imgs.shape[0], p**2 * 3, h, w))
        
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def mim_per_sample_block_masking(self, x, mask_ratio, block_size=16):
        batch, channel, height, width = x.shape
        input_size = self.img_size        
        assert height == width, f"Input height and width doesn't match ({height} != {width})."
        
        mask_size = input_size // block_size
        bw_ratio = height // mask_size
        len_keep = int(mask_size**2 * (1 - mask_ratio))

        noise = torch.rand(batch, mask_size**2, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        loss_mask = torch.ones([batch, mask_size**2], device=x.device)
        loss_mask[:, :len_keep] = 0
        loss_mask = torch.gather(loss_mask, dim=1, index=ids_restore)
        loss_mask = loss_mask.reshape(batch, 1, mask_size, mask_size).long()
        
        mask = loss_mask.repeat(1, bw_ratio**2, 1, 1)
        mask = mask.reshape(batch, bw_ratio, bw_ratio, mask_size, mask_size).permute(
            0, 3, 1, 4, 2).reshape(batch, 1, height, width)
        
        if self.block_size > self.patch_size:
            loss_mask = torch.repeat_interleave(loss_mask, self.block_size//self.patch_size, dim=2)
            loss_mask = torch.repeat_interleave(loss_mask, self.block_size//self.patch_size, dim=3)
        
        return mask, loss_mask

    def forward_encoder(self, imgs, mask_ratio):       
        B, C, H, W = imgs.shape
        mask, loss_mask = self.mim_per_sample_block_masking(imgs, mask_ratio, block_size=self.block_size)
        x = imgs * (1-mask) + (mask) * self.mask_token.repeat(B, 1, H, W)
        
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, loss_mask

    def forward_decoder(self, x):       
        if self.decoder_blocks:
            x = self.decoder_embed(x)
            for blk in self.decoder_blocks:
                x = blk(x)
                
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        return x
    
    def forward_l2_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        B, N, C = pred.shape
        H = W = int(N**0.5)
        pred = pred.transpose(-1,-2).reshape(B, C, H, W)
        
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=1, keepdim=True)
            var = target.var(dim=1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            # target = (target - mean) / torch.clip((var + 1.e-6)**.5, min=0.01)
            
        mask = mask.repeat(1, C, 1, 1).bool()

        loss = (pred[mask] - target[mask]) ** 2
        loss = loss.mean()
        
        return loss
    
    def forward_hog_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        B, N, C = pred.shape
        H = W = int(N**0.5)
        
        hogC = HOGLayerC(nbins=9, pool=8, norm_pix_loss=self.norm_pix_loss).cuda()
        target = hogC(imgs)
        
        mask_size = mask.shape[-1]
        if mask_size > W:
            target_size, target_channel = target.shape[3], target.shape[1]
            target = target.permute(0, 2, 3, 1).flatten(1, 2)
            mask = torch.repeat_interleave(mask, target_size//mask_size, dim=2)
            mask = torch.repeat_interleave(mask, target_size//mask_size, dim=3)
            pred = pred.reshape(B, H, W, -1, target_size//H, target_size//W).permute(0, 1, 4, 2, 5, 3).reshape(B, target_size**2, target_channel)
        else:
            unfold_size = target.shape[-1] // W
            target = (
                target.permute(0, 2, 3, 1)
                .unfold(1, unfold_size, unfold_size)
                .unfold(2, unfold_size, unfold_size)
                .flatten(1, 2).flatten(2)
            )

        mask = mask.flatten(1).bool()
        loss = (pred[mask] - target[mask]) ** 2
        loss = loss.mean()
        
        return loss

    def forward(self, imgs, mask_ratio=0.75, mask_type='random'):
        latent, mask = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent)
        if self.mim_loss == 'l2':
            loss = self.forward_l2_loss(imgs, pred, mask)
        elif self.mim_loss == 'HOG':
            loss = self.forward_hog_loss(imgs, pred, mask)
            
        return loss, pred, mask


def mim_vit_base(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
