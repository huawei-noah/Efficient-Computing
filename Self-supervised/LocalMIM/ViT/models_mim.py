# 2022.3.3-Changed for building LocalMIM
#          Huawei Technologies Co., Ltd. <foss@huawei.com>
#
# Modified from MAE by Haoqing Wang
# MAE: https://github.com/facebookresearch/mae
#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MAE_Decoder(nn.Module):
    def __init__(self, inp_dim, embed_dim=256, out_dim=27, scale=1., num_patches=196, depth=1, num_heads=8, mlp_ratio=4., qkv_bias=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_patches = num_patches
        self.embed = nn.Linear(inp_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+self.num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # pred head
        hidden = embed_dim
        if scale == 4.0:
            layers = [nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2),
                      LayerNorm(embed_dim//2),
                      nn.GELU(),
                      nn.ConvTranspose2d(embed_dim//2, embed_dim//4, kernel_size=2, stride=2)]
            hidden = embed_dim//4
        elif scale == 2.0:
            layers = [nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2)]
            hidden = embed_dim//2
        elif scale == 1.0:
            layers = []
        elif scale == 0.5:
            layers = [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
        layers.append(nn.Conv2d(hidden, out_dim, kernel_size=1))
        self.pred = nn.Sequential(*layers)

        self.initialize_weights()

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize position embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1]+1-x.shape[1], 1)
        x_ = torch.cat([x[:, 1:], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # [B, L, d]

        # predictor projection
        H = W = int(self.num_patches**0.5)
        x = x[:, 1:].transpose(1, 2).reshape(x.size(0), -1, H, W)
        x = self.pred(x)
        x = x.flatten(2, 3).transpose(1, 2)

        return x


class HOGLayer(nn.Module):
    def __init__(self, nbins, pool, bias=False, max_angle=math.pi, stride=1, padding=1, dilation=1):
        super(HOGLayer, self).__init__()
        self.nbins = nbins

        self.conv = nn.Conv2d(1, 2, 3, stride=stride, padding=padding, dilation=dilation, padding_mode='reflect', bias=bias)
        mat = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.conv.weight.data = mat[:, None, :, :]

        self.max_angle = max_angle
        self.pooler = nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False, count_include_pad=True)

    @ torch.no_grad()
    def forward(self, x):  # [B, 1, 224, 224]
        gxy = self.conv(x)

        # 2. Mag/ Phase
        mag = gxy.norm(dim=1)
        norm = mag[:, None, :, :]
        phase = torch.atan2(gxy[:, 0, :, :], gxy[:, 1, :, :])

        # 3. Binning Mag with linear interpolation
        phase_int = phase/self.max_angle*self.nbins
        phase_int = phase_int[:, None, :, :]

        n, c, h, w = gxy.shape
        out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=gxy.device)
        out.scatter_(1, phase_int.floor().long() % self.nbins, norm)

        hog = self.pooler(out)
        hog = nn.functional.normalize(hog, p=2, dim=1)
        return hog


class MaskedAutoencoderViT(nn.Module):
    """
        Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512,
                 decoder_depth=1, decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, hog_nbins=9, hog_bias=False, **kwargs):
        super().__init__()
        # MIM encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)])
        self.ID = [1, 3, depth-3, depth-1]
        self.scale = [4.0, 2.0, 1.0, 0.5]
        self.norm = nn.ModuleList([norm_layer(embed_dim) for _ in range(len(self.ID))])
        self.initialize_weights()

        # MIM decoder specifics
        self.decoder = nn.ModuleList([
            MAE_Decoder(embed_dim, decoder_embed_dim, in_chans*hog_nbins, s, num_patches, decoder_depth, decoder_num_heads, mlp_ratio, True, norm_layer)
            for s in self.scale])

        # target
        self.hog_enc = nn.ModuleList([HOGLayer(nbins=hog_nbins, pool=k, bias=hog_bias) for k in [4, 8, 16, 32]])
        for hog_enc in self.hog_enc:
            for param in hog_enc.parameters():
                param.requires_grad = False

    def initialize_weights(self):
        # initialization
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def HOG(self, imgs, k):  # [B, 3, 224, 224]
        """
        imgs: (N, 3, H, W)
        x: (N, L, d)
        """
        hog_R = self.hog_enc[k](imgs[:, :1, :, :])  # [B, nb, h, w]
        hog_G = self.hog_enc[k](imgs[:, 1:2, :, :])  # [B, nb, h, w]
        hog_B = self.hog_enc[k](imgs[:, 2:, :, :])  # [B, nb, h, w]
        hog_feat = torch.cat([hog_R, hog_G, hog_B], 1)  # [B, 3*nb, h, w]
        hog_feat = hog_feat.flatten(2, 3).transpose(1, 2)
        return hog_feat

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L*(1-mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # [N, len_keep]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # [N, len_keep, D]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)  # [B, num_patches, d]
        # add pos embed
        x = x + self.pos_embed[:, 1:]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        latent = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i in self.ID:
                latent.append(self.norm[self.ID.index(i)](x))

        return latent, mask, ids_restore

    def recal_mask(self, mask, k):
        B, L, s = mask.size(0), mask.size(1), self.scale[k]
        H = W = int(L**.5)
        if s >= 1.:
            s = int(s)
            mask = mask.reshape(B, H, W).unsqueeze(3).unsqueeze(2).repeat(1, 1, s, 1, s).reshape(B, -1)
        else:
            s = int(1/s)
            mask = mask.reshape(B, H//s, s, H//s, s).transpose(2, 3).mean((-2, -1)).reshape(B, -1)

        return mask

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = [self.HOG(imgs, k) for k in range(len(self.hog_enc))]

        loss = 0.
        for k in range(len(pred)):
            M = self.recal_mask(mask, k)
            loss += (((pred[k]-target[k])**2).mean(dim=-1)*M).sum()/M.sum()

        return loss

    def forward(self, imgs, mask_ratio=0.75):  # [B, C, H, W]
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = [self.decoder[i](latent[i], ids_restore) for i in range(len(latent))]
        loss = self.forward_loss(imgs, pred, mask)
        return loss


def MIM_vit_small_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=256,
        decoder_depth=1,
        decoder_num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def MIM_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=256,
        decoder_depth=1,
        decoder_num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def MIM_vit_large_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=256,
        decoder_depth=1,
        decoder_num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def MIM_vit_huge_patch14(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=256,
        decoder_depth=1,
        decoder_num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model