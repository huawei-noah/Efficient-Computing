from types import MethodType

import torch
import torch.nn.functional as F

from timm.models.beit import Beit
from timm.models.resnet import ResNet


class Config:
    _feat_dim = {
        'resnet50': (
            (64, 112, 112), (256, 56, 56), (512, 28, 28), (1024, 14, 14), (2048, 7, 7), (2048, None, None)),
        'resnet152': (
            (64, 112, 112), (256, 56, 56), (512, 28, 28), (1024, 14, 14), (2048, 7, 7), (2048, None, None)),
        'swin_small_patch4_window7_224': (
            (96, 56, 56), (192, 28, 28), (384, 14, 14), (768, 7, 7), (768, 7, 7), (768, None, None)),
        'swin_base_patch4_window7_224': (
            (128, 56, 56), (256, 28, 28), (512, 14, 14), (1024, 7, 7), (1024, 7, 7), (1024, None, None)),
        'swin_large_patch4_window7_224': (
            (192, 56, 56), (384, 28, 28), (768, 14, 14), (1536, 7, 7), (1536, 7, 7), (1536, None, None)),
        'beitv2_large_patch16_224': (
            (64, 56, 56), (64, 56, 56), (256, 28, 28), (1024, 14, 14), (1024, 7, 7), (1024, None, None)),
        'bit_r152x2': (
            (128, 112, 112), (512, 56, 56), (1024, 28, 28), (2048, 14, 14), (4096, 7, 7), (4096, 1, 1)),
    }

    _kd_feat_index = {
        'resnet50': (1, 2, 3, 4),
        'resnet152': (1, 2, 3, 4),
        'swin_small_patch4_window7_224': (0, 1, 2, 4),
        'swin_base_patch4_window7_224': (0, 1, 2, 4),
        'swin_large_patch4_window7_224': (0, 1, 2, 4),
        'beitv2_large_patch16_224': (1, 2, 3, 4),
        'bit_r152x2': (1, 2, 3, 4),
    }

    def get_pre_logit_dim(self, model):
        feat_sizes = self._feat_dim[model]
        if isinstance(feat_sizes, tuple):
            return feat_sizes[-1][0]
        else:
            return feat_sizes

    def get_used_feature_index(self, model):
        index = self._kd_feat_index[model]
        if index is None:
            raise NotImplementedError(f'undefined feature kd for model {model}')
        return index

    def get_feature_size_by_index(self, model, index):
        valid_index = self.get_used_feature_index(model)
        feat_sizes = self._feat_dim[model]
        assert index in valid_index
        return feat_sizes[index]


config = Config()


def register_forward(model):  # only resnet have implemented pre_act feat
    if isinstance(model, ResNet):  # ResNet
        model.forward = MethodType(ResNet_forward, model)
        model.forward_features = MethodType(ResNet_forward_features, model)
    elif isinstance(model, Beit):  # Beit
        model.forward = MethodType(Beitv2_forward, model)
        model.forward_features = MethodType(Beitv2_forward_features, model)
    else:
        raise NotImplementedError('undefined forward method to get feature, check the exp setting carefully!')


def _unpatchify(x, p, remove_token=0):
    """
    x: (N, L, patch_size**2 *C)
    imgs: (N, C, H, W)
    """
    # p = self.patch_embed.patch_size[0]
    x = x[:, remove_token:, :]
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
    return imgs


# ResNet
def bottleneck_forward(self, x):
    shortcut = x

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.drop_block(x)
    x = self.act2(x)
    x = self.aa(x)

    x = self.conv3(x)
    x = self.bn3(x)

    if self.se is not None:
        x = self.se(x)

    if self.drop_path is not None:
        x = self.drop_path(x)

    if self.downsample is not None:
        shortcut = self.downsample(shortcut)
    x += shortcut
    pre_act_x = x
    x = self.act3(pre_act_x)

    return x, pre_act_x


def ResNet_forward_features(self, x, requires_feat):
    pre_act_feat = []
    feat = []
    x = self.conv1(x)
    x = self.bn1(x)
    pre_act_feat.append(x)
    x = self.act1(x)
    feat.append(x)
    x = self.maxpool(x)

    for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
        for bottleneck in layer:
            x, pre_act_x = bottleneck_forward(bottleneck, x)

        pre_act_feat.append(pre_act_x)
        feat.append(x)

    return (x, (pre_act_feat, feat)) if requires_feat else x


def ResNet_forward(self, x, requires_feat=False):
    if requires_feat:
        x, (pre_act_feat, feat) = self.forward_features(x, requires_feat=True)
        x = self.forward_head(x, pre_logits=True)
        feat.append(x)
        pre_act_feat.append(x)
        x = self.fc(x)
        return x, (pre_act_feat, feat)
    else:
        x = self.forward_features(x, requires_feat=False)
        x = self.forward_head(x)
        return x


def Beitv2_forward_features(self, x, requires_feat):
    x = self.patch_embed(x)
    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    if self.pos_embed is not None:
        x = x + self.pos_embed
    x = self.pos_drop(x)

    pre_act_feat = [_unpatchify(x, 4, 1)]  # stem
    feat = [_unpatchify(x, 4, 1)]  # stem

    rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
    for i, blk in enumerate(self.blocks):  # fixme: curremt for beitv2L only
        x = blk(x, shared_rel_pos_bias=rel_pos_bias)
        f = None
        if i == 1:
            f = _unpatchify(x, 4, 1)
        elif i == 3:
            f = _unpatchify(x, 2, 1)
        elif i == 21:
            f = _unpatchify(x, 1, 1)
        elif i == 23:
            f = F.adaptive_avg_pool2d(_unpatchify(x, 1, 1), (7, 7))
        if f is not None:
            pre_act_feat.append(f)
            feat.append(f)
    x = self.norm(x)
    return (x, (pre_act_feat, feat)) if requires_feat else x


def Beitv2_forward(self, x, requires_feat=False):
    if requires_feat:
        x, (pre_act_feat, feat) = self.forward_features(x, requires_feat=True)
        x = self.forward_head(x, pre_logits=True)
        feat.append(x)
        pre_act_feat.append(x)
        x = self.head(x)
        return x, (pre_act_feat, feat)
    else:
        x = self.forward_features(x, requires_feat=False)
        x = self.forward_head(x)
        return x
