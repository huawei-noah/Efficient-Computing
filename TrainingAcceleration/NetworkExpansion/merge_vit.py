# 2023.3.20-Written for building NetworkExpansion
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
from copy import deepcopy
from torch import distributed as dist
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg


def create_custom_vit(num_heads=12, depth=12, **kwargs):
    # fix the dimension of each attn head to be 64
    assert type(num_heads) == type(depth) == int
    embed_dim = int(64 * num_heads)
    model = VisionTransformer(
        patch_size=16, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


def expand_depth_using_ema_model_stack(ori_vit, ema_vit, new_layer=2):
    assert ori_vit.blocks[0].attn.num_heads == ema_vit.blocks[0].attn.num_heads
    assert isinstance(new_layer, int) and new_layer > 0
    assert len(ori_vit.blocks) == len(ema_vit.blocks) >= new_layer

    ori_n_layer = len(ori_vit.blocks)
    for i in range(new_layer):
        ori_vit.blocks.add_module(
            str(ori_n_layer + i), deepcopy(ema_vit.blocks[i - new_layer])
        )
    assert len(ori_vit.blocks) == ori_n_layer + new_layer

    ori_vit.requires_grad_(True)
    return ori_vit


def expand_depth_using_ema_model_insert(ori_vit, ema_vit, new_layer=2):
    assert ori_vit.blocks[0].attn.num_heads == ema_vit.blocks[0].attn.num_heads
    assert isinstance(new_layer, int) and new_layer > 0
    assert len(ori_vit.blocks) == len(ema_vit.blocks) >= new_layer

    ori_n_layer = len(ori_vit.blocks)
    for i in range(new_layer):
        ori_vit.blocks.add_module(
            str(ori_n_layer + i), torch.nn.Identity()
        )  # for placeholder purpose
    total_layer = len(ori_vit.blocks)
    assert total_layer == ori_n_layer + new_layer

    for i in range(new_layer):
        ptr = -1 - i
        ema_idx = - (2 * i + 1)
        ori_vit.blocks[ema_idx] = deepcopy(ema_vit.blocks[ptr])
        if i == (new_layer - 1): break
        ori_idx = - (2 * i + 2)
        ori_vit.blocks[ori_idx] = deepcopy(ori_vit.blocks[ptr-new_layer])

    ori_vit.requires_grad_(True)
    return ori_vit


depth_fn_map = {
    'stack': expand_depth_using_ema_model_stack,
    'insert': expand_depth_using_ema_model_insert,
}


@torch.no_grad()
def dispatch_vit_expand(old_vit, old_ema, new_vit, last_model_config, next_model_config,
                        width_fn, depth_fn, call_barrier=False):

    assert old_vit.num_classes == old_ema.num_classes == new_vit.num_classes
    assert (not hasattr(old_vit, 'module')) and \
           (not hasattr(old_ema, 'module')) and \
           (not hasattr(new_vit, 'module')), 'Do not pass distributed model.'

    new_num_layers, old_num_layers = next_model_config['depth'], last_model_config['depth']
    expand_depth = (new_num_layers > old_num_layers)

    if call_barrier:
        dist.barrier()

    if expand_depth:
        print('using depth expanding function :', depth_fn)
        assert len(old_vit.blocks) == old_num_layers
        drop_path_rate = round(old_vit.blocks[-1].drop_path.drop_prob, 4)
        new_layer = new_num_layers - old_num_layers
        ret_vit = depth_fn_map[depth_fn](old_vit, old_ema, new_layer=new_layer)
        assert len(ret_vit.blocks) == new_num_layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, new_num_layers)]
        for i in range(new_num_layers):  # re-arrange drop_path_rate
            ret_vit.blocks[i].drop_path.drop_prob = dpr[i]

    elif expand_width:
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    if call_barrier:
        dist.barrier()

    return ret_vit
