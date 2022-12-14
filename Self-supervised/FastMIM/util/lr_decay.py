# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
#
# 2022.12.14-Changed for building FastMIM
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#

import json
from collections import defaultdict


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    
    if hasattr(model, 'blocks'):
        if hasattr(model, 'layer1') and hasattr(model, 'layer2'):
            num_layers = len(model.blocks) + 3
        else:
            num_layers = len(model.blocks) + 1
    elif hasattr(model, 'layers'):
        """ setting for swin transformer 26ld"""
        num_layers = 1
        for i in range(len(model.layers)):
            num_layers += len(model.layers[i].blocks)
    else:
        num_layers = model.get_num_layers()
    
    if isinstance (num_layers, list):
        layer_scales = list(layer_decay ** (sum(num_layers) + 1 - i) for i in range(sum(num_layers) + 1 + 1))
    else:
        layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        if model. __class__. __name__ == 'SwinTransformer':
            stage_nlayers = [len(model.layers[0].blocks), len(model.layers[1].blocks), len(model.layers[2].blocks)]
            nlayer = [0, sum(stage_nlayers[0:1]), sum(stage_nlayers[0:2]), sum(stage_nlayers[0:3])]
            layer_id = get_layer_id_for_swin(n, num_layers, nlayer)
        elif model. __class__. __name__ == 'CMT':
            nlayer = [0, sum(num_layers[0:1]), sum(num_layers[0:2]), sum(num_layers[0:3])]
            layer_id = get_layer_id_for_cmt(n, sum(num_layers) + 1, nlayer)
        elif model. __class__. __name__ == 'PyramidVisionTransformerV2':
            nlayer = [0, sum(num_layers[0:1]), sum(num_layers[0:2]), sum(num_layers[0:3])]
            layer_id = get_layer_id_for_pvtv2(n, sum(num_layers) + 1, nlayer)
        else:
            layer_id = get_layer_id_for_vit(n, num_layers)
            
        group_name = "layer_%02d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
    
    if layer_decay == 1.0:
        print("parameter with layer decay rate = 1.0")
    else:
        print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2, sort_keys=True))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks_token_only'):
        return num_layers
    elif name.startswith('blocks'):
        # return int(name.split('.')[1]) + 1 if int(name.split('.')[1]) + 1 < num_layers else num_layers - 1
        return int(name.split('.')[1]) + 1
    else:
        return num_layers
    
def get_layer_id_for_swin(name, num_layers, nlayer):
    if name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        if 'downsample' in name:
            return nlayer[int(name.split('.')[1]) + 1]
        else:
            return nlayer[int(name.split('.')[1])] + int(name.split('.')[3]) + 1
    else:
        return num_layers
    
def get_layer_id_for_cmt(name, num_layers, nlayer):
    if name.startswith("stem"):
        return 0
    elif name.startswith("patch_embed_a") or name.startswith("relative_pos_a"):
        return nlayer[0] + 1
    elif name.startswith("blocks_a"):
        return nlayer[0] + 1 + int(name.split('.')[1])
    elif name.startswith("patch_embed_b") or name.startswith("relative_pos_b"):
        return nlayer[1] + 1
    elif name.startswith("blocks_b"):
        return nlayer[1] + 1 + int(name.split('.')[1])
    elif name.startswith("patch_embed_c") or name.startswith("relative_pos_c"):
        return nlayer[2] + 1
    elif name.startswith("blocks_c"):
        return nlayer[2] + 1 + int(name.split('.')[1])
    elif name.startswith("patch_embed_d") or name.startswith("relative_pos_d"):
        return nlayer[3] + 1
    elif name.startswith("blocks_d"):
        return nlayer[3] + 1 + int(name.split('.')[1])
    else:
        return num_layers

def get_layer_id_for_pvtv2(name, num_layers, nlayer):
    if name.startswith("patch_embed"):
        return nlayer[int(name.split('.')[0][-1]) - 1]
    if name.startswith("norm1"):
        return nlayer[1]
    if name.startswith("norm2"):
        return nlayer[2]
    if name.startswith("norm3"):
        return nlayer[3]
    elif name.startswith("block"):
        return nlayer[int(name.split('.')[0][5]) - 1] + 1 + int(name.split('.')[1])
    else:
        return num_layers
