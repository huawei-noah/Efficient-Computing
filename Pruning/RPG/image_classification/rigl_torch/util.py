# 2023.11-Modified some parts in the code
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
import torchvision


EXCLUDED_TYPES = (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)


def get_weighted_layers(model, i=0, layers=None, linear_layers_mask=None):
    if layers is None:
        layers = []
    if linear_layers_mask is None:
        linear_layers_mask = []

    items = model._modules.items()
    if i == 0:
        items = [(None, model)]

    for layer_name, p in items:
        if isinstance(p, torch.nn.Linear):
            layers.append([p])
            linear_layers_mask.append(1)
        elif hasattr(p, 'weight') and type(p) not in EXCLUDED_TYPES and p.weight.dim() > 1:
            layers.append([p])
            linear_layers_mask.append(0)
        elif isinstance(p, torchvision.models.resnet.Bottleneck) or isinstance(p, torchvision.models.resnet.BasicBlock):
            _, linear_layers_mask, i = get_weighted_layers(p, i=i + 1, layers=layers, linear_layers_mask=linear_layers_mask)
        else:
            _, linear_layers_mask, i = get_weighted_layers(p, i=i + 1, layers=layers, linear_layers_mask=linear_layers_mask)

    return layers, linear_layers_mask, i 



def get_W(model, return_linear_layers_mask=False):
    layers, linear_layers_mask, _ = get_weighted_layers(model)

    W = []
    for layer in layers:
        idx = 0 if hasattr(layer[0], 'weight') else 1
        W.append(layer[idx].weight)

    assert len(W) == len(linear_layers_mask)

    if return_linear_layers_mask:
        return W, linear_layers_mask
    return W
