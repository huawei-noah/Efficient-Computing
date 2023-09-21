import torch

from yolov6.layers.common import RepVGGBlock


def switch_to_deploy(model):
    model = _convert_batchnorm(model)
    model = convert_checkpoint_False(model)
    
    for layer in model.modules():
        if isinstance(layer, (RepVGGBlock)):
            layer.switch_to_deploy()
    return model


def convert_checkpoint_False(module):
    for layer in module.modules():
        if hasattr(layer, 'use_checkpoint'):
            layer.use_checkpoint = False
        if hasattr(layer, 'use_cpt'):
            layer.use_cpt = False
    return module


def convert_checkpoint_True(module):
    for layer in module.modules():
        if hasattr(layer, 'use_checkpoint'):
            layer.use_checkpoint = True
        if hasattr(layer, 'use_cpt'):
            layer.use_cpt = True
    return module


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output
