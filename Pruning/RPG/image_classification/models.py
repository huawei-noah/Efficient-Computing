# 2023.11-Modified some parts in the code
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torchvision.models as models
import torch

def get_models(args):
    model_dict = {
        'resnet18':models.resnet18,
        'resnet34':models.resnet34,
        'resnet50':models.resnet50,
        'resnet101':models.resnet101,
        'resnet152':models.resnet152,
    }
    num_classes = {'imagenet':1000,'cifar10':10,'cifar100':100}[args.dataset]
    model = model_dict[args.arch](pretrained=True, num_classes=num_classes)
    return model