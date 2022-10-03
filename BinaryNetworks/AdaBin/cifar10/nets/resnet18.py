# 2022.09.29-Changed for implementation for AdaBin model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
This script was started from an early version of the PyTorch ImageNet example 
(https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.binarylib import AdaBin_Conv2d, Maxout

class BasicBlock_1w1a(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_1w1a, self).__init__()
        self.conv1 = AdaBin_Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.nonlinear1 = Maxout(planes)

        self.conv2 = AdaBin_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.nonlinear2 = Maxout(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                     )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out += self.shortcut(x)
        out = self.nonlinear1(out)
        x1 = out
        out = self.bn2(self.conv2(out))
        out += x1
        out = self.nonlinear2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.nonlinear1 = Maxout(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(512*block.expansion)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.nonlinear1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        # out = F.softmax(out, dim=1)
        return out 

def resnet18_1w1a():
    return ResNet(BasicBlock_1w1a, [2,2,2,2])
