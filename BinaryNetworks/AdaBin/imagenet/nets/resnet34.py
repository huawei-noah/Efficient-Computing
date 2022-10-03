# 2022.09.29-Changed for implementation for AdaBin model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
This script was started from an early version of the PyTorch ImageNet example 
(https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
"""
import math
import torch.nn as nn
from utils.binarylib import AdaBin_Conv2d, Maxout

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = AdaBin_Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.nonlinear1 = Maxout(planes)

        self.conv2 = AdaBin_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.nonlinear2 = Maxout(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        
        out = self.nonlinear1(out)
        
        residual = out
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.nonlinear2(out)
        
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_stem=False,
                 avg_down=True, bypass_last_bn=False,
                 bn_group_size=1,
                 bn_group=None,
                 bn_sync_stats=False,
                 use_sync_bn=True):

        global BN, bypass_bn_weight_list

        BN = nn.BatchNorm2d

        bypass_bn_weight_list = []


        self.inplanes = 64
        super(ResNet, self).__init__()

        self.deep_stem = deep_stem
        self.avg_down = avg_down

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(32),
                        Maxout(32),
                        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(32),
                        Maxout(32),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.nonlinear1 = Maxout(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)
        self.nonlinear2 = Maxout(512 * block.expansion)

        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, AdaBin_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        if bypass_last_bn:
            for param in bypass_bn_weight_list:
                param.data.zero_()
            print('bypass {} bn.weight in BottleneckBlocks'.format(len(bypass_bn_weight_list)))

    def _make_layer(self, block, planes, blocks, stride=1, avg_down=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    BN(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    BN(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.nonlinear1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)     

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)      
        x = self.fc(x)

        return x

def resnet34_1w1a(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model
