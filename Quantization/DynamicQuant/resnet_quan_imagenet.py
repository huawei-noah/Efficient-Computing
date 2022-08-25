# 2022.07.19-Changed for implementation for DQ model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
This script was started from an early version of the PyTorch ImageNet example 
(https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils.quan_conv import QuanConv
from utils.quan_conv import DynamicQConv as MyConv


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


Conv = nn.Conv2d

def conv3x3(in_planes, out_planes, name_w, name_a, nbit_w, nbit_a, stride=1):
    "3x3 convolution with padding"
    return MyConv(in_planes, out_planes, 3, name_w, name_a, nbit_w, nbit_a, stride=stride, padding=1, bias=False)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
       
        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, name_w, name_a, nbit_w, nbit_a, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, name_w, name_a, nbit_w, nbit_a, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, name_w, name_a, nbit_w, nbit_a)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, one_hot):
        residual = x

        out = self.conv1(x, one_hot[:,0,:].squeeze())
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, one_hot[:,1,:].squeeze())
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample[0](x)
            residual = self.downsample[1](residual)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock_Quan(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, name_w, name_a, nbit_w, nbit_a, stride=1, downsample=None):
        super(BasicBlock_Quan, self).__init__()
        self.conv1 = QuanConv(inplanes, planes, 3, name_w, name_a, nbit_w, nbit_a, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = QuanConv(planes, planes, 3, name_w, name_a, nbit_w, nbit_a, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample[0](x)
            residual = self.downsample[1](residual)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, name_w, name_a, nbit_w, nbit_a, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = MyConv(inplanes, planes, name_w, name_a, nbit_w, nbit_a, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = MyConv(planes, planes, name_w, name_a, nbit_w, nbit_a, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = MyConv(planes, planes * 4, name_w, name_a, nbit_w, nbit_a, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, one_hot):
        residual = x

        out = self.conv1(x, one_hot[:,0,:].squeeze())
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, one_hot[:,1,:].squeeze())
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, one_hot[:,2,:].squeeze())
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample[0](x)
            residual = self.downsample[1](residual)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, name_w, name_a, nbit_w, nbit_a, num_layers, block_layers, num_bits=3, num_classes=1000, width=1.):
        self.inplanes = int(64*width)
        super(ResNet, self).__init__()
        self.conv1 = Conv(3, int(64*width), kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(int(64*width))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock_Quan, int(64*width),  layers[0], name_w, name_a, nbit_w, nbit_a)
        self.layer2 = self._make_layer(block, int(128*width), layers[1], name_w, name_a, nbit_w, nbit_a, stride=2)
        self.layer3 = self._make_layer(block, int(256*width), layers[2], name_w, name_a, nbit_w, nbit_a, stride=2)
        self.layer4 = self._make_layer(block, int(512*width), layers[3], name_w, name_a, nbit_w, nbit_a, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(int(512*width) * block.expansion, num_classes)
        
        
        self.avgpool_policy = nn.AvgPool2d((7,7))
        self.fc1 = nn.Linear(64*8*8, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_bits*num_layers)
        self.num_layers = num_layers
        self.block_layers = block_layers
        self.num_bits = num_bits
        
        for m in self.modules():
            if isinstance(m, Conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, name_w, name_a, nbit_w, nbit_a, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.ModuleList()
            downsample.append(QuanConv(self.inplanes, planes * block.expansion, 
                              1, name_w, name_a, nbit_w, nbit_a, stride=stride, bias=False))
            downsample.append(nn.BatchNorm2d(planes * block.expansion))

        layers = nn.ModuleList()
        layers.append(block(self.inplanes, planes, name_w, name_a, nbit_w, nbit_a, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, name_w, name_a, nbit_w, nbit_a))

        return layers #nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for m in self.layer1:
            x = m(x)
        
        feat = self.avgpool_policy(x)
        feat = self.fc1(feat.view(x.size(0),-1))
        feat = self.dropout(feat)
        feat = self.fc2(feat)
        feat = torch.reshape(feat, (-1, self.num_layers, self.num_bits))
        one_hot = F.gumbel_softmax(feat, tau=1, hard=True)
        
        i = 0
        for m in self.layer2:
            x = m(x, one_hot[:,i:i+self.block_layers,:])
            i += self.block_layers
        for m in self.layer3:
            x = m(x, one_hot[:,i:i+self.block_layers,:])
            i += self.block_layers
        for m in self.layer4:
            x = m(x, one_hot[:,i:i+self.block_layers,:])
            i += self.block_layers
            
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x, one_hot


def resnet18(pretrained=False, name_w='pact', name_a='pact', nbit_w=4, nbit_a=4, num_layers=12, block_layers=2, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], name_w, name_a, nbit_w, nbit_a, num_layers, block_layers, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
