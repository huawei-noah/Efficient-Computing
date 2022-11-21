
#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np

from utils import AverageMeter,accuracy



class MaskBlock(nn.Module):
    def __init__(self, in_channels, out_channels, args=None): 
        super(MaskBlock, self).__init__() 
        
      
        self.clamp_max=args.clamp_max

        if out_channels < 80:
            squeeze_rate = args.squeeze_rate // 2
        else:
            squeeze_rate = args.squeeze_rate
            
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.fc1 = nn.Linear(in_channels, out_channels // squeeze_rate, bias=False)
        self.fc2 = nn.Linear(out_channels // squeeze_rate, out_channels, bias=True)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 1.0)# 
        
        self.register_buffer('mask_sum', torch.zeros(out_channels))
        self.register_buffer('thre',torch.zeros(1))
        self.thre.fill_(args.thre_init)
         
    def forward(self, x):

        
        x_averaged = self.avg_pool(x).squeeze(-1).squeeze(-1)
        y = self.fc1(x_averaged)
        y = F.relu(y,inplace=True)
        y = self.fc2(y)

        mask_before = F.relu(y) 
        mask_before=torch.clamp(mask_before,max=self.clamp_max)
        _lasso=mask_before.mean(dim=-1)##
        
        if self.training:
            self.mask_sum.add_(mask_before.data.sum(dim=0)) 
        
        tmp=torch.ones_like(mask_before)
        tmp[mask_before.data<self.thre]=0 
        mask=mask_before*tmp
        
        return mask,_lasso,mask_before

class SimiBlock(nn.Module):
    def __init__(self, args=None): 
        super(SimiBlock, self).__init__() 
        
                  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)   
        self.fc1 = nn.Linear(1, 1, bias=False)
        self.fc2 = nn.Linear(1, 1, bias=True)
        self.mask=None       
        
    def forward(self, x):
    
        x_averaged = self.avg_pool(x).squeeze(-1).squeeze(-1)        
        self.mask=x_averaged       
        return x_averaged  


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class MaskedBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,args=None):
        super(MaskedBasicblock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    
     
        self.mb1=MaskBlock(inplanes, planes, args=args)
        self.mb2=MaskBlock(planes, planes, args=args)
        

    def forward(self, x):
        x,_mask_list,_lasso_loss,_mask_before_list,_avg_fea_list=x
        
        residual = x
        
 
        mask1,_lasso1,mask1_before=self.mb1(x)
        _mask_list.append(mask1)
        _lasso_loss.append(_lasso1)
        _mask_before_list.append(mask1_before)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        

        _avg_fea_list.append(F.adaptive_avg_pool2d(out,1))
        out=out* mask1.unsqueeze(-1).unsqueeze(-1) 
        mask2,_lasso2,mask2_before=self.mb2(out)
        _mask_list.append(mask2)
        _lasso_loss.append(_lasso2)
        _mask_before_list.append(mask2_before)

        
        
        out = self.conv2(out)
        out = self.bn2(out)
        
    
        _avg_fea_list.append(F.adaptive_avg_pool2d(out,1))
        out=out* mask2.unsqueeze(-1).unsqueeze(-1)
 
        

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return [out,_mask_list,_lasso_loss,_mask_before_list,_avg_fea_list]


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,args=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        
      
  
        
        
        self.conv1_7x7 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],args=args)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,args=args)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,args=args)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,args=args)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,args=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,args=args))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,args=args))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_7x7(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1([x,[],[],[],[]])
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        _mask_list=x[1]
        _lasso_loss = x[2]
        _mask_before_list=x[3]
        _avg_fea_list=x[4]
        
        x = self.avgpool(x[0])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        

        return x,_mask_list,_lasso_loss,_mask_before_list,_avg_fea_list

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(MaskedBasicblock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        print('ResNet-18 Use pretrained model for initalization')
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(MaskedBasicblock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        print('ResNet-34 Use pretrained model for initalization')
    return model


           



