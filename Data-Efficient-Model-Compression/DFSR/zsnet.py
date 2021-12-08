#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import argparse
import os
import numpy as np
import math
import sys
import pdb

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch

import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import moxing as mox

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class GeneratorINE1(nn.Module):
    def __init__(self, img_size, channels, latent=10):
        super(GeneratorINE1, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent, 128*self.init_size**2))
        self.conv_blocks0 = nn.Sequential(
            nn.InstanceNorm2d(num_features=128, affine=True),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=128, eps=0.8,affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=64, eps=0.8,affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
            #nn.BatchNorm2d(opt.channels, affine=False) 
        )
        self.apply(_weights_init)
        
    def forward(self, z, in_feature=False):
        out = self.l1(z)
        l_output = out
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        #img = (img + 1) * 127.5
        img = (img + 1) * 0.5
        if not in_feature:
            return img
        else:
            return img, l_output

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
        