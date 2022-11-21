#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

class Generator(nn.Module):
    def __init__(self,dim=64):
        super(Generator, self).__init__()
        
        self.dim=dim
        
        self.linear1=nn.Linear(128, 4 * 4 * 4 * dim)
        self.bn1=nn.BatchNorm1d(4 * 4 * 4 * dim)#
        self.relu1=nn.ReLU(True)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, stride=2),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 2, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
        )
        self.deconv_out = nn.ConvTranspose2d(dim, 3, 2, stride=2) 
      
        self.tanh = nn.Tanh()

    def forward(self, input):
  
        
        output=self.linear1(input)       
        output=self.bn1(output)
        output=self.relu1(output)
       
        output = output.view(-1, 4 * self.dim, 4, 4)
        
        
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)
