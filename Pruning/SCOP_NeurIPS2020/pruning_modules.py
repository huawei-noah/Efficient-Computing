#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from torch.nn.parameter import Parameter



class Kf_Conv2d(nn.Module):
    def __init__(self, conv_ori,bn_ori):
        super(Kf_Conv2d, self).__init__()
        self.conv=conv_ori 
        self.bn=bn_ori        
        self.out_channels=self.conv.out_channels
        self.kfscale= Parameter(torch.ones(1,self.out_channels,1,1)) 
        self.kfscale.data.fill_(0.5)
    def forward(self,x):
        x=self.conv(x)        
        if self.training:
            num_ori=int(x.shape[0]//2)            
            x=torch.cat([self.kfscale*x[:num_ori]+(1-self.kfscale)*x[num_ori:],x[num_ori:]],dim=0)
        x=self.bn(x)
        return x

class Masked_Conv2d_bn(nn.Module):
    def __init__(self, kf_conv2d_ori):
        super(Masked_Conv2d_bn, self).__init__()
        self.conv=kf_conv2d_ori.conv
        self.bn=kf_conv2d_ori.bn
        self.register_buffer('out_index',kf_conv2d_ori.out_index.clone())
        
        
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
       
        mask=torch.zeros(*x.shape,device=x.device)
       
        mask[:,self.out_index,:,:]=1 
        x=x*mask
        return x

class Pruned_Conv2d_bn1(nn.Module):# 只砍输出
    def __init__(self, masked_module):
        super(Pruned_Conv2d_bn1, self).__init__()
        
        newconv=nn.Conv2d(in_channels=masked_module.conv.in_channels,out_channels=len(masked_module.out_index),
                        kernel_size=masked_module.conv.kernel_size,stride=masked_module.conv.stride,
                          bias=False,padding=masked_module.conv.padding)

        weight_data=masked_module.conv.weight.data.clone()
        weight_data=weight_data.index_select(dim=0,index=masked_module.out_index)
        newconv.weight.data=weight_data

        newbn=nn.BatchNorm2d(len(masked_module.out_index))
        newbn.weight.data=masked_module.bn.weight.data.clone().index_select(dim=0,index=masked_module.out_index)
        newbn.bias.data=masked_module.bn.bias.data.clone().index_select(dim=0,index=masked_module.out_index)
        newbn.running_mean.data=masked_module.bn.running_mean.data.clone().index_select(dim=0,index=masked_module.out_index)
        newbn.running_var.data=masked_module.bn.running_var.data.clone().index_select(dim=0,index=masked_module.out_index)   
        
        self.conv=newconv
        self.bn=newbn
        
        self.oriout_channels=masked_module.conv.out_channels
        self.out_index=masked_module.out_index
        
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        return x

class Pruned_Conv2d_bn_middle(nn.Module):# 输入输出都砍
    def __init__(self, masked_module):
        super(Pruned_Conv2d_bn_middle, self).__init__()
        
        newconv=nn.Conv2d(in_channels=len(masked_module.in_index),out_channels=len(masked_module.out_index),
                        kernel_size=masked_module.conv.kernel_size,stride=masked_module.conv.stride,
                          bias=False,padding=masked_module.conv.padding)

        weight_data=masked_module.conv.weight.data.clone()
        weight_data=weight_data.index_select(dim=0,index=masked_module.out_index).index_select(dim=1,index=masked_module.in_index)
        newconv.weight.data=weight_data

        newbn=nn.BatchNorm2d(len(masked_module.out_index))
        newbn.weight.data=masked_module.bn.weight.data.clone().index_select(dim=0,index=masked_module.out_index)
        newbn.bias.data=masked_module.bn.bias.data.clone().index_select(dim=0,index=masked_module.out_index)
        newbn.running_mean.data=masked_module.bn.running_mean.data.clone().index_select(dim=0,index=masked_module.out_index)
        newbn.running_var.data=masked_module.bn.running_var.data.clone().index_select(dim=0,index=masked_module.out_index)   
        
        self.conv=newconv
        self.bn=newbn
        
        self.oriout_channels=masked_module.conv.out_channels
        self.out_index=masked_module.out_index
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        return x    
    
class Pruned_Conv2d_bn2(nn.Module):
    def __init__(self, masked_module):
        super(Pruned_Conv2d_bn2, self).__init__()
        
        newconv=nn.Conv2d(in_channels=len(masked_module.in_index),out_channels=len(masked_module.out_index),
                        kernel_size=masked_module.conv.kernel_size,stride=masked_module.conv.stride,
                          bias=False,padding=masked_module.conv.padding)

        weight_data=masked_module.conv.weight.data.clone()
        weight_data=weight_data.index_select(dim=0,index=masked_module.out_index).index_select(dim=1,index=masked_module.in_index)
        newconv.weight.data=weight_data

        newbn=nn.BatchNorm2d(len(masked_module.out_index))
        newbn.weight.data=masked_module.bn.weight.data.clone().index_select(dim=0,index=masked_module.out_index)
        newbn.bias.data=masked_module.bn.bias.data.clone().index_select(dim=0,index=masked_module.out_index)
        newbn.running_mean.data=masked_module.bn.running_mean.data.clone().index_select(dim=0,index=masked_module.out_index)
        newbn.running_var.data=masked_module.bn.running_var.data.clone().index_select(dim=0,index=masked_module.out_index)   
        
        self.conv=newconv
        self.bn=newbn
        
        self.oriout_channels=masked_module.conv.out_channels
        self.out_index=masked_module.out_index
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        output=torch.zeros(x.shape[0],self.oriout_channels,x.shape[2],x.shape[3],device=x.device)
        output[:,self.out_index,:,:]=x
        return output        
