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

import dct

def high_loss_sum_sub(sr_data, lr_data, p=2):
    if p > 1:
        sr_res = torch.sum(sr_data.mul(sr_data), dim=1)
        lr_res = torch.sum(lr_data.mul(lr_data), dim=1)
    else: 
        sr_res = torch.sum(torch.abs(sr_data), dim=1)
        lr_res = torch.sum(torch.abs(lr_data), dim=1)
    error = sr_res - lr_res
    #print(sr_res, lr_res, error)
    #print(error.shape, type(error))
    loss = - torch.sum(error)# ** 2
    return loss


def high_loss_norm(sr_data, lr_data, p=2, wabs=0):
    if wabs:
        sr_data = torch.abs(sr_data)
        lr_data = torch.abs(lr_data)
    error = sr_data - lr_data
    #print(error.shape, type(error))
    loss = - torch.norm(error, p=p)** 2
    return loss
    
    
def get_high_frequency_data(data, T=1, li=None):
    batch_size, channel, w, h = data.shape
    if channel > 1 :
        Y_data =  0.299 *data[:,0,0:w,0:h] + 0.587 *data[:,1,0:w,0:h] + 0.114 *data[:,2,0:w,0:h]
    else:
        Y_data = data
    dct_data = dct.dct_2d(Y_data/255.)
    #if not li:
    #    _, li = z_matrix(w,h)
    #print("dct_data shape:", dct_data.shape)
    
    dct_data_view = dct_data.contiguous().view([batch_size, -1])
    #print("dct_data_view shape", dct_data_view.shape)
    high_frequency_data = dct_data_view[:, li[T:]].squeeze()
    #print("high_frequency_data:", high_frequency_data.shape)
    return high_frequency_data
    

def z_matrix_wh(w, h):
    #print(rect)
    #w, h = rect.shape
    n =w
    i,j = 0,0
    index_list = []
    index = []
    up = True
    for t in range(n*n):
        index_list.append([j, i])
        index.append(j*w + i)

        if up:#右上
            if i==n-1:    
                j += 1      
                up=False     
            elif j==0:   
                i += 1       
                up=False   
            else:         
                i += 1       
                j -= 1
        else:#左下
            if j==n-1:  
                i += 1       
                up=True     
            elif i==0:   
                j += 1       
                up=True      
            else:        
                i -= 1      
                j += 1
    index = np.array(index)
    return index_list, index


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

    
class GeneratorBNE(nn.Module):
    def __init__(self, img_size, channels, latent=10):
        super(GeneratorBNE, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent, 128*self.init_size**2))
        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            #nn.Tanh(),
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
        if not in_feature:
            return img
        else:
            return img, l_output


class GeneratorINE(nn.Module):
    def __init__(self, img_size, channels, latent=10):
        super(GeneratorINE, self).__init__()

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
            #nn.Tanh(),
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
        if not in_feature:
            return img
        else:
            return img, l_output
        
        
        
class GeneratorBNE255(nn.Module):
    def __init__(self, img_size, channels, latent=10):
        super(GeneratorBNE255, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent, 128*self.init_size**2))
        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
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
        img = (img + 1) * 127.5
        if not in_feature:
            return img
        else:
            return img, l_output

class GeneratorBNELIN(nn.Module):
    def __init__(self, img_size, channels, latent=10):
        super(GeneratorBNELIN, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent, 128*self.init_size**2))
        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            # nn.Tanh(),
            nn.BatchNorm2d(channels, affine=True) 
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
        if not in_feature:
            return img
        else:
            return img, l_output
        

class GeneratorINE255(nn.Module):
    def __init__(self, img_size, channels, latent=10):
        super(GeneratorINE255, self).__init__()

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
        img = (img + 1) * 127.5
        if not in_feature:
            return img
        else:
            return img, l_output

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

class GeneratorINELIN(nn.Module):
    def __init__(self, img_size, channels, latent=10):
        super(GeneratorINELIN, self).__init__()

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
            # nn.Tanh(),
            nn.InstanceNorm2d(channels, affine=True) 
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
        if not in_feature:
            return img
        else:
            return img, l_output
        
class GeneratorINE255IN(nn.Module):
    def __init__(self, img_size, channels, latent=10):
        super(GeneratorINE255IN, self).__init__()

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
            nn.InstanceNorm2d(channels, affine=True), 
            nn.Tanh(),
            #
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
        img = (img + 1) * 127.5
        if not in_feature:
            return img
        else:
            return img, l_output 
        
        
class GeneratorINETAN(nn.Module):
    def __init__(self, img_size, channels, latent=10):
        super(GeneratorINETAN, self).__init__()

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
            nn.InstanceNorm2d(channels, affine=True), 
            #
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
        # img = (img + 1) * 127.5
        if not in_feature:
            return img
        else:
            return img, l_output 

class GeneratorBNETAN(nn.Module):
    def __init__(self, img_size, channels, latent=10):
        super(GeneratorBNETAN, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent, 128*self.init_size**2))
        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(channels) 
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
        