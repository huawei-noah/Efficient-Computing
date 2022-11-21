# 2019.11.25-Changed for constructing pruned model 
#            Huawei Technologies Co., Ltd. <foss@huawei.com> 

import torch.nn as nn
import numpy as np

from models import Generator
from models import ResidualBlock


first_conv_out=64
mask_chns=[]
mask_chns.append(first_conv_out) #1st conv
mask_chns.append(first_conv_out*2) #2nd conv
mask_chns.append(first_conv_out*4) #3rd conv 1~9 res_block
mask_chns.append(first_conv_out*2) #1st trans_conv
mask_chns.append(first_conv_out) #2nd trans_conv
bit_len=0
for mask_chn in mask_chns:
    bit_len+= mask_chn


def compute_layer_mask(mask,mask_chns):
    cfg_mask=[]
    start_id=0
    end_id=start_id+mask_chns[0]
    
    cfg_mask.append(mask[:end_id])
    start_id=end_id
    end_id=start_id+mask_chns[1]
    cfg_mask.append(mask[start_id:end_id])
    start_id=end_id
    end_id=start_id+mask_chns[2]
    for i in range(19):
        cfg_mask.append(mask[start_id:end_id])
    start_id=end_id
    end_id=start_id+mask_chns[3]
    cfg_mask.append(mask[start_id:end_id])
    start_id=end_id
    end_id=start_id+mask_chns[4]
    cfg_mask.append(mask[start_id:end_id])
    cfg_mask.append(np.ones(3))
    
    return cfg_mask

class Generator_Prune(nn.Module):
    def __init__(self, cfg_mask, n_residual_blocks=9):
        super(Generator_Prune, self).__init__()

        first_conv_out= int(sum(cfg_mask[0]))
        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(3, first_conv_out, 7),
                    nn.InstanceNorm2d(first_conv_out),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = int(sum(cfg_mask[0]))
        out_features =int(sum(cfg_mask[1]))
        
        model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
        in_features = int(sum(cfg_mask[1]))
        out_features =int(sum(cfg_mask[2]))
        model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
        
        in_features= int(sum(cfg_mask[2]))
     

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = int(sum(cfg_mask[21]))
        
        
        model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
        in_features= out_features
        out_features= int(sum(cfg_mask[22]))
        
        model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]

        

        # Output layer
        in_features= out_features
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(in_features, 3, 7),  #nn.Conv2d(64, output_nc, 7)
                    nn.Tanh() ]
        

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

