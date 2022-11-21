# 2019.11.25-Changed for testing pruned model 
#            Huawei Technologies Co., Ltd. <foss@huawei.com> 

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np

from models_prune import Generator_Prune
from datasets import ImageDataset
import datetime

#copy model files to test environment
import moxing as mox
mox.file.copy("s3://models/GA/horse2zebra/netG_A2B_prune_200.pth",'/cache/log/GA/horse2zebra/netG_A2B_prune_200.pth')
mox.file.copy("s3://models/GA/horse2zebra/netG_B2A_prune_200.pth",'/cache/log/GA/horse2zebra/netG_B2A_prune_200.pth')
mox.file.copy_parallel("s3://models/GA/txt","/cache/GA/txt/")


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/cache/data/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', type=bool, default=True , help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='/cache/log/output/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='/cache/log/output/netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)

#construct mask
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

mask_input_A2B=np.loadtxt("/cache/GA/txt/best_fitness_A2B.txt")
cfg_mask_A2B=compute_layer_mask(mask_input_A2B,mask_chns)

model_A2B = Generator_Prune(cfg_mask_A2B)
model_A2B.load_state_dict(torch.load('/cache/log/GA/horse2zebra/netG_A2B_prune_200.pth'))

mask_input_B2A=np.loadtxt("/cache/GA/txt/best_fitness_B2A.txt")
cfg_mask_B2A=compute_layer_mask(mask_input_B2A,mask_chns)

model_B2A = Generator_Prune(cfg_mask_B2A)
model_B2A.load_state_dict(torch.load('/cache/log/GA/horse2zebra/netG_B2A_prune_200.pth'))



if opt.cuda:
    model_A2B.cuda()
    model_B2A.cuda()
    

model_A2B.eval()
model_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######
log_dir='/cache/log/GA/horse2zebra/'

# Create output dirs if they don't exist
if not os.path.exists(log_dir+'A'):
    os.makedirs(log_dir+'A')
if not os.path.exists(log_dir+'B'):
    os.makedirs(log_dir+'B')

for i, batch in enumerate(dataloader):
    # Set model input
    
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    # Generate output
    fake_A = 0.5*(model_B2A(real_B).data + 1.0)
    fake_B = 0.5*(model_A2B(real_A).data + 1.0)


    time_.append((endtime - starttime).microseconds)
   
    # Save image files
    save_image(fake_B, log_dir+'B/%04d.png' % (i+1))
    save_image(fake_A, log_dir+'A/%04d.png' % (i+1))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

 
sys.stdout.write('\n')
###################################
