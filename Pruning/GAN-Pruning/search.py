# 2019.11.25-Changed for searching pruned model 
#            Huawei Technologies Co., Ltd. <foss@huawei.com> 

import torch

import torch.multiprocessing as mp

import os
import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler


default_collate_func = dataloader.default_collate


def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]

import argparse
import itertools
import datetime
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn as nn
import numpy as np

from models import Generator
from models import Discriminator
from models_prune import compute_layer_mask
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal
from datasets import ImageDataset

from GA import *

#copy dataset and pretrained model to training environment
import moxing as mox
mox.file.copy_parallel("s3://data/horse2zebra_train_val_test","/cache/data/horse2zebra")
mox.file.copy("s3://models/CycleGAN/horse2zebra/netG_A2B_200.pth","/cache/models/netG_A2B.pth")
mox.file.copy("s3://models/CycleGAN/horse2zebra/netG_B2A_200.pth","/cache/models/netG_B2A.pth")
mox.file.copy("s3://models/CycleGAN/horse2zebra/netD_A_200.pth","/cache/models/output/netD_A.pth")
mox.file.copy("s3://models/CycleGAN/horse2zebra/netD_B_200.pth","/cache/models/output/netD_B.pth")

parser = argparse.ArgumentParser()
parser.add_argument('--data_url', type=str, default='', help='root directory of the dataset')
parser.add_argument('--train_url', type=str, default='', help='root directory of the dataset')
parser.add_argument('--num_gpu', type=int, default=8, help='num_gpu')

parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/cache/data/horse2zebra', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda',type=bool, default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()

population=32
current_fitness_base_A2B=mp.Array('f',range(population))
current_fitness_A2B=np.asarray(current_fitness_base_A2B.get_obj(),dtype=np.float32)

current_fitness_base_B2A=mp.Array('f',range(population))
current_fitness_B2A=np.asarray(current_fitness_base_B2A.get_obj(),dtype=np.float32)


# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()


def caculate_fitness_for_first_time(mask_input,gpu_id,fitness_id,A2B_or_B2A):
    
    
        ###### Definition of variables ######
    torch.cuda.set_device(gpu_id)
    #print("GPU_ID is%d\n"%(gpu_id))
    if A2B_or_B2A=='A2B':
        netG_A2B = Generator(opt.input_nc, opt.output_nc)
        netD_B = Discriminator(opt.output_nc)
        netG_A2B.cuda(gpu_id)
        netD_B.cuda(gpu_id)
        model = Generator(opt.input_nc, opt.output_nc)
        model.cuda(gpu_id)
        netG_A2B.load_state_dict(torch.load('/cache/models/netG_A2B.pth'))
        netD_B.load_state_dict(torch.load('/cache/models/netD_B.pth'))
        model.load_state_dict(torch.load('/cache/models/netG_A2B.pth'))
        model.eval()
        netD_B.eval()
        netG_A2B.eval()
        
    elif A2B_or_B2A=='B2A':
        netG_B2A = Generator(opt.output_nc, opt.input_nc)
        netD_A = Discriminator(opt.input_nc)
        netG_B2A.cuda(gpu_id)
        netD_A.cuda(gpu_id)
        model = Generator(opt.input_nc, opt.output_nc)
        model.cuda(gpu_id)
        netG_B2A.load_state_dict(torch.load('/cache/models/netG_B2A.pth'))
        netD_A.load_state_dict(torch.load('/cache/models/netD_A.pth'))
        model.load_state_dict(torch.load('/cache/models/netG_B2A.pth'))
        model.eval()
        netD_A.eval()
        netG_B2A.eval()
        
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    fitness=0   
    cfg_mask=compute_layer_mask(mask_input,mask_chns)
    cfg_full_mask=[y for x in cfg_mask for y in x]
    cfg_full_mask=np.array(cfg_full_mask)
    cfg_id=0
    start_mask=np.ones(3)
    end_mask=cfg_mask[cfg_id]
  
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
 
            mask=np.ones(m.weight.data.shape)
            
            mask_bias=np.ones(m.bias.data.shape)
            
            cfg_mask_start=np.ones(start_mask.shape)-start_mask
            cfg_mask_end=np.ones(end_mask.shape)-end_mask
            idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask_start)))
            idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask_end)))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            
            mask[:,idx0.tolist(),:,:]=0
            mask[idx1.tolist(),:,:,:]=0
            mask_bias[idx1.tolist()]=0
            
            
            m.weight.data=m.weight.data*torch.FloatTensor(mask).cuda(gpu_id)
            
         
            
            m.bias.data= m.bias.data*torch.FloatTensor(mask_bias).cuda(gpu_id)
            
            idx_mask=np.argwhere(np.asarray(np.ones(mask.shape)-mask))
            
            m.weight.data[:,idx0.tolist(),:,:].requires_grad= False
            m.weight.data[idx1.tolist(),:,:,:].requires_grad= False
            m.bias.data[idx1.tolist()].requires_grad=False
               
            cfg_id += 1
            start_mask=end_mask
            if cfg_id<len(cfg_mask):
                end_mask=cfg_mask[cfg_id]
            continue
        elif isinstance(m,nn.ConvTranspose2d):
       
            mask=np.ones(m.weight.data.shape)
            mask_bias=np.ones(m.bias.data.shape)
            
            cfg_mask_start=np.ones(start_mask.shape)-start_mask
            cfg_mask_end=np.ones(end_mask.shape)-end_mask
            
            idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask_start)))
            idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask_end)))

            mask[idx0.tolist(),:,:,:]=0
        
            mask[:,idx1.tolist(),:,:]=0
      
            mask_bias[idx1.tolist()]=0
        
            m.weight.data=m.weight.data*torch.FloatTensor(mask).cuda(gpu_id)
            m.bias.data= m.bias.data*torch.FloatTensor(mask_bias).cuda(gpu_id)
            
            m.weight.data[idx0.tolist(),:,:,:].requires_grad= False
            m.weight.data[:,idx1.tolist(),:,:].requires_grad= False
            m.bias.data[idx1.tolist()].requires_grad=False
                     
            cfg_id += 1
            start_mask=end_mask
            end_mask=cfg_mask[cfg_id]
            continue
         
     # Dataset loader
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()        
        
    lamda_loss_ID=5.0
    lamda_loss_G=1.0
    lamda_loss_cycle=10.0
    
    with torch.no_grad():
    
        transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    
        
        dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='val'), 
                        batch_size=opt.batchSize, shuffle=False, drop_last=True)
    
                             
    
    
        Loss_resemble_G=0
        if A2B_or_B2A=='A2B':
            for i, batch in enumerate(dataloader):
                # Set model input
                real_A = Variable(input_A.copy_(batch['A']))



                # GAN loss
                fake_B = model(real_A)
                fake_B_full_model=netG_A2B(real_A)
        
                # Fake loss
                pred_fake = netD_B(fake_B.detach())
    
      
                pred_fake_full=netD_B(fake_B_full_model.detach())
    
                loss_D_fake = criterion_GAN(pred_fake.detach(),pred_fake_full.detach())
                Loss_resemble_G=Loss_resemble_G+loss_D_fake 
                
                lambda_prune=0.001
       
            fitness = 500/Loss_resemble_G.detach() + sum(np.ones(cfg_full_mask.shape)-cfg_full_mask)*lambda_prune
            print("A2B first generation")
            print("GPU_ID is %d"%(gpu_id))
            print("channel num is: %d"%(sum(cfg_full_mask)))    
            print("Loss_resemble_G is %f prune_loss is %f "%(500/Loss_resemble_G,sum(np.ones(cfg_full_mask.shape)-cfg_full_mask)))
            print("fitness is %f \n"%(fitness))
    
   
            current_fitness_A2B[fitness_id]= fitness.item()
                
        elif A2B_or_B2A=='B2A':
            for i, batch in enumerate(dataloader):
  
                real_B = Variable(input_B.copy_(batch['B']))
 


       
                fake_A = model(real_B)
                fake_A_full_model=netG_B2A(real_B)

        
  
                pred_fake = netD_A(fake_A.detach())
  
      
                pred_fake_full=netD_A(fake_A_full_model.detach())

        
                loss_D_fake = criterion_GAN(pred_fake.detach(),pred_fake_full.detach())
                Loss_resemble_G=Loss_resemble_G+loss_D_fake        
        
                lambda_prune=0.001
       
            fitness = 500/Loss_resemble_G.detach() + sum(np.ones(cfg_full_mask.shape)-cfg_full_mask)*lambda_prune
            print("B2A first generation")
            print("GPU_ID is %d"%(gpu_id))
            print("channel num is: %d"%(sum(cfg_full_mask)))    
            print("Loss_resemble_G is %f prune_loss is %f "%(500/Loss_resemble_G,sum(np.ones(cfg_full_mask.shape)-cfg_full_mask)))
            print("fitness is %f \n"%(fitness))
    
   
            current_fitness_B2A[fitness_id]= fitness.item()
    

def caculate_fitness(mask_input_A2B,mask_input_B2A,gpu_id,fitness_id,A2B_or_B2A):

    torch.cuda.set_device(gpu_id)
    #print("GPU_ID is%d\n"%(gpu_id))

    model_A2B = Generator(opt.input_nc, opt.output_nc)
    model_B2A = Generator(opt.input_nc, opt.output_nc)
    
    netD_A = Discriminator(opt.input_nc)
    netD_B = Discriminator(opt.output_nc)

    
    netD_A.cuda(gpu_id)
    netD_B.cuda(gpu_id)
    model_A2B.cuda(gpu_id)
    model_B2A.cuda(gpu_id)

    model_A2B.load_state_dict(torch.load('/cache/models/netG_A2B.pth'))
    model_B2A.load_state_dict(torch.load('/cache/models/netG_B2A.pth'))                         
    netD_A.load_state_dict(torch.load('/cache/models/netD_A.pth'))
    netD_B.load_state_dict(torch.load('/cache/models/netD_B.pth'))




    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    
    fitness=0   
    cfg_mask_A2B=compute_layer_mask(mask_input_A2B,mask_chns)
    cfg_mask_B2A=compute_layer_mask(mask_input_B2A,mask_chns)
    cfg_full_mask_A2B=[y for x in cfg_mask_A2B for y in x]
    cfg_full_mask_A2B=np.array(cfg_full_mask_A2B)
    cfg_full_mask_B2A=[y for x in cfg_mask_B2A for y in x]
    cfg_full_mask_B2A=np.array(cfg_full_mask_B2A)
    cfg_id=0
    start_mask=np.ones(3)
    end_mask=cfg_mask_A2B[cfg_id]
 
    for m in model_A2B.modules():
        if isinstance(m, nn.Conv2d):

            #print("conv2d")
            #print(m.weight.data.shape)
            #out_channels = m.weight.data.shape[0]    
            mask=np.ones(m.weight.data.shape)
            
            mask_bias=np.ones(m.bias.data.shape)
            
            cfg_mask_start=np.ones(start_mask.shape)-start_mask
            cfg_mask_end=np.ones(end_mask.shape)-end_mask
            idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask_start)))
            idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask_end)))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            
            mask[:,idx0.tolist(),:,:]=0
            mask[idx1.tolist(),:,:,:]=0
            mask_bias[idx1.tolist()]=0
            
            
            m.weight.data=m.weight.data*torch.FloatTensor(mask).cuda(gpu_id)
            
         
            
            m.bias.data= m.bias.data*torch.FloatTensor(mask_bias).cuda(gpu_id)
            
            idx_mask=np.argwhere(np.asarray(np.ones(mask.shape)-mask))
            
            m.weight.data[:,idx0.tolist(),:,:].requires_grad= False
            m.weight.data[idx1.tolist(),:,:,:].requires_grad= False
            m.bias.data[idx1.tolist()].requires_grad=False
               
            cfg_id += 1
            start_mask=end_mask
            if cfg_id<len(cfg_mask):
                end_mask=cfg_mask_A2B[cfg_id]
            continue
        elif isinstance(m,nn.ConvTranspose2d):
       
            mask=np.ones(m.weight.data.shape)
            mask_bias=np.ones(m.bias.data.shape)
            
            cfg_mask_start=np.ones(start_mask.shape)-start_mask
            cfg_mask_end=np.ones(end_mask.shape)-end_mask
            
            idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask_start)))
            idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask_end)))

            mask[idx0.tolist(),:,:,:]=0
        
            mask[:,idx1.tolist(),:,:]=0
      
            mask_bias[idx1.tolist()]=0
        
            m.weight.data=m.weight.data*torch.FloatTensor(mask).cuda(gpu_id)
            m.bias.data= m.bias.data*torch.FloatTensor(mask_bias).cuda(gpu_id)
            
            m.weight.data[idx0.tolist(),:,:,:].requires_grad= False
            m.weight.data[:,idx1.tolist(),:,:].requires_grad= False
            m.bias.data[idx1.tolist()].requires_grad=False
                     
            cfg_id += 1
            start_mask=end_mask
            end_mask=cfg_mask_A2B[cfg_id]
            continue
        
        
    cfg_id=0
    start_mask=np.ones(3)
    end_mask=cfg_mask_B2A[cfg_id]
 
    for m in model_B2A.modules():
        if isinstance(m, nn.Conv2d):

            #print("conv2d")
            #print(m.weight.data.shape)
            #out_channels = m.weight.data.shape[0]    
            mask=np.ones(m.weight.data.shape)
            
            mask_bias=np.ones(m.bias.data.shape)
            
            cfg_mask_start=np.ones(start_mask.shape)-start_mask
            cfg_mask_end=np.ones(end_mask.shape)-end_mask
            idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask_start)))
            idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask_end)))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            
            mask[:,idx0.tolist(),:,:]=0
            mask[idx1.tolist(),:,:,:]=0
            mask_bias[idx1.tolist()]=0
            
            
            m.weight.data=m.weight.data*torch.FloatTensor(mask).cuda(gpu_id)
            
         
            
            m.bias.data= m.bias.data*torch.FloatTensor(mask_bias).cuda(gpu_id)
            
            idx_mask=np.argwhere(np.asarray(np.ones(mask.shape)-mask))
            
            m.weight.data[:,idx0.tolist(),:,:].requires_grad= False
            m.weight.data[idx1.tolist(),:,:,:].requires_grad= False
            m.bias.data[idx1.tolist()].requires_grad=False
               
            cfg_id += 1
            start_mask=end_mask
            if cfg_id<len(cfg_mask):
                end_mask=cfg_mask_B2A[cfg_id]
            continue
        elif isinstance(m,nn.ConvTranspose2d):
       
            mask=np.ones(m.weight.data.shape)
            mask_bias=np.ones(m.bias.data.shape)
            
            cfg_mask_start=np.ones(start_mask.shape)-start_mask
            cfg_mask_end=np.ones(end_mask.shape)-end_mask
            
            idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask_start)))
            idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask_end)))

            mask[idx0.tolist(),:,:,:]=0
        
            mask[:,idx1.tolist(),:,:]=0
      
            mask_bias[idx1.tolist()]=0
        
            m.weight.data=m.weight.data*torch.FloatTensor(mask).cuda(gpu_id)
            m.bias.data= m.bias.data*torch.FloatTensor(mask_bias).cuda(gpu_id)
            
            m.weight.data[idx0.tolist(),:,:,:].requires_grad= False
            m.weight.data[:,idx1.tolist(),:,:].requires_grad= False
            m.bias.data[idx1.tolist()].requires_grad=False
                     
            cfg_id += 1
            start_mask=end_mask
            end_mask=cfg_mask_B2A[cfg_id]
            continue        
         
     # Dataset loader
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()        
        
    lamda_loss_ID=5.0
    lamda_loss_G=1.0
    lamda_loss_cycle=10.0
    optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad,model_A2B.parameters()), filter(lambda p: p.requires_grad,model_B2A.parameters())),lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    transforms_ = [ 
           transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(opt.size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    
    
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True,mode='train'), batch_size=opt.batchSize, shuffle=True,drop_last=True)
    

    
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

        # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = model_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*lamda_loss_ID #initial 5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = model_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*lamda_loss_ID #initial 5.0

            # GAN loss
            fake_B = model_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)*lamda_loss_G  #initial 1.0

            fake_A = model_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)*lamda_loss_G #initial 1.0

            # Cycle loss
            recovered_A = model_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*lamda_loss_cycle  #initial 10.0

            recovered_B = model_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*lamda_loss_cycle  #initial 10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            
            optimizer_G.step()
            
            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
        
            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()
    
    with torch.no_grad():
    
        transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    
 
        
        dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='val'), 
                        batch_size=opt.batchSize, shuffle=False, drop_last=True)
    

    

        Loss_resemble_G=0
        if A2B_or_B2A=='A2B':
            netG_A2B = Generator(opt.output_nc, opt.input_nc)
            netD_B = Discriminator(opt.output_nc)
    

            netG_A2B.cuda(gpu_id)
            netD_B.cuda(gpu_id)
    
            model_A2B.eval()
            netD_B.eval()
            netG_A2B.eval()
    
            netD_B.load_state_dict(torch.load('/cache/models/netD_B.pth'))        
            netG_A2B.load_state_dict(torch.load('/cache/models/netG_A2B.pth'))

    
            

            for i, batch in enumerate(dataloader):

                real_A = Variable(input_A.copy_(batch['A']))


                fake_B = model_A2B(real_A)
                fake_B_full_model=netG_A2B(real_A)
                recovered_A=model_B2A(fake_B)
            
        
                pred_fake = netD_B(fake_B.detach())

      
                pred_fake_full=netD_B(fake_B_full_model.detach())
 
        
                loss_D_fake = criterion_GAN(pred_fake.detach(),pred_fake_full.detach())
                cycle_loss = criterion_cycle(recovered_A,real_A)*lamda_loss_cycle
                Loss_resemble_G=Loss_resemble_G+loss_D_fake+ cycle_loss
                
                lambda_prune=0.001
       
            fitness = 500/Loss_resemble_G.detach() + sum(np.ones(cfg_full_mask_A2B.shape)-cfg_full_mask_A2B)*lambda_prune
        
            print('A2B')
            print("GPU_ID is %d"%(gpu_id))
            print("channel num is: %d"%(sum(cfg_full_mask_A2B)))    
            print("Loss_resemble_G is %f prune_loss is %f "%(500/Loss_resemble_G,sum(np.ones(cfg_full_mask_A2B.shape)-cfg_full_mask_A2B)))
            print("fitness is %f \n"%(fitness))
    
   
            current_fitness_A2B[fitness_id]= fitness.item()
                
                
        if A2B_or_B2A=='B2A':
            netG_B2A = Generator(opt.output_nc, opt.input_nc)
            netD_A = Discriminator(opt.output_nc)
    

            netG_B2A.cuda(gpu_id)
            netD_A.cuda(gpu_id)
    
            model_B2A.eval()
            netD_A.eval()
            netG_B2A.eval()
    
            netD_A.load_state_dict(torch.load('/cache/models/netD_A.pth'))        
            netG_B2A.load_state_dict(torch.load('/cache/models/netG_B2A.pth'))

    
            

            for i, batch in enumerate(dataloader):

                real_B = Variable(input_B.copy_(batch['B']))


                fake_A = model_B2A(real_B)
                fake_A_full_model=netG_B2A(real_B)
                recovered_B=model_A2B(fake_A)
            
        
                pred_fake = netD_A(fake_A.detach())

      
                pred_fake_full=netD_A(fake_A_full_model.detach())

        
                loss_D_fake = criterion_GAN(pred_fake.detach(),pred_fake_full.detach())
                cycle_loss = criterion_cycle(recovered_B,real_B)*lamda_loss_cycle
                Loss_resemble_G=Loss_resemble_G+loss_D_fake+ cycle_loss
    
    
                lambda_prune=0.001
       
            fitness = 500/Loss_resemble_G.detach() + sum(np.ones(cfg_full_mask_B2A.shape)-cfg_full_mask_B2A)*lambda_prune
                 
            print('B2A')
            print("GPU_ID is %d"%(gpu_id))
            print("channel num is: %d"%(sum(cfg_full_mask_B2A)))    
            print("Loss_resemble_G is %f prune_loss is %f "%(500/Loss_resemble_G,sum(np.ones(cfg_full_mask_B2A.shape)-cfg_full_mask_B2A)))
            print("fitness is %f \n"%(fitness))
    
   
            current_fitness_B2A[fitness_id]= fitness.item()
     

    

max_generation=50
layer_id = 0
cfg = []
cfg_mask = []
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
#print(bit_len)    

s1=0.2 #prob for selection
s2=0.7 #prob for crossover
s3=0.1 #prob for mutation
print("A new start training")

if os.path.exists('/cache/log/GA')==False:
    os.makedirs('/cache/log/GA')

mask_all_A2B=[]
for i in range(population):
    mask_all_A2B.append(np.random.randint(2,size=bit_len))
mask_all_B2A=[]
for i in range(population):
    mask_all_B2A.append(np.random.randint(2,size=bit_len))    




starttime = datetime.datetime.now()

for i in range(int(population/8)):

    process1=mp.Process(target=caculate_fitness_for_first_time,args=(mask_all_A2B[i*8],0,i*8,'A2B'))
    process2=mp.Process(target=caculate_fitness_for_first_time,args=(mask_all_A2B[i*8+1],1,i*8+1,'A2B'))
    process3=mp.Process(target=caculate_fitness_for_first_time,args=(mask_all_A2B[i*8+2],2,i*8+2,'A2B'))
    process4=mp.Process(target=caculate_fitness_for_first_time,args=(mask_all_A2B[i*8+3],3,i*8+3,'A2B'))
    process5=mp.Process(target=caculate_fitness_for_first_time,args=(mask_all_A2B[i*8+4],4,i*8+4,'A2B'))
    process6=mp.Process(target=caculate_fitness_for_first_time,args=(mask_all_A2B[i*8+5],5,i*8+5,'A2B'))
    process7=mp.Process(target=caculate_fitness_for_first_time,args=(mask_all_A2B[i*8+6],6,i*8+6,'A2B'))
    process8=mp.Process(target=caculate_fitness_for_first_time,args=(mask_all_A2B[i*8+7],7,i*8+7,'A2B'))
    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process5.start()
    process6.start()
    process7.start()
    process8.start()
    process1.join()
    process2.join()
    process3.join()
    process4.join()
    process5.join()
    process6.join()
    process7.join()
    process8.join()


    
endtime = datetime.datetime.now()
print("The time is:")
print((endtime - starttime).seconds)
mask_best_A2B=mask_all_A2B[np.argmax(current_fitness_A2B)]
best_fitness_A2B=max(current_fitness_A2B)
ave_fitness_A2B=np.mean(current_fitness_A2B)
print('The best fitness is: %4f'%(best_fitness_A2B))
mask_best_A2B_full=compute_layer_mask(mask_best_A2B,mask_chns)
mask_best_A2B_full=[y for x in mask_best_A2B_full for y in x]
mask_best_A2B_full=np.array(mask_best_A2B_full)
print('The best model channel num is:%d'%(sum(mask_best_A2B_full)))
print('The ave fitness is: %4f'%(ave_fitness_A2B))
np.savetxt('/cache/log/GA/A2B_%d_th.txt'%(0),mask_best_A2B)

for i in range(int(population/8)):

    process1=mp.Process(target=caculate_fitness_for_first_time,args=(mask_all_B2A[i*8],0,i*8,'B2A'))
    process2=mp.Process(target=caculate_fitness_for_first_time,args=(mask_all_B2A[i*8+1],1,i*8+1,'B2A'))
    process3=mp.Process(target=caculate_fitness_for_first_time,args=(mask_all_B2A[i*8+2],2,i*8+2,'B2A'))
    process4=mp.Process(target=caculate_fitness_for_first_time,args=(mask_all_B2A[i*8+3],3,i*8+3,'B2A'))
    process5=mp.Process(target=caculate_fitness_for_first_time,args=(mask_all_B2A[i*8+4],4,i*8+4,'B2A'))
    process6=mp.Process(target=caculate_fitness_for_first_time,args=(mask_all_B2A[i*8+5],5,i*8+5,'B2A'))
    process7=mp.Process(target=caculate_fitness_for_first_time,args=(mask_all_B2A[i*8+6],6,i*8+6,'B2A'))
    process8=mp.Process(target=caculate_fitness_for_first_time,args=(mask_all_B2A[i*8+7],7,i*8+7,'B2A'))
    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process5.start()
    process6.start()
    process7.start()
    process8.start()
    process1.join()
    process2.join()
    process3.join()
    process4.join()
    process5.join()
    process6.join()
    process7.join()
    process8.join()


#current_prob=calculate_prob(current_fitness)
mask_best_B2A=mask_all_B2A[np.argmax(current_fitness_B2A)]
best_fitness_B2A=max(current_fitness_B2A)
ave_fitness_B2A=np.mean(current_fitness_B2A)
print('The best fitness is: %4f'%(best_fitness_B2A))
mask_best_B2A_full=compute_layer_mask(mask_best_B2A,mask_chns)
mask_best_B2A_full=[y for x in mask_best_B2A_full for y in x]
mask_best_B2A_full=np.array(mask_best_B2A_full)
print('The best model channel num is:%d'%(sum(mask_best_B2A_full)))
print('The ave fitness is: %4f'%(ave_fitness_B2A))
np.savetxt('/cache/log/GA/B2A_%d_th.txt'%(0),mask_best_B2A)





for j in range(max_generation):
    mask_all_current_A2B=[]
    
    rest_population=population
    mask_all_current_A2B.append(mask_best_A2B)
    #fitness.append(best_fitness)
    rest_population-=1

    

    while(rest_population>0):

        s=np.random.uniform(0,1)
        #selection
        #print(s)
        if s<s1:
            #print("into selection")
            mask_,_=roulette(mask_all_A2B,population,current_fitness_A2B)
            mask_all_current_A2B.append(mask_)
            #fitness.append(fitness_)
            rest_population-=1

        #cross over   
        elif (s>s1)&(s<=s1+s2):
            #print("into cross over")
            mask1,mask2 = crossover(mask_all_A2B,population,current_fitness_A2B,bit_len)
            
            if rest_population<=1:
                mask_all_current_A2B.append(mask1)
                rest_population-=1
                
            else:
                mask_all_current_A2B.append(mask1)
                mask_all_current_A2B.append(mask2)
                rest_population-=2
        
        #mutation
        else :

            mask_=mutation(mask_all_A2B,population,current_fitness_A2B,bit_len)
            mask_all_current_A2B.append(mask_)
            rest_population-=1
            
  
    mask_all_A2B= mask_all_current_A2B
    
        
    for i in range(int(population/8)):

        process1=mp.Process(target=caculate_fitness,args=(mask_all_A2B[i*8],mask_best_B2A,0,i*8,'A2B'))
        process2=mp.Process(target=caculate_fitness,args=(mask_all_A2B[i*8+1],mask_best_B2A,1,i*8+1,'A2B'))
        process3=mp.Process(target=caculate_fitness,args=(mask_all_A2B[i*8+2],mask_best_B2A,2,i*8+2,'A2B'))
        process4=mp.Process(target=caculate_fitness,args=(mask_all_A2B[i*8+3],mask_best_B2A,3,i*8+3,'A2B'))
        process5=mp.Process(target=caculate_fitness,args=(mask_all_A2B[i*8+4],mask_best_B2A,4,i*8+4,'A2B'))
        process6=mp.Process(target=caculate_fitness,args=(mask_all_A2B[i*8+5],mask_best_B2A,5,i*8+5,'A2B'))
        process7=mp.Process(target=caculate_fitness,args=(mask_all_A2B[i*8+6],mask_best_B2A,6,i*8+6,'A2B'))
        process8=mp.Process(target=caculate_fitness,args=(mask_all_A2B[i*8+7],mask_best_B2A,7,i*8+7,'A2B'))
        process1.start()
        process2.start()
        process3.start()
        process4.start()
        process5.start()
        process6.start()
        process7.start()
        process8.start()
        process1.join()
        process2.join()
        process3.join()
        process4.join()
        process5.join()
        process6.join()
        process7.join()
        process8.join()
    
    
    print('A2B')
    mask_best_A2B=mask_all_A2B[np.argmax(current_fitness_A2B)]
    mask_best_A2B_full=compute_layer_mask(mask_best_A2B,mask_chns)
    mask_best_A2B_full=[y for x in mask_best_A2B_full for y in x]
    mask_best_A2B_full=np.array(mask_best_A2B_full)
    print('The %d th best model channel num is: %d'%(j,sum(mask_best_A2B_full)))
    best_fitness_A2B=max(current_fitness_A2B)
    print('The %d th best fitness is: %4f'%(j,best_fitness_A2B))
    ave_fitness_A2B=np.mean(current_fitness_A2B)
    print('The %d th ave fitness is: %4f'%(j,ave_fitness_A2B))
        
    
       
    np.savetxt('/cache/log/GA/A2B_%d_th.txt'%(j),mask_best_A2B)
    
    
    
    mask_all_current_B2A=[]
    
    rest_population=population
    mask_all_current_B2A.append(mask_best_B2A)
    #fitness.append(best_fitness)
    rest_population-=1

    

    while(rest_population>0):
        s=np.random.uniform(0,1)
        #selection
        #print(s)
        if s<s1:
            mask_,_=roulette(mask_all_B2A,population,current_fitness_B2A)
            mask_all_current_B2A.append(mask_)
            rest_population-=1

        #cross over   
        elif (s>s1)&(s<=s1+s2):
            #print("into cross over")
            mask1,mask2 = crossover(mask_all_B2A,population,current_fitness_B2A,bit_len)
            
            if rest_population<=1:
                mask_all_current_B2A.append(mask1)
                rest_population-=1
                
            else:
                mask_all_current_B2A.append(mask1)
                mask_all_current_B2A.append(mask2)
                rest_population-=2
        
        #mutation
        else :
            #print("into mutation")
            mask_=mutation(mask_all_B2A,population,current_fitness_B2A,bit_len)
            mask_all_current_B2A.append(mask_)
            rest_population-=1
            
    
    mask_all_B2A= mask_all_current_B2A

    
        
    for i in range(int(population/8)):

        process1=mp.Process(target=caculate_fitness,args=(mask_best_A2B,mask_all_B2A[i*8],0,i*8,'B2A'))
        process2=mp.Process(target=caculate_fitness,args=(mask_best_A2B,mask_all_B2A[i*8+1],1,i*8+1,'B2A'))
        process3=mp.Process(target=caculate_fitness,args=(mask_best_A2B,mask_all_B2A[i*8+2],2,i*8+2,'B2A'))
        process4=mp.Process(target=caculate_fitness,args=(mask_best_A2B,mask_all_B2A[i*8+3],3,i*8+3,'B2A'))
        process5=mp.Process(target=caculate_fitness,args=(mask_best_A2B,mask_all_B2A[i*8+4],4,i*8+4,'B2A'))
        process6=mp.Process(target=caculate_fitness,args=(mask_best_A2B,mask_all_B2A[i*8+5],5,i*8+5,'B2A'))
        process7=mp.Process(target=caculate_fitness,args=(mask_best_A2B,mask_all_B2A[i*8+6],6,i*8+6,'B2A'))
        process8=mp.Process(target=caculate_fitness,args=(mask_best_A2B,mask_all_B2A[i*8+7],7,i*8+7,'B2A'))
        process1.start()
        process2.start()
        process3.start()
        process4.start()
        process5.start()
        process6.start()
        process7.start()
        process8.start()
        process1.join()
        process2.join()
        process3.join()
        process4.join()
        process5.join()
        process6.join()
        process7.join()
        process8.join()
    
    

    mask_best_B2A=mask_all_B2A[np.argmax(current_fitness_B2A)]
    print('B2A')
    mask_best_B2A_full=compute_layer_mask(mask_best_B2A,mask_chns)
    mask_best_B2A_full=[y for x in mask_best_B2A_full for y in x]
    mask_best_B2A_full=np.array(mask_best_B2A_full)
    print('The %d th best model channel num is: %d'%(j,sum(mask_best_B2A_full)))
    best_fitness_B2A=max(current_fitness_B2A)
    print('The %d th best fitness is: %4f'%(j,best_fitness_B2A))
    ave_fitness_B2A=np.mean(current_fitness_B2A)
    print('The %d th ave fitness is: %4f'%(j,ave_fitness_B2A))
   
        
    
       
    np.savetxt('/cache/log/GA/B2A_%d_th.txt'%(j),mask_best_B2A)
    


 




    
        
        
        
     

    
    


