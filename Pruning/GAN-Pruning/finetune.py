# 2019.11.25-Changed for finetuning pruned model 
#            Huawei Technologies Co., Ltd. <foss@huawei.com> 

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn as nn
import numpy as np

from models import Generator
from models import Discriminator
from models_prune import Generator_Prune
from models_prune import compute_layer_mask
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal
from datasets import ImageDataset

import numpy as np

#copy data and pretrained model to finetuning environment 
import moxing as mox
mox.file.copy_parallel("s3://data/horse2zebra","/cache/data/horse2zebra")
mox.file.copy_parallel("s3://models/CycleGAN/horse2zebra/output","/cache/log/output")
mox.file.copy_parallel("s3://models/GA/txt","/cache/GA/txt")

parser = argparse.ArgumentParser()
parser.add_argument('--data_url', type=str, default='', help='root directory of the dataset')
parser.add_argument('--train_url', type=str, default='', help='root directory of the dataset')
parser.add_argument('--num_gpus', type=int, default=8, help='num_gpu')
parser.add_argument('--init_method', type=str, default='', help='init method')
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=201, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/cache/data/horse2zebra', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', type=bool, default =True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


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


def train_from_mask():
    
	#load best fitness binary masks 
    mask_input_A2B=np.loadtxt("/cache/GA/txt/best_fitness_A2B.txt")
    mask_input_B2A=np.loadtxt("/cache/GA/txt/best_fitness_B2A.txt")


    cfg_mask_A2B=compute_layer_mask(mask_input_A2B,mask_chns)
    cfg_mask_B2A=compute_layer_mask(mask_input_B2A,mask_chns)
    

    netG_B2A = Generator(opt.output_nc, opt.input_nc)
    netG_A2B = Generator(opt.output_nc, opt.input_nc)
    model_A2B = Generator_Prune(cfg_mask_A2B)
    model_B2A = Generator_Prune(cfg_mask_B2A)
    netD_A = Discriminator(opt.input_nc)
    netD_B = Discriminator(opt.output_nc)

    
 



    netG_A2B.load_state_dict(torch.load('/cache/log/output/netG_A2B.pth'))
    netG_B2A.load_state_dict(torch.load('/cache/log/output/netG_B2A.pth'))
    
    netD_A.load_state_dict(torch.load('/cache/log/output/netD_A.pth'))
    netD_B.load_state_dict(torch.load('/cache/log/output/netD_B.pth'))
     
      



    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    

    
    
    layer_id_in_cfg=0
    start_mask=torch.ones(3)
    end_mask=cfg_mask_A2B[layer_id_in_cfg]
    
    for [m0, m1] in zip(netG_A2B.modules(), model_A2B.modules()):
  
        if isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            
            m1.bias.data =m0.bias.data[idx1.tolist()].clone()
            
            layer_id_in_cfg += 1
            start_mask = end_mask
            if layer_id_in_cfg < len(cfg_mask_A2B):  # do not change in Final FC
                end_mask = cfg_mask_A2B[layer_id_in_cfg]
                print(layer_id_in_cfg)
        elif isinstance(m0, nn.ConvTranspose2d):
            print('Into ConvTranspose...')
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        

            w1 = m0.weight.data[idx0.tolist(),:, :, :].clone()
            w1 = w1[:,idx1.tolist(), :, :].clone()
            m1.weight.data = w1.clone()
            m1.bias.data =m0.bias.data[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask
            if layer_id_in_cfg < len(cfg_mask_A2B):  
                end_mask = cfg_mask_A2B[layer_id_in_cfg] 

    layer_id_in_cfg=0
    start_mask=torch.ones(3)
    end_mask=cfg_mask_B2A[layer_id_in_cfg]
    
    for [m0, m1] in zip(netG_B2A.modules(), model_B2A.modules()):
  
        if isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            
            m1.bias.data =m0.bias.data[idx1.tolist()].clone()
            
            layer_id_in_cfg += 1
            start_mask = end_mask
            if layer_id_in_cfg < len(cfg_mask_B2A):  
                end_mask = cfg_mask_B2A[layer_id_in_cfg]
                print(layer_id_in_cfg)
        elif isinstance(m0, nn.ConvTranspose2d):
            print('Into ConvTranspose...')
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        
            w1 = m0.weight.data[idx0.tolist(),:, :, :].clone()
            w1 = w1[:,idx1.tolist(), :, :].clone()
            m1.weight.data = w1.clone()
            m1.bias.data =m0.bias.data[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask
            if layer_id_in_cfg < len(cfg_mask_B2A):  
                end_mask = cfg_mask_B2A[layer_id_in_cfg] 

    
    
         
     # Dataset loader
    
    netD_A=torch.nn.DataParallel(netD_A).cuda()
    netD_B=torch.nn.DataParallel(netD_B).cuda()
    model_A2B=torch.nn.DataParallel(model_A2B).cuda()
    model_B2A=torch.nn.DataParallel(model_B2A).cuda()   

    
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
    optimizer_G = torch.optim.Adam(itertools.chain(model_A2B.parameters(), model_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    
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
            
        
        print("epoch:%d  Loss G:%4f  LossID_A:%4f LossID_B:%4f  Loss_G_A2B:%4f  Loss_G_B2A:%4f  Loss_Cycle_ABA:%4f  Loss_Cycle_BAB:%4f "%(epoch,loss_G,loss_identity_A, loss_identity_B, loss_GAN_A2B, loss_GAN_B2A, loss_cycle_ABA, loss_cycle_BAB))

         # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        
        if epoch%20==0:

        # Save models checkpoints
            torch.save(model_A2B.module.state_dict(), '/cache/log/output/A2B_%d.pth'%(epoch))
            torch.save(model_B2A.module.state_dict(), '/cache/log/output/B2A_%d.pth'%(epoch))


###################################

if __name__ == "__main__":
    train_from_mask()

            
    
