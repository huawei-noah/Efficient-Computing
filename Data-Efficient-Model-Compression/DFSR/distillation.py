#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os, sys
os.system("pip install torch==1.0.0")
os.system("pip install torchtext==0.3.1")
os.system("pip install torchvision==0.2.1")
os.system("pip install numpy==1.14.5")
os.system("pip install scipy==1.5.0")
os.system("pip install mpmath==1.0.0")
os.system("pip install glob2==0.6")
os.system("pip install matplotlib==1.5.3")

import argparse, os,logging,sys
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vdsr_block import Net
from vdsr_half_block import NetHalf

from dataset import DatasetFromHdf5
import numpy as np
import math,glob
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from zsnet import TVLoss, GeneratorINE1
from bicubic import * # bicubic package from https://github.com/tonyzzzt/bicubic-interpolation-pytorch-version-the-same-results-with-matlab-imresize/blob/master/bicubic.py

# Training settings
parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--val_dataset", default="Set5", type=str, help="val dataset")
parser.add_argument("--model_type", default="half", type=str, help="model architecture")
parser.add_argument('--ge_type', type=str, default="IN", help='type of generator', choices=["IN"])
parser.add_argument('--lr_G', type=float, default=1e-3, help='generate learning rate')
parser.add_argument('--latent', type=int, default=256, help='latent dimension')
parser.add_argument('--input_size', type=int, default=41, help='input size')
parser.add_argument('--stu_step', type=int, default=50, metavar='S')
parser.add_argument('--reconw', type=float, default=0., help='reconstrution loss weight')
parser.add_argument('--gw', type=float, default=1., help='generator loss weight')
parser.add_argument('--align', type=float, default=0., help='input_align loss')
parser.add_argument("--loss_type", default="MSE", type=str, help="type of loss", choices=["MSE", "L1"])
parser.add_argument('--scale', type=int, default=0, help='scale')
parser.add_argument('--iter', type=int, default=360, metavar='S')
parser.add_argument('--inc_step', type=int, default=1)
parser.add_argument('--inc_num', type=int, default=18, choices=[18,9,6,3,2,1])


parser.add_argument('--tmp_data_dir', default='/cache/dataset/VDSR', help='temp data dir')
parser.add_argument('--tmp_save_dir', default='/cache/save/VDSR/noise', help='temp save dir')
  
args, unparsed = parser.parse_known_args()
print("os.path.dirname(args.tmp_data_dir)", os.path.dirname(args.tmp_data_dir))


target = os.path.join(args.tmp_data_dir)


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.tmp_save_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("args = %s",args)

lossList,psnrList=[],[]
genlossList = []
bicubic = bicubic()


def main():
    print("args:")
    print(args)
    best_psnr,best_epoch=0,0
    #args.seed = random.randint(1, 10000)
    args.seed = 1
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    
    tmp_train_dataPath=os.path.join(args.tmp_data_dir,'data/train.h5')
    tmp_val_dataPath=os.path.join(args.tmp_data_dir,'Set5_mat')
    
    train_set = DatasetFromHdf5(tmp_train_dataPath)
    training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)
    
    val_imglist=glob.glob(tmp_val_dataPath+"/*.*") 
    
    assert args.pretrained != ""
    teacher = Net()
    teacher = torch.nn.DataParallel(teacher).cuda()
    
    teacher.load_state_dict(torch.load(args.pretrained))
    print("The architecture of Teacher:")
    print(teacher)
    psnr= validate(teacher,-1,val_imglist)
    print("The  PSNR of Teacher is ", psnr)

    ## Generator 
    generator1 = GeneratorINE1(img_size=args.input_size // 2, channels=1, latent=args.latent).cuda()  
    generator2 = GeneratorINE1(img_size=args.input_size // 3, channels=1, latent=args.latent).cuda()   
    generator3 = GeneratorINE1(img_size=args.input_size // 4, channels=1, latent=args.latent).cuda()  

    print("The architecture of generator: ")
    print(generator1)
    optimizer_G1 = torch.optim.Adam(generator1.parameters(), lr=args.lr_G)
    optimizer_G2 = torch.optim.Adam(generator2.parameters(), lr=args.lr_G)
    optimizer_G3 = torch.optim.Adam(generator3.parameters(), lr=args.lr_G)
    
    lr_schedulerG1 = optim.lr_scheduler.StepLR(optimizer_G1, step_size=10,gamma=0.1)
    lr_schedulerG2 = optim.lr_scheduler.StepLR(optimizer_G2, step_size=10,gamma=0.1)
    lr_schedulerG3 = optim.lr_scheduler.StepLR(optimizer_G3, step_size=10,gamma=0.1)
    
    if args.loss_type == "MSE":
        criterion = nn.MSELoss(reduction='sum')
    else:
        criterion = nn.L1Loss(reduction='sum')

    model_epoch_list = [i* args.inc_step for i in range(args.inc_num)]
    epoch_index = 0
    print("model_epoch_list: ", )
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        
        if epoch_index < len(model_epoch_list) and epoch == 1+ model_epoch_list[epoch_index]:
            epoch_index += 1
            best_psnr = -1
            if args.model_type == "origin":
                model=Net(block_num=min(18, epoch * 18 // args.inc_num))
            else:
                model=NetHalf(block_num=min(18, epoch * 18 // args.inc_num))

            model = torch.nn.DataParallel(model)
            model = model.cuda()
            for name, param in model.named_parameters():
                print(name)
            if epoch_index > 1:
                save_data = torch.load(os.path.join(args.tmp_save_dir, 'model_vdsr_{}.pth'.format(epoch-1)))
                model.load_state_dict(save_data, strict=False)
                print("Load model from {}".format(os.path.join(args.tmp_save_dir, 'model_vdsr_{}.pth'.format(epoch-1))))
            criterion = criterion.cuda()
            
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            lr_scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
        
        logging.info('current epoch {}, lr {:.5e}'.format(epoch,optimizer.param_groups[0]['lr']))
        train(training_data_loader, optimizer, model, criterion, epoch, teacher, generator1,generator2,generator3, optimizer_G1, optimizer_G2, optimizer_G3)
        plot(epoch,lossList,'mseLoss')
        lr_scheduler.step()
        lr_schedulerG1.step()
        lr_schedulerG2.step()
        lr_schedulerG3.step()
        psnr= validate(model,epoch,val_imglist)
        plot(epoch,psnrList[1:],'psnr')
        is_best=psnr>best_psnr
        if is_best:
            best_epoch=epoch
            best_psnr=psnr
            torch.save(model.state_dict(), os.path.join(args.tmp_save_dir, 'model_vdsr_best.pth'))
        torch.save(model.state_dict(), os.path.join(args.tmp_save_dir, 'model_vdsr_{}.pth'.format(epoch)))
        torch.save(generator1.state_dict(), os.path.join(args.tmp_save_dir, 'generator1_vdsr_{}.pth'.format(epoch)))
        torch.save(generator2.state_dict(), os.path.join(args.tmp_save_dir, 'generator2_vdsr_{}.pth'.format(epoch)))
        torch.save(generator3.state_dict(), os.path.join(args.tmp_save_dir, 'generator3_vdsr_{}.pth'.format(epoch)))
        logging.info('best psnr: {}, best epoch: {}'.format(best_psnr,best_epoch))
        source = os.path.join(args.tmp_save_dir)
        target = os.path.join(args.train_url)


def train(training_data_loader, optimizer, model, criterion, epoch, teacher, generator1,generator2,generator3, optimizer_G1, optimizer_G2, optimizer_G3):
    losses = AverageMeter()
    model.train()
    generator1.train()
    generator2.train()
    generator3.train()
    print("Train studentand update generator [multi scale]  iteration {}  and scheduler lr_G  after 18 !!! ".format(args.iter))
    #for iteration, batch in enumerate(training_data_loader, 1):
    for iteration in range(args.iter):
        for t in range(args.stu_step):
            z = Variable(torch.randn(args.batchSize, args.latent)).cuda()
            scale = random.randint(2, 4)
            if scale == 2:
                input_data = generator1(z)
            elif scale == 3:
                input_data = generator2(z)
            else:
                input_data = generator3(z)
            input_data = bicubic(input_data, scale=scale)
            input_data = torch.clamp(input_data, min=0, max=1)
            
            teacher_sr = teacher(input_data)
            loss = criterion(model(input_data), teacher_sr.detach())
            optimizer.zero_grad()
            loss.backward() 
            nn.utils.clip_grad_norm_(model.parameters(),args.clip) 
            optimizer.step()
            
        losses.update(loss.data.item(),input_data.size(0))
        optimizer_G1.zero_grad()
        optimizer_G2.zero_grad()
        optimizer_G3.zero_grad()
        z = Variable(torch.randn(args.batchSize, args.latent)).cuda()
        scale = random.randint(2, 4)
        if scale == 2:
            lr_gens = generator1(z)
        elif scale == 3:
            lr_gens = generator2(z)
        else:
            lr_gens = generator3(z)

        lr_gens_bicubic = bicubic(lr_gens, scale=scale)
        lr_gens_bicubic = torch.clamp(lr_gens_bicubic, min=0, max=1)
        
        teacher_sr = teacher(lr_gens_bicubic)
        genkd_loss = - torch.log (1 + criterion(model(lr_gens_bicubic), teacher_sr))
        teacher_sr_lr = bicubic(teacher_sr, scale=1./scale)  # new bicubic ; align
        teacher_sr_lr = torch.clamp(teacher_sr_lr, min=0, max=1)
        recon_loss = criterion(lr_gens, teacher_sr_lr)  
        
        gen_loss = args.gw * genkd_loss + args.reconw * recon_loss 
        gen_loss.backward()
        
        optimizer_G1.step()
        optimizer_G2.step()
        optimizer_G3.step()
        if iteration%50 == 0:
            logging.info("===> Epoch[{}]({}/{}): Loss: {:.10f} GEN-KD:{:.10f} RW:{:.10f}".format(epoch, iteration, len(training_data_loader), loss.item(), genkd_loss.item(), recon_loss.item()))
    global lossList
    lossList.append(losses.avg)
    

def validate(model,epoch,image_list,scales=[2]):
    model.eval()
    for scale in scales:
        avg_psnr_predicted = 0.0
        # avg_psnr_bicubic = 0.0
        count = 0.0
        for image_name in image_list:
            if str(scale) in image_name:
                count += 1
                # print("Processing ", image_name)
                im_gt_y = sio.loadmat(image_name)['im_gt_y']
                im_b_y = sio.loadmat(image_name)['im_b_y']
                           
                im_gt_y = im_gt_y.astype(float)
                im_b_y = im_b_y.astype(float)

                im_input = im_b_y/255.

                im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

                model = model.cuda()
                im_input = im_input.cuda()
                HR = model(im_input)
                HR = HR.cpu()

                im_h_y = HR.data[0].numpy().astype(np.float32)

                im_h_y = im_h_y * 255.
                im_h_y[im_h_y < 0] = 0
                im_h_y[im_h_y > 255.] = 255.
                im_h_y = im_h_y[0,:,:]

                psnr_predicted = PSNR(im_gt_y, im_h_y,shave_border=scale)
                avg_psnr_predicted += psnr_predicted

        logging.info("current epoch: {}, validate psnr: {} ".format(epoch,avg_psnr_predicted/count))
        global psnrList
        psnrList.append(avg_psnr_predicted/count)
    return avg_psnr_predicted/count

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def plot(epoch,valList,label):
    axis=np.linspace(1,epoch,epoch)
    fig=plt.figure()
    plt.title(label)
    plt.plot(axis,valList,label=label)
    plt.xlabel('epoch')
    plt.ylabel(label)
    plt.grid(True)
    plt.savefig(os.path.join(args.tmp_save_dir,'{}.pdf'.format(label)))
    plt.clf()
    plt.close(fig)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    main()