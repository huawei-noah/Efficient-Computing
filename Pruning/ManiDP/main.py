

#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.



import os
os.chdir('./ManiDP/')


import sys, shutil, time, random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset 
import torchvision.transforms as transforms
import time

import math





from torch.nn import init



import numpy as np
import pickle
from scipy.spatial import distance
import pdb
import copy
from torch.nn.parameter import Parameter

from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, timing,accuracy
import models

from models import resnet_cifar,resnet_imagenet,mobilenetv2_imagenet



from torch.distributions.multivariate_normal import MultivariateNormal
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Training on CIFAR or ImageNet',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='/cache/tyh/cifar10/',help='Path to dataset')
parser.add_argument('--dataset', type=str,default='cifar10', choices=['cifar10', 'imagenet'],
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='dyresnet20')

parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--decay_branch', type=float, default=0.0, help='Weight decay (L2 penalty).')
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='/cache/dypruning_results/', help='Folder to save checkpoints and log.')
parser.add_argument('--pretrain_path', default='/cache/pretrain_path/dynamic_pruning/', type=str, help='..path of pre-trained model')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--squeeze_rate', type=int, default=2)
parser.add_argument('--thre_freq', type=int, default=1)
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05, help='The Learning Rate.') 
parser.add_argument('--thre_init', type=float, default=-10000.0)
parser.add_argument('--thre_cls', type=float, default=0.5)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--clamp_max', type=float, default=1000.0)



parser.add_argument('--lambda_lasso', type=float, default=1e-1,help='group lasso loss weight')
parser.add_argument('--lambda_graph', type=float, default=0.1)
parser.add_argument('--target_remain_rate', type=float, default=0.65)




args,unparsed = parser.parse_known_args()



args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True



if args.dataset=='imagenet':
    args.workers=16


if args.dataset in ['cifar10']:    
    from models.resnet_cifar import MaskBlock
elif args.dataset in ['imagenet']:    
    from models.resnet_imagenet import MaskBlock



if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

log =None


if not os.path.isdir(args.data_path):
    os.makedirs(args.data_path)

if args.dataset == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
elif args.dataset == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
elif args.dataset=='imagenet':
    normalize_imgnet = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
else:
    assert False, "Unknow dataset : {}".format(args.dataset)
if args.dataset=='imagenet':
    pass
else:
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

if args.dataset == 'cifar10':
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 10
elif args.dataset == 'imagenet':   
    train_data = dset.ImageFolder(os.path.join(args.data_path, 'train'),transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_imgnet,
    ]))
    train_data_test = dset.ImageFolder(os.path.join(args.data_path, 'train'),transforms.Compose([
        transforms.Scale(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize_imgnet,
    ]))
    
    test_data=dset.ImageFolder(os.path.join(args.data_path, 'val'), transforms.Compose([
        transforms.Scale(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize_imgnet,
    ]))
   
else:
    assert False, 'Do not support dataset : {}'.format(args.dataset)

  
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.workers, pin_memory=True)


print("=> creating model '{}'".format(args.arch))


if args.dataset=='imagenet':    
    if args.arch=='resnet34':
        net=resnet_imagenet.resnet34(args=args)
    elif args.arch=='resnet18':
        net=resnet_imagenet.resnet18(args=args)

else:
    if args.arch=='dyresnet20':
        net=resnet_cifar.resnet20(num_classes=num_classes,args=args)
  
    elif args.arch=='dyresnet32':
        net=resnet_cifar.resnet32(num_classes=num_classes,args=args)
     
    elif args.arch=='dyresnet56':
        net=resnet_cifar.resnet56(num_classes=num_classes,args=args)

    
if args.dataset=='imagenet':

    if args.arch=='resnet34':
        state_dict = torch.load(os.path.join(args.pretrain_path,'resnet34-333f7ec4.pth'))
    elif args.arch=='resnet18':
        state_dict = torch.load(os.path.join(args.pretrain_path,'resnet18-5c106cde.pth'))
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        if k=='conv1.weight':
            new_state_dict['conv1_7x7.weight'] = v
        else:
            new_state_dict[k] = v    
  
    net.load_state_dict(new_state_dict,strict=False)
    
net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu))).cuda() 


if args.dataset=='cifar10':
    if args.arch=='dyresnet20':      
        pretrain_dict = torch.load(os.path.join(args.pretrain_path,'cifar_pretrained_nets/','ck_resnet20_cifar10.pth'))
    elif args.arch=='dyresnet32':
        pretrain_dict = torch.load(os.path.join(args.pretrain_path,'cifar_pretrained_nets/','ck_resnet32_cifar10.pth'))
    elif args.arch=='dyresnet56':
        pretrain_dict = torch.load(os.path.join(args.pretrain_path,'cifar_pretrained_nets/','ck_resnet56_cifar10.pth'))
    key_list=list(pretrain_dict.keys())
    for key in key_list:
        if 'mb' in key:
            del pretrain_dict[key]  

    net.load_state_dict(pretrain_dict,strict=False)

criterion = torch.nn.CrossEntropyLoss()

def cal_similarity_loss( f_s, f_t):
    bsz = f_s.shape[0]
    f_s = f_s.view(bsz, -1)
    f_t = f_t.view(bsz, -1) 
    f_s=F.normalize(f_s)
    G_s = torch.mm(f_s, torch.t(f_s))   
    f_t=F.normalize(f_t)
    G_t = torch.mm(f_t, torch.t(f_t))
    G_diff = G_t - G_s 
    loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
    
    return loss
def train(train_loader, model, criterion, optimizer, epoch, log,
          lambda_lasso=None,total_epochs=None,lr_init=None,train_los_pre=None):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    loss_lasso_record=AverageMeter()
    loss_graph_record=AverageMeter()
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    
    loss_ce_list=[]
    
    train_los_pre=train_los_pre.cuda()
    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        
        lr=adjust_learning_rate(optimizer, epoch, args, batch=i,
                              nBatch=len(train_loader), total_epochs=total_epochs,lr_init=lr_init)
     
        
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        
        output,_mask_list,lasso_list,_mask_before_list,_avg_fea_list= model(input_var)
        

        loss_ce = F.cross_entropy(output, target_var,reduction='none')
        
        losses.update(loss_ce.data.mean().item(), input.size(0))s
        
        gamma=args.gamma
        
        loss_lasso=0.0  
        train_los_pre=train_los_pre.mean().to(loss_ce.device)
        w1=(loss_ce < args.thre_cls*train_los_pre).float() 
        w2=((args.thre_cls*train_los_pre-loss_ce)/(args.thre_cls*train_los_pre)).clamp(min=1e-10).pow(gamma)
        w=w1*w2                  
        for ilasso in range(len(lasso_list)):
            loss_lasso=loss_lasso+(lasso_list[ilasso]*w).mean()     
        loss_lasso_record.update(loss_lasso.data.item(), input.size(0))

        loss_ce_list.append(loss_ce.data.cpu())
        loss=loss_ce.mean() + lambda_lasso*loss_lasso

         
        graph_loss=0.0
        for igraph in range(len(_mask_before_list)):
            graph_loss=graph_loss+cal_similarity_loss(_mask_before_list[igraph],_avg_fea_list[igraph])


        loss_graph_record.update(graph_loss.data.item(), input.size(0))

        loss=loss + args.lambda_graph*graph_loss
            
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        
        
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string())
    print('**Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                              error1=100 - top1.avg))   
    
    print('lr:',lr)
 
    if lambda_lasso!=0.0:
        print('Loss_lasso {loss_lasso.avg:.4f}   '.format(loss_lasso=loss_lasso_record,end=''))   
    if args.lambda_graph!=0.0:
        print('Loss_graph {loss_graph.avg:.4f}   '.format(loss_graph=loss_graph_record,end=''))
        
    print('Loss {loss.avg:.4f}   '.format(loss=losses))
    
    loss_ce_list=torch.cat(loss_ce_list,dim=0)
    
    return top1.avg, loss_ce_list


def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output,_,_,_,_ = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    print('**Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, 
                                                                                                   top5=top5,error1=100 - top1.avg))

    return top1.avg, top5.avg, losses.avg


def validate_ontrain(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    loss_ce_list=[]
    
    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output,_,_,_,_ = model(input_var)
            #loss = criterion(output, target_var)
            loss_ce = F.cross_entropy(output, target_var,reduction='none')
            loss_ce_list.append(loss_ce.data.cpu())
            loss=loss_ce.mean()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        
        
    loss_ce_list=torch.cat(loss_ce_list,dim=0)
    print('**Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, 
                                                                                                   top5=top5,error1=100 - top1.avg))

    return top1.avg, loss_ce_list

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, args, batch=None,nBatch=None, total_epochs=None,lr_init=None):
    T_total = total_epochs * nBatch
    T_cur = (epoch % total_epochs) * nBatch + batch 
    lr = 0.5 * lr_init * (1 + math.cos(math.pi * T_cur / T_total))  
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_remain_rate(epoch,total_epochs,target_remain_rate):  
    remain_rate_init=1.0
    remain_rate_last=target_remain_rate
    remain_rate=(remain_rate_init-remain_rate_last)/2.0*math.cos(math.pi * (epoch+1) / total_epochs)+(remain_rate_init+ remain_rate_last)/2.0

    return remain_rate

        



def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



maskblock_list=[]
for module in net.modules():
    if isinstance(module,MaskBlock):
        maskblock_list.append(module)

params_main=[]
params_branch=[]
for nam, parm in net.named_parameters():
    if 'mb' in nam:
        params_branch.append(parm)
    else:
        params_main.append(parm)

optimizer = torch.optim.SGD([{'params':params_main,'weight_decay':args.decay},
                            {'params':params_branch,'weight_decay':args.decay_branch}], 
                            args.lr, momentum=args.momentum,  nesterov=True)


start_time = time.time()
epoch_time = AverageMeter()  
recorder = RecorderMeter(args.epochs)
recorder_top5=RecorderMeter(args.epochs)

train_acc_1,train_los=validate_ontrain(train_loader, net, criterion, log)

for epoch in range(0, args.epochs):

    
    remain_rate=adjust_remain_rate(epoch,total_epochs=args.epochs,target_remain_rate=args.target_remain_rate)  
    print('remain_rate:',remain_rate)
    
    
    
    need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
    need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

    print(
        '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(time_string(), epoch, args.epochs,
                                                                               need_time) \
        + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                           100 - recorder.max_accuracy(False)))

    for mblock in maskblock_list:
        mblock.mask_sum.fill_(0.0)
        
    train_acc, train_los = train(train_loader, net, criterion, optimizer, epoch, log,
                                 lambda_lasso=args.lambda_lasso,
                                total_epochs=args.epochs,lr_init=args.lr,train_los_pre=train_los)
    
    val_acc_1,val_acc_5, val_los_1 = validate(test_loader, net, criterion, log)
    
    if epoch % args.thre_freq==0 and epoch>=1:

        
        num_data=float(len(train_loader.dataset))/args.ngpu
        mask_mean_list=[mblock.mask_sum/num_data for mblock in maskblock_list]
        threshold_list=[torch.topk(mask_mean_list[imodule],k=math.ceil(remain_rate*len(mask_mean_list[imodule])),
                                       sorted=True)[0][-1] for imodule in range(len(mask_mean_list))]    
                
        imask=0
        for module in net.modules():
            if isinstance(module,MaskBlock):
                module.thre.fill_(threshold_list[imask])
                imask=imask+1


    is_best = recorder.update(epoch, train_los.mean(), train_acc, val_los_1, val_acc_1)

    recorder_top5.update(epoch, train_los.mean(), train_acc, val_los_1, val_acc_5)

    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': net,
        'recorder': recorder,
        'optimizer': optimizer.state_dict(),
    }, is_best, args.save_path, 'checkpoint.pth.tar')

  
    epoch_time.update(time.time() - start_time)
    start_time = time.time() 
acc,acc5, loss = validate(test_loader, net, criterion, log)
print('last acc:',acc)
print('best acc:',recorder.max_accuracy(False))
