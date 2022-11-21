#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

from __future__ import division


import os, sys, shutil, time, random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset 
import torchvision.transforms as transforms
import time
import moxing as mox
import math
import numpy as np
import pickle
from scipy.spatial import distance
import pdb
import copy
from torch.nn.parameter import Parameter
import warnings
warnings.filterwarnings("ignore")
mox.file.shift('os','mox')

os.chdir('./SCOP_NeurIPS2020/')
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, timing
import models
from models import resnet_imagenet, generator_cifar,generator_imagenet
from pruning_modules import Kf_Conv2d,Masked_Conv2d_bn,Pruned_Conv2d_bn1,Pruned_Conv2d_bn_middle,Pruned_Conv2d_bn2


parser = argparse.ArgumentParser(description='SCOP',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='/cache/tyh/cifar10/',help='Path to dataset')
parser.add_argument('--dataset', type=str,default='imagenet', choices=['cifar10','imagenet'])
parser.add_argument('--arch',  default='resnet50')
# Optimization options

parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
parser.add_argument('--batch_size_kf', type=int, default=512, help='Batch size.')

parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--learning_rate', type=float, default=0.004, help='The Learning Rate.')

parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay (L2 penalty).')

parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./tmp/', help='Folder to save checkpoints and log.')

parser.add_argument('--evaluate',type=int,default=1, help='evaluate model on validation set')

parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')

parser.add_argument('--manualSeed', type=int, help='manual seed')


parser.add_argument('--prune_rate', type=float, default=0.4, help='the reducing ratio of pruning based on knockoff')


parser.add_argument('--epochs_ft', type=int, default=120)
parser.add_argument('--lr_ft', type=float, default=0.2, help='The Learning Rate.')

parser.add_argument('--pretrain_path', default='', type=str, help='..path of pre-trained model')





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
def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)


        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
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
    print(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                              error1=100 - top1.avg))
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    print('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, 
                                                                                                   top5=top5,error1=100 - top1.avg))

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def adjust_learning_rate(optimizer, epoch, lr_init):
    lr = lr_init * (0 + (1 - 0) * (1 + math.cos(float(epoch) / args.epochs_ft * math.pi)) * 1/2)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)




if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
log =None

# Init dataset
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
    train_data_test = dset.CIFAR10(args.data_path, train=True, transform=test_transform, download=True)
    
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

    
    
#args.batch_size=12    
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)
train_loader_kf = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_kf, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.workers, pin_memory=True)






if args.dataset=='cifar10':
    netG=generator_cifar.Generator(dim=64)
    netG=torch.nn.DataParallel(netG, device_ids=list(range(args.ngpu))).cuda() 
    netG.load_state_dict(torch.load(os.path.join(args.pretrain_path,'netG_cifar.pth'))) 
elif args.dataset=='imagenet':
    netG=generator_imagenet.ResNetGenerator(64, 128, 4,activation=F.relu, num_classes=0, distribution='normal')
    netG=torch.nn.DataParallel(netG, device_ids=list(range(args.ngpu))).cuda()  
    netG.module.load_state_dict(torch.load(os.path.join(args.pretrain_path,'netG_imagenet.pth.tar'))['model'])
def train_with_kf(train_loader, model, criterion, optimizer, epoch, log,kfclass):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    
    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        
        if input.shape[0]%(args.ngpu)!=0: 
            continue
        
        if args.use_cuda:
            target = target.cuda()
            input = input.cuda()
        
        data_time.update(time.time() - end)
        
        if args.dataset=='cifar10':
            with torch.no_grad():
                kf_input=kfclass(torch.randn(input.shape[0], 128).cuda())
        elif args.dataset=='imagenet':
            with torch.no_grad():
                kf_input=kfclass(torch.empty(input.shape[0], 128, dtype=torch.float32).normal_().cuda())
                kf_input=F.interpolate(kf_input,size=224)
        
        input_list=[]
        num_pgpu=input.shape[0]//args.ngpu
        for igpu in range(args.ngpu):
            input_list.append(torch.cat([input[igpu*num_pgpu:(igpu+1)*num_pgpu],kf_input[igpu*num_pgpu:(igpu+1)*num_pgpu]],dim=0))
        input=torch.cat(input_list,dim=0)        


        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)


        
        output = model(input_var)
        
        output_list=[]
        for igpu in range(args.ngpu):
            output_list.append(output[igpu*num_pgpu*2:igpu*num_pgpu*2+num_pgpu])
        output=torch.cat(output_list,dim=0)  
        
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        for module in net.modules():
            if isinstance(module,Kf_Conv2d):
                module.kfscale.data.clamp_(min=0,max=1) 
        
        
        # measure elapsed time
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
    print(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                              error1=100 - top1.avg),
        log)
    return top1.avg, losses.avg


print("=> creating model '{}'".format(args.arch))


if args.dataset=='imagenet':    
   
    if args.arch=='resnet101':
        net=resnet_imagenet.resnet101()
    elif args.arch=='resnet50':
        net=resnet_imagenet.resnet50()
    elif args.arch=='resnet34':
        net=resnet_imagenet.resnet34()
    elif args.arch=='resnet18':
        net=resnet_imagenet.resnet18()
else:
    if args.arch=='resnet110':
        net=models.resnet110(num_classes=10)
    elif args.arch=='resnet56':
        net=models.resnet56(num_classes=10)
    elif args.arch=='resnet32':
        net=models.resnet32(num_classes=10)
    elif args.arch=='resnet20':
        net=models.resnet20(num_classes=10)
        

if args.dataset=='imagenet':
  
    if args.arch=='resnet101':
        state_dict = torch.load(os.path.join(args.pretrain_path,'resnet101-5d3b4d8f.pth'))       
    elif args.arch=='resnet50':
        state_dict = torch.load(os.path.join(args.pretrain_path,'resnet50-19c8e357.pth'))
    elif args.arch=='resnet34':
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
    net.load_state_dict(new_state_dict)         

net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.dataset!='imagenet':
    if args.arch=='resnet110':
        pretrain = torch.load(os.path.join(args.pretrain_path,'cifar_pretrained_nets/','resnet110.pth.tar'))
    elif args.arch=='resnet56':
        pretrain = torch.load(os.path.join(args.pretrain_path,'cifar_pretrained_nets/','resnet56.pth.tar'))
    elif args.arch=='resnet32':
        pretrain = torch.load(os.path.join(args.pretrain_path,'cifar_pretrained_nets/','resnet32.pth.tar'))
    elif args.arch=='resnet20':
        pretrain = torch.load(os.path.join(args.pretrain_path,'cifar_pretrained_nets/','resnet20.pth.tar'))
    net.load_state_dict(pretrain['state_dict'].state_dict())










net=net.cpu()
if args.dataset=='imagenet':
    def transform_conv( net):
  
        def _inject(modules):
            keys = list(modules.keys())
            #print(keys)
            for ik, k in enumerate(keys):
                if isinstance(modules[k], nn.Conv2d): 
 
                    if k!='0' and k!='conv1_7x7': 
                      
                        modules[k] = Kf_Conv2d(modules[k],modules[keys[ik+1]])
                        modules[keys[ik+1]]=nn.Sequential() 
                elif (not isinstance(modules[k], Kf_Conv2d)) and len(modules[k]._modules) > 0: 
                    _inject(modules[k]._modules)
        _inject(net._modules)
else:    
    def transform_conv( net):

        def _inject(modules):
            keys = list(modules.keys())
          
            for ik, k in enumerate(keys):
                if isinstance(modules[k], nn.Conv2d): 
                    if k!='0' and k!='conv_1_3x3': 
                       
                        modules[k] = Kf_Conv2d(modules[k],modules[keys[ik+1]])
                        modules[keys[ik+1]]=nn.Sequential() 
                elif (not isinstance(modules[k], Kf_Conv2d)) and len(modules[k]._modules) > 0: 
                    _inject(modules[k]._modules)
        _inject(net._modules)
transform_conv(net) 
kfconv_list=[]
for module in net.modules():
    if isinstance(module,Kf_Conv2d):
        kfconv_list.append(module)
kfscale_list=[[] for _ in range(len(kfconv_list))]
net=net.cuda()

criterion = torch.nn.CrossEntropyLoss().cuda()
recorder = RecorderMeter(args.epochs)


for param in net.parameters(): 
    param.requires_grad=False
for module in net.modules():
    if isinstance(module,Kf_Conv2d):
        module.kfscale.requires_grad=True
    
        
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate, betas=(0.5, 0.9))


start_time = time.time()
epoch_time = AverageMeter()


netG.eval()
for epoch in range(0, args.epochs):
    current_learning_rate =args.learning_rate
    need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
    need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

    print(
        '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs,
        need_time, current_learning_rate) + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                           100 - recorder.max_accuracy(False)))

    train_acc, train_los =train_with_kf(train_loader_kf, net, criterion, optimizer, epoch, log,kfclass=netG)

    
    for ikf in range(len(kfconv_list)):
        kfscale_list[ikf].append(kfconv_list[ikf].kfscale.data.clone().cpu())


    epoch_time.update(time.time() - start_time)
    start_time = time.time()
    
for param in net.parameters():
    param.requires_grad=True 
for kfscale in kfscale_list[10]:
    print(kfscale.squeeze().numpy())
for kfscale_last in kfscale_list: 
    print(kfscale_last[-1].squeeze().numpy())
net=net.cpu()
for imd, (nam,module) in enumerate(net.named_modules()):
    if isinstance(module, Kf_Conv2d):
        module.score=module.bn.weight.data.abs()*(module.kfscale.data-(1-module.kfscale.data)).squeeze() 
      

for kfconv in kfconv_list:
    kfconv.prune_rate=args.prune_rate
for imd, (nam,module) in enumerate(net.named_modules()):
    if isinstance(module, Kf_Conv2d):
        _,index=module.score.sort()
        num_pruned_channel=int(module.prune_rate*module.score.shape[0])
        
        module.out_index=index[num_pruned_channel:]


def recover_conv(net):

    def _inject(modules):
        keys = list(modules.keys())

        for ik, k in enumerate(keys):
            if isinstance(modules[k], Kf_Conv2d): #### Kf_Conv2d里面没有k=0的
                modules[k] =Masked_Conv2d_bn(modules[k])
                    
            elif (not isinstance(modules[k], Kf_Conv2d)) and len(modules[k]._modules) > 0: # nn.Conv2d的_modules的长度为0，但是Biased_Conv2d的长度为1
                _inject(modules[k]._modules)

    _inject(net._modules)
recover_conv(net)
net=net.cuda()
for input,target in train_loader:
    net.train()
    with torch.no_grad():
        net(input)
acc_before_ft,acc5_before_ft, loss_before_ft = validate(test_loader, net, criterion, log)


optimizer_ft = torch.optim.SGD(net.parameters(), args.lr_ft, momentum=args.momentum,
                            weight_decay=args.decay, nesterov=True)

recorder_ft = RecorderMeter(args.epochs_ft)

recorder_ft_top5=RecorderMeter(args.epochs_ft)


for epoch in range(0, args.epochs_ft):
    current_learning_rate = adjust_learning_rate(optimizer_ft, epoch, lr_init=args.lr_ft)

    need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs_ft - epoch))
    need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

    print(
        '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs_ft,
                                                                               need_time, current_learning_rate) \
        + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder_ft.max_accuracy(False),
                                                           100 - recorder_ft.max_accuracy(False)))

 
    train_acc, train_los = train(train_loader, net, criterion, optimizer_ft, epoch, log)

   
    val_acc_1,val_acc_5, val_los_1 = validate(test_loader, net, criterion, log)


    is_best_ft = recorder_ft.update(epoch, train_los, train_acc, val_los_1, val_acc_1)

    recorder_ft_top5.update(epoch, train_los, train_acc, val_los_1, val_acc_5)

    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': net,
        'recorder_ft': recorder_ft,
        'optimizer': optimizer_ft.state_dict(),
    }, is_best_ft, args.save_path, 'checkpoint_ft.pth.tar')

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time() 
print('top1:',recorder_ft.max_accuracy(False))

masked_conv_list=[]
for imd, (nam,module) in enumerate(net.named_modules()):
    if isinstance(module, Masked_Conv2d_bn):
        masked_conv_list.append((nam,module))
if args.dataset=='imagenet':
    for imd in range(len(masked_conv_list)):
        
        if 'conv2' in masked_conv_list[imd][0] or 'conv3' in masked_conv_list[imd][0]:
            masked_conv_list[imd][1].in_index=masked_conv_list[imd-1][1].out_index
else:
    for imd in range(len(masked_conv_list)):
        
        if 'conv_b' in masked_conv_list[imd][0]:
            masked_conv_list[imd][1].in_index=masked_conv_list[imd-1][1].out_index


if args.dataset=='imagenet':
    def pruning_conv( net):
       
        def _inject(modules):
            keys = list(modules.keys())
       
            for ik, k in enumerate(keys):
                if isinstance(modules[k], Masked_Conv2d_bn): 
                 
                    if args.arch=='resnet18' or args.arch=='resnet34':
                        if 'conv1' in k:
                            modules[k] = Pruned_Conv2d_bn1(modules[k])
                        elif 'conv2' in k:      
                            modules[k] = Pruned_Conv2d_bn2(modules[k])
                    else: ##### bottleneck结构
                        if 'conv1' in k:
                            modules[k] = Pruned_Conv2d_bn1(modules[k])
                        elif 'conv2' in k:      
                            modules[k] = Pruned_Conv2d_bn_middle(modules[k])
                        elif 'conv3' in k:      
                            modules[k] = Pruned_Conv2d_bn2(modules[k])

                elif (not isinstance(modules[k], Kf_Conv2d)) and len(modules[k]._modules) > 0: 
                    _inject(modules[k]._modules)
        _inject(net._modules)

else:
    def pruning_conv( net):
     
        def _inject(modules):
            keys = list(modules.keys())
       
            for ik, k in enumerate(keys):
                if isinstance(modules[k], Masked_Conv2d_bn): 
                  
                    if 'conv_a' in k:
                        modules[k] = Pruned_Conv2d_bn1(modules[k])
                    elif 'conv_b' in k:      
                        modules[k] = Pruned_Conv2d_bn2(modules[k])

                elif (not isinstance(modules[k], Kf_Conv2d)) and len(modules[k]._modules) > 0: 
                    _inject(modules[k]._modules)
        _inject(net._modules)
pruning_conv(net)
