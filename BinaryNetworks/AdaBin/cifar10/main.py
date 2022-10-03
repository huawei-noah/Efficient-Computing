# 2022.09.29-Changed for main script for AdaBin model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
This script was started from an early version of the IR-Net repository
(https://github.com/htqin/IR-Net/blob/master/CIFAR-10/ResNet20/1w1a/trainer.py)
"""
import argparse
import os
import time
import random
import numpy as np
import math
import torch
import logging
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils.binarylib import AdaBin_Conv2d
from utils.utils import *
from nets.resnet20 import resnet20_1w1a
from nets.resnet18 import resnet18_1w1a

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    help='model architecture')
parser.add_argument('--data', '-d', default='/opt/data/private/dataset/',
                    help='cifar10 dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--dropout', default=0, type=float,help='dropout rate')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--progressive', dest='progressive', action='store_true',
                    help='progressive train ')
parser.add_argument('--gpu-id',default="0", type=str,
                    help='gpu devices id')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=1)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# Check the save_dir exists or not
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
# log
if args.evaluate:
    log = open(os.path.join(args.save_dir, "log.txt"),"a+")
else:
    log = open(os.path.join(args.save_dir, "log.txt"),"w")
    log.write(f"arch : {args.arch}\n")
    log.flush()

best_prec1 = 0

def main():
    global log

    logging.info(args)

    model = torch.nn.DataParallel(eval(args.arch)()) 
    model.cuda()

    global best_prec1
    # pretrained
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}"
                  .format(args.evaluate))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            try:
                args.start_epoch = checkpoint['epoch']
            except:
                pass
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, args.start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=args.data, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4), 
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=args.data, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min = 0, last_epoch=-1)

    if args.evaluate:
        log.write("Evaluate : \n")
        validate(val_loader, model, criterion)
        return 
    
    print (f"arch : {args.arch} : \n",model)

    # file = open("./checkpoints/center_dist.txt", "w+")
    for epoch in range(args.start_epoch, args.epochs):
        if args.progressive:
            t = Log_UP(epoch, args.epochs)
            if (t < 1):
                k = 1 / t 
            else:
                k = torch.tensor([1]).float().cuda() 

            layer_cnt = 0 
            param = []
            for m in model.modules():
                if isinstance(m, AdaBin_Conv2d):
                    m.t = t
                    m.k = k
                    layer_cnt +=1
            
            line = f"layer : {layer_cnt}, k = {k.cpu().detach().numpy()[0]:.5f}, t = {t.cpu().detach().numpy()[0]:.5f}"
            log.write("=> "+line+"\n")
            log.flush()
            print(line) 

        # file.flush()
        # train for one epoch
        line = 'current lr {:.5e}'.format(optimizer.param_groups[0]['lr'])
        log.write("=> "+line+"\n")
        log.flush()
        print(line)
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        # prec1 = 0
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))
            # }, is_best, filename=os.path.join(args.save_dir, f'checkpoint_{epoch}.th'))
        if is_best:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

        line = f" *Prec@1 {best_prec1:.3f}"
        log.write("=> "+line+"\n")
        log.flush()
        print(line) 


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # target = target.cuda(async=True)
        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            line = 'Epoch: [{0}][{1}/{2}]\t' \
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch+1, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1)
            log.write(line+"\n")
            log.flush()
            print(line)

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target)

        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            line = 'Test: [{0}/{1}]\t' \
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1)
            log.write(line+"\n")
            log.flush()
            print(line)

    line = 'val Prec@1 {top1.avg:.3f}'.format(top1=top1)
    log.write("=> "+line+"\n")
    log.flush()
    print(line) 

    return top1.avg


if __name__ == '__main__':
    main()
