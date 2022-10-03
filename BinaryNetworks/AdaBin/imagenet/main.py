# 2022.09.29-Changed for main script for AdaBin model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
This script was started from an early version of the pytorch-classification repository
(https://github.com/bearpaw/pytorch-classification/blob/master/imagenet.py)
"""
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from prefetch_generator import BackgroundGenerator

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.binarylib import AdaBin_Conv2d

from nets.resnet18 import resnet18_1w1a
from nets.resnet34 import resnet34_1w1a

class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run') 
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=256, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18_1w1a') 
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id 
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # laod models
    print(f"arch : {args.arch}")
    model = eval(f"{args.arch}()")

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = DataLoaderX(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = DataLoaderX(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min = 0, last_epoch=-1)

    # Resume
    title = 'ImageNet-' + args.arch
    if args.pretrained:
        # Load checkpoint.
        print('==> Resuming from pretrained..')
        assert os.path.isfile(args.pretrained), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.pretrained)
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return
    
    T_min, T_max = 1e-3, 1e1 
    # T_min, T_max = 1e-1, 1e1 
    import math 
    def Log_UP(K_min, K_max, epoch):
        Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
        return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / args.epochs * epoch)]).float().cuda()

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        # progressively
        t = Log_UP(T_min, T_max, epoch)
        if (t < 1):
            k = 1 / t
        else:
            k = torch.tensor([1]).float().cuda() 
        layer_cnt = 0
        for m in model.modules():
            if isinstance(m, AdaBin_Conv2d): 
                m.t = t.cpu()
                m.k = k.cpu()
                layer_cnt +=1

        # train for one epoch
        print('\nEpoch: [%02d|%02d] LR: %.5f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        print(f"layer : {layer_cnt}, k = {k.cpu().detach().numpy()[0]:.5f}, t = {t.cpu().detach().numpy()[0]:.5f}") 

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)
        lr_scheduler.step()
        
        # append logger file
        logger.append([optimizer.param_groups[0]['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Train Process', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # plot progress
        bar.suffix  = '({:05d}/{:05d}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: 2.2f}%% | top5: {top5: 2.2f}%%'.format(
                    batch_idx + 1,
                    len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Test Process', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({:05d}/{:05d}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: 2.2f}%% | top5: {top5: 2.2f}%%'.format(
                    batch_idx + 1,
                    len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
        
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == '__main__':
    main()
