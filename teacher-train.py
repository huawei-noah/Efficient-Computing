#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os
from lenet import LeNet5
import resnet
import torch
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import argparse

parser = argparse.ArgumentParser(description='train-teacher-network')

# Basic model parameters.
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='/cache/data/')
parser.add_argument('--output_dir', type=str, default='/cache/models/')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)  

acc = 0
acc_best = 0

if args.dataset == 'MNIST':
    
    data_train = MNIST(args.data,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           ]))
    data_test = MNIST(args.data,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                          ]))

    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

    net = LeNet5().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
if args.dataset == 'cifar10':
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_train = CIFAR10(args.data,
                       transform=transform_train)
    data_test = CIFAR10(args.data,
                      train=False,
                      transform=transform_test)

    data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=100, num_workers=0)

    net = resnet.ResNet34().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

if args.dataset == 'cifar100':
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_train = CIFAR100(args.data,
                       transform=transform_train)
    data_test = CIFAR100(args.data,
                      train=False,
                      transform=transform_test)
                      
    data_train_loader = DataLoader(data_train, batch_size=opt.batchsize, shuffle=True, num_workers=0)
    data_test_loader = DataLoader(data_test, batch_size=opt.batchsize, num_workers=0)
    net = resnet.ResNet34(num_classes=100).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < 80:
        lr = 0.1
    elif epoch < 120:
        lr = 0.01
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(epoch):
    if args.dataset != 'MNIST':
        adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
 
        optimizer.zero_grad()
 
        output = net(images)
 
        loss = criterion(output, labels)
 
        loss_list.append(loss.data.item())
        batch_list.append(i+1)
 
        if i == 1:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))
 
        loss.backward()
        optimizer.step()
 
 
def test():
    global acc, acc_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
 
    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    if acc_best < acc:
        acc_best = acc
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
 
 
def train_and_test(epoch):
    train(epoch)
    test()
 
 
def main():
    if args.dataset == 'MNIST':
        epoch = 10
    else:
        epoch = 200
    for e in range(1, epoch):
        train_and_test(e)
    torch.save(net,args.output_dir + 'teacher')
 
 
if __name__ == '__main__':
    main()