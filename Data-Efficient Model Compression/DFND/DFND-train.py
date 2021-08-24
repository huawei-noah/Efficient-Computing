#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os
import resnet
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from resnet import ResNet18,ResNet34
from torchvision.datasets import CIFAR100,ImageFolder,CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pdb
import numpy as np
import warnings

warnings.filterwarnings('ignore')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_select', type=int, default=600000)
parser.add_argument('--data_cifar', type=str, default='/cache/data/cifar/')
parser.add_argument('--data_imagenet', type=str, default='/cache/data/imagenet/train')
parser.add_argument('--teacher_dir', type=str, default='/cache/models/')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','cifar100'])
parser.add_argument('--epochs', type=float, default=800)
args,_ = parser.parse_known_args()


acc = 0
acc_best = 0

teacher = torch.load(opt.teacher_dir + 'teacher').cuda()
teacher.eval()
for parameter in teacher.parameters():
    parameter.requires_grad = False

def kdloss(y, teacher_scores, T=4):
    weights = weights.unsqueeze(1)
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, reduce=False)
    loss = torch.sum(l_kl) / y.shape[0]
    return loss * (T**2)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
data_train = ImageFolder(args.data_imagenet, transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    normalize,
]))

data_train_transform = ImageFolder(args.data_imagenet, transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
]))


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

if args.dataset == 'cifar100':
    data_test = CIFAR100(args.data_cifar,
                      train=False,
                      transform=transform_test)
    teacher_acc = torch.tensor([0.7774])
    n_classes = 100
if args.dataset == 'cifar10':
    data_test = CIFAR10(args.data_cifar,
                      train=False,
                      transform=transform_test)
    teacher_acc = torch.tensor([0.9523])
    n_classes = 10
    

data_test_loader = DataLoader(data_test, batch_size=1000, num_workers=0)

noise_adaptation = torch.nn.Parameter(torch.zeros(n_classes,n_classes-1))
def noisy(noise_adaptation):
    noise_adaptation_softmax = torch.nn.functional.softmax(noise_adaptation,dim=1) * (1 - teacher_acc)
    noise_adaptation_layer = torch.zeros(n_classes,n_classes)
    for i in range(n_classes):
        if i == 0:
            noise_adaptation_layer[i] = torch.cat([teacher_acc,noise_adaptation_softmax[i][i:]])
        if i == n_classes-1:
            noise_adaptation_layer[i] = torch.cat([noise_adaptation_softmax[i][:i],teacher_acc])
        else:
            noise_adaptation_layer[i] = torch.cat([noise_adaptation_softmax[i][:i],teacher_acc,noise_adaptation_softmax[i][i:]])
    return noise_adaptation_layer.cuda()
net = ResNet18(n_classes).cuda()
net = torch.nn.DataParallel(net)
criterion = torch.nn.CrossEntropyLoss().cuda()
celoss = torch.nn.CrossEntropyLoss(reduction = 'none').cuda()
optimizer = torch.optim.SGD(list(net.parameters()), lr=0.1, momentum=0.9, weight_decay=5e-4)

optimizer_noise = torch.optim.Adam([noise_adaptation], lr=0.001)
 
data_train_loader_noshuffle = DataLoader(data_train, batch_size=256, shuffle=False, num_workers=8)
    
def identify_outlier():
    value = []
    pred_list = []
    index = 0
    
    teacher.eval()
    for i,(inputs, labels) in enumerate(data_train_loader_noshuffle):
        inputs = inputs.cuda()
        outputs = teacher(inputs)
        pred = outputs.data.max(1)[1]
        loss = celoss(outputs, pred)
        value.append(loss.detach().clone())
        index += inputs.shape[0]
        pred_list.append(pred)
    return torch.cat(value,dim=0), torch.cat(pred_list,dim=0) 
    
def train(epoch, trainloader, nll):
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(trainloader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()

        optimizer.zero_grad()
        optimizer_noise.zero_grad()

        output = net(images)
        
        output_t = teacher(images).detach()
        pred = output_t.data.max(1)[1]
            
        loss = kdloss(output, output_t)         
               
        output_s = F.softmax(output, dim=1)
        output_s_adaptation = torch.matmul(output_s, noisy(noise_adaptation))
        loss += nll(torch.log(output_s_adaptation), pred)

        loss_list.append(loss.data.item())
        batch_list.append(i+1)

        if i == 1:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))

        loss.backward()
        optimizer.step()
        optimizer_noise.step()


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


def train_and_test(epoch, trainloader3, num_class, nll):
    train(epoch, trainloader3, num_class, nll)
    test()

def adjust_learning_rate(optimizer, epoch, max_epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < (max_epoch/200.0*80.0):
        lr = 0.1
    elif epoch < (max_epoch/200.0*160.0):
        lr = 0.01
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    global acc_best
    
    value, pred = identify_outlier()
    positive_index = value.topk(args.num_select,largest=False)[1]
    nll = torch.nn.NLLLoss().cuda()
    positive_index = positive_index.tolist()

    data_train_select = torch.utils.data.Subset(data_train_transform, positive_index)
    trainloader3 = torch.utils.data.DataLoader(data_train_select, batch_size=256, shuffle=True, num_workers=32)
    epoch = int(40000/args.num_select * 512)

    for e in range(1, epoch):
        adjust_learning_rate(optimizer, e, epoch)
        train_and_test(e, trainloader3, nll)
    print(acc_best)


if __name__ == '__main__':
    main()
        
    
