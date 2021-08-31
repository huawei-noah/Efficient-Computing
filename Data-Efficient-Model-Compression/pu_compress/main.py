#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from model import ResNet34, ResNet34_PU, ResNet18
from train_model import train, train_pu

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pu_lr', type=float, default=0.001)
parser.add_argument('--pu_weight_decay', type=float, default=5e-3)
parser.add_argument('--pu_num_epochs', type=int, default=170)
parser.add_argument('--pu_batchsize', type=int, default=4096)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--batchsize', type=int, default=2048)

parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--perturb_num', type=int, default=1)

parser.add_argument('--pos_num', type=int, default=100)
parser.add_argument('--prior', type=float, default=0.21)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--label_dir', type=str, default='/cache/data/cifar10/')
parser.add_argument('--unlabel_dir', type=str, default='/cache/data/imagenet/train')
parser.add_argument('--teacher_model_dir', type=str, default='/cache/resnet34.pth')
opt,_ = parser.parse_known_args()
print(opt)

def initialize_model(use_pu=True, num_classes=10):
    model = None
    
    if(use_pu):
        model = ResNet34_PU(num_classes=num_classes)
        model.load_state_dict(torch.load(opt.teacher_model_dir), strict=False)
        num_ftrs = 512+256+128+64
        model.fc = nn.Linear(num_ftrs, 1)
    else:
        model = ResNet34(num_classes=num_classes)
        model.load_state_dict(torch.load(opt.teacher_model_dir))
    
    return model

def get_positive(model, dataloader):
    positive_index = []
    index = 0
    
    model.eval()
    for i,(inputs, labels) in enumerate(dataloader):
        inputs = inputs.cuda()
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            for j in range(inputs.shape[0]):
                if(outputs[j] > 0):
                    positive_index.append(index+j)
            index += inputs.shape[0]
    return positive_index  

def get_class_weight(model, dataloader, num_classes=10, T=1):
    classes_outputs = np.zeros(num_classes)
    
    model.eval()
    for i,(inputs, labels) in enumerate(dataloader):
        inputs = inputs.cuda()
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = F.softmax(outputs/T, dim=1)
            for j in range(inputs.shape[0]):
                classes_outputs += outputs[j].cpu().data.numpy()
    
    class_weights = 1/classes_outputs
    weights_sum = np.sum(class_weights)
    class_weights /= weights_sum
    class_weights *= opt.num_classes
    
    return class_weights 

def perturb(weight, epsilon=0.1, perturb_num=1):
    weights = []
    weights.append(weight)
    for i in range(perturb_num):
        p = np.random.rand(weight.shape[0]) * epsilon
        weight_new = weight + p
        weights.append(weight_new)
    return weights

def main():
    prior = torch.tensor(opt.prior)
    input_size = 32
    
    # prepare training data
    transform = {
        'train': transforms.Compose([
         transforms.Resize(input_size),
         transforms.RandomCrop(input_size, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ]),
        'val': transforms.Compose([
         transforms.Resize(input_size),
         transforms.CenterCrop(input_size),
         transforms.ToTensor(),
         transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ]),
    }
    label_trainset = datasets.CIFAR10(root=opt.label_dir, train=True, download=False, transform=transform['train'])
    label_testset = datasets.CIFAR10(root=opt.label_dir, train=False, download=False, transform=transform['val'])
    labelset_index=[]
    classnum = np.ones(opt.num_classes) * opt.pos_num
    i = 0
    while(np.sum(classnum) > 0):
        image = label_trainset[i][0]
        label = label_trainset[i][1]
        if(classnum[label] > 0):
            labelset_index.append(i)
            classnum[label] -= 1
        i += 1
    label_train_subset = torch.utils.data.Subset(label_trainset, labelset_index)
    target_transform = transforms.Lambda(lambda target: target + opt.num_classes)
    unlabel_set = datasets.ImageFolder(opt.unlabel_dir, transform['train'], target_transform=target_transform)
    trainloader = torch.utils.data.DataLoader(label_train_subset+unlabel_set, batch_size=opt.pu_batchsize, shuffle=True, num_workers=32)
    testloader = torch.utils.data.DataLoader(label_testset, batch_size=opt.pu_batchsize, shuffle=False, num_workers=32)
    dataloader = {}
    dataloader['train'] = trainloader
    dataloader['val'] = testloader
    

    # stage1: get positive data from unlabeled dataset
    model_pu = nn.DataParallel(initialize_model(use_pu=True,num_classes=opt.num_classes)).cuda()
    optimizer_pu = optim.SGD(model_pu.parameters(), lr=opt.pu_lr, momentum=opt.momentum, weight_decay=opt.pu_weight_decay)
    scheduler_pu = optim.lr_scheduler.MultiStepLR(optimizer_pu, milestones=[50,100,150], gamma=0.1, last_epoch=-1)
    model_pu = train_pu(model_pu, dataloader, optimizer_pu, scheduler_pu, prior=prior, num_classes = opt.num_classes, num_epochs=opt.pu_num_epochs)
    trainloader2 = torch.utils.data.DataLoader(unlabel_set, batch_size=opt.pu_batchsize, shuffle=False, num_workers=32)
    positive_index_all = get_positive(model_pu, trainloader2)
    print("We have {} positive unlabeled images for all!".format(len(positive_index_all)))
    
    unlabel_positive_set = torch.utils.data.Subset(unlabel_set, positive_index_all)
    trainloader3 = torch.utils.data.DataLoader(label_train_subset + unlabel_positive_set, batch_size=opt.batchsize, shuffle=True, num_workers=32)
    dataloader['train'] = trainloader3
    
    
    # stage2: train student model with rkd method
    teacher = nn.DataParallel(initialize_model(use_pu=False,num_classes=opt.num_classes)).cuda()
    student = nn.DataParallel(ResNet18(num_classes=opt.num_classes)).cuda()
    class_weight = get_class_weight(teacher, trainloader3, num_classes=opt.num_classes)
    print(class_weight)
    
    class_weights = perturb(class_weight, opt.epsilon, opt.perturb_num)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(student.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.1, last_epoch=-1)
    model_s, hist = train(student, teacher, class_weights, dataloader, criterion, optimizer, scheduler, num_epochs=opt.num_epochs)
    
if __name__ == '__main__':
    main()
