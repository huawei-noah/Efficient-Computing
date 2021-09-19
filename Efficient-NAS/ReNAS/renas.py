  
#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import numpy as np
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.autograd import Function, Variable
import time
import os
import copy
import math
import random
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--total_num', type=int, default=423624, help='total number of data')
parser.add_argument('--data_path', type=str, default='./nasbench101/', help='dataset path')
parser.add_argument('--train_ratio', type=float, default=0.001, help='ratio of train data')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--count', type=int, default=10, help='number of architecture validated after training the predictor')
args = parser.parse_args()


def get_data_from_file(file_path='', args=None):
    total_num = args.total_num

    data = np.zeros((total_num, 19, 7, 7))
    label = np.zeros((total_num, 8))  # label= halfway_training_time, halfway_train_accuracy, halfway_validation_accuracy, halfway_test_accuracy
                                      #        final_training_time, final_train_accuracy, final_validation_accuracy, final_test_accuracy
    for i in range(43):
        tmp_data = np.load(file_path + 'data_'+str(i)+'.npy')
        tmp_label = np.load(file_path + 'label_'+str(i)+'.npy')
        data[i*10000 : min(i*10000+10000,total_num)] = tmp_data
        label[i*10000 : min(i*10000+10000,total_num)] = tmp_label
    order = list(range(total_num))
    random.shuffle(order)
    data = data[order]
    label = label[order]
    label = label[:,-1]
    
    return data, label

def get_dataset(data, label, args=None):
    total_num = args.total_num
    train_num = int(math.floor(total_num * args.train_ratio))
    test_num = total_num - train_num

    train_data = data[:-test_num]
    test_data = data[-test_num:]
    alldata = np.array(data)
    train_data = train_data.reshape(-1, 19*7*7)
    test_data = test_data.reshape(-1,19*7*7)
    alldata = alldata.reshape(-1,19*7*7)
    train_label = label[:-test_num]
    train_label = np.squeeze(train_label)
    test_label = label[-test_num:]
    test_label = np.squeeze(test_label)
    all_label = np.array(label)
    
                            
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data_standard = scaler.transform(train_data)
    test_data_standard = scaler.transform(test_data)
    alldata_standard = scaler.transform(alldata)
                            
    max_train=max(train_label)
    min_train=min(train_label)
    train_label = (train_label-min_train)/(max_train-min_train)
    test_label = (test_label-min_train)/(max_train-min_train)
    all_label = (all_label-min_train)/(max_train-min_train)
    train_data_standard = train_data_standard.reshape(-1,19,7,7)
    test_data_standard = test_data_standard.reshape(-1,19,7,7)
    alldata_standard = alldata_standard.reshape(-1,19,7,7)

    train_data_standard = torch.from_numpy(train_data_standard).float()
    test_data_standard = torch.from_numpy(test_data_standard).float()
    alldata_standard = torch.from_numpy(alldata_standard).float()
    train_label = torch.from_numpy(train_label).float()
    test_label = torch.from_numpy(test_label).float()
    all_label = torch.from_numpy(all_label).float()
    train_set = []
    test_set = []
    all_set = []
    for i in range(train_data_standard.shape[0]):
        train_set.append((train_data_standard[i],train_label[i]))
    for i in range(test_data_standard.shape[0]):
        test_set.append((test_data_standard[i],test_label[i]))
    for i in range(all_label.shape[0]):
        all_set.append((alldata_standard[i],all_label[i]))

    return train_set, test_set, all_set

def right_pair(output, label):
    labels = np.array(label)
    outputs = np.array(output)
    tmp = sorted(zip(labels,outputs))
    labels = np.array([t[0] for t in tmp])
    outputs = np.array([t[1] for t in tmp])
    temp = outputs.argsort()
    a = np.arange(len(outputs))[temp.argsort()]
    acc=0
    for i in range(a.shape[0]-1):
        acc += np.sum(a[i+1:]>a[i])
    total = a.shape[0]*(a.shape[0]-1)//2
    tau = (acc - (total-acc)) / total
    return tau

def pair_loss(outputs, labels):
    output = outputs.unsqueeze(1)
    output1 = output.repeat(1,outputs.shape[0])
    label = labels.unsqueeze(1)
    label1 = label.repeat(1,labels.shape[0])
    tmp = (output1-output1.t())*torch.sign(label1-label1.t())
    tmp = torch.log(1+torch.exp(-tmp))
    eye_tmp = tmp*torch.eye(len(tmp)).cuda()
    new_tmp = tmp - eye_tmp
    loss = torch.sum(new_tmp)/(outputs.shape[0]*(outputs.shape[0]-1))
    return loss

def train(model, dataloaders, criterion, optimizer, num_epochs=25, use_pair_loss=False, beta=0.5):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
                
#         for phase in ['train', 'val']:
        for phase in ['train']:
            if phase == 'train':
                optimizer.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            # Iterate over data.
            total = 0
            acc = 0
            for i,(inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)   
                    if use_pair_loss:
                        loss = pair_loss(outputs,labels)
                        
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    acc += right_pair(outputs.detach().cpu().numpy(),labels.detach().cpu().numpy())
                    total += 1
                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = acc / total
            print('{} Loss: {:.4f} right pair: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'train' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        te = time.time()-start
        print("This epoch takes{:.0f}m {:.0f}s".format(te//60,te%60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best train KTau: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test(model, dataloaders):
    model.eval()
    total_output = []
    total_label = []
    for i,(inputs, labels) in enumerate(dataloaders['all']):
        total_label = total_label+list(labels)
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = list(model(inputs).detach().cpu().numpy())
        total_output = total_output + outputs
    return np.array(total_output), np.array(total_label)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(19, 38, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(38)
        self.conv2 = nn.Conv2d(38, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1   = nn.Linear(128*7*7, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out.view(-1)

def main():
    ########### Reproducable ###############
    random.seed(0)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    def worker_init_fn(worker_id):
        np.random.seed(0+worker_id)
    print('random number: ', torch.randn(2,3))
    ########################################

    data, label = get_data_from_file(args.data_path, args)
    train_set, test_set, all_set = get_dataset(data, label, args)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                              shuffle=True, num_workers=1, worker_init_fn=worker_init_fn)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                             shuffle=False, num_workers=1, worker_init_fn=worker_init_fn)
    allloader = torch.utils.data.DataLoader(all_set, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=1)
    dataloader = {}
    dataloader['train'] = trainloader
    dataloader['val'] = testloader
    dataloader['all'] = allloader

    criterion = nn.MSELoss()
    model = LeNet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = train(model, dataloader, criterion, optimizer, num_epochs=args.num_epochs, use_pair_loss=True)

    pred_labels, labels = test(model, dataloader)
    s = pred_labels.argsort()
    
    max_l = 0
    for i in range(args.count):
        if max_l < label[s[-i-1]]:
            max_l = label[s[-i-1]]
            
    print("search model acc: %.2f"%(max_l*100))
    
if __name__ == "__main__":
    main()