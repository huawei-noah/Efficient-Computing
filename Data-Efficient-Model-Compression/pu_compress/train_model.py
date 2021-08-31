#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import torch
import time
import copy
import numpy as np
from loss import pu_loss, kdloss

def train_pu(model, dataloaders, optimizer, scheduler, prior, num_classes=10, num_epochs=25):
    since = time.time()
    val_acc_history = []
    
    best_acc = 0.0

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        start = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for i,(inputs, labels) in enumerate(dataloaders[phase]):
                for j in range(inputs.shape[0]):
                    if labels[j] < num_classes:
                        labels[j]=1
                    else:
                        labels[j]=-1
                inputs = inputs.cuda()
                labels = labels.cuda()
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    module = pu_loss(labels,prior)
                    loss = module(outputs)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            time_elapsed = time.time() - start
            print('Epoch: {}/{}, lr: {}, {} loss: {:.4f}, time: {:.0f}m {:.0f}s'.format(epoch, num_epochs - 1, scheduler.get_lr()[0], phase, epoch_loss, time_elapsed // 60, time_elapsed % 60))
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model

def train(student, teacher, class_weights, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    val_acc_history = []
    best_model_wts = copy.deepcopy(student.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            start = time.time()
            if phase == 'train':
                scheduler.step()
                student.train()  # Set model to training mode
            else:
                student.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for i,(inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.cuda()
                labels = labels.cuda()
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = student(inputs)
                    outputs_t = teacher(inputs).detach()
                    
                    _, preds_t = torch.max(outputs_t, 1)
                    preds_t = preds_t.cpu().data.numpy()
                    
                    max_value = 0
                    loss = None
                    for class_weight in class_weights:
                        weights = torch.from_numpy(class_weight[preds_t]).float().cuda()
                        tmp_loss = kdloss(outputs, outputs_t, weights)
                        if tmp_loss.item() > max_value:
                            max_value = tmp_loss.item()
                            loss = tmp_loss
                            
                    test_loss = 0
                    if(phase=='val'):
                        test_loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # statistics
                if(phase == 'train'):
                    running_loss += loss.item() * inputs.size(0)
                else:
                    running_loss += test_loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            time_elapsed = time.time() - start
            if(phase=='train'):
                print('Epoch: {}/{}, lr: {}, {} loss: {:.4f}, time: {:.0f}m {:.0f}s'.format(epoch, num_epochs - 1, scheduler.get_lr()[0], phase, epoch_loss, time_elapsed // 60, time_elapsed % 60))
            else:
                print('Epoch: {}/{}, lr: {}, {} loss: {:.4f}, Acc: {:.4f}, time: {:.0f}m {:.0f}s'.format(epoch, num_epochs - 1, scheduler.get_lr()[0], phase, epoch_loss, epoch_acc, time_elapsed // 60, time_elapsed % 60))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(student.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    student.load_state_dict(best_model_wts)
    return student, val_acc_history