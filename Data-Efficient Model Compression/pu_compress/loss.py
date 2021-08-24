#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch._jit_internal import weak_module, weak_script_method
from torch.autograd import Function, Variable

class PULoss(Function):
    def __init__(self):
        self.prior = 0
        self.label = 0

    @staticmethod
    def forward(self, input, label, prior):
        self.input = input
        self.label = label
        self.prior = prior.cuda().float()
        self.positive = 1
        self.unlabeled = -1
        self.loss_func = lambda x: F.sigmoid(-x)
        self.beta = 0
        self.gamma = 1
        
        self.positive_x = (self.label==self.positive).float()
        self.unlabeled_x = (self.label==self.unlabeled).float()
        self.positive_num = torch.max(torch.sum(self.positive_x), torch.tensor(1).cuda().float())
        self.unlabeled_num = torch.max(torch.sum(self.unlabeled_x), torch.tensor(1).cuda().float())
        self.positive_y = self.loss_func(self.input)
        self.unlabeled_y = self.loss_func(-self.input)
        self.positive_loss = torch.sum(self.prior * self.positive_x / self.positive_num * self.positive_y.squeeze())
        self.negative_loss = torch.sum((self.unlabeled_x / self.unlabeled_num - self.prior * self.positive_x / self.positive_num) * self.unlabeled_y.squeeze())
        objective = self.positive_loss + self.negative_loss
        
        if self.negative_loss.data < -self.beta:
            objective = self.positive_loss - self.beta
            self.x_out = -self.gamma * self.negative_loss
        else:
            self.x_out = objective
        return objective
            
    @staticmethod
    def backward(self,grad_output):
        d_input = torch.zeros(self.input.shape).cuda().float()
        d_positive_loss = -self.prior * self.positive_x / self.positive_num * self.positive_y.squeeze() * (1-self.positive_y.squeeze())
        d_negative_loss = (self.unlabeled_x / self.unlabeled_num - self.prior * self.positive_x / self.positive_num) * self.unlabeled_y.squeeze() * (1-self.unlabeled_y.squeeze())
        if self.negative_loss.data < -self.beta:
            d_input = -self.gamma * d_negative_loss
        else:
            d_input = d_positive_loss + d_negative_loss
        d_input = d_input.unsqueeze(1)
        d_input = d_input * grad_output
        return d_input,None,None
        
class pu_loss(nn.Module):
    def __init__(self, label, prior):
        super(pu_loss, self).__init__()
        self.prior = prior
        self.label = label
    def forward(self, input):
        return PULoss.apply(input, self.label, self.prior)
    
    
def kdloss(y, teacher_scores, weights, T=1):
    weights = weights.unsqueeze(1)
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, reduce=False)
    loss = torch.sum(l_kl*weights) / y.shape[0]
    return loss * T