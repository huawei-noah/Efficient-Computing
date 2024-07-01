# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .quant import Quantization

def judge(elem, eleList):
    for ele in eleList:
        if ele in elem:
            return True
    return False

class Quant_Conv2d(nn.Conv2d):
    '''
        Quantized Convolution layer
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, q_config=None, module=None):
        super(Quant_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.a_bit  = q_config['a_bit']
        self.w_bit  = q_config['w_bit']
        self.a_init = q_config['a_init']
        self.w_init = q_config['w_init']
        self.layer_name = q_config['layer_name']

        self.per_channel = q_config['per_channel']

        # quantize partially
        if judge(self.layer_name, q_config['FP_scope']):
            self.a_bit = 32
            self.w_bit = 32
        elif judge(self.layer_name, q_config['int8_scope']):
            self.a_bit = 8
            self.w_bit = 8

        self.a_bit = self.a_bit if q_config['a_bit']!=32 else 32
        self.w_bit = self.w_bit if q_config['w_bit']!=32 else 32

        # copy parameters
        if module!=None:
            self.weight.data = module.weight.data
            if bias:
                self.bias.data = module.bias.data

        if self.a_bit!=32:
            self.act_quantizer = Quantization(bits=self.a_bit, groups=1, tag="activation", initializer=self.a_init, per_channel=False, \
                quant_calib=q_config['quant_calib'], layer_name=self.layer_name)

        if self.w_bit!=32:
            if self.per_channel:
                self.wgt_quantizer = Quantization(bits=self.w_bit, groups=self.weight.size()[0], tag="weight", initializer=self.w_init, \
                    per_channel=True, quant_calib=q_config['quant_calib'], layer_name=self.layer_name)
            else:
                self.wgt_quantizer = Quantization(bits=self.w_bit, groups=1, tag="weight", initializer=self.w_init, per_channel=False, \
                    quant_calib=q_config['quant_calib'], weight_clip=q_config['weight_clip'], layer_name=self.layer_name)

    def forward(self, x):

        if self.a_bit!=32:
            x = self.act_quantizer(x)

        if self.w_bit!=32:
            weight = self.wgt_quantizer(self.weight)
        else:
            weight = self.weight

        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Quant_Linear(nn.Linear):
    '''
        Quantized Linear layer
    '''
    def __init__(self, in_features, out_features, bias=True, q_config=None, module=None):
        super(Quant_Linear, self).__init__(in_features, out_features, bias)

        self.a_bit  = q_config['a_bit']
        self.w_bit  = q_config['w_bit']
        self.a_init = q_config['a_init']
        self.w_init = q_config['w_init']
        self.layer_name = q_config['layer_name']

        # quantize partially
        if judge(self.layer_name, q_config['FP_scope']):
            self.a_bit = 32
            self.w_bit = 32
        elif judge(self.layer_name, q_config['int8_scope']):
            self.a_bit = 8
            self.w_bit = 8

        # copy parameters
        if module!=None:
            self.weight.data = module.weight.data
            if bias:
                self.bias.data = module.bias.data

        if self.a_bit!=32:
            self.act_quantizer = Quantization(bits=self.a_bit, groups=1, tag="activation", initializer=self.a_init, per_channel=False)
        if self.w_bit!=32:
            self.wgt_quantizer = Quantization(bits=self.w_bit, groups=1, tag="weight", initializer=self.w_init, per_channel=False)

    def forward(self, x):
        if self.a_bit!=32:
            x = self.act_quantizer(x)

        if self.w_bit!=32:
            weight = self.wgt_quantizer(self.weight)
        else:
            weight = self.weight

        return F.linear(x, weight, self.bias)

def feature_loss(fm1, fm2):
    fm1_norm = F.normalize(fm1.pow(2).mean(1).view(fm1.size(0), -1))
    fm2_norm = F.normalize(fm2.pow(2).mean(1).view(fm2.size(0), -1))
    loss = (fm1_norm - fm2_norm).pow(2).mean()
    return loss