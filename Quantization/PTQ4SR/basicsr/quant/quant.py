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

from re import L
from cv2 import MSER
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

from sklearn.cluster import DBSCAN

def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor

class Quantization(nn.Module):
    def __init__(self, bits=4, groups=1, tag="weight", initializer = "percentile", per_channel=False, quant_calib=False, weight_clip=0, layer_name=None, percentile=1e-4):
        super(Quantization, self).__init__()
        self.groups = groups
        self.bits = bits
        self.tag = tag
        self.range = 2**bits-1
        # calibration
        self.initializer = initializer
        self.calibration  = False
        self.ratio = 0.9
        # grad scale
        self.scale = 1
        # quantizer
        self.quantizer = None
        self.init = True
        self.per_channel = per_channel

        self.quant_calib = quant_calib

        self.weight_clip = weight_clip

        self.layer_name = layer_name

        if self.bits==8:
            self.percentile = 1e-5
        else:
            self.percentile = percentile

        self.calib_iter = 1

        assert self.tag in ['weight', 'activation']
        if self.tag=="weight" and self.per_channel:
            self.lower_clip_val = nn.Parameter(torch.tensor([float("inf")]*self.groups))
            self.upper_clip_val = nn.Parameter(torch.tensor([float("inf")]*self.groups))
        else:
            self.lower_clip_val = nn.Parameter(torch.tensor(float("inf")))
            self.upper_clip_val = nn.Parameter(torch.tensor(float("inf")))

    def calibrate(self, x):
        assert self.initializer in ["percentile", "L1_Loss"]
        if self.initializer=="percentile":
            lower_clip, upper_clip = self.Percentile_init(x)
        elif self.initializer=="L1_Loss":
            lower_clip, upper_clip = self.L1_Loss_init(x)
        if torch.isinf(self.lower_clip_val.data).any() and torch.isinf(self.upper_clip_val.data).any():
            self.lower_clip_val.data = lower_clip
            self.upper_clip_val.data = upper_clip
        else:
            # EMA
            self.ratio = 1-1/(1+self.calib_iter)
            # moving average
            self.lower_clip_val.data = self.lower_clip_val.data * self.ratio + lower_clip * (1-self.ratio)
            self.upper_clip_val.data = self.upper_clip_val.data * self.ratio + upper_clip * (1-self.ratio)

            self.calib_iter += 1

        self.lower_clip_val.data = reduce_mean(self.lower_clip_val.data)
        self.upper_clip_val.data = reduce_mean(self.upper_clip_val.data)

    def Percentile_init(self, x, bins=2048):
        def helper(x_):
            min_val, max_val = x_.min(), x_.max()
            assert self.tag in ["weight", "activation"]
            if self.tag=="weight":
                if self.weight_clip!=0:
                    min_val, max_val = torch.quantile(x_, torch.tensor([self.weight_clip, 1-self.weight_clip]).cuda())
                return min_val, max_val
            hist = torch.histc(x_, bins)
            lower_idx, upper_idx = 0, len(hist)-1
            while torch.sum(hist[lower_idx:upper_idx+1])>(1-self.percentile) * x_.numel():
                if hist[lower_idx]<hist[upper_idx]:
                    lower_idx = lower_idx+1
                else:
                    upper_idx = upper_idx-1

            interval = (max_val-min_val)/bins
            lower_clip = interval*(lower_idx+1) + min_val
            upper_clip = interval*(upper_idx+1) + min_val
            return lower_clip, upper_clip

        if self.per_channel:
            lower_clip, upper_clip = [], []
            for c in range(self.groups):
                lower_sub_clip, upper_sub_clip = helper(x[c])
                lower_clip.append(lower_sub_clip)
                upper_clip.append(upper_sub_clip)
            lower_clip = torch.tensor(lower_clip)
            upper_clip = torch.tensor(upper_clip)
        else:
            lower_clip, upper_clip = helper(x)
        lower_clip = lower_clip.cuda()
        upper_clip = upper_clip.cuda()
        return lower_clip, upper_clip

    def gradient_scale(self, x):
        yOut = x.clone()
        yGrad = x * self.scale
        y = (yOut - yGrad).detach() + yGrad
        return y

    def round_pass(self, x):
        '''
            Straight Thought Estimator
        '''
        yOut = x.round()
        yGrad = x
        y = (yOut - yGrad).detach() + yGrad
        return y

    def quant_tensor(self, x):
        '''
            quant to [0, 2^b-1]
        '''
        upper_clip = self.gradient_scale(self.upper_clip_val)
        lower_clip = self.gradient_scale(self.lower_clip_val)
        if self.per_channel:
            upper_clip = upper_clip.view(-1,1,1,1)
            lower_clip = lower_clip.view(-1,1,1,1)
        quant_x = self.round_pass(((x-lower_clip)/(upper_clip-lower_clip)).clamp(0, 1)*self.range)
        dequant_x = quant_x * (upper_clip-lower_clip)/self.range + lower_clip
        return dequant_x.cuda()

    def forward(self, x):
        # inilialize the quantizer in the first iter
        if self.init:
            self.scale = 1.0/np.sqrt(x.numel() * (2.0**(self.bits-1)-1))
            self.init = False
        # calibrate the clip_val
        if self.calibration:
            self.calibrate(x)
            if self.tag=="weight":   # calibration once is ok
                self.calibration = False
                # x = self.quant_tensor(x)   # quant weight when calibration
            return x
        else:
            # quant and de-quant
            dquant_x = self.quant_tensor(x)
            return dquant_x
