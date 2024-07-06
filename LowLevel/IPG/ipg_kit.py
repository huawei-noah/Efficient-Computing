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
import einops
import torch.nn.functional as F


def get_mask(idx, array):
    '''
    array: b m, records # of elements to be masked
    '''
    b, m = array.shape
    n = idx.size(-1)
    A = torch.arange(n, dtype=idx.dtype, device=idx.device).unsqueeze(0).unsqueeze(0).expand(b,m,n) # 1 1 n -> b m n
    mask = A < array.unsqueeze(-1)
    return mask


def alloc(var, rest, budget, tp, maximum, times=0, fast=False):
    '''
    var: (b m) variance of each pixel POSITIVE VALUE
    rest: (b m) list of already allocated budgets
    budget: (b) remaining to be allocated 
    tp: mean type, plain/softmax
    maximum: maximum budget for each pixel
    '''
    b, m = var.shape
    if tp == 'plain':
        var_p = var * (rest < maximum)
        var_sum = var_p.sum(dim=-1, keepdim=True) # b 1
        proportion = var_p / var_sum # b m
    elif tp == 'softmax':
        var_p = var.clone()
        var_p[rest >= maximum] = -float('inf') # maximum
        proportion = torch.nn.functional.softmax(var_p, dim=-1) # b m
    allocation = torch.round(proportion * budget.unsqueeze(1)) # b m
    new_rest = torch.clamp(rest + allocation, 0, maximum) # b m
    remain_budget = budget - (new_rest - rest).sum(dim=-1) # b m allocated
    negative_remain = (remain_budget < 0)
    while negative_remain.sum() > 0:
        offset = torch.eye(m, device=rest.device)[torch.randint(m, (negative_remain.sum().int().item(),), device=rest.device)]
        new_rest[negative_remain] = torch.clamp(new_rest[negative_remain] - offset, 1, maximum) # reduce by one
        
        # update remain budget
        remain_budget = budget - (new_rest - rest).sum(dim=-1) # b m allocated
        negative_remain = (remain_budget < 0)
        

    if (remain_budget > 0).sum() > 0:
        if times < 3:
            new_rest[remain_budget>0] = alloc(var[remain_budget>0], new_rest[remain_budget>0], remain_budget[remain_budget>0], tp, maximum, times+1, fast=fast)
        elif not fast: # precise budget allocation
            positive_remain = (remain_budget > 0)
            while positive_remain.sum() > 0:
                offset = torch.eye(m, device=rest.device)[torch.randint(m, (positive_remain.sum().int().item(),), device=rest.device)]
                new_rest[positive_remain] = torch.clamp(new_rest[positive_remain] + offset, 1, maximum) # add by one
                # update remain budget
                remain_budget = budget - (new_rest - rest).sum(dim=-1) # b m allocated
                positive_remain = (remain_budget > 0)
    return new_rest


def flex(D_:torch.Tensor, X:torch.Tensor, idx:torch.Tensor, flex_type, topk_, current_iter, total_iters, X_diff, fast=False, return_maskarray=False):
    '''
    D: (b m n) Gram matrix, sorted on last dim, descending
    X: (b numh numw he) c (sh sw) X_data
    idx: (b m n) sorted index of D
    x_size: (h, w) 2-tuple tensor
    OUT: (b m n) Binary mask
    '''
    b, m, n = D_.shape
    if flex_type is None or flex_type == 'none':
        mask_array = topk_ * torch.ones((b,m), dtype=torch.int, device=D_.device)

    elif flex_type == 'gsort':
        D = D_.clone()
        D -= (D == D.max(dim=-1, keepdim=True))* 100000 # neglect max position 
        val, g_idx = torch.sort(D.view(b, -1), dim=-1, descending=True) # global sort
        # g_idx: (b m*n)
        g_idx += m*n*torch.arange(b, dtype=g_idx.dtype, device=g_idx.device).unsqueeze(-1) # b 1
        non_topk_idx = g_idx[:, topk_*(m-1):] # select top k, neglect max
        
        mask_ = torch.ones_like(D).bool()
        mask_.view(-1)[non_topk_idx.reshape(-1)] = False # set to negative value
        mask_array = mask_.sum(dim=-1)
        mask_array += 1 # include max, ensure each pixel has at least one match

    elif flex_type == 'interdiff_plain': # interpolate and diff

        rest = torch.ones_like(X_diff)
        budget = torch.ones(b,dtype=torch.int, device=idx.device) * (topk_-1) * idx.size(1)
        mask_array = alloc(X_diff, rest, budget, tp='plain', maximum=idx.size(-1), fast=fast)
    else:
        raise NotImplementedError(f'Graph type {flex_type} not implemented...')

    if return_maskarray:
        return mask_array

    mask = ~get_mask(idx, mask_array) # negated

    return mask


def cossim(X_sample, Y_sample, graph=None):
    if graph is not None:
        return torch.einsum('a b m c, a b n c -> a b m n', F.normalize(X_sample, dim=-1), F.normalize(Y_sample, dim=-1)) + (-100.) * (~graph)
    return torch.einsum('a b m c, a b n c -> a b m n', F.normalize(X_sample, dim=-1), F.normalize(Y_sample, dim=-1))

def local_sampling(x, group_size, unfold_dict, output=0, tp='bhwc'):
    '''
        output: 
        x (grouped) [B, nn, c]
        x_unfold [B, NN, C]
        0/1/2: grouped, sampled, both
    '''
    if isinstance(group_size, int):
        group_size = (group_size, group_size)

    if output != 1:
        if tp == 'bhwc':
            x_grouped = einops.rearrange(x, 'b (numh sh) (numw sw) c-> (b numh numw) (sh sw) c', sh=group_size[0], sw=group_size[1])
        elif tp == 'bchw':
            x_grouped = einops.rearrange(x, 'b c (numh sh) (numw sw)-> (b numh numw) (sh sw) c', sh=group_size[0], sw=group_size[1])

        if output == 0:
            return x_grouped


    if tp== 'bhwc':
        x = einops.rearrange(x, 'b h w c -> b c h w')
        
    x_sampled = einops.rearrange(F.unfold(x, **unfold_dict), 'b (c k0 k1) l -> (b l) (k0 k1) c', k0=unfold_dict['kernel_size'][0], k1=unfold_dict['kernel_size'][1])

    if output == 1:
        return x_sampled

    assert x_grouped.size(0) == x_sampled.size(0)
    return x_grouped, x_sampled


def global_sampling(x, group_size, sample_size, output=0, tp='bhwc'):
    '''
        output: 
        x (grouped) [B, nn, c]
        x_unfold [B, NN, C]
    '''
    if isinstance(group_size, int):
        group_size = (group_size, group_size)
    if isinstance(sample_size, int):
        sample_size = (sample_size, sample_size)

    if output != 1:
        if tp == 'bchw':
            x_grouped = einops.rearrange(x, 'b c (sh numh) (sw numw) -> (b numh numw) (sh sw) c', sh=group_size[0], sw=group_size[1])
        elif tp == 'bhwc':
            x_grouped = einops.rearrange(x, 'b (sh numh) (sw numw) c -> (b numh numw) (sh sw) c', sh=group_size[0], sw=group_size[1])

        if output == 0:
            return x_grouped

    if tp == 'bchw':
        x_sampled = einops.rearrange(x, 'b c (sh extrah numh) (sw extraw numw) -> b extrah numh extraw numw c sh sw', sh=sample_size[0], sw=sample_size[1], extrah=1, extraw=1)
    elif tp == 'bhwc':
        x_sampled = einops.rearrange(x, 'b (sh extrah numh) (sw extraw numw) c -> b extrah numh extraw numw c sh sw', sh=sample_size[0], sw=sample_size[1], extrah=1, extraw=1)
    b_y, _, numh, _, numw, c_y, sh_y, sw_y = x_sampled.shape
    ratio_h, ratio_w = sample_size[0] // group_size[0], sample_size[1] // group_size[1]
    x_sampled = x_sampled.expand(b_y, ratio_h, numh, ratio_w, numw, c_y, sh_y, sw_y).reshape(-1, c_y, sh_y*sw_y).permute(0, 2, 1)

    if output == 1:
        return x_sampled

    assert x_grouped.size(0) == x_sampled.size(0)
    return x_grouped, x_sampled