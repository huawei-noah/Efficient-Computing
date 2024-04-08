#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math as m
from image_classification.rigl_torch.util import get_W
import time

def find_k(S2,loss_value):
    assert 0 <= loss_value <=1
    Sp = torch.flip(S2,dims=[0])#reverse order
    acc_A = torch.cumsum(Sp,dim=0)

    acc_A -= loss_value
    acc_A = acc_A.abs()
    k =  acc_A.argmin()
    return len(acc_A) - k.item() -1

def bsearch(normed_w, U,S,V,k_start,k_end,loss_value = 0.5):
    return find_k(S**2, loss_value)

def get_sing_loss(w,loss_value,error=None,args=None):
    sp=w.shape
    def normed(mat):
        normer = torch.norm(mat)
        if normer !=0:
            return mat / normer
        print('Normer = 0 !!!')
        return 0

    normed_w = normed(w.view(sp[0],-1))
    if isinstance(normed_w,int) and normed_w == 0:
        return 0.,0
    try:
        U,S,V = torch.svd(normed_w.detach())
        # flops calc for reb
        m, n = normed_w.shape
        args.svd_flops['solve'] += (2*m*n*n+11*n*n*n)
        args.svd_flops['iter'] += (4*m*n*n-4/3*n*n*n)
        args.svd_flops['min_solve'] += (2*m*n*n+11*n*n*n)
        args.svd_flops['min_iter'] += (4*m*n*n-4/3*n*n*n)
    except:
        rand_num = int(time.time())
        file_name = f'results/err{rand_num}.pkl'
        print(f'Error Detected ... Error matrix saved in {file_name}!')
        torch.save(normed_w.detach(),file_name)
        return str(rand_num),0
    
    k = bsearch(normed_w,U,S,V,0,normed_w.size(0), loss_value=loss_value)
    assert k <= sp[0]
    w_approx = U[:,:k] @ torch.diag(S)[:k, :k]@V.t()[:k,:]
    loss =  -F.mse_loss(normed_w,w_approx,reduction='sum')
    if error is None:
        return loss,k
    else:
        l = (loss - error)
        if l < 0:
            return  l ** 2,k
        else:
            return 0.,k
def print_loss(losses,ks=None,head = 'Loss'):
    print(head,end = ": ")
    if ks is None:
        for idx,loss in enumerate(losses):
            if isinstance(loss,float) or isinstance(loss,int):
                print(f'{idx}:{loss:.4f}',end =' ')
            else:
                print(f'{idx}:{loss.item():.4f}',end =' ')
    else:
        for idx,(k,loss) in enumerate(zip(ks,losses)):
            if isinstance(loss,float) or isinstance(loss,int):
                print(f'{idx}@{k}:{loss:.3f}',end =' ')
            else:
                print(f'{idx}@{k}:{loss.item():.3f}',end =' ')
            
    print()

def get_k(w,partial_k):
    sz=w.shape
    if 0. < partial_k <= 1.0:
        k = round(sz[0] * partial_k)
    else:
        k = sz[0] * round(partial_k)
    return k

def get_loss_para(model,partial_k,errors,sparsity_thres=0.9,pretrainedUR = None,args =None):
    if pretrainedUR is not None and pretrainedUR[0]==pretrainedUR[1] == None:
        pretrainedUR =None
    if not hasattr(args,'ks'):
        args.ks = []
    layers,linear_masks = get_W(model, return_linear_layers_mask=True)
    W=[]
    L=[]
    for l,lmask in zip(layers,linear_masks):
        if not lmask:
            W.append(l.data)
            L.append(l)
    loss = 0.
    activated_layers = []
    losses = []
    ks = []
    if errors is None:
        errors = [None] * len(L)
    for numlayer,(w,error) in enumerate(zip(L,errors)):
        sparsity = (w == 0.).sum().item() / w.numel()
        if sparsity >= sparsity_thres:
            this_loss,this_k = get_sing_loss(w,partial_k,error,args)
            if isinstance(this_loss,str):
                print(f'Layer {numlayer} SVD Error!!!')
                torch.save(model.state_dict(),f'results/modeldict{this_loss}.pkl')
                torch.save(model.module,f'results/model{this_loss}.pkl')

                this_loss = 0.

            if this_loss != 0.:
                activated_layers.append(numlayer)
            losses.append(this_loss)
            ks.append(this_k)
            if this_loss !=0.:
                loss += this_loss
    # print('Rank Loss Activated Layers:', activated_layers)
    # print_loss(losses,ks)
    args.ks.append(ks)
    if len(activated_layers) == 0:
        loss = 0.0
    else:
        loss /= len(activated_layers)
    return loss, None, None















