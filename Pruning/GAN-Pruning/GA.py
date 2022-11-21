# 2019.11.25-Changed for GA operations 
#            Huawei Technologies Co., Ltd. <foss@huawei.com> 

import os
import torch
import numpy as np

def roulette(mask_all,N,fitness):  #N is num of population
    prob = np.cumsum(fitness) / np.sum(fitness)
    rnd = np.random.rand(1).item()
    idx = np.sum(prob < rnd)
    if idx >= N:
        print(prob, rnd, N, idx)
        idx = self.N - 1
    mask_=mask_all[idx]
    fitness_=fitness[idx]
    return mask_.copy(),fitness_

def crossover(mask_all,N,fitness,L):
    individual1_mask,_ = roulette(mask_all,N,fitness)
    individual2_mask,_ = roulette(mask_all,N,fitness)
    idx = np.random.randint(0, L, 2)
    start_idx, end_idx = np.min(idx), np.max(idx)
    individual1_mask_copy=individual1_mask.copy()
    individual2_mask_copy=individual2_mask.copy()
    individual1_mask_copy[start_idx: end_idx] = individual2_mask[start_idx: end_idx]
    individual2_mask_copy[start_idx: end_idx] = individual1_mask[start_idx: end_idx]
    return individual1_mask_copy, individual2_mask_copy

def mutation(mask_all,N,fitness,L):
    individual_mask, _ = roulette(mask_all,N,fitness)
    idx = np.random.randint(0, L, 2)
    start_idx, end_idx = np.min(idx), np.max(idx)
    individual_mask_copy=individual_mask.copy()
    individual_mask_copy[start_idx: end_idx]=np.ones(end_idx-start_idx)-individual_mask[start_idx: end_idx]
    return individual_mask_copy
