# 2023.11-Modified some parts in the code
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import numpy as np
import torch
import shutil
import torch.distributed as dist
import time
import sys

def should_backup_checkpoint(args):
    def _sbc(epoch):
        return args.gather_checkpoints #and (epoch < 10 or epoch % 10 == 0)

    return _sbc


def save_checkpoint(
    state,
    is_best,
    filename="checkpoint.pth.tar",
    checkpoint_dir="./checkpoints/",
    backup_filename=None,
    epoch=-1,
    args=None
):
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        filename = os.path.join(checkpoint_dir, filename)
        print("SAVING {}".format(filename))
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(
                filename, os.path.join(checkpoint_dir, "model_best.pth.tar")
            )
            # shutil.copyfile(
            #     filename, os.path.join(checkpoint_dir, "best_{}.pth.tar".format(epoch))
            # )
        if backup_filename is not None:
            shutil.copyfile(filename, os.path.join(checkpoint_dir, backup_filename))



def timed_generator(gen):
    start = time.time()
    for g in gen:
        end = time.time()
        t = end - start
        yield g, t
        start = time.time()


def timed_function(f):
    def _timed_function(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        return ret, time.time() - start

    return _timed_function

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))


    '''
    print(output.data.shape)
    np_data = np.array(output.data.detach().cpu())
    np.set_printoptions(threshold=sys.maxsize)
    print(np_data[0][:20])
    exit()
    #np_pred = np.argmax(np_data,axis=1)


    with open('temp.txt', 'a') as f:
        for _pred in np_pred:
            f.write(str(_pred))
            f.write("\n")
    #input("?")
    '''
    '''
    np_target = np.array(target.cpu())
    #print(np_target)
    #print(np_pred)
    np_res = np.sum(np_target - np_pred == 0)
    prec1 = (np_res + 0.0)*100.0/batch_size
    #print(prec1)
    prec1 = torch.tensor(prec1).cuda()
    '''

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    return rt


def first_n(n, generator):
    for i, d in zip(range(n), generator):
        yield d
