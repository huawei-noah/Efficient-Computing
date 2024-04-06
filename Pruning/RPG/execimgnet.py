#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.
import os
import argparse
import numpy as np

def get_no():
    import torch
    return torch.cuda.device_count()


def exec(save, density=1., dataset = None,lr= None, model = 'resnet50' , optim =None, dist='uniform', wd = None, momentum = 0.9,save_inter = 0,epochs = 120,batch_size  = 1024,delta=100,alpha = 0.3,amp=False,distributed=True,eval_batch_size=1024,scheduler='multistep',\
    lamb=None,sparsity_thres=None,grad_accumulation_n=None,partial_k = None,lrconfig = None,mixup=0.0,data_backend='pytorch',label_smoothing = 0.0,pretrained=False,warmup=5,lr_decay_epochs=None,iterative_T_end_percent=0.0,T_end_percent=0.8, args=None):
    
    parallel_str = '' 
    num_classes={'cifar10':10,'cifar100':100,'imagenet':1000}[dataset]
    distributed_cmd=f'./multiproc.py --nproc_per_node {get_no()}{parallel_str}' if distributed else ''
    cmd = f'python {distributed_cmd} ./main.py --data {args.data} --arch {model} --dataset {dataset} --num-classes {num_classes} --iterative_T_end_percent {iterative_T_end_percent}'
    cmd += f' --checkpoint-dir {save} --seed 42 --delta {delta} --alpha {alpha} --label-smoothing {label_smoothing}'
    cmd += f' --lr {lr} --sp-distribution {dist} --lr-schedule {scheduler} --mixup {mixup} --data-backend {data_backend} --T-end-percent {T_end_percent}'
    cmd += f' --epochs {epochs} --warmup {warmup} --batch-size {batch_size} --eval-batch-size {eval_batch_size} --momentum {momentum}'
    if density != 1.:
        cmd += f' --dense-allocation {density}'
    if optim is not None:
        cmd += f' --optimizer {optim}'
    if wd is not None:
        cmd += f' --wd {wd}'
    if save_inter > 0:
        cmd += f' --save_inter {save_inter}'
    if amp:
        cmd += f' --amp'
    if lamb is not None:
        cmd += f' --lamb {lamb}'
    if sparsity_thres is not None:
        cmd += f' --sparsity_thres {sparsity_thres}'
    if grad_accumulation_n is not None:
        cmd += f' --grad-accumulation-n {grad_accumulation_n}'
    if partial_k is not None:
        cmd += f' --partial_k {partial_k}'
    if lrconfig is not None:
        cmd += f' --lrconfig {lrconfig}'
    if pretrained:
        cmd += f' --pretrained-weights {pretrained}'
    if lr_decay_epochs is not None:
        cmd += f' --lr-decay-epochs {lr_decay_epochs}'
    print("="*15)
    print(cmd)
    print("-"*15)
    os.system(cmd)
    print("="*15)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data',type=str,default = '../data/imagenet',help = 'datadir')
    parser.add_argument('--dataset',type=str,default = 'imagenet',help = 'benchmark')
    parser.add_argument('--model',type=str,default = 'resnet50',help = 'model')

    parser.add_argument('--optim',type=str,default = None,help = 'optimizer')

    
    parser.add_argument('--lr',type=float,default = 0.4,help = 'lr')
    parser.add_argument('--alpha',type=float,default = 0.3,help = 'alpha')

    parser.add_argument('--distribution',type=str,default = 'magnitude-exponential',help = 'sparsity distribution')
    parser.add_argument('--wd',type=float,default = 1e-4,help = '1e-4 for imgnet')
    parser.add_argument('--sparsity',type=float,default = 0.90)
    parser.add_argument('--save_inter',type=int,default = 0)
    parser.add_argument('--epochs',type=int,default = 100)
    parser.add_argument('--batch_size',type=int,default = 128)
    parser.add_argument('--eval_batch_size',type=int,default = 25)
    parser.add_argument('--delta',type=int,default = 100)
    parser.add_argument('--amp',action='store_true')
    parser.add_argument('--distributed',action='store_true')
    parser.add_argument('--pretrained',type=str,default='')
    parser.add_argument('--scheduler',type=str,default='cosine2')
    parser.add_argument('--momentum',type=float,default=0.9)

    parser.add_argument('--lamb',default=1.0,type=float)
    parser.add_argument('--sparsity_thres',default=0.0,type=float)
    parser.add_argument('--grad-accumulation-n',default=1,type=int,
                            help='number of gradients to accumulate before scoring for rigl')
    parser.add_argument('--partial_k',default=0.2,type=float,help='partial rank constraint')

    parser.add_argument('--lrconfig',default=None,type=str)
    parser.add_argument('--mixup',default=0.0,type=float,help='mixup')
    parser.add_argument('--label_smoothing',default=0.1,type=float,help='label-smoothing.....set to 0.1')
    parser.add_argument('--data_backend',default='dali-cpu',type=str,choices=['pytorch','dali-cpu','dali-gpu','synthetic'])
    
    parser.add_argument('--warmup',default=5,type=int)
    parser.add_argument('--allconfig',default=None,type=str)
    parser.add_argument('--lr-decay-epochs',default=None,type=str)

    parser.add_argument('--scheme',default=None,type=str,help='hyper tuning scheme')
    parser.add_argument('--iterative_T_end_percent',type=float,default=0.9,help='iterative_T_end_percent')
    parser.add_argument('--T_end_percent',type=float,default=0.901,help='T_end_percent')
    
    parser.add_argument('--init_method',default=None,type=str,help='parallel training')
    args, unparsed = parser.parse_known_args()
    ''' Config designation '''
    args.distributed = True # default: distributed data parallel
    argslib = [args]

    for args in argslib:
        print('Please check args: ', args)
        print(f'Sparsity: {args.sparsity}')

        nopt_tag ='ptd' if args.pretrained else ''
        lrconfig_tag = f'conf{args.lrconfig}_' if args.lrconfig is not None else ''
        if lrconfig_tag == '' and args.iterative_T_end_percent != 0.:
            lrconfig_tag = f'conf{args.iterative_T_end_percent}'
        mixup_tag=f'mix{args.mixup}' if args.mixup !=0. else ''
        data_backend_tag='' if args.data_backend == 'pytorch' else f'_{args.data_backend}'
        allconfig_tag = f'_ac{args.allconfig}' if args.allconfig is not None else ''
        savedir = f'./results/{args.model}{args.dataset}{args.distribution}_{args.sparsity}_{args.epochs}/{lrconfig_tag}{nopt_tag}{args.lr}_{args.batch_size}_{args.delta}_{args.alpha}{mixup_tag}/{args.lamb}_{args.sparsity_thres}_{args.grad_accumulation_n}_{args.partial_k}{data_backend_tag}'
        exec(savedir, np.round(1.-args.sparsity, 3), args.dataset, args.lr, args.model, optim=args.optim, dist=args.distribution, wd=args.wd, momentum=args.momentum, save_inter=args.save_inter, epochs=args.epochs, batch_size=args.batch_size,\
            delta=args.delta, alpha=args.alpha, amp=args.amp, distributed=args.distributed, eval_batch_size=args.eval_batch_size, scheduler=args.scheduler,\
                lamb=args.lamb, sparsity_thres=args.sparsity_thres, grad_accumulation_n=args.grad_accumulation_n, partial_k=args.partial_k, lrconfig=args.lrconfig, mixup=args.mixup, data_backend=args.data_backend, \
                label_smoothing=args.label_smoothing, pretrained=args.pretrained, warmup=args.warmup, lr_decay_epochs=args.lr_decay_epochs, iterative_T_end_percent=args.iterative_T_end_percent, T_end_percent=args.T_end_percent, args=args)

        print(f'=====Save Dir: {savedir}')

            

        






                        











