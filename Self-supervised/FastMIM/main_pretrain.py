# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
#
# 2022.12.14-Changed for building FastMIM
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch

import warnings
warnings.filterwarnings("ignore")  # Del ImageNet Warnings
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    has_apex = True
except ImportError:
    has_apex = False     
from util.misc import ApexScaler


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--apex_amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--opt_level', default='O1', help='Opt level for Apex AMP mixed precision')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    # Decoder parameters
    parser.add_argument('--decoder_embed_dim', default=256, type=int, help='Decoder dim')
    parser.add_argument('--decoder_depth', default=4, type=int, help='Decoder depth')

    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--mim_pt_size', nargs='+', type=int, default=None, help='dynamic mim input size')
    parser.add_argument('--mim_pt_bs', nargs='+', type=int, default=None, help='dynamic mim batch size')
    parser.add_argument('--mim_pt_accum', nargs='+', type=int, default=None, help='dynamic mim batch size')
    parser.add_argument('--mim_pt_step', nargs='+', type=int, default=None, help='dynamic mim input step')
    parser.add_argument('--window_size', default=7, type=int, help='swin window size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mask_type', default='random',
                        help='Masking type: random, grid_random, grid_normal')
    parser.add_argument('--block_size', default=16, type=int, help='Block wise masking size.')
    parser.add_argument('--mim_loss', default='l2', help='MIM forward loss.')
    parser.add_argument('--rrc_scale', nargs='+', type=float, default=(0.2, 1.0), help='RandomResizedCrop ratio.')
    parser.add_argument('--adamw_betas', nargs='+', type=float, default=(0.9, 0.95))
    parser.add_argument('--color_jitter', default=None, type=float)

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--bmin_lr', type=float, default=0., metavar='LR',
                        help='base min learning rate: min_lr = bmin_lr * total_batch_size / 256')
    parser.add_argument('--decay_epoch', type=int, default=None, metavar='LR',
                        help='learning rate (absolute lr)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_new_sched', action='store_true', help='resume with new schedule')
    parser.set_defaults(resume_new_sched=False)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    tfs = [
        transforms.RandomResizedCrop(args.input_size, scale=tuple(args.rrc_scale), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
    ]
    if args.color_jitter is not None:
        if isinstance(args.color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(args.color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            args.color_jitter = (float(args.color_jitter),) * 3
        tfs += [transforms.ColorJitter(*args.color_jitter)]
    tfs += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    transform_train = transforms.Compose(tfs)
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    
    if args.mim_pt_bs:
        data_loader_trains = []
        for bs in args.mim_pt_bs:
            data_loader_trains.append(
                torch.utils.data.DataLoader(
                    dataset_train, sampler=sampler_train, batch_size=bs,
                    num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
            )
        data_loader_train = data_loader_trains[0]
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
    
    # define the model
    model = models_mae.__dict__[args.model](
        norm_pix_loss = args.norm_pix_loss,
        img_size = args.input_size,
        block_size = args.block_size,
        window_size = args.window_size,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        mim_loss = args.mim_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    args.min_lr = args.bmin_lr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        if args.apex_amp and has_apex:
            print("Using APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=tuple(args.adamw_betas))
    print(optimizer)
    
    if args.apex_amp and has_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
        loss_scaler = ApexScaler()
        print('Using NVIDIA APEX AMP. Training in ApexScaler().')
    else:
        loss_scaler = NativeScaler()
        print('APEX AMP not enabled. Training in NativeScaler().')
    
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.mim_pt_size:
            assert len(args.mim_pt_size) == len(args.mim_pt_step)
            idx = None
            for i, size in enumerate(args.mim_pt_step):
                if epoch >= size:
                    idx = i
            data_loader_train = data_loader_trains[idx]
            model.module.mim_pt_size = args.mim_pt_size[idx]
            data_loader_train.dataset.transform.transforms[0].size = (args.mim_pt_size[idx], args.mim_pt_size[idx])
            args.accum_iter = args.mim_pt_accum[idx]
            eff_batch_size = data_loader_train.batch_size * args.accum_iter * misc.get_world_size()
            args.lr = args.blr * eff_batch_size / 256
            args.min_lr = args.bmin_lr * eff_batch_size / 256
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr
            print('bs:{}; accum_iter:{}; transforms:{};'.format(
                data_loader_train.batch_size, args.accum_iter, data_loader_train.dataset.transform.transforms[0]))
        
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
