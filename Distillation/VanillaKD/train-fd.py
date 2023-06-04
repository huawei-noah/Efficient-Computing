# !/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)

# 2023-Changed for building vanillakd
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""


import argparse
import logging
import logging.handlers
import os
import time
from collections import OrderedDict
from contextlib import suppress
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torchvision import transforms

from losses import Correlation, ReviewKD, RKD
from register import config, register_forward
from timm.data import AugMixDataset, create_dataset, create_loader, FastCollateMixup, resolve_data_config
from timm.loss import *
from timm.models import convert_splitbn_model, create_model, model_parameters, safe_model_name
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
from timm.utils import *
from timm.utils import ApexScaler, NativeScaler
from utils import CheckpointSaverWithLogger, MultiSmoothingMixup, process_feat, setup_default_logging, TimePredictor
import models

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# ---------------------------------------------------------------------------------------
# KD parameters: CC, review, and RKD
parser.add_argument('--teacher', default='beitv2_base_patch16_224', type=str)
parser.add_argument('--teacher-pretrained', default=None, type=str)  # teacher checkpoint path
parser.add_argument('--teacher-resize', default=None, type=int)
parser.add_argument('--student-resize', default=None, type=int)
parser.add_argument('--input-size', default=None, nargs=3, type=int)

# use torch.cuda.empty_cache() to save GPU memory
parser.add_argument('--economic', action='store_true')

# eval every 'eval-interval' epochs before epochs * eval_interval_end
parser.add_argument('--eval-interval', type=int, default=1)
parser.add_argument('--eval-interval-end', type=float, default=0.75)

# one teacher forward, multiple students trained for saving time
# e.g. --kd_loss kd dist --kd_loss_weight 1 2
# then there are 2 settings which are (kd_loss=kd, kd_loss_weight=1) and (kd_loss=dist, kd_loss_weight=2),
# but not 2 * 2 = 4 in total
_nargs_attrs = ['kd_loss', 'kd_loss_weight', 'ori_loss_weight', 'model', 'opt', 'clip_grad', 'lr',
                'weight_decay', 'drop', 'drop_path', 'model_ema_decay', 'smoothing', 'bce_loss']

parser.add_argument('--kd-loss', default=['kd'], type=str, nargs='+')
parser.add_argument('--kd-loss-weight', default=[1.], type=float, nargs='+')
parser.add_argument('--ori-loss-weight', default=[1.], type=float, nargs='+')
parser.add_argument('--model', default=['resnet50'], type=str, nargs='+')
parser.add_argument('--opt', default=['sgd'], type=str, nargs='+')
parser.add_argument('--clip-grad', type=float, default=[None], nargs='+')
parser.add_argument('--lr', default=[0.05], type=float, nargs='+')
parser.add_argument('--weight-decay', type=float, default=[2e-5], nargs='+')
parser.add_argument('--drop', type=float, default=[0.0], nargs='+')
parser.add_argument('--drop-path', type=float, default=[None], nargs='+')
parser.add_argument('--model-ema-decay', type=float, default=[0.9998], nargs='+')
parser.add_argument('--smoothing', type=float, default=[0.1], nargs='+')
parser.add_argument('--bce-loss', type=int, default=[0], nargs='+')  # 0: disable; others: enable
# ---------------------------------------------------------------------------------------

# Dataset parameters
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

# Model parameters
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='Validation batch size override (default: None)')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='torch.jit.script the full model')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")

# Optimizer parameters
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
parser.add_argument('--layer-decay', type=float, default=None,
                    help='layer-wise learning rate decay (default: None)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-repeats', type=float, default=0,
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd-loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    _logger.parent = None
    setup_default_logging(_logger, log_path='train.log')
    args, args_text = _parse_args()

    # parse nargs
    setting_num = 1
    setting_dicts = [dict()]
    for attr in _nargs_attrs:
        value = getattr(args, attr)
        if isinstance(value, list):
            if len(value) == 1:
                for d in setting_dicts:
                    d[attr] = value[0]
            else:
                if setting_num == 1:
                    setting_num = len(value)
                    setting_dicts = [deepcopy(setting_dicts[0]) for _ in range(setting_num)]
                else:  # ensure that args with multiple values have the same length
                    assert setting_num == len(value)
                for i, v in enumerate(value):
                    setting_dicts[i][attr] = v
        else:
            for d in setting_dicts:
                d[attr] = value

    # merge duplicating settings, only for non-nested dict
    setting_dicts = [dict(t) for t in sorted(list({tuple(sorted(d.items())) for d in setting_dicts}))]

    # merge settings with only different 'model_ema_decay'
    model_ema_decay_list = []
    assist_dict = dict()
    for i, d in enumerate(deepcopy(setting_dicts)):
        model_ema_decay_list.append(d['model_ema_decay'])
        del d['model_ema_decay']
        h = hash(tuple(sorted(d.items())))
        if h not in assist_dict:
            assist_dict[h] = [i]
        else:
            assist_dict[h].append(i)

    merged_setting_dict_list = []
    for v in assist_dict.values():
        d = setting_dicts[v[0]]
        d['model_ema_decay'] = tuple([model_ema_decay_list[index] for index in v])
        merged_setting_dict_list.append(d)

    # update
    setting_dicts = merged_setting_dict_list
    setting_num = len(setting_dicts)

    logger_list = [_logger]
    if setting_num > 1:
        if args.local_rank == 0:
            _logger.info(f'there are {setting_num} settings in total. creating individual logger for each setting')

        logger_list = []
        for i in range(setting_num):
            logger = logging.getLogger(f'train-setting-{i}')
            logger.parent = None
            setup_default_logging(logger, log_path=f'train-setting-{i}.log')
            logger_list.append(logger)
            if args.local_rank == 0:
                logger.info(f'settings of index {i}: ' +
                            ', '.join(f"{k}={setting_dicts[i][k]}" for k in setting_dicts[i]))
    else:
        _logger.info(f'settings: ' + ', '.join(f"{k}={setting_dicts[0][k]}" for k in setting_dicts[0]))

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        assert 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.device = int(os.environ['LOCAL_RANK'])

        torch.cuda.set_device(args.device)

        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))

    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)

    if args.fuser:
        set_jit_fuser(args.fuser)

    teacher = create_model(
        args.teacher,
        checkpoint_path=args.teacher_pretrained,
        num_classes=args.num_classes)
    register_forward(teacher)
    teacher = teacher.cuda()
    teacher.eval()

    model_list = []
    for i in range(setting_num):
        model = create_model(
            setting_dicts[i]['model'],
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=setting_dicts[i]['drop'],
            drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
            drop_path_rate=setting_dicts[i]['drop_path'],
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path=args.initial_checkpoint)
        register_forward(model)
        model_list.append(model)

        if args.local_rank == 0:
            logger_list[i].info(f'Model {safe_model_name(setting_dicts[i]["model"])} created, '
                                f'param count:{sum([m.numel() for m in model.parameters()])}')

    # all settings must have the same data config to assure consistent teacher prediction
    if args.num_classes is None:
        assert hasattr(model_list[0],
                       'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model_list[0].num_classes  # FIXME handle model default vs config num_classes more elegantly

    data_config = resolve_data_config(vars(args), model=model_list[0], verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    for i, model in enumerate(model_list):
        # enable split bn (separate bn stats per batch-portion)
        if args.split_bn:
            assert num_aug_splits > 1 or args.resplit
            model = convert_splitbn_model(model, max(num_aug_splits, 2))

        # move model to GPU, enable channels last layout if set
        if setting_dicts[i]['kd_loss'] == 'correlation':
            kd_loss_fn = Correlation(feat_s_channel=config.get_pre_logit_dim(setting_dicts[i]['model']),
                                     feat_t_channel=config.get_pre_logit_dim(args.teacher))
        elif setting_dicts[i]['kd_loss'] == 'review':
            feat_index_s = config.get_used_feature_index(setting_dicts[i]['model'])
            feat_index_t = config.get_used_feature_index(args.teacher)
            in_channels = [config.get_feature_size_by_index(setting_dicts[i]['model'], j)[0] for j in feat_index_s]
            out_channels = [config.get_feature_size_by_index(args.teacher, j)[0] for j in feat_index_t]
            in_channels = in_channels + [config.get_pre_logit_dim(setting_dicts[i]['model'])]
            out_channels = out_channels + [config.get_pre_logit_dim(args.teacher)]

            kd_loss_fn = ReviewKD(feat_index_s, feat_index_t, in_channels, out_channels)
        elif setting_dicts[i]['kd_loss'] == 'rkd':
            kd_loss_fn = RKD()
        else:
            raise NotImplementedError(f'unknown kd loss {args.kd_loss}')

        model.kd_loss_fn = kd_loss_fn
        model.cuda()
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)

        # setup synchronized BatchNorm for distributed training
        if args.distributed and args.sync_bn:
            assert not args.split_bn
            if has_apex and use_amp == 'apex':
                # Apex SyncBN preferred unless native amp is activated
                model = convert_syncbn_model(model)
            else:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if args.local_rank == 0:
                _logger.info(
                    'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                    'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

        if args.torchscript:
            assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
            assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
            model = torch.jit.script(model)

    optimizer_list = []
    # if setting_num == 1:
    #     optimizer_list.append(create_optimizer_v2(model_list[0], **optimizer_kwargs(cfg=args)))
    # else:
    for i in range(setting_num):
        optimizer_list.append(create_optimizer_v2(model_list[i],
                                                  opt=setting_dicts[i]['opt'],
                                                  lr=setting_dicts[i]['lr'],
                                                  weight_decay=setting_dicts[i]['weight_decay'],
                                                  momentum=args.momentum))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        new_model_list = []
        new_optimizer_list = []
        for model, optimizer in zip(model_list, optimizer_list):
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
            new_model_list.append(model)
            new_optimizer_list.append(optimizer)
        model_list = new_model_list
        optimizer_list = new_optimizer_list
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema_list = [(None,) for _ in range(setting_num)]
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        for i in range(setting_num):
            model_emas = []
            for decay in setting_dicts[i]['model_ema_decay']:
                model_ema = ModelEmaV2(model_list[i], decay=decay,
                                       device='cpu' if args.model_ema_force_cpu else None)
                model_emas.append((model_ema, decay))
            model_ema_list[i] = tuple(model_emas)

    # setup distributed training
    if args.distributed:
        new_model_list = []
        for i in range(setting_num):
            if has_apex and use_amp == 'apex':
                # Apex DDP preferred unless native amp is activated
                if args.local_rank == 0:
                    _logger.info("Using NVIDIA APEX DistributedDataParallel.")
                model = ApexDDP(model_list[i], delay_allreduce=True)
            else:
                if args.local_rank == 0:
                    _logger.info("Using native Torch DistributedDataParallel.")
                model = NativeDDP(model_list[i], device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)
            # NOTE: EMA model does not need to be wrapped by DDP
            new_model_list.append(model)
        model_list = new_model_list

    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch

    # setup learning rate schedule and starting epoch
    lr_scheduler_list = []
    for i in range(setting_num):
        lr_scheduler, num_epochs = create_scheduler(args, optimizer_list[i])
        if lr_scheduler is not None and start_epoch > 0:
            lr_scheduler.step(start_epoch)

        lr_scheduler_list.append(lr_scheduler)

        if args.local_rank == 0:
            logger_list[i].info('Scheduled epochs: {}'.format(num_epochs))

    # create the train and eval datasets
    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        repeats=args.epoch_repeats)
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size)

    # setup mixup / cutmix
    # smoothing is implemented in data loader when prefetcher=True,
    # so prefetcher should be turned off when multiple smoothing settings are used
    smoothing_setting_num = len(set([d['smoothing'] for d in setting_dicts]))

    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        if smoothing_setting_num == 1 and args.prefetcher:
            mixup_args = dict(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=setting_dicts[0]['smoothing'], num_classes=args.num_classes)
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            smoothings = tuple([d['smoothing'] for d in setting_dicts])
            mixup_args = dict(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                smoothings=smoothings, num_classes=args.num_classes)
            mixup_fn = MultiSmoothingMixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=(3, 224, 224),
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    # setup loss function
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    train_loss_fn_list = []
    for i in range(setting_num):
        if args.jsd_loss:
            assert num_aug_splits > 1  # JSD only valid with aug splits set
            train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=setting_dicts[i]['smoothing'])
        elif mixup_active:
            # smoothing is handled with mixup target transform which outputs sparse, soft targets
            if setting_dicts[i]['bce_loss']:
                train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
            else:
                train_loss_fn = SoftTargetCrossEntropy()
        elif setting_dicts[i]['smoothing']:
            if setting_dicts[i]['bce_loss']:
                train_loss_fn = BinaryCrossEntropy(smoothing=setting_dicts[i]['smoothing'],
                                                   target_threshold=args.bce_target_thresh)
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(smoothing=setting_dicts[i]['smoothing'])
        else:
            train_loss_fn = nn.CrossEntropyLoss()
        train_loss_fn = train_loss_fn.cuda()
        train_loss_fn_list.append(train_loss_fn)

    teacher_resizer = student_resizer = None
    if args.teacher_resize is not None:
        teacher_resizer = transforms.Resize(args.teacher_resize).cuda()
    if args.student_resize is not None:
        student_resizer = transforms.Resize(args.student_resize).cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    saver_list = [None for _ in range(setting_num)]
    ema_saver_list = [tuple([None for _ in range(len(model_ema_list[i]))]) for i in range(setting_num)]
    for ema, saver in zip(model_ema_list, ema_saver_list):
        assert len(ema) == len(saver)
    output_dir_list = [None for _ in range(setting_num)]
    for i in range(setting_num):
        if args.rank == 0:
            if args.experiment:
                exp_name = args.experiment + f'-setting-{i}'
            else:
                exp_name = '-'.join([
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(setting_dicts[i]['model']),
                    str(data_config['input_size'][-1]),
                    f'-setting-{i}'
                ])
            output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
            output_dir_list[i] = output_dir
            decreasing = True if eval_metric == 'loss' else False
            saver_dir = os.path.join(output_dir, 'checkpoint')
            os.makedirs(saver_dir)
            saver = CheckpointSaverWithLogger(
                logger=logger_list[i], model=model_list[i], optimizer=optimizer_list[i], args=args,
                amp_scaler=loss_scaler, checkpoint_dir=saver_dir, recovery_dir=saver_dir,
                decreasing=decreasing, max_history=args.checkpoint_hist)
            saver_list[i] = saver
            if model_ema_list[i][0] is not None:
                ema_savers = []
                for ema, decay in model_ema_list[i]:
                    ema_saver_dir = os.path.join(output_dir, f'ema{decay}_checkpoint')
                    os.makedirs(ema_saver_dir)
                    ema_saver = CheckpointSaverWithLogger(
                        logger=logger_list[i], model=model_list[i], optimizer=optimizer_list[i], args=args,
                        model_ema=ema, amp_scaler=loss_scaler, checkpoint_dir=ema_saver_dir,
                        recovery_dir=ema_saver_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
                    ema_savers.append(ema_saver)
                ema_saver_list[i] = tuple(ema_savers)
            with open(os.path.join(get_outdir(args.output if args.output else './output/train'),
                                   'args.yaml'), 'w') as f:
                f.write(args_text)

    best_metric_list = [None for _ in range(setting_num)]
    best_epoch_list = [None for _ in range(setting_num)]
    try:
        tp = TimePredictor(num_epochs - start_epoch)
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            ori_loss_weight_list = tuple([d['ori_loss_weight'] for d in setting_dicts])
            kd_loss_weight_list = tuple([d['kd_loss_weight'] for d in setting_dicts])
            clip_grad_list = tuple([d['clip_grad'] for d in setting_dicts])
            train_metrics_list = train_one_epoch(
                epoch, model_list, teacher, loader_train, optimizer_list, train_loss_fn_list, args,
                lr_scheduler_list=lr_scheduler_list, amp_autocast=amp_autocast,
                loss_scaler=loss_scaler, model_ema_list=model_ema_list, mixup_fn=mixup_fn,
                teacher_resizer=teacher_resizer, student_resizer=student_resizer,
                ori_loss_weight_list=ori_loss_weight_list, kd_loss_weight_list=kd_loss_weight_list,
                clip_grad_list=clip_grad_list, logger_list=logger_list)

            for i in range(setting_num):
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    if args.local_rank == 0:
                        logger_list[i].info("Distributing BatchNorm running means and vars")
                    distribute_bn(model_list[i], args.world_size, args.dist_bn == 'reduce')

                is_eval = epoch > int(args.eval_interval_end * args.epochs) or epoch % args.eval_interval == 0
                if is_eval:
                    eval_metrics = validate(model_list[i], loader_eval, validate_loss_fn, args,
                                            logger=logger_list[i], amp_autocast=amp_autocast)

                    if saver_list[i] is not None:
                        # save proper checkpoint with eval metric
                        save_metric = eval_metrics[eval_metric]
                        best_metric, best_epoch = saver_list[i].save_checkpoint(epoch, metric=save_metric)
                        best_metric_list[i] = best_metric
                        best_epoch_list[i] = best_epoch

                    if model_ema_list[i][0] is not None and not args.model_ema_force_cpu:
                        for j, ((ema, decay), saver) in enumerate(zip(model_ema_list[i], ema_saver_list[i])):

                            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                                distribute_bn(ema, args.world_size, args.dist_bn == 'reduce')

                            ema_eval_metrics = validate(ema.module, loader_eval, validate_loss_fn, args,
                                                        logger=logger_list[i], amp_autocast=amp_autocast,
                                                        log_suffix=f' (EMA {decay:.5f})')

                            if saver is not None:
                                # save proper checkpoint with eval metric
                                save_metric = ema_eval_metrics[eval_metric]
                                saver.save_checkpoint(epoch, metric=save_metric)

                    if output_dir_list[i] is not None:
                        update_summary(
                            epoch, train_metrics_list[i], eval_metrics, os.path.join(output_dir_list[i], 'summary.csv'),
                            write_header=best_metric_list[i] is None, log_wandb=args.log_wandb and has_wandb)

                    metrics = eval_metrics[eval_metric]
                else:
                    metrics = None

                if lr_scheduler_list[i] is not None:
                    # step LR for next epoch
                    lr_scheduler_list[i].step(epoch + 1, metrics)

            tp.update()
            if args.rank == 0:
                print(f'Will finish at {tp.get_pred_text()}')
                print(f'Avg running time of latest {len(tp.time_list)} epochs: {np.mean(tp.time_list):.2f}s/ep.')

    except KeyboardInterrupt:
        pass

    for i in range(setting_num):
        if best_metric_list[i] is not None:
            logger_list[i].info('*** Best metric: {0} (epoch {1})'.format(best_metric_list[i], best_epoch_list[i]))

    if args.rank == 0:
        if setting_num == 1:
            os.system(f'mv train.log {args.output}')
        else:
            os.system(f'mv train.log {args.output}')
            for i in range(setting_num):
                os.system(f'mv train-setting-{i}.log {args.output}')


def train_one_epoch(
        epoch, model_list, teacher, loader, optimizer_list, loss_fn_list, args,
        lr_scheduler_list=(None,), amp_autocast=suppress, loss_scaler=None, model_ema_list=(None,), mixup_fn=None,
        teacher_resizer=None, student_resizer=None, ori_loss_weight_list=(None,),
        kd_loss_weight_list=(None,), clip_grad_list=(None,), logger_list=(None,)):
    setting_num = len(model_list)

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order_list = [hasattr(o, 'is_second_order') and o.is_second_order for o in optimizer_list]
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m_list = [AverageMeter() for _ in range(setting_num)]
    losses_ori_m_list = [AverageMeter() for _ in range(setting_num)]
    losses_kd_m_list = [AverageMeter() for _ in range(setting_num)]

    for model in model_list:
        model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, targets = mixup_fn(input, target)
            else:
                targets = [target for _ in range(setting_num)]
        else:
            targets = [target for _ in range(setting_num)]

        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # teacher forward
        with amp_autocast():
            if teacher_resizer is not None:
                teacher_input = teacher_resizer(input)
            else:
                teacher_input = input

            if args.economic:
                torch.cuda.empty_cache()
            with torch.no_grad():
                output_t, feat_t = teacher(teacher_input, requires_feat=True)

            if args.economic:
                torch.cuda.empty_cache()

        # student forward
        for i in range(setting_num):

            if setting_num != 1:  # more than 1 model
                torch.cuda.empty_cache()

            with amp_autocast():
                if student_resizer is not None:
                    student_input = student_resizer(input)
                else:
                    student_input = input

                output, feat = model_list[i](student_input, requires_feat=True)

                loss_ori = ori_loss_weight_list[i] * loss_fn_list[i](output, targets[i])

                try:
                    kd_loss_fn = model_list[i].module.kd_loss_fn
                except AttributeError:
                    kd_loss_fn = model_list[i].kd_loss_fn

                loss_kd = kd_loss_weight_list[i] * kd_loss_fn(z_s=output, z_t=output_t.detach(),
                                                              target=targets[i],
                                                              epoch=epoch,
                                                              feature_student=process_feat(kd_loss_fn, feat),
                                                              feature_teacher=process_feat(kd_loss_fn, feat_t))
                loss = loss_ori + loss_kd

            if not args.distributed:
                losses_m_list[i].update(loss.item(), input.size(0))
                losses_ori_m_list[i].update(loss_ori.item(), input.size(0))
                losses_kd_m_list[i].update(loss_kd.item(), input.size(0))

            optimizer_list[i].zero_grad()
            if loss_scaler is not None:
                loss_scaler(
                    loss, optimizer_list[i],
                    clip_grad=clip_grad_list[i], clip_mode=args.clip_mode,
                    parameters=model_parameters(model_list[i], exclude_head='agc' in args.clip_mode),
                    create_graph=second_order_list[i])
            else:
                loss.backward(create_graph=second_order_list[i])
                if clip_grad_list[i] is not None:
                    dispatch_clip_grad(
                        model_parameters(model_list[i], exclude_head='agc' in args.clip_mode),
                        value=clip_grad_list[i], mode=args.clip_mode)
                optimizer_list[i].step()

            if model_ema_list[i][0] is not None:
                for ema, _ in model_ema_list[i]:
                    ema.update(model_list[i])

            torch.cuda.synchronize()
            batch_time_m.update(time.time() - end)
            if last_batch or batch_idx % args.log_interval == 0:
                lrl = [param_group['lr'] for param_group in optimizer_list[i].param_groups]
                lr = sum(lrl) / len(lrl)

                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data, args.world_size)
                    reduced_loss_ori = reduce_tensor(loss_ori.data, args.world_size)
                    reduced_loss_kd = reduce_tensor(loss_kd.data, args.world_size)
                    losses_m_list[i].update(reduced_loss.item(), input.size(0))
                    losses_ori_m_list[i].update(reduced_loss_ori.item(), input.size(0))
                    losses_kd_m_list[i].update(reduced_loss_kd.item(), input.size(0))

                if args.local_rank == 0:
                    logger_list[i].info(
                        'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'Loss_ori: {loss_ori.val:#.4g} ({loss_ori.avg:#.3g})  '
                        'Loss_kd: {loss_kd.val:#.4g} ({loss_kd.avg:#.3g})  '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'LR: {lr:.3e}  '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            epoch,
                            batch_idx, len(loader),
                            100. * batch_idx / last_idx,
                            loss=losses_m_list[i],
                            loss_ori=losses_ori_m_list[i],
                            loss_kd=losses_kd_m_list[i],
                            batch_time=batch_time_m,
                            rate=input.size(0) * args.world_size / batch_time_m.val,
                            rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                            lr=lr,
                            data_time=data_time_m))

            if setting_num != 1:  # more than 1 model
                torch.cuda.empty_cache()

        num_updates += 1
        for i in range(setting_num):
            if lr_scheduler_list[i] is not None:
                lr_scheduler_list[i].step_update(num_updates=num_updates, metric=losses_m_list[i].avg)

        end = time.time()
        # end for

    for i in range(setting_num):
        if hasattr(optimizer_list[i], 'sync_lookahead'):
            optimizer_list[i].sync_lookahead()

    return [OrderedDict([('loss', losses_m.avg)]) for losses_m in losses_m_list]


def validate(model, loader, loss_fn, args, logger, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    main()
