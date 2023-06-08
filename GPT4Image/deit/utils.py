# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# 2023.6.5-Changed for building GPT4Image
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
import datetime
from collections import defaultdict, deque

import torch
import random
import numpy as np
from torch import nn
import torch.distributed as dist
import torchvision
from torch.nn import functional as F
from timm.models.layers import trunc_normal_


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def seed_training(seed):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


class ImageFolderWithEmbed(torchvision.datasets.ImageFolder):
    def __init__(self, *args, text_emb=None, **kwargs):
        super(ImageFolderWithEmbed, self).__init__(*args, **kwargs)

        self.text_emb_path = text_emb
        assert os.path.exists(self.text_emb_path)
        self.text_embeddings = torch.load(self.text_emb_path).float()
        print(self.text_emb_path, f'loaded. contain {len(self.text_embeddings)} items')
        # assert len(self.text_embeddings) == len(self.samples)

    def __getitem__(self, index):
        img, target = super(ImageFolderWithEmbed, self).__getitem__(index)
        text_emb = self.text_embeddings[index]
        return img, target, text_emb


def concat_all_gather(tensor, rank=None, world_size=1):
    """
    rank=None means no gradient will be retained.
    Specify rank with a int to retain gradient on local rank.
    """
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    if rank is not None:
        tensors_gather[rank] = tensor  # retain gradients on local rank
    output = torch.cat(tensors_gather, dim=0)
    return output


def byol_loss(img_emb, text_emb, args):
    img_emb = F.normalize(img_emb, p=2, dim=-1)
    loss = 2 - 2 * (img_emb * text_emb).sum(dim=-1)
    return loss.mean()


def ctrastive_loss(img_emb, text_emb, args):
    # half of clip_loss ?
    img_emb = F.normalize(img_emb, p=2, dim=-1)
    logits = img_emb @ text_emb.T / args.tau  # temperature
    labels = torch.arange(logits.shape[0], dtype=torch.long, device=img_emb.device)
    return F.cross_entropy(logits, labels)


def clip_loss(img_emb, text_emb, args):
    img_emb = F.normalize(img_emb, p=2, dim=-1)
    all_image_features = concat_all_gather(img_emb, rank=args.rank, world_size=args.world_size)
    all_text_features = concat_all_gather(text_emb, rank=None, world_size=args.world_size)
    logits = (all_image_features @ all_text_features.T) / args.tau
    labels = torch.arange(logits.shape[0], dtype=torch.long, device=img_emb.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.
    return loss


# text_emb is already normalized
text_loss_getter = {
    'byol': byol_loss,
    'ctr': ctrastive_loss,
    'clip': clip_loss
}


def build_mlp(in_dim=512, hidden_dim=2048, out_dim=1024, bn=True, GELU=False):
    layers = [nn.Linear(in_dim, hidden_dim, bias=False if bn else True)]
    if bn:
        layers.append(nn.BatchNorm1d(hidden_dim))
    if GELU:
        layers.append(nn.GELU())
    else:
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)
