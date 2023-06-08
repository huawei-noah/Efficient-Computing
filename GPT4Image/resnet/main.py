# 2023.6.5-Changed for building GPT4Image
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import argparse
import os
import time
from enum import Enum
import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as tv_models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import model_lib
from utils import seed_training, sync_and_time, save_checkpoint, init_distributed_mode, ImageFolderWithEmbed, WrapperModel, text_loss_getter
from contextlib import nullcontext
scaler = torch.cuda.amp.GradScaler()

def parse_args():
    model_names = sorted(name for name in tv_models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(tv_models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
    #                     help='path to dataset (default: imagenet)')
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='/cache/output/')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--no_scale_lr', action='store_true', default=False)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=30, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--use_amp', action='store_true', default=False)

    parser.add_argument('--text_emb', type=str, default='imagenet_clip_text_emb_0_1281166.pth')
    parser.add_argument('--proj_type', type=str, default='mlp', choices=['mlp', 'linear'])
    parser.add_argument('--loss_type', type=str, default='clip', choices=text_loss_getter.keys())
    parser.add_argument('--lamb', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--sync_bn', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_training(args.seed)
    init_distributed_mode(args)

    if args.no_scale_lr:
        pass
    else:
        args.lr = args.lr * args.world_size * args.batch_size / 256.
        print('lr rescaled to', args.lr)

    print("{}".format(args).replace(', ', ',\n'))

    if args.data is None:
        train_dataset = datasets.FakeData(2560-256+1, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(1000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        val_dataset = datasets.ImageFolder(
            os.path.join(args.data, 'val'),
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
        if args.text_emb is not None:
            train_dataset = ImageFolderWithEmbed(
                os.path.join(args.data, 'train'), transform=train_transform, text_emb=args.text_emb)
        else:
            train_dataset = datasets.ImageFolder(
                os.path.join(args.data, 'train'), transform=train_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=250, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    
    print("=> creating model '{}'".format(args.arch))
    if args.text_emb is not None:
        model = getattr(model_lib, args.arch)()
        print(f'text_emb={args.text_emb}\nproj_type={args.proj_type}\nloss_type={args.loss_type}\nlamb={args.lamb}\ntau={args.tau}')
        _proj_dim = train_dataset.text_embeddings.shape[-1]
        model = WrapperModel(model, proj_type=args.proj_type, out_dim=_proj_dim)
    else:
        model = tv_models.__dict__[args.arch]()
        print('baseline training')

    if args.sync_bn:
        print('using sync_bn')
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    device = torch.device('cuda:{}'.format(args.gpu))
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        args.tb_writer = SummaryWriter(log_dir=args.save_dir)

    best_acc1 = 0
    best_epo = 0
    total_train_time = []
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_start_time = sync_and_time()
        train(train_loader, model, criterion, optimizer, epoch, device, args)
        epoch_time = sync_and_time() - train_start_time
        total_train_time.append(epoch_time)
        print('time of this epoch :', str(datetime.timedelta(seconds=int(epoch_time))))

        acc1, val_loss = validate(val_loader, model, criterion, args)
        if args.rank == 0:
            args.tb_writer.add_scalar('Eval/ce_loss', val_loss, epoch)
            args.tb_writer.add_scalar('Eval/acc1', acc1, epoch)
            args.tb_writer.add_scalar('Train/lr', optimizer.param_groups[-1]['lr'], epoch)

        scheduler.step()

        # remember best acc1 and save checkpoint
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
            best_epo = epoch
            save_checkpoint(model, acc1, epoch, args)

    total_train_time = int(sum(total_train_time))
    total_train_time_str = str(datetime.timedelta(seconds=total_train_time))
    print(f'Training time {total_train_time_str}')
    print(f'best_acc1 = {best_acc1:.4f} @ {best_epo}epoch')
    if args.rank == 0:
        args.tb_writer.close()


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}/{args.epochs}]")

    text_loss_func = text_loss_getter[args.loss_type]
    model.train()
    end = time.time()

    amp_cm = torch.cuda.amp.autocast() if args.use_amp else nullcontext()

    for i, loader_out in enumerate(train_loader):
        data_time.update(time.time() - end)
        global_step = i + epoch * len(train_loader)
        if args.text_emb is not None:
            images, target, text_emb = loader_out
            text_emb = text_emb.to(device, non_blocking=True)
        else:
            images, target = loader_out
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with amp_cm:
            if args.text_emb is not None:
                feat, output = model(images, get_feat=True)
                text_loss = text_loss_func(feat, text_emb, args)
                ce_loss = criterion(output, target)
                loss = ce_loss + args.lamb * text_loss
                if args.rank == 0:
                    args.tb_writer.add_scalar('Train/ce_loss', ce_loss.item(), global_step)
                    args.tb_writer.add_scalar('Train/text_loss', text_loss.item(), global_step)
            else:
                output = model(images)
                loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
    
    if args.distributed:
        top1.all_reduce()
    if args.rank == 0:
        args.tb_writer.add_scalar('Train/acc1', top1.avg, epoch)


def validate(val_loader, model, criterion, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                with torch.cuda.amp.autocast():
                    output = model(images)
                    loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()
        losses.all_reduce()
    # if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
    #     aux_val_dataset = Subset(val_loader.dataset,
    #                              range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
    #     aux_val_loader = torch.utils.data.DataLoader(
    #         aux_val_dataset, batch_size=args.batch_size, shuffle=False,
    #         num_workers=args.workers, pin_memory=True)
    #     run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()
    return top1.avg, losses.avg


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
