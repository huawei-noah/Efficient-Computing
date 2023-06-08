# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# 2023.6.5-Changed for building GPT4Image
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    i=0
    text_loss_func = utils.text_loss_getter[args.loss_type]

    for loader_out in metric_logger.log_every(data_loader, print_freq, header):
        if args.text_emb is not None:
            samples, targets, text_emb = loader_out
            text_emb = text_emb.to(device, non_blocking=True)
        else:
            samples, targets = loader_out
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
                    
        with torch.cuda.amp.autocast():
            feat, outputs = model(samples, get_feat=True)
            ce_loss = criterion(samples, outputs, targets)
            if args.text_emb is not None:
                text_loss = text_loss_func(feat, text_emb, args)
                loss = ce_loss + args.lamb * text_loss
            else:
                loss = ce_loss

        optimizer.zero_grad()

        loss_value = loss.item()
        if math.isfinite(loss_value):
            if args.text_emb is not None:
                metric_logger.update(ce_loss=ce_loss.item())
                metric_logger.update(text_loss=text_loss.item())
            else:
                metric_logger.update(loss=loss_value)
        else:
            print(f"Loss is {loss_value}, stopping training. This is rank {args.rank}.", force=True)
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
