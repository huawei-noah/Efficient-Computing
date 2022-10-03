import math
import torch
import torch.nn as nn

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# schedule
def Log_UP(epoch, total_epochs):
    K_min, K_max = 1e-3, 1e1 
    # K_min, K_max = 1e-1, 1e1
    Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
    return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / total_epochs * epoch)]).float().cuda()
