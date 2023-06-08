# 2023.6.5-Changed for building GPT4Image
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
import numpy
import random
import time
import os
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import ImageFolder
from timm.models.layers import trunc_normal_


def seed_training(seed):
    if seed is not None:
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def sync_and_time():
    torch.cuda.synchronize()
    return time.time()


def save_checkpoint(model, acc1, epoch, args):
    if args.rank == 0:
        torch.save(
            {'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
             'acc1': acc1,
             'epoch': epoch},
            os.path.join(args.save_dir, 'best_model.pth.tar')
        )


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}, world_size={}'.format(
        args.rank, args.dist_url, args.gpu, args.world_size), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


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


class WrapperModel(nn.Module):
    def __init__(self, model, proj_type='linear', out_dim=1024):
        super().__init__()
        self.in_dim = model.fc.weight.shape[1]
        self.out_dim = out_dim
        self.model = model
        self.proj_type = proj_type.lower()
        if self.proj_type == 'linear':
            self.proj_layer = nn.Linear(self.in_dim, self.out_dim, bias=False)
        elif self.proj_type == 'mlp':
            self.proj_layer = build_mlp(
                in_dim=self.in_dim, hidden_dim=int(self.out_dim * 2),
                out_dim=self.out_dim, bn=True, GELU=False)
        else:
            raise NotImplementedError('Unsupported proj_type.')
        self._init_weight()

    def _init_weight(self):
        if self.proj_type == 'linear':
            trunc_normal_(self.proj_layer.weight, std=1024 ** -0.5)
        elif self.proj_type.lower() == 'mlp':
            trunc_normal_(self.proj_layer[0].weight, std=1024 ** -0.5)
            trunc_normal_(self.proj_layer[-1].weight, std=1024 ** -0.5)

    def forward(self, x, get_feat=False):
        outputs = self.model(x, get_feat=get_feat)
        if isinstance(outputs, tuple):
            feature, pred = outputs
            feature = self.proj_layer(feature)
            return feature, pred
        else:
            return outputs


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


class ImageFolderWithEmbed(ImageFolder):
    def __init__(self, *args, text_emb=None, **kwargs):
        super(ImageFolderWithEmbed, self).__init__(*args, **kwargs)

        self.text_emb_path = text_emb
        assert os.path.exists(self.text_emb_path)
        print('loading', self.text_emb_path)
        self.text_embeddings = torch.load(self.text_emb_path).float()
        # assert len(self.text_embeddings) == len(self.samples)

    def __getitem__(self, index):
        img, target = super(ImageFolderWithEmbed, self).__getitem__(index)
        text_emb = self.text_embeddings[index]
        return img, target, text_emb
