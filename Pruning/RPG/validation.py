import torch
import torch.nn as nn
import argparse


from image_classification.dataloaders import *
from image_classification.training import *
from image_classification.rigl_torch.util import get_W


def add_parser_arguments(parser):
    #model_names = models.resnet_versions.keys()
    #model_configs = models.resnet_configs.keys()

    parser.add_argument("--data", default="../data/imagenet", help="path to dataset")
    parser.add_argument(
        "--data-backend",
        metavar="BACKEND",
        default="dali-cpu",
        # choices=DATA_BACKEND_CHOICES,
        # help="data backend: "
        # + " | ".join(DATA_BACKEND_CHOICES)
        # + " (default: dali-cpu)",
    )

    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="resnet50",
        help="model architecture: (default: resnet50), or mobilenetv2",
    )

    parser.add_argument(
        "--model-config",
        "-c",
        metavar="CONF",
        default="classic",
        #choices=model_configs,
        help="model configs: (default: classic)",
    )

    parser.add_argument(
        "--num-classes",
        metavar="N",
        default=1000,
        type=int,
        help="number of classes in the dataset",
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 8)",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256) per gpu",
    )

    parser.add_argument("--fp16", action="store_true", help="Run model fp16 mode.")

    parser.add_argument('--dataset', default="imagenet", type=str)

    parser.add_argument(
        "--pretrained-weights",
        default="",
        type=str,
        metavar="PATH",
        help="load weights from here",
    )

    parser.add_argument(
        "--memory-format",
        type=str,
        default="nchw",
        choices=["nchw", "nhwc"],
        help="memory layout, nchw or nhwc",
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch ImageNet Inference")
    add_parser_arguments(parser)
    args = parser.parse_args()

    args.short_train = False

    if args.data_backend == "pytorch":
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    elif args.data_backend == "dali-gpu":
        get_train_loader = get_dali_train_loader(dali_cpu=False)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == "dali-cpu":
        get_train_loader = get_dali_train_loader(dali_cpu=True)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == "syntetic":
        get_val_loader = get_syntetic_loader
        get_train_loader = get_syntetic_loader

    memory_format = (
        torch.channels_last if args.memory_format == "nhwc" else torch.contiguous_format
    )
    
    pretrained_weights = torch.load(args.pretrained_weights, map_location='cpu')
    

    model_and_loss = ModelAndLoss(
        (args.arch, args.model_config, args.num_classes),
        nn.CrossEntropyLoss,
        pretrained_weights=pretrained_weights['state_dict'],
        cuda=True,
        fp16=args.fp16,
        memory_format=memory_format,
        args=args,
    )

    # calculate sparsity
    W = get_W(model_and_loss.model)
    total_params = 0
    zero_params = 0
    for w in W:
        total_params += w.numel()
        zero_params += (w==0).sum().item()
    print(f'Model sparsity: {zero_params/total_params}')
    val_loader, val_loader_len = get_val_loader(
        args.data,
        args.batch_size,
        args.num_classes,
        False,
        workers=args.workers,
        fp16=args.fp16,
        memory_format=memory_format,
    )

    prec1, prec5, val_loss, nimg = validate(
    val_loader,
    model_and_loss,
    fp16=None,
    logger=None,
    epoch=0,
    prof=-1,
    register_metrics=False,
    args=args,
    )
    print(prec1, prec5, val_loss, nimg)