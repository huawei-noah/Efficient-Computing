import os, sys
os.system("pip install torch==1.0.0")
os.system("pip install torchtext==0.3.1")
os.system("pip install torchvision==0.2.1")
os.system("pip install numpy==1.14.5")
os.system("pip install scipy==1.5.0")
os.system("pip install mpmath==1.0.0")
os.system("pip install glob2==0.6")
os.system("pip install matplotlib==1.5.3")

import argparse, os,logging,sys
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vdsr_block import Net
from vdsr_half_block import NetHalf

from dataset import DatasetFromHdf5
import numpy as np
import math,glob
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--val_dataset", default="Set5", type=str, help="val dataset")
parser.add_argument("--model_type", default="origin", type=str, help="model architecture")

parser.add_argument('--tmp_data_dir', default='/cache/dataset/VDSR', help='temp data dir')
parser.add_argument('--tmp_save_dir', default='/cache/save/VDSR/noise', help='temp save dir')
  
args, unparsed = parser.parse_known_args()
print("os.path.dirname(args.tmp_data_dir)", os.path.dirname(args.tmp_data_dir))


target = os.path.join(args.tmp_data_dir)


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.tmp_save_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("args = %s",args)

lossList,psnrList=[],[]



def main():
    print("args:")
    print(args)
    best_psnr,best_epoch=0,0
    #args.seed = random.randint(1, 10000)
    args.seed = 1
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    
    tmp_train_dataPath=os.path.join(args.tmp_data_dir,'data/train.h5')
    tmp_val_dataPath=os.path.join(args.tmp_data_dir,'Set5_mat')
    
    train_set = DatasetFromHdf5(tmp_train_dataPath)
    training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)
    
    val_imglist=glob.glob(tmp_val_dataPath+"/*.*") 
    
    model = Net()
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.MSELoss(reduction='sum')
    criterion = criterion.cuda()

    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
    
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        
        logging.info('current epoch {}, lr {:.5e}'.format(epoch,optimizer.param_groups[0]['lr']))
        train(training_data_loader, optimizer, model, criterion, epoch,)
        plot(epoch,lossList,'mseLoss')
        lr_scheduler.step()
        psnr= validate(model,epoch,val_imglist)
        plot(epoch,psnrList[1:],'psnr')
        is_best=psnr>best_psnr
        if is_best:
            best_epoch=epoch
            best_psnr=psnr
            torch.save(model.state_dict(), os.path.join(args.tmp_save_dir, 'model_vdsr_best.pth'))
        logging.info('best psnr: {}, best epoch: {}'.format(best_psnr,best_epoch))



def train(training_data_loader, optimizer, model, criterion, epoch):
    losses = AverageMeter()
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        input = input.cuda()
        target = target.cuda()
        loss = criterion(model(input_data), target)    
        optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm_(model.parameters(),args.clip) 
        optimizer.step()
            
        losses.update(loss.data.item(),input_data.size(0))

        if iteration%200 == 0:
            logging.info("===> Epoch[{}]({}/{}): Loss: {:.10f} ".format(epoch, iteration, len(training_data_loader), loss.item()))
    global lossList
    lossList.append(losses.avg)
    

def validate(model,epoch,image_list,scales=[2]):
    model.eval()
    for scale in scales:
        avg_psnr_predicted = 0.0
        # avg_psnr_bicubic = 0.0
        count = 0.0
        for image_name in image_list:
            if str(scale) in image_name:
                count += 1
                # print("Processing ", image_name)
                im_gt_y = sio.loadmat(image_name)['im_gt_y']
                im_b_y = sio.loadmat(image_name)['im_b_y']
                           
                im_gt_y = im_gt_y.astype(float)
                im_b_y = im_b_y.astype(float)

                im_input = im_b_y/255.

                im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

                model = model.cuda()
                im_input = im_input.cuda()
                HR = model(im_input)
                HR = HR.cpu()

                im_h_y = HR.data[0].numpy().astype(np.float32)

                im_h_y = im_h_y * 255.
                im_h_y[im_h_y < 0] = 0
                im_h_y[im_h_y > 255.] = 255.
                im_h_y = im_h_y[0,:,:]

                psnr_predicted = PSNR(im_gt_y, im_h_y,shave_border=scale)
                avg_psnr_predicted += psnr_predicted

        logging.info("current epoch: {}, validate psnr: {} ".format(epoch,avg_psnr_predicted/count))
        global psnrList
        psnrList.append(avg_psnr_predicted/count)
    return avg_psnr_predicted/count

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def plot(epoch,valList,label):
    axis=np.linspace(1,epoch,epoch)
    fig=plt.figure()
    plt.title(label)
    plt.plot(axis,valList,label=label)
    plt.xlabel('epoch')
    plt.ylabel(label)
    plt.grid(True)
    plt.savefig(os.path.join(args.tmp_save_dir,'{}.pdf'.format(label)))
    plt.clf()
    plt.close(fig)


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

if __name__ == "__main__":
    main()