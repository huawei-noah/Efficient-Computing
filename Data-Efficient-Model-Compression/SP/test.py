import argparse
import os
os.system("pip install astropy")
import torch
import torch.nn as nn
import dataloader_test
import network
from torch.utils.data import DataLoader 

parser = argparse.ArgumentParser(description='ResNets for FRBdata in pytorch')

parser.add_argument('--batch_size', '--bs', default=128, type=int,
                    metavar='BS', help='batch size (default: 2)')
parser.add_argument('--toc', default=1.5, type=float)
parser.add_argument('--path', type=str, default='data_np_cut/')
parser.add_argument('--load', type=str,default='FRBnew2model_dedm3_moredata')
parser.add_argument('--the', default=0.9, type=float)


args, unparsed = parser.parse_known_args()

data_test = dataloader_test.FRBdataset(path = args.path, toc=args.toc, train=False)


net = torch.nn.DataParallel(network.resnet20()).cuda()

data_test_loader = DataLoader(data_test, batch_size=args.batch_size)

if args.load != None:
    net.load_state_dict(torch.load(args.load))
    
f = open("results.txt",'w')    

def test():
    global acc, acc_best
    net.eval()
    with torch.no_grad():
        for i, (images, filename, split_time) in enumerate(data_test_loader):
            images = images.cuda()

            images -= images.mean(dim=2,keepdim=True)
            
            output, reg = net(images, True)
            reg = (reg.squeeze(1)*4000+split_time.cuda()*10000)
            output = output.squeeze(1)
            pred = (torch.sigmoid(output.data)>args.the)
            
            for j in range(images.shape[0]):
                if pred[j] == 1:
                    name = filename[j].split('/')[-1].split('.fits')[0]
                    f.write(name+" "+str(int(reg[j]))+' '+str(torch.sigmoid(output[j].data).item())) 
                    f.write("\n")        

        
def main():
    test()

if __name__ == '__main__':
    main()