import astropy.io.fits as pyfits
#import fitsio

from torchvision import transforms, utils
import torch.nn.functional as F
import numpy as np
import random
import os
import torch
import argparse
import math

parser = argparse.ArgumentParser(description='ResNets for FRBdata in pytorch')

parser.add_argument('--path', default='data/', type=str, metavar='N')
parser.add_argument('--output', default='data_np_cut/', type=str, metavar='N')
parser.add_argument('--dm', default=564.4, type=float, metavar='N')
args, unparsed = parser.parse_known_args() 

def roll_by_gather(mat, shifts: torch.LongTensor):
    # assumes 2D array
    n, n_rows, n_cols = mat.shape

    arange1 = torch.arange(n_rows).view((1, n_rows, 1)).repeat((n, 1, n_cols)).cuda()
    arange2 = (arange1 - shifts.unsqueeze(1)) % n_rows
    return torch.gather(mat, 1, arange2)

chan_freqs = torch.linspace(1000,1500,1024).cuda()

def dedisperse(data, dm):
    data = data.squeeze(1)
    _, nt, nf = data.shape
    delay_time = (4148808.0* dm* (1. / (chan_freqs[-1]) ** 2 - 1. / (chan_freqs.unsqueeze(0)) ** 2)/ 1000.)
    delay_bins = torch.round(delay_time / 0.000786432)
    return roll_by_gather(data,delay_bins.long())


path = args.path
directory = args.output
import os
if not os.path.exists(directory):
    os.makedirs(directory)
        
for filename in os.listdir(path):
    if not filename.endswith('fits'):
        continue
    hdulist = pyfits.open(os.path.join(path,filename))
    hdu1 = hdulist[1]
    data1 = hdu1.data['data']
    tsamp = hdu1.header['TBIN']
    fchannel = hdulist['SUBINT'].data[0]['DAT_FREQ']
    fch1 = fchannel[0]
    startf=1000
    endf=1500
    df = hdu1.header['CHAN_BW']
    startfreq = int((startf - fch1)/df)
    endfreq = int((endf - fch1)/df)
    numChannel = int(endfreq - startfreq)
    #data1 = torch.from_numpy(data1)
    dataPol0 = data1[:,:,0,:,:].squeeze().reshape((-1,numChannel))
    dataPol1 = data1[:,:,1,:,:].squeeze().reshape((-1,numChannel))
    Scale = np.mean(dataPol0)/np.mean(dataPol1) ###new
    dataI = (dataPol0+Scale*dataPol1)/2.
    dataI = torch.from_numpy(dataI).cuda()
    dataI = F.avg_pool2d(dataI.unsqueeze(0).float(), [8,4])
    data_dedms = dedisperse(dataI, args.dm).cpu()
    for i in range(0,int(dataI.shape[1]/1250-1)):
        data_divide_dedms = data_dedms[:,i*1250:(i+2)*1250,:]
        save_name = filename+"_"+str(i)+".npy"
        print(save_name)
        np.save(directory+save_name,data_divide_dedms.numpy())
