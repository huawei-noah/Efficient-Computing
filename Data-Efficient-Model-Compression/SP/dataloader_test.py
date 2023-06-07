import astropy.io.fits as pyfits
#import fitsio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import numpy as np
import random
import os
import torch
    
class FRBdataset(Dataset):
    def __init__(self, path='/home/ma-user/work/FRBtest_np_cut/20190911/', data_len=1000, toc=2, datatoc=2, train=True, transform= None):

        valpath = path
        self.vallist = []
        for filename in os.listdir(valpath):
            if 'npy' in filename:
                self.vallist.append(os.path.join(valpath,filename))
        self.train=train
        self.transform = transform
        self.toc = toc
        self.datatoc = datatoc


    def __getitem__(self, index):
        name = self.vallist[index//2]
        img = np.load(name)
        name_split = name.split('_')
        split_time = int(name_split[-1].split('.')[0])
        img = torch.from_numpy(img)
        if index%2 == 0:
            img = img[:,:int(self.toc*img.size(1)/self.datatoc),:]
        else:
            img = img[:,img.shape[1]-int(self.toc*img.size(1)/self.datatoc):,:]
        return img, name, split_time

    def __len__(self):
        return len(self.vallist)*2
