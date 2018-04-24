"""
Author: Sebastien Lachapelle
"""
import os
import shutil
import cPickle
import zipfile
import tarfile
import gzip
from copy import deepcopy
import csv
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Denoising(Dataset):

    def __init__(self, path, set_, transform=None, normalize=False, data_augment=True):
        """
        Args:
            path (string): Path to a folder containing the 
                mnist.pkl.gz file.
            set_ (string): Must be 'train', 'valid', 'test'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert set_ == 'train' or set_ == 'valid' or set_ == 'test'

        print 'Loading Denoising '+set_+' set...'

        if set_ == 'valid':
            flist_name = 'val.lst'
        elif set_ == 'train':
            flist_name = 'train.lst'

        self.path = path+'/train'
        flist_name = os.path.join(self.path, flist_name)
        with open(flist_name, 'r') as f:
            lines = f.readlines()

        self.num_data = len(lines)

        self.x_names = []
        self.y_names = []
        for line in lines:
            _, x_name, y_name = line.strip('\n').split(" ")

            self.x_names += [x_name]
            self.y_names += [y_name]

        self.data_augment = data_augment
        #self.transform = transform

    def _read_img(self, img1_name, label_name):

        data_buffer = np.fromfile(os.path.join(self.path, img1_name), dtype=np.int16)
        noisy = data_buffer[2:]
        noisy = noisy.reshape((data_buffer[0], data_buffer[1]))
        noisy = np.clip(noisy.astype(np.float32)/5000, 0, 1)

        data_buffer = np.fromfile(os.path.join(self.path, label_name), dtype=np.int16)
        clean = data_buffer[2:]
        clean = clean.reshape((data_buffer[0], data_buffer[1]))
        clean = np.clip(clean.astype(np.float32)/5000, 0, 1)
        sz = clean.shape
        if (sz[0] > sz[1]):
            clean = np.swapaxes(clean, 0, 1)
            noisy = np.swapaxes(noisy, 0, 1)

        if self.data_augment:
            if (np.random.uniform(0, 1.0) > 0.5):
                clean = np.fliplr(clean)
                noisy = np.fliplr(noisy)

            left = int(np.random.uniform(0, 214-129))
            right = 128 + left
            up = int(np.random.uniform(0, 160-97))
            down = 96 + up
        else:
            left = 42
            right = 128 + left
            up = 63
            down = 96 + up

        clean = clean[up:down,left:right]
        noisy = noisy[up:down,left:right]

        clean = np.expand_dims(clean, axis=2)  # (1, c, h, w)
        clean = np.swapaxes(clean, 0, 2)
        clean = np.swapaxes(clean, 1, 2)  # (c, h, w)
        clean = np.expand_dims(clean, axis=0)  # (1, c, h, w)


        noisy = np.expand_dims(noisy, axis=2)  # (1, c, h, w)
        noisy = np.swapaxes(noisy, 0, 2)
        noisy = np.swapaxes(noisy, 1, 2)  # (c, h, w)
        noisy = np.expand_dims(noisy, axis=0)  # (1, c, h, w)
        return (noisy.squeeze(0).copy(), clean.squeeze(0).copy())

    def __len__(self):

        return self.num_data

    def __getitem__(self, idx):

        x_name, y_name = self.x_names[idx], self.y_names[idx]
        x, y = self._read_img(x_name, y_name)
        
        return {'input': x, 'label': y}

def setupData(dataset, local_folder='/Tmp/lachaseb'):
    """ Copies the data to your local_folder.
    WARNING: Works only if you're working on MILA's servers.
    Args:
        dataset: 'mnist' or ...
        local_folder (string): path to the GPU local disk.
    """
    if not os.path.exists(local_folder+'/data'):
        os.makedirs(local_folder+'/data')
    if not os.path.exists(local_folder+'/data/proximalnet_data.tar.gz'):
        print 'Copying data to local disk...'
        shutil.copy2('/data/lisa/data/seven_scenes/proximalnet_data.tar.gz', local_folder+'/data')

    if not os.path.exists(local_folder+'/data/train'):
        print 'Extracting data...'
        tar = tarfile.open(local_folder+'/data/proximalnet_data.tar.gz')
        tar.extractall(path=local_folder+'/data')
        tar.close()

def getLoaders(hyper, data_modes=['train','valid','test'], data_augment=True):
    """ Returns dictionnary of desired DataLoader instances.
    Args:
        hyper (dict): See hyper.py
        data_modes (list): List of strings specifying the desired DataLoaders  
    """
    dataloaders = {}
    if hyper['DATASET'] == 'Denoising':
        for data_mode in data_modes:
            dataset = Denoising('/Tmp/lachaseb/data/',
                                data_mode,
                                normalize=hyper['norm_data'],
                                data_augment=data_augment)

            loader = DataLoader(dataset, batch_size=hyper['bs'], shuffle=hyper['shuffle'])

            dataloaders[data_mode] = loader

    return dataloaders