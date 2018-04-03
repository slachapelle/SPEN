"""
Author: Sebastien Lachapelle
"""
import os
import shutil
import cPickle
import zipfile
import gzip
from copy import deepcopy
import csv
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, ToPILImage, RandomResizedCrop

class Denoising(Dataset):

    def __init__(self, path, set_, transform=None, normalize=False):
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

        # TODO: To implement

        """
        if normalize:
            with open(path+'/catsAndDogs_stat.pkl', 'rb') as f:
                mean, std = pickle.load(f)
        """


        self.transform = transform

    def __len__(self):

        # TODO

    def __getitem__(self, idx):

        #if self.transform:
            
        #else:
            
        # TODO

        return sample

def setupData(dataset, local_folder='/Tmp/lachaseb'):
    """
    Args:
        dataset: 'mnist' or ...
        local_folder (string): path to the GPU local disk.
    """
    if not os.path.exists(local_folder+'/data'):
        os.makedirs(local_folder+'/data')

    # TODO: To implement. This function should copy the data to the local_folder. 


def getLoaders(hyper, data_modes=['train','valid','test']):
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
                                normalize=hyper['norm_data'] )

            loader = DataLoader(dataset, batch_size=hyper['bs'], shuffle=hyper['shuffle'])

            dataloaders[data_mode] = loader

    return dataloaders