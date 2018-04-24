#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import argparse

import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

from model import *
from train import TrainLoop
import data 
from hyper import getDefaultHyper
from post_training import *
from utils import *

LOCAL_FOLDER = '/Tmp/lachaseb' # GPU interact with this local hard memory (must be local to the GPU)

def experiment(args, reset_epoch=False):
    """
    Execute a full experiment.
    """
    # Don't execute if the experiment is completed.
    if finished(args.exp_folder):
        print 'Experiment already completed.'
        return None

    hyper = getDefaultHyper(args.model_class,
                            args.exp_folder,
                            args.dataset,
                            resume=True)

    print args.exp_folder

    # model
    model = eval(hyper['MODEL_CLASS']+'(hyper)')

    # data
    dataloaders = data.getLoaders(hyper,
                                  data_modes=['train','valid'])
    print 'train set has ', len(dataloaders['train'].dataset), 'examples'
    print 'valid set has ', len(dataloaders['valid'].dataset), 'examples'
    # optimizer
    optimizer = eval(hyper['optimizer'])

    # train
    train_loop = TrainLoop(model, optimizer, phases=['train','valid'])
    if reset_epoch:
        train_loop.epoch = 1
    model = train_loop.train_model({s:dataloaders[s] for s in ['train','valid']})
    
    # post training analysis
    postTrain = PostTrainAnalysis(model, dataloaders)
    postTrain.measurePerformance()

    which = ['loss','psnr']
    if 'sigma_2' in train_loop.stat_list:
        which += ['sigma_2']
    postTrain.graph(which=which)

    finishing(hyper['EXPERIMENT_FOLDER'])

def restartExperiment(args):
    experiment(args, reset_epoch=True )

def seedExperiment(args):
    """ Produce a hyper.pkl with the desired hparams """
    getDefaultHyper(args.model_class,
                    args.exp_folder,
                    args.dataset,
                    resume=True)

def evaluate(args):
    hyper = getDefaultHyper(args.model_class,
                            args.exp_folder,
                            args.dataset,
                            resume=True)
    hyper['shuffle'] = False

    # model
    model = eval(hyper['MODEL_CLASS']+'(hyper)')

    # data
    dataloaders = data.getLoaders(hyper,
                                  data_modes=['train','valid'],
                                  data_augment=False)

    print 'train set has ', len(dataloaders['train'].dataset), 'examples'
    print 'valid set has ', len(dataloaders['valid'].dataset), 'examples'
    # optimizer
    optimizer = eval(hyper['optimizer'])

    with open(args.exp_folder+'/checkpoint.pkl', 'rb') as f:
        checkpoint = torch.load(f)

    model.load_state_dict(checkpoint['best_model'])
    
    # post training analysis
    postTrain = PostTrainAnalysis(model, dataloaders)
    postTrain.measurePerformance()

    which = ['loss','psnr']
    if isinstance(model, GradientDescentPredictor):
        which += ['sigma_2']
    postTrain.graph(which=which)

    for dat in dataloaders['valid']:
        x = Variable(dat['input'].cuda()) 
        y = Variable(dat['label'].cuda())
        break

    y_pred = model(x)

    if isinstance(model, ConvInit):
        y_pred = F.sigmoid(y_pred)
    else:
        # 25th appears to be interesting visually...
        vizDenoise(x[25].data.cpu().numpy(),
                   y_pred[:,25,:,:,:].data.cpu().numpy(),
                   y[25].data.cpu().numpy(),
                   hyper['EXPERIMENT_FOLDER'])
        y_pred = y_pred[-1]

    visualizePredictions(x.data.cpu().numpy(), 
                         y_pred.data.cpu().numpy(),
                         y.data.cpu().numpy(),
                         hyper['EXPERIMENT_FOLDER'],
                         name='viz_Final', 
                         nb_ex=5, 
                         seed=56063)

def showData(args):

    hyper = {}
    hyper['shuffle'] = False
    hyper['norm_data'] = False
    hyper['bs'] = 32
    hyper['DATASET'] = args.dataset
    
    # data
    dataloaders = data.getLoaders(hyper,
                                  data_modes=['train','valid'],
                                  data_augment=False)

    print dataloaders

    k = 0 
    for dat in dataloaders['valid']:
        x = dat['input'].cpu().numpy()
        y = dat['label'].cpu().numpy()
        if k == 3:
            break
        else:
            k += 1

    fig = plt.figure(figsize=(10,10))
    plt.title('Noisy image, denoised image and ground truth image')
    plt.axis('off')
    for i in range(3):
        for j in range(5):

            ax1 = fig.add_subplot(5,3,i+j*3+1)
            ax1.imshow(x[3*i + 5*j].reshape(-1, y.shape[-1]))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('/u/lachaseb/SPEN/images.jpg')

    plt.close(fig)

def trainVStest(args):

    hyper = getDefaultHyper(args.model_class,
                            args.exp_folder,
                            args.dataset,
                            resume=True)

    # data
    dataloaders = data.getLoaders(hyper,
                                  data_modes=['train','valid'])
    train_list = dataloaders['train'].dataset.x_names
    train_set = set(train_list)
    valid_list = dataloaders['valid'].dataset.x_names
    valid_set = set(valid_list)

    print 'size train list: ', len(train_list)
    print 'size train set: ', len(train_set)
    print 'size valid list: ', len(valid_list)
    print 'size valid set: ', len(valid_set)
    print 'size intersection: ', len(train_set.intersection(valid_set))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Running mode for the program
    parser.add_argument('mode', help="Enter the name of the desired function from main.py")
    # Model class to use
    parser.add_argument('model_class', help="Enter the name of the desired Class from model.py")
    # The experiment folder accessible from any machine.
    parser.add_argument('exp_folder', help="Choose an experiment folder accessible from any machine.")

    parser.add_argument('dataset', help="Only choice: 'Denoising'")
    args = parser.parse_args()

    if args.mode not in []: # List of modes which doesn't require data.
        data.setupData(args.dataset, local_folder=LOCAL_FOLDER)

    if not os.path.exists(args.exp_folder):
        os.makedirs(args.exp_folder)

    # Run a function from main.py
    f = eval(args.mode)
    f(args)
