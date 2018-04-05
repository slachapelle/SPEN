#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import argparse

import numpy as np
from torch import optim

from model import GradientDescentPredictor
from train import TrainLoop
import data 
from hyper import getDefaultHyper
from post_training import *
from utils import *

LOCAL_FOLDER = '/Tmp/lachaseb' # GPU interact with this local hard memory (must be local to the GPU)

def experiment(args):
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
    model = train_loop.train_model({s:dataloaders[s] for s in ['train','valid']})
    
    # post training analysis
    postTrain = PostTrainAnalysis(model, dataloaders)
    postTrain.measurePerformance()
    postTrain.graph()

    finishing(hyper['EXPERIMENT_FOLDER'])

def evaluate(args):
    hyper = getDefaultHyper(args.model_class,
                            args.exp_folder,
                            args.dataset,
                            resume=True)

    # model
    model = eval(hyper['MODEL_CLASS']+'(hyper)')

    # data
    dataloaders = data.getLoaders(hyper,
                                  data_modes=['train','valid'])
    print len(dataloaders['train'])
    # optimizer
    optimizer = eval(hyper['optimizer'])

    with open(args.exp_folder+'/checkpoint.pkl', 'rb') as f:
        checkpoint = torch.load(f)

    model.load_state_dict(checkpoint['current_model'])
    
    # post training analysis
    postTrain = PostTrainAnalysis(model, dataloaders)
    postTrain.measurePerformance()
    postTrain.graph()




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
