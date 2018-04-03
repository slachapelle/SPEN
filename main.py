#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import argparse

import numpy as np

from model import GradienDescentPredictor
from train import TrainLoop
import data 
from hyper import getDefaultHyper
from post_training import *

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
                                  data_modes=['train','valid','test'])
    # optimizer
    optimizer = eval(hyper['optimizer'])
    # train
    train_loop = TrainLoop(model, optimizer)
    model = train_loop.train_model({s:dataloaders[s] for s in ['train','valid']})
    
    # post training analysis
    postTrain = PostTrainAnalysis(model, dataloaders)
    postTrain.measurePerformance()
    postTrain.graph()

    finishing(hyper['EXPERIMENT_FOLDER'])

def finishing(experiment_folder):
    """Let a marker in the experiment_folder.
    See finished methods."""
    with open(experiment_folder+'/finished.txt','w') as f:
        f.write('This experiment is fully completed.')

def finished(exp_folder):
    """Verifies if the "finished.txt" marker is present. 
    See finishing method
    """
    return os.path.isfile(exp_folder+'/finished.txt')

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
        data.setupData(args.dataset)

    if not os.path.exists(args.exp_folder):
    	os.makedirs(args.exp_folder)

    # Run a function from main.py
    f = eval(args.mode)
    f(args)