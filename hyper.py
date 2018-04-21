import os
import pickle
from collections import OrderedDict

import numpy as np
import torch

def getDefaultHyper(model_class,exp_folder,dataset, resume=True):
    """
    Sets up the basic hparams needed to run an experiment.
    """

    # If the experiment have already been started. (resuming)
    if resume and os.path.isfile(exp_folder+'/hyper.pkl'):

        with open(exp_folder+'/hyper.pkl','rb') as f:
            hyper = pickle.load(f)

        if hyper['use_gpu']:
            print 'Using the GPU'
        else:
            print 'Using the CPU'

        torch.set_default_tensor_type(hyper['float'])

        hyper['nb_of_times_resumed'] += 1
        saveHyper(hyper)
        return hyper

    hyper = OrderedDict()
    hyper['use_gpu'] = torch.cuda.is_available()

    if hyper['use_gpu']:
        print 'Using the GPU'
        # setting the default float and int(long) types
        hyper['float'] = 'torch.cuda.FloatTensor'
        hyper['long'] = 'torch.cuda.LongTensor'
    else:
        print 'Using the CPU'
        # setting the default float and int(long) types
        hyper['float'] = 'torch.FloatTensor'
        hyper['long'] = 'torch.LongTensor'

    torch.set_default_tensor_type(hyper['float'])

    # fixing the seed
    #hyper['seed'] = 23549
    #np.random.seed(hyper['seed'])

    hyper['MODEL_CLASS'] = model_class
    hyper['EXPERIMENT_FOLDER'] = exp_folder
    hyper['DATASET'] = dataset
    hyper['nb_of_times_resumed'] = 0

    if model_class == 'GradientDescentPredictor':
        """Setting standards hparams, will be changed in utils.py"""
        #---OPTIMIZATION---#
        hyper['bs'] = 32
        #lr = '0.001' (default)
        #weight_decay=5*10**(-4) (to try)
        hyper['optimizer'] = 'optim.Adam(model.parameters())'
        hyper['patience'] = 50
        hyper['time_between_save'] = 20 # in minutes
        hyper['time_between_save'] *= 60 # convert in seconds
        hyper['epochs_between_valid'] = 1
        # Maximal number of epochs of training
        hyper['n_epochs_max'] = 1000

        #---ARCHITECTURE---#
        hyper['filter_size'] = 5
        hyper['entropy_decay'] = 1.
        hyper['T'] = 3 # number of gradient steps
        
        #---Initializer architecture---#
        hyper['init_procedure'] = 'ConvInit'
        if hyper['init_procedure'] == 'ConvInit':
            # Set pre_train to None if you want to learn the model from scratch
            # Set it to a folder from an experiment if you want a pre_trained instance.
            hyper['pre_train'] = '/data/milatmp1/lachaseb/SPEN/exp_simple_conv2'
            # Set freeze to True if you don't want to learn the initializer's parameters
            hyper['freeze'] = True

            ##---Initializer hparams---##
            # The follwing hyperparams will be used if hyper['pre_train'] == None
            # else, the hparams from the pretrained architecture will be used
            hyper['active_func'] = 'F.relu'
            hyper['ker_size'] = 5
            hyper['nb_feat_map'] = 64
            hyper['nb_hid_layer'] = 10
            hyper['bn'] = True # batchnorm

        hyper['norm_data'] = False # TODO: not implemented yet.
        hyper['shuffle'] = True # DataLoader argument

    elif model_class == 'GDMomentumPredictor':
        """Setting standards hparams, will be changed in utils.py"""
        #---OPTIMIZATION---#
        hyper['bs'] = 32
        #lr = '0.001' (default)
        #weight_decay=5*10**(-4) (to try)
        hyper['optimizer'] = 'optim.Adam(model.parameters())'
        hyper['patience'] = 50
        hyper['time_between_save'] = 20 # in minutes
        hyper['time_between_save'] *= 60 # convert in seconds
        hyper['epochs_between_valid'] = 1
        # Maximal number of epochs of training
        hyper['n_epochs_max'] = 1000

        #---ARCHITECTURE---#
        hyper['filter_size'] = 7
        hyper['entropy_decay'] = 1.
        hyper['momentum'] = 0.25
        hyper['T'] = 3 # number of gradient steps

        #---Initializer architecture---#
        hyper['init_procedure'] = 'ConvInit'
        if hyper['init_procedure'] == 'ConvInit':
            # Set pre_train to None if you want to learn the model from scratch
            # Set it to a folder from an experiment if you want a pre_trained instance.
            hyper['pre_train'] = '/data/milatmp1/lachaseb/SPEN/exp_simple_conv2'
            # Set freeze to True if you don't want to learn the initializer's parameters
            hyper['freeze'] = True

            ##---Initializer hparams---##
            # The follwing hyperparams will be used if hyper['pre_train'] == None
            # else, the hparams from the pretrained architecture will be used
            hyper['active_func'] = 'F.relu'
            hyper['ker_size'] = 5
            hyper['nb_feat_map'] = 64
            hyper['nb_hid_layer'] = 10
            hyper['bn'] = True # batchnorm

        hyper['norm_data'] = False # TODO: not implemented yet.
        hyper['shuffle'] = True # DataLoader argument

    elif model_class == 'GDLossLogitPredictor':
        """Setting standards hparams, will be changed in utils.py"""
        #---OPTIMIZATION---#
        hyper['bs'] = 32
        #lr = '0.001' (default)
        #weight_decay=5*10**(-4) (to try)
        hyper['optimizer'] = 'optim.Adam(model.parameters())'
        hyper['patience'] = 50
        hyper['time_between_save'] = 20 # in minutes
        hyper['time_between_save'] *= 60 # convert in seconds
        hyper['epochs_between_valid'] = 1
        # Maximal number of epochs of training
        hyper['n_epochs_max'] = 1000

        #---ARCHITECTURE---#
        hyper['filter_size'] = 5
        hyper['entropy_decay'] = 1.
        hyper['momentum'] = 0.25
        hyper['T'] = 3 # number of gradient steps

        #---Initializer architecture---#
        hyper['init_procedure'] = 'ConvInit'
        if hyper['init_procedure'] == 'ConvInit':
            # Set pre_train to None if you want to learn the model from scratch
            # Set it to a folder from an experiment if you want a pre_trained instance.
            hyper['pre_train'] = '/data/milatmp1/lachaseb/SPEN/exp_simple_conv2'
            # Set freeze to True if you don't want to learn the initializer's parameters
            hyper['freeze'] = True

            ##---Initializer hparams---##
            # The follwing hyperparams will be used if hyper['pre_train'] == None
            # else, the hparams from the pretrained architecture will be used
            hyper['active_func'] = 'F.relu'
            hyper['ker_size'] = 5
            hyper['nb_feat_map'] = 64
            hyper['nb_hid_layer'] = 10
            hyper['bn'] = True # batchnorm



        hyper['norm_data'] = False # TODO: not implemented yet.
        hyper['shuffle'] = True # DataLoader argument

    elif model_class == 'ConvInit':
        """Setting standards hparams, will be changed in utils.py"""
        #---OPTIMIZATION---#
        hyper['bs'] = 32
        #lr = '0.001' (default)
        #weight_decay=5*10**(-4) (to try)
        hyper['optimizer'] = 'optim.Adam(model.parameters())'
        hyper['patience'] = 100
        hyper['time_between_save'] = 20 # in minutes
        hyper['time_between_save'] *= 60 # convert in seconds
        hyper['epochs_between_valid'] = 1
        # Maximal number of epochs of training
        hyper['n_epochs_max'] = 1000

        #---ARCHITECTURE---#
        hyper['active_func'] = 'F.relu'
        hyper['ker_size'] = 5
        hyper['nb_feat_map'] = 64
        hyper['nb_hid_layer'] = 10
        hyper['bn'] = True # batchnorm

        hyper['norm_data'] = False # TODO: not implemented yet.
        hyper['shuffle'] = True # DataLoader argument

    saveHyper(hyper)

    return hyper

def getRandomHyper(model_class,exp_folder,dataset, resume=True):
    """
    Sets up the basic hparams needed to run an experiment.
    """

    # If the experiment have already been started. (resuming)
    if resume and os.path.isfile(exp_folder+'/hyper.pkl'):

        with open(exp_folder+'/hyper.pkl','rb') as f:
            hyper = pickle.load(f)

        if hyper['use_gpu']:
            print 'Using the GPU'
        else:
            print 'Using the CPU'

        torch.set_default_tensor_type(hyper['float'])

        hyper['nb_of_times_resumed'] += 1
        saveHyper(hyper)
        return hyper

    hyper = OrderedDict()
    hyper['use_gpu'] = torch.cuda.is_available()

    if hyper['use_gpu']:
        print 'Using the GPU'
        # setting the default float and int(long) types
        hyper['float'] = 'torch.cuda.FloatTensor'
        hyper['long'] = 'torch.cuda.LongTensor'
    else:
        print 'Using the CPU'
        # setting the default float and int(long) types
        hyper['float'] = 'torch.FloatTensor'
        hyper['long'] = 'torch.LongTensor'

    torch.set_default_tensor_type(hyper['float'])

    # fixing the seed
    #hyper['seed'] = 23549
    #np.random.seed(hyper['seed'])

    hyper['MODEL_CLASS'] = model_class
    hyper['EXPERIMENT_FOLDER'] = exp_folder
    hyper['DATASET'] = dataset
    hyper['nb_of_times_resumed'] = 0

    if model_class == 'GradientDescentPredictor':
        # TODO
        hyper

    elif model_class == 'GDMomentumPredictor':
        # TODO
        hyper

    elif model_class == 'GDLossLogitPredictor':
        #TODO
        hyper

    elif model_class == 'ConvInit':

        """Setting standards hparams, will be changed in utils.py"""
        #---OPTIMIZATION---#
        hyper['bs'] = 32
        #lr = '0.001' (default)
        #weight_decay=5*10**(-4) (to try)
        hyper['optimizer'] = 'optim.Adam(model.parameters())'
        hyper['patience'] = 50
        hyper['time_between_save'] = 20 # in minutes
        hyper['time_between_save'] *= 60 # convert in seconds
        hyper['epochs_between_valid'] = 1
        # Maximal number of epochs of training
        hyper['n_epochs_max'] = 1000

        #---ARCHITECTURE---#
        hyper['active_func'] = 'F.relu'
        hyper['ker_size'] = 5
        hyper['nb_feat_map'] = 50
        hyper['nb_hid_layer'] = 10
        hyper['bn'] = True # batchnorm

        hyper['norm_data'] = False # TODO: not implemented yet.
        hyper['shuffle'] = True # DataLoader argument

    saveHyper(hyper)

    return hyper

def saveHyper(hyper):
    """
    Saves hyper.pkl and hyper.txt in hyper['EXPERIMENT_FOLDER']
    """
    with open(hyper['EXPERIMENT_FOLDER']+'/hyper.txt','w') as f:
        f.write('HYPERPARAMS\n')
        for k,v in hyper.iteritems():
            f.write(k+': '+str(v)+'\n')
    with open(hyper['EXPERIMENT_FOLDER']+'/hyper.pkl','wb') as f:
        pickle.dump(hyper,f)