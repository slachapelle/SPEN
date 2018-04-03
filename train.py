import os
import time
from copy import deepcopy
import pickle

import numpy as np
import torch
from torch.autograd import Variable

from post_training import PostTrainAnalysis
import data_iter
class TrainLoop(object):
    def __init__(self, model, optimizer, scheduler=None, phases=['train','valid'], stat_list=['loss']):

        self.model = model
        self.optimizer = optimizer
        #self.scheduler = scheduler
        self.phases = phases
        self.stat_list = stat_list

        if os.path.isfile(self.model.hyper['EXPERIMENT_FOLDER']+\
                                                        '/checkpoint.pkl'):
            with open(self.model.hyper['EXPERIMENT_FOLDER']+\
                                                '/checkpoint.pkl','rb') as f:
                checkpoint = torch.load(f)

            self.epoch = checkpoint['epoch']
            self.time_elapsed = checkpoint['time_elapsed'] # Time passed in train_model
            self.model.load_state_dict(checkpoint['current_model'])
            self.best_loss = checkpoint['best_loss']
            self.best_epoch = checkpoint['best_epoch']
            self.best_model = checkpoint['best_model']
            self.patience = checkpoint['patience']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            #self.scheduler.load_state_dict(checkpoint['schefuler'])
            self.stat = checkpoint['stat']
        else:
            self.epoch = 0
            self.time_elapsed = 0
            self.best_loss = np.inf
            self.best_epoch = 0
            self.patience = model.hyper['patience']
            self.stat = {}
            for phase in self.phases:
            	self.stat[phase] = {}
            	for measure in self.stat_list:
    	            self.stat[phase][measure] = []
        self.train_dataiter = data_iter.FileIter(
        root_dir             = "/Tmp/sai/train",
        flist_name           = "train.lst",
        # cut_off_size         = 400,
        rgb_mean             = (113, 113, 113),
        batch_size           = b_size,
        )
        self.val_dataiter = data_iter.FileIter(
        root_dir             = "/Tmp/sai/train",
        flist_name           = "val.lst",
        rgb_mean             = (113, 113, 113),
        batch_size           = b_size,
        )

    def train_model(self, dataloaders):
        """
        Args:
            dataloaders (dict): Contains train and valid data loaders
            track_iter (bool): If True, will track and count iterations instead of epochs.
            num_iter (int): Number of desired iterations. Use iff track_iter is True.
        """
        self.time_0 = time.time() - self.time_elapsed

        since_save = time.time()

        # Getting the datasets sizes
        n = {}
        for phase in self.phases:
        	n[phase] = len(dataloaders[phase].dataset)

        #-----MAIN LOOP-----#
        num_epochs = self.model.hyper['n_epochs_max']

        while self.epoch <= num_epochs and self.patience > 0:
            print 'After {}/{} epochs'.format(self.epoch, num_epochs)
            print '-' * 10

            # Each epoch has a training and validation phase
            for phase in self.phases:
                if phase == 'train':
                    #if self.scheduler:
                    #        self.scheduler.step()
                    self.model.train(True)  # Set model to training mode
                else:
                    self.model.train(False)  # Set model to evaluate mode

                # Initialize running statistics
                run_stat = {}
                for measure in self.stat_list:
                    run_stat[measure] = 0.0

                # Iterate over data.
                self.train_dataiter.reset()
                for data in self.train_dataiter:

                    # casting to default float (will also send data to gpu if necessary)
                    inputs = data['input'].type(self.model.hyper['float'])
                    labels = data['label'].type(self.model.hyper['float'])

                    # Wrap data in Variable
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    out = self.model(inputs)
                    # averaged loss
                    loss, pred = self.model.getLossPred(out,
                                 labels.type(self.model.hyper['long']))

                    # backward + optimize only if in training phase
                    # Don't update params if epoch == 0.
                    if phase == 'train' and self.epoch>0:
                        loss.backward()
                        self.optimizer.step()

                    # Accumulate the running statistics
                    # (.data since only need Tensors)
                    # accumulate unaveraged loss
                    run_stat['loss'] += loss.data[0]*labels.size(0)

                # Compute and print the epoch statistics
                to_print = str(phase)+': '
                for measure in self.stat_list:
                    self.stat[phase][measure] += [run_stat[measure] / n[phase]]
                    to_print += measure+\
                                ': {:.4f} '.format(self.stat[phase][measure][-1])

                print to_print

                # Validation check
                if phase == 'valid' and\
                   (self.epoch+1) % self.model.hyper['epochs_between_valid'] == 0 \
                   and self.stat['valid']['loss'][-1] < self.best_loss:

                    self.best_loss = self.stat['valid']['loss'][-1]
                    self.best_model = deepcopy(self.model.state_dict())
                    self.best_epoch = self.epoch
                    self.patience = self.model.hyper['patience']

                elif phase == 'valid' and\
                     (self.epoch+1) % self.model.hyper['epochs_between_valid'] == 0:

                    self.patience -= 1

            print 'Best valid loss rate: {:.4f}'.format(self.best_loss)
            print 'Patience: {}'.format(self.patience)
            # Safety saving
            if time.time() - since_save >= self.model.hyper['time_between_save']:
                print 'Safety saving...'
                self.save_checkpoint()
                since_save = time.time()

            self.epoch += 1
            print ''

        self.time_elapsed = time.time() - self.time_0
        m, s = divmod(self.time_elapsed, 60)
        h, m = divmod(m, 60)
        print "Training completed in %dh%02dm%02ds" % (h, m, s)

        #-----SAVING----#
        print 'Final saving...'
        self.save_checkpoint()

        #-----FINAL PRINT----#
        #loading best paramters
        self.model.load_state_dict(self.best_model)

        print 'Final results:'
        # Compute and print the epoch statistics
        to_print = ''
        for phase in self.phases:
            to_print += str(phase)+': '
            for measure in self.stat_list:
                to_print += measure+': {:.4f} '.format(self.stat[phase][measure][-1])
            to_print += '\n'
        print to_print

        # return best model
        return self.model

    def save_checkpoint(self):
        """Take a snapshot of the training loop and save it on the
        EXPERIMENT_FOLDER accessible from any machine."""
        self.time_elapsed = time.time() - self.time_0

        checkpoint = {'epoch': self.epoch,
                   'current_model': self.model.state_dict(),
                   'best_model': self.best_model,
                   'best_loss': self.best_loss,
                   'best_epoch': self.best_epoch,
                   'patience': self.patience,
                   'optimizer': self.optimizer.state_dict(),
                   #'scheduler': self.scheduler.stat_dict(),
                   'stat': self.stat,
                   'time_elapsed': self.time_elapsed}
        torch.save(checkpoint,
                   self.model.hyper['EXPERIMENT_FOLDER']+'/checkpoint.pkl')
