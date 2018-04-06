from collections import OrderedDict
import pickle
import cPickle

import numpy as np
import matplotlib
# To avoid displaying the figures
matplotlib.use('Agg')
# style template
#matplotlib.style.use('seaborn')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from utils import *

class PostTrainAnalysis(object):

    def __init__(self, model, dataloaders, folder=None):
        self.model = model
        self.dataloaders = dataloaders
        if folder is None:
            self.folder = model.hyper['EXPERIMENT_FOLDER']
        else:
            self.folder = folder

    def measurePerformance(self,data_modes=['train','valid'], save=True):

        print 'Measuring performance...'
        # Setting the model to evaluation mode
        self.model.train(False)

        # Initialize statistics 
        stat_list = ['loss', 'psnr']

        stats = {}

        to_save = ''

        # Iterate over different datasets
        for data_mode in data_modes:

            # Initialize the dict containing the final stats
            stats[data_mode] = OrderedDict()
            for measure in stat_list:
                stats[data_mode][measure] = 0.

            # Iterate over data.
            for data in self.dataloaders[data_mode]:
                # casting to default float (will also send data to gpu if necessary)
                inputs = data['input'].type(self.model.hyper['float']) 
                labels = data['label'].type(self.model.hyper['float'])

                # Wrap data in Variable
                inputs = Variable(inputs)
                labels = Variable(labels)

                # forward
                # averaged loss
                out = self.model(inputs)
                loss, pred = self.model.getLossPred(out,
                             labels)

                # Accumulate the loss and errors
                # not averaged loss
                stats[data_mode]['loss'] += loss.data[0]*labels.size(0)
                stats[data_mode]['psnr'] += computePSNR(pred.data, labels.data)
                

            n = len(self.dataloaders[data_mode].dataset)
            stats[data_mode]['loss'] /= n
            stats[data_mode]['psnr'] /= n

            # Compute and print the epoch statistics
            to_print = str(data_mode)+': '
            for measure in stat_list:
                to_print += measure+': {:.4f} '.format(stats[data_mode]
                                                           [measure])

            print to_print
            to_save += '\n'+to_print

        visualizePredictions(inputs.data.cpu().numpy(), 
                             pred.data.cpu().numpy(), 
                             labels.data.cpu().numpy(), 
                             self.folder,
                             name = 'best_valid', 
                             nb_ex=5)

        if save:
            # save in .pkl format
            with open(self.folder+'/results.pkl','wb') as f:
                pickle.dump(stats,f)

            # save in .txt format
            with open(self.folder+'/results.txt','w') as f:
                f.write(to_save)

        return to_save

    def graph(self, phases=['train','valid'], which=['loss','psnr']):
        """Given an experiment folder, it produces graphs and saves them.

        Graph_1.png: Training and validation loss
        Graph_2.png: Training and validation PSNR
        """
        print 'Plotting experiment\'s graphs...' 

        with open(self.folder+'/checkpoint.pkl', 'rb') as f:
            checkpoint = torch.load(f)
            stat = checkpoint['stat']

        # The epoch where the model early stopped
        nb_epoch = len(stat['train']['loss'])

        if 'loss' in which:
            # Plotting and saving graph2
            plt.figure(3)
            ymax = np.max([stat[phase]['loss'] for phase in phases])
            plt.grid(True)
            for phase in phases:
            	plt.plot(range(nb_epoch),stat[phase]['loss'], label=phase)
            
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss')
            plt.legend()
            plt.savefig(self.folder+"/Graph_1(Loss).png")
            plt.clf()
        if 'psnr' in which:
            # Plotting and saving graph2
            plt.figure(3)
            ymax = np.max([stat[phase]['psnr'] for phase in phases])
            plt.grid(True)
            for phase in phases:
                plt.plot(range(nb_epoch),stat[phase]['psnr'], label=phase)
            
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.title('PSNR')
            plt.legend()
            plt.savefig(self.folder+"/Graph_2(PSNR).png")
            plt.clf()

def computePSNR(y_pred, y):
    # y shape: (bs,1,H,W)
    
    rmse = torch.mean((y_pred - y)**2,3)
    rmse = torch.mean(rmse, 2)
    rmse = torch.sqrt(torch.mean(rmse,1))
    log10 = torch.log(1./rmse)/torch.log(torch.Tensor([10.]))
    psnr = 20*log10

    return torch.sum(psnr) # size: (1,) unaveraged over minibatch