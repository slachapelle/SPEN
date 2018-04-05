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


class PostTrainAnalysis(object):

    def __init__(self, model, dataloaders, folder=None):
        self.model = model
        self.dataloaders = dataloaders
        if folder is None:
            self.folder = model.hyper['EXPERIMENT_FOLDER']
        else:
            self.folder = folder

    def measurePerformance(self,data_modes=['train','valid','test'], save=True):

        print 'Measuring performance...'
        # Setting the model to evaluation mode
        self.model.train(False)

        # Initialize statistics 
        stat_list = ['loss']

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
                inputs = data['input'].type(self.model.hyper['float']) # TODO: adapt this to new DataLoader
                labels = data['label'].type(self.model.hyper['float'])

                # Wrap data in Variable
                inputs = Variable(inputs)
                labels = Variable(labels)

                # forward
                # averaged loss
                out = self.model(inputs)
                loss, pred = self.model.getLossPred(out,
                             labels.type(self.model.hyper['long']))

                # Accumulate the loss and errors
                # not averaged loss
                stats[data_mode]['loss'] += loss.data[0]*labels.size(0)
                

            n = len(self.dataloaders[data_mode].dataset)
            stats[data_mode]['loss'] /= n

            # Compute and print the epoch statistics
            to_print = str(data_mode)+': '
            for measure in stat_list:
                to_print += measure+': {:.4f} '.format(stats[data_mode]
                                                           [measure])

            print to_print
            to_save += '\n'+to_print

        if save:
            # save in .pkl format
            with open(self.folder+'/results.pkl','wb') as f:
                pickle.dump(stats,f)

            # save in .txt format
            with open(self.folder+'/results.txt','w') as f:
                f.write(to_save)
        
        if misclass:
            return highly_misclass_np[1:], uncertain_np[1:], inputs.data.cpu().numpy()

        return to_save

    def graph(self, phases=['train','valid'], which=['loss']):
        """Given an experiment folder, it produces graphs and saves them.

        Graph_1.png: error rate on the validation set and on the training set
        Graph_2.png: Training and validation loss
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
            plt.ylabel('Loss') #TODO: NLL?
            plt.title('Loss')
            plt.legend()
            plt.savefig(self.folder+"/Graph_2(Loss).png")
            plt.clf()