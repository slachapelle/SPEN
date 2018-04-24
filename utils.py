import os

import numpy as np
import matplotlib
# To avoid displaying the figures
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def visualizePredictions(x, y_pred, y_gt, folder, name='viz_pred', nb_ex=5, seed=123):
    """Produces visualization of the predictions"""

    np.random.seed(seed)
    rnd_idx = np.array(range(x.shape[0]))
    np.random.shuffle(rnd_idx)

    rnd_idx = rnd_idx[:nb_ex]

    fig = plt.figure(figsize=(7,10))
    plt.title('Noisy image, denoised image and ground truth image')
    plt.axis('off')
    for i in range(nb_ex):

        ax1 = fig.add_subplot(nb_ex,3,3*i+1)
        ax1.imshow(x[rnd_idx[i]].reshape(-1, y_gt.shape[-1]))
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

        ax1 = fig.add_subplot(nb_ex,3,3*i+2)
        ax1.imshow(y_pred[rnd_idx[i]].reshape(-1, y_gt.shape[-1]))
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

        ax1 = fig.add_subplot(nb_ex,3,3*i+3)
        ax1.imshow(y_gt[rnd_idx[i]].reshape(-1, y_pred.shape[-1]))
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(folder+'/'+name+'.jpg')

    plt.close(fig)

def vizDenoise(x,y_seq,y,folder):

    fig = plt.figure(figsize=(12,3))
    plt.title('Noisy image, denoising process and ground truth image')
    plt.axis('off')


    print x.shape, y_seq.shape, y.shape

    ax1 = fig.add_subplot(1,2+y_seq.shape[0],1)
    ax1.imshow(x.reshape(-1, y.shape[-1]))
    ax1.axis('off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    for i in range(y_seq.shape[0]):

        ax1 = fig.add_subplot(1,2+y_seq.shape[0],i+2)
        ax1.imshow(y_seq[i].reshape(-1, y.shape[-1]))
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    ax1 = fig.add_subplot(1,2+y_seq.shape[0],i+3)
    ax1.imshow(y.reshape(-1, y.shape[-1]))
    ax1.axis('off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(folder+'/vizDenoise.jpg')

    plt.close(fig)


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