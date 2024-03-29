import pickle 
import string

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F

from derivativeGlobalE import ComputeDerivativeGlobalE_wrt_y

class DerivativeGlobalE_wrt_y(nn.Module):
    """Encapsulates the operation of taking the derivative of E_global wrt to y, 
    with all the learnable Parameters envolved.
    """

    def __init__(self,hyper):
        super(DerivativeGlobalE_wrt_y,self).__init__()

        self.hyper = hyper
        filter_size = hyper['filter_size']
        # supposing input image are greyscale
        self.filter1 = nn.Parameter(torch.Tensor(32, 1, filter_size, filter_size))
        self.bias1 = nn.Parameter(torch.Tensor(32))
        self.filter2 = nn.Parameter(torch.Tensor(32, 32, filter_size, filter_size))
        self.bias2 = nn.Parameter(torch.Tensor(32))
        self.filter3 = nn.Parameter(torch.Tensor(1, 32, 1, 1))
        self.bias3 = nn.Parameter(torch.Tensor(1))

    def initParams(self):

        nn.init.xavier_uniform(self.filter1)
        nn.init.xavier_uniform(self.filter2)
        nn.init.xavier_uniform(self.filter3)

    def forward(self, y):

        return ComputeDerivativeGlobalE_wrt_y.apply(y,
                                                    self.filter1,
                                                    self.bias1,
                                                    self.filter2,
                                                    self.bias2,
                                                    self.filter3,
                                                    self.bias3)

class DerivativeE_wrt_y(nn.Module):
    """Encapsulates the operation of taking the derivative of E wrt to y
    """

    def __init__(self, hyper):
        super(DerivativeE_wrt_y, self).__init__()

        self.hyper = hyper
        self.dGlobalE_dy = DerivativeGlobalE_wrt_y(hyper)
        self.sigma_2 = nn.Parameter(torch.Tensor([25.]))

    def initParams(self):

        # TODO: init self.sigma_2 ??

        for child in self.children():
            child.initParams()

    def forward(self,F_x, y):

        # The analytical derivative of L2_norm(y - x)**2 
        dLocalE_dy = 2*(y - F_x)

        # The analytical derivative of the entropy term See Belanger, Yang and McAllum 2017
        dEntropy_dy = torch.log(1 - y + 1e-16) - torch.log(y + 1e-16)
        
        dE_dy = dLocalE_dy - 2 * F.softplus(self.sigma_2) * self.dGlobalE_dy(y) + self.hyper['entropy_decay']*dEntropy_dy

        
        return dE_dy

class GradientDescentPredictor(nn.Module):
    """Encapsulates the whole end-to-end process to compute the loss and the prediction
    Energy minimization: GD on logits. See Belanger, Yang and McAllum 2017
    Loss: MSE on y
    """

    def __init__(self,hyper):
        super(GradientDescentPredictor, self).__init__()
        self.hyper = hyper
        self.T = hyper['T']

        self.dE_dy = DerivativeE_wrt_y(hyper)
        if hyper['init_procedure'] == 'Identity':
            self.init = Identity()
        if hyper['init_procedure'] == 'ConvInit':
            
            if hyper['pre_train'] is not None:

                with open(hyper['pre_train']+'/hyper.pkl','rb') as f:
                    hyper_init = pickle.load(f)
                self.init = ConvInit(hyper_init)

                with open(hyper['pre_train']+'/checkpoint.pkl','rb') as f:
                    best_model = torch.load(f)['best_model']
                self.init.load_state_dict(best_model)

                if hyper['freeze']:
                    for param in self.init.parameters():
                        param.require_grad = False

            else:
                self.init = ConvInit(hyper)

        # learnable learning rate.
        self.lr = nn.Parameter(torch.Tensor([1.]*self.T))

        # computing the weights used in the loss weighting
        # TODO: Set the first weight (t=0) to 0 ?
        self.weight = Variable(torch.Tensor([ 1.0 / (self.T - t + 1) for t in xrange(self.T+1)]))
        self.weight = self.weight.view(-1,1,1,1,1)

        self.initParams()

    def initParams(self):

        for child in self.children():
            child.initParams()

    def forward(self, x):
        # shape of x: (bs, 1, H, W)

        bs = x.size(0)

        # The initialization procedure returns an image with pixels in [0,1],
        # so it makes sense to use it directly as our initial y value.
        F_x = self.init(x)
        if isinstance(self.init, ConvInit):
            F_x = F.sigmoid(F_x)

        y_tab = F_x.unsqueeze(0)
        
        # Since we're doing gradient descent on the logit version of y (pre-sigmoid),
        # we need to find the corresponding logit that yield our initial y.
        # To do so, we simply apply the inverse of the sigmoid function to y.
        logit = (torch.log(y_tab+ 1e-16) - torch.log(1-y_tab+ 1e-16)).squeeze(0)
        
        for t in xrange(1,self.T+1):

            # See Belanger, Yang and McAllum 2017 for this trick 
            logit = logit.clone() - F.softplus(self.lr[t-1]) * y_tab[t-1] * (1 - y_tab[t-1]) * self.dE_dy(F_x, y_tab[t-1])

            y_tab = torch.cat([y_tab, F.sigmoid(logit).unsqueeze(0)], 0)

        return y_tab # shape: (T+1, bs, 1 , H, W)

    def getLossPred(self, y_tab, y_gt):
        # y_tab shape: (T+1, bs, 1, H, W)
        # y_gt shape: (bs,1,H,W)

        #Use all weighted iterations in the loss
        loss = torch.sum((y_tab - y_gt.unsqueeze(0))**2 * self.weight)

        return loss, y_tab[-1]

"""
loss = F.binary_cross_entropy(y_tab,
                              y_gt.unsqueeze(0).expand(y_tab.size(0),-1,-1,-1,-1),
                              weight=self.weight.view(-1,1,1,1,1),
                              size_average=False)
"""

class GDLossLogitPredictor(nn.Module):
    """Encapsulates the whole end-to-end process to compute the loss and the prediction
    Energy minimization: GD on logits. See Belanger, Yang and McAllum 2017
    Loss: MSE on logits
    """

    def __init__(self,hyper):
        super(GDLossLogitPredictor, self).__init__()
        self.hyper = hyper
        self.T = hyper['T']

        self.dE_dy = DerivativeE_wrt_y(hyper)
        if hyper['init_procedure'] == 'Identity':
            self.init = Identity()

        # learnable learning rate. 
        self.lr = nn.Parameter(torch.Tensor([1.]*self.T))

        # computing the weights used in the loss weighting
        # TODO: Set the first weight (t=0) to 0 ?
        self.weight = Variable(torch.Tensor([ 1.0 / (self.T - t + 1) for t in xrange(self.T+1)]))
        self.weight = self.weight.view(-1,1,1,1,1)

        self.initParams()

    def initParams(self):

        for child in self.children():
            child.initParams()

    def forward(self, x):
        # shape of x: (bs, 1, H, W)

        bs = x.size(0)
        
        # The initialization procedure returns an image with pixels in [0,1],
        # so it makes sense to use it directly as our initial y value.
        y0 = self.init(x)

        # Since we're doing gradient descent on the logit version of y (pre-sigmoid),
        # we need to find the corresponding logit that yield our initial y.
        # To do so, we simply apply the inverse of the sigmoid function to y.
        logit_tab = (torch.log(y0 + 1e-16) - torch.log(1-y0 + 1e-16)).unsqueeze(0)
        

        for t in xrange(1,self.T+1):

            y = F.sigmoid(logit_tab[-1])
            # See Belanger, Yang and McAllum 2017 for this trick 
            logit = logit_tab[-1] - F.softplus(self.lr[t-1]) * y * (1 - y) * self.dE_dy(x, y)

            logit_tab = torch.cat([logit_tab, logit.unsqueeze(0)], 0)

        return logit_tab # shape: (T+1, bs, 1 , H, W)

    def getLossPred(self, logit_tab, y_gt):
        # logit_tab shape: (T+1, bs, 1, H, W)
        # y_gt shape: (bs,1,H,W)

        bs = y_gt.size(0)

        logit_gt = torch.log(y_gt + 1e-16 ) - torch.log(1-y_gt + 1e-16)
        
        loss = torch.mean((logit_tab - logit_gt.unsqueeze(0))**2 * self.weight)

        return loss*bs, F.sigmoid(logit_tab[-1])

class GDMomentumPredictor(nn.Module):
    """Encapsulates the whole end-to-end process to compute the loss and the prediction
    Energy minimization: GD+Momentum on logits. See Belanger, Yang and McAllum 2017
    Loss: MSE on y
    """

    def __init__(self,hyper):
        super(GDMomentumPredictor, self).__init__()
        self.hyper = hyper
        self.T = hyper['T']
        self.momentum = hyper['momentum']

        self.dE_dy = DerivativeE_wrt_y(hyper)
        if hyper['init_procedure'] == 'Identity':
            self.init = Identity()

        # learnable initial velocity
        self.initial_velocity = nn.Parameter(torch.Tensor(96,128))
        # learnable learning rate. 
        self.lr = nn.Parameter(torch.Tensor([1.]*self.T))

        # computing the weights used in the loss weighting
        # TODO: Set the first weight (t=0) to 0 ?
        self.weight = Variable(torch.Tensor([ 1.0 / (self.T - t + 1) for t in xrange(self.T+1)]))
        self.weight = self.weight.view(-1,1,1,1,1)

        self.initParams()

    def initParams(self):

        #self.initial_velocity.zero_()

        for child in self.children():
            child.initParams()

    def forward(self, x):
        # shape of x: (bs, 1, H, W)
        
        bs = x.size(0)
        
        # The initialization procedure returns an image with pixels in [0,1],
        # so it makes sense to use it directly as our initial y value.
        y_tab = self.init(x).unsqueeze(0)

        # Since we're doing gradient descent on the logit version of y (pre-sigmoid),
        # we need to find the corresponding logit that yield our initial y.
        # To do so, we simply apply the inverse of the sigmoid function to y.
        logit = (torch.log(y_tab+ 1e-16) - torch.log(1-y_tab+ 1e-16)).squeeze(0)

        # initialize the velocity state
        velocity = self.initial_velocity.clone()
        

        for t in xrange(1,self.T+1):

            # See Belanger, Yang and McAllum 2017 for this trick 

            velocity = self.momentum * velocity.clone() \
                       - F.softplus(self.lr[t-1]) * y_tab[t-1] * (1 - y_tab[t-1]) * self.dE_dy(x, y_tab[t-1])

            logit = logit.clone() + velocity

            y_tab = torch.cat([y_tab, F.sigmoid(logit).unsqueeze(0)], 0)

        return y_tab # shape: (T+1, bs, 1 , H, W)

    def getLossPred(self, y_tab, y_gt):
        # y_tab shape: (T+1, bs, 1, H, W)
        # y_gt shape: (bs,1,H,W)

        #Use multiple iteration in the loss
        loss = torch.sum((y_tab - y_gt.unsqueeze(0))**2 * self.weight)


        return loss, y_tab[-1]

class Identity(nn.Module):
    """An very simple initialization procedure.
    It simply initializes y0 to the value of x.
    """
    def __init__(self):
        super(Identity,self).__init__()

    def initParams(self):
        # no params to init
        return None

    def forward(self, x):

        return x

class ConvInit(nn.Module):
    """An initialization procedure (convNet)"""
    def __init__(self, hyper):
        # WARNING: expects odd ker_size
        super(ConvInit, self).__init__()
        assert hyper['ker_size'] % 2 == 1

        self.hyper = hyper

        nb_hid_layer = hyper['nb_hid_layer']
        nb_feat_map = hyper['nb_feat_map']
        ker_size = hyper['ker_size']
        self.active_func = eval(hyper['active_func'])
        self.bn = hyper['bn']

        self.layers = nn.ModuleList()
        self.BNs = nn.ModuleList()

        pad = ker_size / 2

        if nb_hid_layer > 0 :
            self.layers.append(nn.Conv2d(1, 
                                         nb_feat_map,
                                         ker_size,
                                         padding=pad))
            if self.bn:
                self.BNs.append(nn.BatchNorm2d(nb_feat_map))

            for i in range(nb_hid_layer - 1):
                self.layers.append(nn.Conv2d(nb_feat_map, 
                                             nb_feat_map,
                                             ker_size,
                                             padding=pad))
                if self.bn:
                    self.BNs.append(nn.BatchNorm2d(nb_feat_map))
            
            self.layers.append(nn.Conv2d(nb_feat_map, 
                                         1,
                                         ker_size,
                                         padding=pad))
            if self.bn:
                self.BNs.append(nn.BatchNorm2d(1))

        else:
            self.layers.append(nn.Conv2d(nb_feat_map,
                                         1,
                                         1,
                                         padding=pad))
            if self.bn:
                self.BNs.append(nn.BatchNorm2d(1))

        self.initParams()

    def initParams(self):

        if self.active_func == F.relu:
            activ = 'relu'
        elif self.active_func == F.tanh:
            activ = 'tanh'
        elif self.active_func == F.sigmoid:
            activ = 'sigmoid'

        for name, param in self.named_parameters():
            if string.find(name,'weight') != -1 and string.find(name,'layers') != -1:
                
                nn.init.xavier_uniform(param, gain=nn.init.calculate_gain(activ))
            elif string.find(name,'bias') != -1:
                param.data.zero_()

    def forward(self, x):

        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = self.active_func(x)
            if self.bn:
                x = self.BNs[i](x)
        
        out = self.layers[-1](x)

        return out

    def getLossPred(self,out, y):

        pred = F.sigmoid(out)
        loss = torch.mean((pred - y)**2, 1)
        loss = torch.mean(loss,1)
        loss = torch.mean(loss,1)
        loss = torch.sum(loss)

        return loss, pred














