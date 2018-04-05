# TODO: add the entropy term to the energy. (see p.5 of Belanger, Yang and McAllum 2017)

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
		# supposing input image are greyscale
		self.filter1 = nn.Parameter(torch.Tensor(32, 1, 5, 5))
		self.bias1 = nn.Parameter(torch.Tensor(32))
		self.filter2 = nn.Parameter(torch.Tensor(32, 32, 5, 5))
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
		self.sigma_2 = nn.Parameter(torch.Tensor([0.]))

	def initParams(self):

		# TODO: init self.sigma_2 ??

		for child in self.children():
			child.initParams()

	def forward(self,x, y):

		#y = y.unsqueeze(1)
		#print x.shape, y.shape

		dLocalE_dy = 2*(y - x)
		dEntropy_dy = torch.log(1 - y + 1e-16) - torch.log(y + 1e-16)
		#print self.dGlobalE_dy(y).size()
		dE_dy = dLocalE_dy - 2 * F.softplus(self.sigma_2) * self.dGlobalE_dy(y) + self.hyper['entropy_decay']*dEntropy_dy

		#print 'dE_dy size is ', dE_dy.size()
		return dE_dy

class GradientDescentPredictor(nn.Module):
	"""Encapsulates the whole end-to-end process to compute the loss and the prediction"""

	def __init__(self,hyper):
		super(GradientDescentPredictor, self).__init__()
		self.hyper = hyper
		self.T = hyper['T']

		self.dE_dy = DerivativeE_wrt_y(hyper)
		if hyper['init_procedure'] == 'Identity':
			self.init = Identity()
		# learnable learning rate. 
		# TODO: having different lr for each step ?
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
		#print 'size of x ', x.size()
		bs = x.size(0)
		"""
		#logit are the unormalized logits
		logit = self.init(x) # shape (bs,1,H,W)

		y_tab = F.sigmoid(logit).unsqueeze(0)
		"""

		#TODO: Try the following alternative:
		
		# The initialization procedure returns an image with pixels in [0,1],
		# so it makes sense to use it directly as our initial y value.
		y_tab = self.init(x).unsqueeze(0)

		# Since we're doing gradient descent on the logit version of y (pre-sigmoid),
		# we need to find the corresponding logit that yield our initial y.
		# To do so, we simply apply the inverse of the sigmoid function to y.
		logit = (torch.log(y_tab) - torch.log(1-y_tab)).squeeze(0)
		

		for t in xrange(1,self.T+1):

			# See Belanger, Yang and McAllum 2017 for this trick 
			logit = logit.clone() - F.softplus(self.lr[t-1]) * y_tab[t-1] * (1 - y_tab[t-1]) * self.dE_dy(x, y_tab[t-1])

			y_tab = torch.cat([y_tab, F.sigmoid(logit).unsqueeze(0)], 0)

		return y_tab # shape: (T+1, bs, 1 , H, W)

	def getLossPred(self, y_tab, y_gt):
		# y_tab shape: (T+1, bs, 1, H, W)
		# y_gt shape: (bs,1,H,W)

		#print y_gt.size()

		#Use multiple iteration in the loss
		loss = torch.sum((y_tab - y_gt.unsqueeze(0))**2 * self.weight)

		# use only the last iteration
		#loss = torch.sum((y_tab[-1] - y_gt)**2)
						  
		"""
		loss = F.binary_cross_entropy(y_tab,
									  y_gt.unsqueeze(0).expand(y_tab.size(0),-1,-1,-1,-1),
									  weight=self.weight.view(-1,1,1,1,1),
									  size_average=False)
		"""

		return loss, y_tab[-1]

class GDMomentumPredictor(nn.Module):
	"""Encapsulates the whole end-to-end process to compute the loss and the prediction"""

	def __init__(self,hyper):
		super(GradientDescentPredictor, self).__init__()
		self.hyper = hyper
		self.T = hyper['T']
		self.momentum = hyper['momentum']

		self.dE_dy = DerivativeE_wrt_y(hyper)
		if hyper['init_procedure'] == 'Identity':
			self.init = Identity()

		self.initial_velocity = nn.Parameter(torch.Tensor(96,128))
		# learnable learning rate. 
		# TODO: having different lr for each step ?
		self.lr = nn.Parameter(torch.Tensor([1.]*self.T))

		# computing the weights used in the loss weighting
		# TODO: Set the first weight (t=0) to 0 ?
		self.weight = Variable(torch.Tensor([ 1.0 / (self.T - t + 1) for t in xrange(self.T+1)]))
		self.weight = self.weight.view(-1,1,1,1,1)

		self.initParams()

	def initParams(self):

		self.initial_velocity.zero_()

		for child in self.children():
			child.initParams()

	def forward(self, x):
		# shape of x: (bs, 1, H, W)
		#print 'size of x ', x.size()
		bs = x.size(0)
		"""
		#logit are the unormalized logits
		logit = self.init(x) # shape (bs,1,H,W)

		y_tab = F.sigmoid(logit).unsqueeze(0)
		"""

		#TODO: Try the following alternative:
		
		# The initialization procedure returns an image with pixels in [0,1],
		# so it makes sense to use it directly as our initial y value.
		y_tab = self.init(x).unsqueeze(0)

		# Since we're doing gradient descent on the logit version of y (pre-sigmoid),
		# we need to find the corresponding logit that yield our initial y.
		# To do so, we simply apply the inverse of the sigmoid function to y.
		logit = (torch.log(y_tab) - torch.log(1-y_tab)).squeeze(0)

		# initialize the velocity state
		velocity = self.initial_velocity.clone()
		

		for t in xrange(1,self.T+1):

			velocity = self.momentum * velocity.clone() \
			           - F.softplus(self.lr[t-1]) * y_tab[t-1] * (1 - y_tab[t-1]) * self.dE_dy(x, y_tab[t-1])

			# See Belanger, Yang and McAllum 2017 for this trick 
			logit = logit.clone() + velocity

			y_tab = torch.cat([y_tab, F.sigmoid(logit).unsqueeze(0)], 0)

		return y_tab # shape: (T+1, bs, 1 , H, W)

	def getLossPred(self, y_tab, y_gt):
		# y_tab shape: (T+1, bs, 1, H, W)
		# y_gt shape: (bs,1,H,W)

		#print y_gt.size()

		#Use multiple iteration in the loss
		loss = torch.sum((y_tab - y_gt.unsqueeze(0))**2 * self.weight)

		# use only the last iteration
		#loss = torch.sum((y_tab[-1] - y_gt)**2)
						  
		"""
		loss = F.binary_cross_entropy(y_tab,
									  y_gt.unsqueeze(0).expand(y_tab.size(0),-1,-1,-1,-1),
									  weight=self.weight.view(-1,1,1,1,1),
									  size_average=False)
		"""

		return loss, y_tab[-1]

class Identity(nn.Module):

	def __init__(self):
		super(Identity,self).__init__()

	def initParams(self):
		# no params to init
		return None

	def forward(self, x):

		return x















