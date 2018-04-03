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
		super(computeGlobalE,self).__init__()

		self.hyper = hyper
		# supposing input image are greyscale
		self.filter1 = Parameter(torch.Tensor(32, 1, 7, 7))
		self.bias1 = Parameter(torch.Tensor(32))
		self.filter2 = Parameter(torch.Tensor(32, 32, 7, 7))
		self.bias2 = Parameter(torch.Tensor(32))
		self.filter3 = Parameter(torch.Tensor(1, 32, 1, 1))
		self.bias3 = Parameter(torch.Tensor(1))

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
		self.sigma_2 = Parameter(torch.Tensor([1]))

	def forward(self,x, y):

		dLocalE_dy = 2*(y - x)

		dE_dy = dLocalE_dy - 2* self.sigma_2 * self.dGlobalE_dy(y)

		return dE_dy

class GradientDescentPredictor(nn.Module):
	"""Encapsulates the whole end-to-end process to compute the loss and the prediction"""

	def __init__(self,hyper):
		super(Predictor, self).__init__()
		self.hyper = hyper
		self.T = hyper['T']

		self.dE_dy = DerivativeE_wrt_y(hyper)
		self.init = Initializer(hyper) # TODO
		# learnable learning rate. 
		# TODO: having different lr for each step ?
		self.lr = Parameter(torch.Tensor([0.1]))

		# computing the weights used in the loss weighting
		# TODO: Set the first weight (t=0) to 0 ?
		self.weight = Variable(torch.Tensor([ 1.0 / (self.T - t + 1) for t in xrange(T+1)]))

	def forward(self, x):
		# shape of x: (bs, 1, H, W)

		bs = x.size(0)
		#logit are the unormalized logits
		logit = self.init(x) # shape (bs,H,W)

		# shape: (T+1, bs, H, W)
		y_tab = Variable(torch.zeros(self.T+1,bs, x.size(2), x.size(3)))

		y_tab[0] = F.sigmoid(logit)

		for t in xrange(1,self.T+1):

			# See Belanger, Yang and McAllum 2017 for this trick
			dE_dlogit = y_tab[t-1] * (1 - y_tab[t-1]) * dE_dy(x, y_tab[t-1])
			logit -= self.lr * dE_dlogit

			y_tab[t] = F.sigmoid(logit)

		return y_tab

	def computeLossPred(self, y_tab, y_gt):
		# y_tab shape: (T+1, bs, H, W)
		# y_gt shape: (bs,H,W)

		loss = F.binary_cross_entropy(y_tab,
									  y_gt.unsqueeze(0).expand(y_tab.size(0),-1,-1,-1),
									  weight=self.weight.view(-1,1,1,1),
									  size_average=False)

		return loss, y_tab[-1]















