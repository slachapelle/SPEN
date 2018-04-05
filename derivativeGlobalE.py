import numpy as np
import torch
from torch.autograd import Variable, Function
import torch.nn as nn
from torch.nn import functional as F

def computeGlobalE(y, filter1, bias1, filter2, bias2, filter3, bias3):
	""" Computes log P(y) from Belanger, McAllum 2017.
	Args:
		y (Variable): image of size (bs, W, H)
		filter1(Variable):
		bias1(Variable):
		filter2(Variable):
		bias2(Variable):
		filter3(Variable):
		bias3(Variable):
	"""

	layer1 = F.conv2d(y, filter1, bias=bias1)
	layer1 = F.softplus(layer1)

	layer2 = F.conv2d(layer1, filter2, bias=bias2)
	layer2 = F.softplus(layer2)

	layer3 = F.conv2d(layer2, filter3, bias=bias3)
	layer3 = F.avg_pool2d(layer3, (layer3.size(-2), layer3.size(-1)))

	DNN_y = layer3.squeeze(3).squeeze(2).squeeze(1)

	return  - DNN_y # shape (bs,)

class ComputeDerivativeGlobalE_wrt_y(Function):
	"""Implement the differentiable operation of taking the derivative of E_global wrt y
	"""

	@staticmethod
	def forward(ctx, y, filter1, bias1, filter2, bias2, filter3, bias3):
		
		ctx.save_for_backward(y, filter1, bias1, filter2, bias2, filter3, bias3)

		y = Variable(y, requires_grad=True)
		filter1 = Variable(filter1)
		bias1 = Variable(bias1)
		filter2 = Variable(filter2)
		bias2 = Variable(bias2)
		filter3 = Variable(filter3)
		bias3 = Variable(bias3)

		e = computeGlobalE(y, filter1, bias1, filter2, bias2, filter3, bias3)

		e.backward(torch.ones(e.size()))

		return y.grad.data

	@staticmethod
	def backward(ctx, output_grad):

		y, filter1, bias1, filter2, bias2, filter3, bias3 = ctx.saved_tensors

		eps = torch.Tensor([1e-9]) # 1e-2 to 1e-9 works ok 
		r = torch.sqrt(eps)*(1 + torch.max(torch.abs(y)))
		r /= (torch.max(torch.abs(output_grad.data)) + eps)

		y1 = Variable(y + r*output_grad.data, requires_grad=True)
		y0 = Variable(y - r*output_grad.data, requires_grad=True)

		filter1 = Variable(filter1, requires_grad=True)
		bias1 = Variable(bias1, requires_grad=True)
		filter2 = Variable(filter2, requires_grad=True)
		bias2 = Variable(bias2, requires_grad=True)
		filter3 = Variable(filter3, requires_grad=True)
		bias3 = Variable(bias3, requires_grad=True)

		e0 = computeGlobalE(y0, filter1, bias1, filter2, bias2, filter3, bias3)
		e0.backward(torch.ones(e0.size()))

		dE_dy_0 = y0.grad.clone()
		dE_dfilter1_0 = filter1.grad.clone()
		dE_dbias1_0 = bias1.grad.clone()
		dE_dfilter2_0 = filter2.grad.clone()
		dE_dbias2_0 = bias2.grad.clone()
		dE_dfilter3_0 = filter3.grad.clone()
		dE_dbias3_0 = bias3.grad.clone()

		y0.grad.zero_()
		filter1.grad.zero_()
		bias1.grad.zero_()
		filter2.grad.zero_()
		bias2.grad.zero_()
		filter3.grad.zero_()
		bias3.grad.zero_()

		e1 = computeGlobalE(y1, filter1, bias1, filter2, bias2, filter3, bias3)
		e1.backward(torch.ones(e1.size()))

		dE_dy_1 = y1.grad.clone()
		dE_dfilter1_1 = filter1.grad.clone()
		dE_dbias1_1 = bias1.grad.clone()
		dE_dfilter2_1 = filter2.grad.clone()
		dE_dbias2_1 = bias2.grad.clone()
		dE_dfilter3_1 = filter3.grad.clone()
		dE_dbias3_1 = bias3.grad.clone()

		r = Variable(r)

		grad_L_y = (dE_dy_1 - dE_dy_0) / (2*r)
		grad_L_dfilter1 = (dE_dfilter1_1 - dE_dfilter1_0) / (2*r)
		grad_L_dbias1 = (dE_dbias1_1 - dE_dbias1_0) / (2*r)
		grad_L_dfilter2 = (dE_dfilter2_1 - dE_dfilter2_0) / (2*r)
		grad_L_dbias2 = (dE_dbias2_1 - dE_dbias2_0) / (2*r)
		grad_L_dfilter3 = (dE_dfilter3_1 - dE_dfilter3_0) / (2*r)
		grad_L_dbias3 = (dE_dbias3_1 - dE_dbias3_0) / (2*r)

		return grad_L_y, grad_L_dfilter1, grad_L_dbias1, grad_L_dfilter2, grad_L_dbias2, grad_L_dfilter3, grad_L_dbias3