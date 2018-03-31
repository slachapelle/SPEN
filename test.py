from copy import deepcopy

import numpy as np
import torch
from torch.autograd import Variable, Function, grad, backward, gradcheck
import torch.nn as nn
from torch.nn import functional as F


#np.random.seed(123456)
torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


COUNT_forward = 0
COUNT_backward = 0

class DE_dy(Function):

	@staticmethod
	def forward(ctx, y, F_x, b, C1, c2):

		global COUNT_forward

		# y: (bs, L)
		# F_x: (bs, k1)
		# b: (L,k1)
		# C1: (k2,L)
		# c2: (k2,)

		ctx.save_for_backward(y, F_x, b, C1, c2)

		# y: (bs, L)
		y = Variable(y, requires_grad=True)
		# F_x: (bs, k1)
		F_x = Variable(F_x)
		# b: (L,k1)
		b = Variable(b) #, requires_grad=True)
		# C1: (k2,L)
		C1 = Variable(C1)#, requires_grad=True)
		# c2: (k2,)
		c2 = Variable(c2)#, requires_grad=True)

		e = energy(y, F_x, b, C1, c2)

		e.backward(torch.ones(e.size()))

		COUNT_forward += 1
		#print 'forward', COUNT_forward
		return y.grad.data #, b.grad.data, C1.grad.data, c2.grad.data

	@staticmethod
	def backward(ctx, output_grad):

		global COUNT_backward

		y, F_x, b, C1, c2 = ctx.saved_tensors

		
		#print y
		#print output_grad
		eps = torch.Tensor([1e-9]) # 1e-2 to 1e-9 works ok 
		r = torch.sqrt(eps)*(1 + torch.max(torch.abs(y)))
		#print r
		#print torch.max(torch.abs(output_grad.data))
		r /= (torch.max(torch.abs(output_grad.data)) + eps)
		#print r
		
		
		#r = 1e-4
		#print r

		# y1: (bs, L)
		y1 = Variable(y + r*output_grad.data, requires_grad=True)
		# y2: (bs,L)
		y0 = Variable(y - r*output_grad.data, requires_grad=True)
		#assert (y1 - y0 != 0.).all()

		# F_x: (bs, k1)
		F_x = Variable(F_x)
		# b: (L,k1)
		b = Variable(b, requires_grad=True)
		# C1: (k2,L)
		C1 = Variable(C1, requires_grad=True)
		# c2: (k2,)
		c2 = Variable(c2, requires_grad=True)

		e1 = energy(y1, F_x, b, C1, c2)
		e1.backward(torch.ones(e1.size()))
		dE_dy_1 = y1.grad.clone()
		dE_db_1 = b.grad.clone()
		dE_dC1_1 = C1.grad.clone()
		dE_dc2_1 = c2.grad.clone()

		# zeroing the grad buffers before the next grad computation
		b.grad.zero_()
		C1.grad.zero_()
		c2.grad.zero_()

		e0 = energy(y0, F_x, b, C1, c2)
		e0.backward(torch.ones(e0.size()))
		dE_dy_0 = y0.grad.clone()
		dE_db_0 = b.grad.clone()
		dE_dC1_0 = C1.grad.clone()
		dE_dc2_0 = c2.grad.clone()

		# Computing the Hessian-vector approximation for each inputs
		r = Variable(r)
		#if (output_grad != 0.).any():
		#	r = r*output_grad.norm(p=2)
		
		grad_L_y = (dE_dy_1 - dE_dy_0) / (2*r)
		grad_L_b = (dE_db_1 - dE_db_0) / (2*r)
		grad_L_C1 = (dE_dC1_1 - dE_dC1_0) / (2*r)
		grad_L_c2 = (dE_dc2_1 - dE_dc2_0) / (2*r)
		"""
		print 'y ', torch.sum(dE_dy_1 - dE_dy_0!=0.).data[0] / np.prod(dE_dy_1.size()).astype('float32')
		print 'b ', torch.sum(dE_db_1 - dE_db_0!=0.).data[0] / np.prod(dE_db_1.size()).astype('float32')
		print 'C1 ', torch.sum(dE_dC1_1 - dE_dC1_0!=0.).data[0] / np.prod(dE_dC1_1.size()).astype('float32')
		print 'c2 ', torch.sum(dE_dc2_1 - dE_dc2_0!=0.).data[0] / np.prod(dE_dc2_1.size()).astype('float32')
		"""
		COUNT_backward += 1
		#print 'backward', COUNT_backward

		# 	   y,        F_x,  b,        C1,        c2
		return grad_L_y, None, grad_L_b, grad_L_C1, grad_L_c2

def energy(y, F_x, b, C1, c2):

	# Local energy
	temp1 = torch.sum(b.unsqueeze(0)*F_x.unsqueeze(1), 2) # sum over the k1 dim
	# (bs,L)
	E_local = torch.sum(y*temp1, 1)
	# (bs,)

	# Global energy
	temp2 = torch.matmul(C1.unsqueeze(0),y.unsqueeze(2)).squeeze(2) # TODO: check this...
	# (bs, k2)

	# TODO: ok activation?
	temp3 = F.softplus(temp2)
	# (bs, k2)
	E_global = torch.matmul(temp3,c2)
	# (bs,)

	return E_local + E_global

class DE_dy_module(nn.Module):

	def __init__(self):

		super(DE_dy_module, self).__init__()
		bs = 5
		L = 10
		k1 = 50
		k2 = 60
		# b: (L,k1)
		self.b = nn.Parameter(torch.Tensor(L,k1).normal_(std=0.1))
		# C1: (k2,L)
		self.C1 = nn.Parameter(torch.Tensor(k2,L).normal_(std=0.1))
		# c2: (k2,)
		self.c2 = nn.Parameter(torch.Tensor(k2).normal_(std=0.1))

	def forward(self, y, F_x):
		return DE_dy.apply(y, F_x, self.b, self.C1, self.c2)

def test(bs = 5, L = 10, k1 = 50, k2 = 60):

	# y: (bs, L)
	y = Variable(torch.Tensor(bs,L).uniform_(), requires_grad=True)
	# F_x: (bs, k1)
	F_x = Variable(torch.Tensor(bs,k1).normal_(std=0.1))
	
	b = Variable(nn.init.xavier_normal(torch.Tensor(L,k1)), requires_grad=True)
	# C1: (k2,L)
	C1 = Variable(nn.init.xavier_normal(torch.Tensor(k2,L)), requires_grad=True)
	# c2: (k2,)
	c2 = Variable(torch.Tensor(k2).normal_(std=0.1), requires_grad=True)

	print gradcheck(DE_dy.apply, (y, F_x, b, C1, c2), eps = 1e-4, atol=0 ,rtol=2)
	# r= 1e-4 -> atol=15e-3
	# r= 1e-3 -> atol=15e-3
	# r= 1e-2
	# eps = 1e-9 -> 1.4e-2
	
	"""
	# y: (bs, L)
	y = Variable(torch.Tensor(bs,L).normal_(std=0.1), requires_grad=True)
	# F_x: (bs, k1)
	F_x = Variable(torch.Tensor(bs,k1).normal_(std=0.1))

	dE_dy = DE_dy_module()

	de0 = dE_dy(y, F_x)

	scalar0 = torch.sum(de0)

	de1 = dE_dy(y, F_x)
	scalar1 = torch.sum(de1)

	print scalar0 == scalar1 

	scalar0.backward(retain_graph=True)

	y_grad0 = y.grad.clone()
	b_grad0 = dE_dy.b.grad.clone()
	C1_grad0 = dE_dy.C1.grad.clone()
	c2_grad0 = dE_dy.c2.grad.clone()

	y.grad.zero_()
	dE_dy.b.grad.zero_()
	dE_dy.C1.grad.zero_()
	dE_dy.c2.grad.zero_()

	scalar0.backward(retain_graph=True)

	y_grad1 = y.grad.clone()
	b_grad1 = dE_dy.b.grad.clone()
	C1_grad1 = dE_dy.C1.grad.clone()
	c2_grad1 = dE_dy.c2.grad.clone()

	#print y_grad0 - y_grad1
	#print y_grad1
	print torch.sum(y_grad0 != y_grad1)
	print torch.sum(b_grad0 != b_grad1)
	print torch.sum(C1_grad0 != C1_grad1)
	print torch.sum(c2_grad0 != c2_grad1)
	"""

"""
# y: (bs, L)
y = Variable(torch.Tensor(bs,L).normal_(std=0.1), requires_grad=True)
# F_x: (bs, k1)
F_x = Variable(torch.Tensor(bs,k1).normal_(std=0.1))

de = dE_dy(y, F_x)

#print de
print 'size of de: ', de.size() 



# summing just to have a scalar
scalar = torch.sum(de)
scalar.backward()

print y.grad.size()
print dE_dy.b.grad.size()
print dE_dy.C1.grad.size()
print dE_dy.c2.grad.size()
"""

#test()

bs = 5
L = 10
k1 = 50
k2 = 60

# y: (bs, L)
y = Variable(torch.Tensor(bs,L).uniform_())
# F_x: (bs, k1)
F_x = Variable(torch.Tensor(bs,k1).normal_(std=0.1))

dE_dy = DE_dy_module()

optimizer = torch.optim.Adam(dE_dy.parameters(), lr = 0.0001)

params0 = deepcopy(list(dE_dy.parameters()))

for i in range(1000):

	de0 = dE_dy(y, F_x)
	de1 = dE_dy(y + 1, F_x)
	scalar = torch.sum(de0 - de1)

	optimizer.zero_grad()
	scalar.backward()
	optimizer.step()

	print scalar.data[0]

	params1 = dE_dy.parameters()

	diff_squared_norm = 0.
	for param0, param1 in zip(params0,params1):
		#print torch.sum((param0.data - param1.data)**2)
		diff_squared_norm += torch.sum((param0.data - param1.data)**2)

	diff_norm = np.sqrt(diff_squared_norm)
	#print 'diff_norm: ', diff_norm

	params0 = deepcopy(list(params1))

