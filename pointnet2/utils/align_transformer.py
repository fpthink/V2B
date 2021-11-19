import torch
import numpy as np
from torch import nn

from torch.nn import functional as F

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
class NLBlockND(nn.Module):

	def __init__(self, in_channels, inter_channels=None, mode='embedded', bn_layer=True):

		"""Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick

		args:

			in_channels: original channel size (1024 in the paper)

			inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)

			mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation

			dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)

			bn_layer: whether to add batch norm

		"""

		super(NLBlockND, self).__init__()


		if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
			raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

		self.mode = mode


		self.in_channels = in_channels

		self.inter_channels = inter_channels

		# the channel size is reduced to half inside the block

		if self.inter_channels is None:

			self.inter_channels = in_channels // 2

			if self.inter_channels == 0:
				self.inter_channels = 1

		# self.T_net=STNkd(k=64)

		# add BatchNorm layer after the last conv layer

		# if bn_layer:
		#
		# 	self.W_z = nn.Sequential(
		#
		# 		nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
		#
		# 		nn.BatchNorm1d(self.in_channels)
		#
		# 	)
		#
		# 	# from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
		#
		# 	nn.init.constant_(self.W_z[1].weight, 0)
		#
		# 	nn.init.constant_(self.W_z[1].bias, 0)
		#
		# else:
		#
		# 	self.W_z = nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
		#
		# 	# from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
		#
		# 	nn.init.constant_(self.W_z.weight, 0)
		#
		# 	nn.init.constant_(self.W_z.bias, 0)

		# define theta and phi for all operations except gaussian

		if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
			self.theta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

			# self.phi = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

		if self.mode == "concatenate":
			self.W_f = nn.Sequential(

				nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),

				nn.ReLU()

			)

	def forward(self, x,y,simi):

		"""

		args

			x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1

		"""

		batch_size = x.size(0)

		# (N, C, THW)

		# this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
		#(b,c,n1),(b,c,n1+n2)
		# g_y = self.g(y)

		# g_y = g_y.permute(0, 2, 1).contiguous()

		if self.mode == "gaussian":

			theta_x = x.permute(0, 2, 1).contiguous()
			#b 64 128
			f = torch.matmul(theta_x, y)
			simi=simi>0.8
			simi=simi.float()
			f=f*simi


		elif self.mode == "embedded" or self.mode == "dot":

			theta_x = self.theta(x)
			theta_y = self.theta(y)

			# phi_x = self.phi(y)

			theta_x = theta_x.permute(0, 2, 1).contiguous()

			f = torch.matmul(theta_x, theta_y)



		elif self.mode == "concatenate":

			theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)

			phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

			h = theta_x.size(2)

			w = phi_x.size(3)

			theta_x = theta_x.repeat(1, 1, 1, w)

			phi_x = phi_x.repeat(1, 1, h, 1)

			concat = torch.cat([theta_x, phi_x], dim=1)

			f = self.W_f(concat)

			f = f.view(f.size(0), f.size(2), f.size(3))

		if self.mode == "gaussian" or self.mode == "embedded":

			# f_div_C = F.softmax(f, dim=-1)
			# N = f.size(-1)  # number of position in x
			#b 64

			f_div_C = F.max_pool1d(f, kernel_size=[f.size(2)])


			f_div_C = f_div_C / self.in_channels


		# contiguous here just allocates contiguous chunk of memory

		f_div_C = f_div_C.permute(0, 2, 1).contiguous()
		f_div_C=f_div_C*x


		# W_y = self.W_z(y)

		# residual connection

		z = f_div_C + x

		return z


if __name__ == '__main__':

	import torch

	for bn_layer in [False,True]:
		# img = torch.zeros(2, 3, 20)

		# net = NLBlockND(in_channels=3, mode='concatenate', dimension=1, bn_layer=bn_layer)
		#
		# out = net(img)
		#
		# print(out.size())
		import time
		# img = torch.randn(48, 256, 2, 128).cuda()
		#['gaussian', 'embedded', 'dot', 'concatenate']:
		search = torch.randn(48, 128,128).cuda()
		template = torch.randn(48, 128,64).cuda()
		net = NLBlockND(in_channels=128, mode='gaussian')
		net.cuda()
		t0=time.time()
		final_out_cla = F.cosine_similarity(template.unsqueeze(-1).expand(48, 128, 64, 128),
		                            search.unsqueeze(2).expand(48, 128, 64, 128))
		out = net(template,search,final_out_cla.detach())
		t1 = time.time()

		print(t1-t0)

		print(out.size())

		# img = torch.randn(2, 3, 8, 20, 20)
		#
		# net = NLBlockND(in_channels=3, mode='concatenate', dimension=3, bn_layer=bn_layer)
		#
		# out = net(img)
		#
		# print(out.size())