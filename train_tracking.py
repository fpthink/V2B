import argparse
import os
import random
import time
import logging
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from Dataset import SiameseTrain
from pointnet2.models import Pointnet_Tracking

from loss.PCLosses import ChamferLoss
from loss.losses import FocalLoss
from loss.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from loss.utils import _sigmoid, _tranpose_and_gather_feat

parser = argparse.ArgumentParser()
parser.add_argument('--which_dataset', type=str, default = 'NUSCENES',  help='datasets:NUSCENES,KITTI')
parser.add_argument('--debug', type=bool, default = False,  help='debug')
parser.add_argument('--batchSize', type=int, default=48, help='input batch size')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=2, help='# GPUs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')
parser.add_argument('--input_feature_num', type=int, default = 0,  help='number of input point features')
parser.add_argument('--data_dir', type=str, default = "/opt/data/common/nuScenes/KITTI_style/train_track",  help='dataset path')
parser.add_argument('--val_data_dir', type=str, default = "/opt/data/common/nuScenes/KITTI_style/val",  help='dataset path')
parser.add_argument('--category_name', type=str, default = 'Car',  help='Object to Track (Car/Pedestrian/Van/Cyclist)')
parser.add_argument('--save_root_dir', type=str, default='./nusc_model',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model name for training resume')
parser.add_argument('--optimizer', type=str, default = '',  help='optimizer name for training resume')

opt = parser.parse_args()
print (opt)

#torch.cuda.set_device(opt.main_gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

opt.manualSeed = 1
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.backends.cudnn.benchmark=True

save_dir = os.path.join(opt.save_root_dir,'{}'.format(opt.category_name))

try:
	os.makedirs(save_dir)
except OSError:
	pass

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
					filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
logging.info('======================================================')

kitti_to_nusc_category={'car':'car','pedestrian':'pedestrian','van':'truck','cyclist':'bicycle'}
if opt.which_dataset.upper()=='NUSCENES':
	opt.category_name=kitti_to_nusc_category[opt.category_name.lower()]
# 1. Load data

train_data = SiameseTrain(
			which_dataset=opt.which_dataset,
            input_size=1024,
            path= opt.data_dir,
            split='TRAIN' if not opt.debug else 'TRAIN_tiny',
            category_name=opt.category_name,
            offset_BB=0,
            scale_BB=1.25,
			voxel_size=[0.3,0.3,0.3],
			xy_size=[0.3,0.3])

train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    pin_memory=True)
test_data = SiameseTrain(
	which_dataset=opt.which_dataset,
    input_size=1024,
    path=opt.data_dir if opt.which_dataset.upper()=="KITTI" else opt.val_data_dir,
    split='VALID' if not opt.debug else 'VALID_tiny',
    category_name=opt.category_name,
    offset_BB=0,
    scale_BB=1.25,
	voxel_size=[0.3, 0.3, 0.3],
	xy_size=[0.3, 0.3])

test_dataloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=int(opt.batchSize/2),
    shuffle=False,
    num_workers=int(opt.workers),
    pin_memory=True)

										  
print('#Train data:', len(train_data), '#Test data:', len(test_data))
print (opt)

# 2. Define model, loss and optimizer
netR = Pointnet_Tracking(input_channels=opt.input_feature_num, use_xyz=True,mode=True,voxel_size=torch.from_numpy(train_data.voxel_size.copy()).float(),voxel_area=train_data.voxel_grid_size,scene_ground=torch.from_numpy(train_data.scene_ground.copy()).float())
if opt.ngpu > 1:
	netR = torch.nn.DataParallel(netR, range(opt.ngpu))
if opt.model != '':
	netR.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))
	  
netR.cuda()
print(netR)
logging.info(opt.__repr__())
logging.info(netR.__repr__())
criterion_completion=ChamferLoss().cuda()
criterion_hm=FocalLoss().cuda()
criterion_loc=RegL1Loss().cuda()
criterion_z_axis=RegL1Loss().cuda()
optimizer = optim.Adam(netR.parameters(), lr=opt.learning_rate, betas = (0.9, 0.999))
if opt.optimizer != '':
	optimizer.load_state_dict(torch.load(os.path.join(save_dir, opt.optimizer)))
	start_epoch=int(opt.optimizer[-6:-4])
else:
	start_epoch=-1
if opt.which_dataset.upper()=="KITTI":
	scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.2,last_epoch=start_epoch)
else:
	scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2, last_epoch=start_epoch)

f = open(os.path.join(save_dir,'loss.txt'), mode='w')
# 3. Training and testing
for epoch in range(start_epoch+1,opt.nepoch):
	print('======>>>>> Online epoch: #%d, lr=%f <<<<<======' %(epoch, scheduler.get_last_lr()[0]))
	# 3.1 switch to train mode
	torch.cuda.synchronize()
	netR.train()
	train_mse = 0.0
	timer = time.time()

	batch_comp_part_loss = 0.0
	batch_hm_loss = 0.0
	batch_loc_loss = 0.0
	batch_z_loss = 0.0
	batch_num = 0.0
	for i, data in enumerate(tqdm(train_dataloader, 0)):
		if len(data['search_pc']) == 1:
			continue
		torch.cuda.synchronize()
		# 3.1.1 load inputs and targets
		for k in data:
			data[k] = Variable(data[k], requires_grad=False).cuda()
		optimizer.zero_grad()
		# 3.1.2 compute output
		# BX128，(b,3,1024,128),(b,1,36,56),(b,3,36,56),(b,3,36,56)
		completion_points,pred_hm, pred_loc,pred_z_axis= netR(data['model_pc'], data['search_pc'])
		loss_comp_part=criterion_completion(completion_points,data['completion_PC_part'],None)

		pred_hm=_sigmoid(pred_hm)
		loss_hm=criterion_hm(pred_hm,data['hm'])
		loss_loc=criterion_loc(pred_loc,data['ind_offsets'],data['loc_reg'])
		loss_z=criterion_z_axis(pred_z_axis,data['ind_ct'],data['z_axis'])

		loss = 1e-6*loss_comp_part+1.0*loss_hm+1.0*loss_loc+2.0*loss_z
		# 3.1.3 compute gradient and do Adam step
		loss.backward()
		optimizer.step()
		# torch.cuda.empty_cache()

		f.write('epoch:{},batch:{},loss:{:.3f}\n'.format(epoch , i, float(loss)))

		torch.cuda.synchronize()

		# 3.1.4 update training error

		train_mse = train_mse + loss.data*len(data['search_pc'])
		batch_comp_part_loss+=loss_comp_part.data
		batch_hm_loss += loss_hm.data
		batch_loc_loss += loss_loc.data
		batch_z_loss += loss_z.data
		batch_num += len(data['search_pc'])
		if (i+1)%20 == 0:
			print('\n ---- batch: %03d ----' % (i+1))
			print('comp_part_loss: %f,hm_loss: %f,loc_loss: %f,z_loss: %f' % (batch_comp_part_loss/20,batch_hm_loss/20,batch_loc_loss/20,batch_z_loss/20))
			# print('pos_points:{} neg_points:{}'.format(num_pos,num_neg))
			batch_comp_part_loss=0.0
			batch_hm_loss = 0.0
			batch_loc_loss = 0.0
			batch_z_loss = 0.0
			batch_num = 0.0

	# time taken
	train_mse = train_mse/len(train_data)
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(train_data)
	print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

	torch.save(netR.state_dict(), '%s/netR_%d.pth' % (save_dir, epoch))
	torch.save(optimizer.state_dict(), '%s/optimizer_%d.pth' % (save_dir, epoch))

	# 3.2 switch to evaluate mode
	torch.cuda.synchronize()
	netR.eval()
	test_comp_part_loss = 0.0
	test_hm_loss = 0.0
	test_loc_loss = 0.0
	test_z_loss = 0.0
	timer = time.time()
	with torch.no_grad():
		for i, data in enumerate(tqdm(test_dataloader, 0)):
			torch.cuda.synchronize()
			# 3.2.1 load inputs and targets
			for k in data:
				data[k] = Variable(data[k], requires_grad=False).cuda()
			# 3.2.2 compute output
			completion_points,pred_hm, pred_loc,pred_z_axis= netR(data['model_pc'], data['search_pc'])
			loss_comp_part = criterion_completion(completion_points, data['completion_PC_part'],None)

			pred_hm = _sigmoid(pred_hm)
			loss_hm = criterion_hm(pred_hm, data['hm'])
			loss_loc = criterion_loc(pred_loc, data['ind_offsets'], data['loc_reg'])
			loss_z = criterion_z_axis(pred_z_axis, data['ind_ct'], data['z_axis'])

			loss = 1e-6*loss_comp_part+1.0*loss_hm+1.0*loss_loc+2.0*loss_z

			torch.cuda.synchronize()
			test_comp_part_loss = test_comp_part_loss + loss_comp_part.data * len(data['search_pc'])
			test_hm_loss = test_hm_loss + loss_hm.data * len(data['search_pc'])
			test_loc_loss = test_loc_loss + loss_loc.data * len(data['search_pc'])
			test_z_loss = test_z_loss + loss_z.data * len(data['search_pc'])

	# time taken
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(test_data)
	print('==> time to learn 1 sample = %f (ms)' %(timer*1000))
	# print mse
	test_comp_part_loss = test_comp_part_loss / len(test_data)
	test_hm_loss = test_hm_loss / len(test_data)
	test_loc_loss = test_loc_loss / len(test_data)
	test_z_loss = test_z_loss / len(test_data)

	print('comp_part_loss: %f,hm_loss: %f,loc_loss: %f,z_loss: %f, #test_data = %d' %(test_comp_part_loss,test_hm_loss,test_loc_loss,test_z_loss,len(test_data)))
	# log
	logging.info('Epoch#%d: train error=%e, test error=%e,%e,%e,%e,lr = %f' %(epoch, train_mse,test_comp_part_loss,test_hm_loss,test_loc_loss,test_z_loss,scheduler.get_last_lr()[0]))
	scheduler.step()

