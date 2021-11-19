import time
import os
import logging
import argparse
import random

import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
import torch

import kitty_utils as utils
import copy
from datetime import datetime
from metrics import AverageMeter, Success, Precision
from metrics import estimateOverlap, estimateAccuracy
from data_classes import PointCloud
from Dataset import SiameseTest

import torch.nn.functional as F
from torch.autograd import Variable

from pointnet2.models import Pointnet_Tracking
from decode import mot_decode


def test(loader, model, epoch=-1, shape_aggregation="", reference_BB="", model_fusion="pointcloud", max_iter=-1,
         IoU_Space=3,model_dir="",sparse_point=[0,100,500,1000]): #[0,100,500,1000] [0,150,1000,2500]
	batch_time = AverageMeter()
	data_time = AverageMeter()
	post_process_time = AverageMeter()

	Success_main = Success()
	Precision_main = Precision()
	interval_Success = []
	interval_Precision = []
	for num in range(4):
		interval_Success.append(Success())
		interval_Precision.append(Precision())
	Success_batch = Success()
	Precision_batch = Precision()
	Success_key_main = Success()
	Precision_key_main = Precision()

	# switch to evaluate mode
	model.eval()
	end = time.time()
	dataset = loader.dataset
	batch_num = 0
	tracklet_nums = 0
	track_anno_nums = 0
	key_track_anno_nums = 0
	with tqdm(enumerate(loader), total=len(loader.dataset.list_of_anno)) as t:
		for batch in loader:
			batch_num = batch_num + 1
			# measure data loading time
			data_time.update((time.time() - end))
			for PCs, BBs, list_of_anno in batch:  # tracklet
				results_BBs = []
				if dataset.which_dataset=='NUSCENES':
					if list_of_anno[0]['num_lidar_pts']==0:
						t.update(len(PCs))
						break
				track_anno_nums += len(PCs)
				tracklet_nums += 1
				for i, _ in enumerate(PCs):
					this_anno = list_of_anno[i]
					this_BB = BBs[i]
					this_PC = PCs[i]
					gt_boxs = []
					result_boxs = []
					if dataset.which_dataset=='NUSCENES' and this_anno['is_key_frame'] == 1:
						key_track_anno_nums += 1

					new_PC = utils.cropPC(this_PC, this_BB, offset=2 * dataset.offset_BB, scale=4 * dataset.scale_BB)
					new_label, align_gt_PC = utils.getlabelPC(new_PC, this_BB, offset=dataset.offset_BB, scale=dataset.scale_BB)
					points_on_target=np.sum(new_label)
					# INITIAL FRAME
					if i == 0:
						data_time.update((time.time() - end))
						end = time.time()
						box = BBs[i]
						results_BBs.append(box)

					else:
						previous_BB = BBs[i - 1]

						# DEFINE REFERENCE BB
						if ("previous_result".upper() in reference_BB.upper()):
							ref_BB = results_BBs[-1]
						elif ("previous_gt".upper() in reference_BB.upper()):
							ref_BB = previous_BB
						elif ("current_gt".upper() in reference_BB.upper()):
							ref_BB = this_BB

						candidate_PC, candidate_label, candidate_reg, new_ref_box, new_this_box ,align_gt_PC= utils.cropAndCenterPC_label_test(
							this_PC,
							ref_BB, this_BB,
							offset=dataset.offset_BB,
							scale=dataset.scale_BB,
							limit_area=dataset.area_extents)

						candidate_PCs, candidate_labels, candidate_reg,align_gt_PC,all_search_label = utils.regularizePCwithlabel(candidate_PC,
						                                                                             align_gt_PC,
						                                                                             candidate_label,
						                                                                             candidate_reg,
						                                                                             dataset.input_size,
						                                                                             istrain=False)

						candidate_PCs_torch = candidate_PCs.unsqueeze(0).cuda()

						# AGGREGATION: IO vs ONLY0 vs ONLYI vs ALL
						if ("firstandprevious".upper() in shape_aggregation.upper()):
							model_PC = utils.getModel([PCs[0], PCs[i - 1]], [results_BBs[0], results_BBs[i - 1]],
							                          offset=dataset.offset_BB, scale=dataset.scale_BB)
						elif ("first".upper() in shape_aggregation.upper()):
							model_PC = utils.getModel([PCs[0]], [results_BBs[0]], offset=dataset.offset_BB,
							                          scale=dataset.scale_BB)
						elif ("previous".upper() in shape_aggregation.upper()):
							model_PC = utils.getModel([PCs[i - 1]], [results_BBs[i - 1]], offset=dataset.offset_BB,
							                          scale=dataset.scale_BB)
						elif ("all".upper() in shape_aggregation.upper()):
							model_PC = utils.getModel(PCs[:i], results_BBs, offset=dataset.offset_BB,
							                          scale=dataset.scale_BB)
						else:
							model_PC = utils.getModel(PCs[:i], results_BBs, offset=dataset.offset_BB,
							                          scale=dataset.scale_BB)

						model_PC_torch = utils.regularizePC(model_PC, dataset.input_size, istrain=False).unsqueeze(0)
						model_PC_torch = Variable(model_PC_torch, requires_grad=False).cuda()
						candidate_PCs_torch = Variable(candidate_PCs_torch, requires_grad=False).cuda()

						data_time.update((time.time() - end))
						end = time.time()
						# (B,128) (B, 3, 128)，(B, 3+2, 64)，(B, 64, 3)
						completion_points,pred_hm, pred_loc,pred_z_axis= model(model_PC_torch,candidate_PCs_torch)
						batch_time.update(time.time() - end)
						end = time.time()

						with torch.no_grad():
							hm=pred_hm.sigmoid_()
							xy_img_z_ry=mot_decode(hm,pred_loc,pred_z_axis,K=1)

						xy_img_z_ry_cpu = xy_img_z_ry.squeeze(0).detach().cpu().numpy()
						xy_img_z_ry_cpu[:,:2]=(xy_img_z_ry_cpu[:,:2]+dataset.min_img_coord)*dataset.xy_size
						estimation_box_cpu=xy_img_z_ry_cpu[0]
						score=estimation_box_cpu[4]
						ry = estimation_box_cpu[3]

						box = utils.getOffsetBBtest(ref_BB, estimation_box_cpu[0:4])
						results_BBs.append(box)
					# estimate overlap/accuracy fro current sample

					this_overlap = estimateOverlap(BBs[i], results_BBs[-1], dim=IoU_Space)
					this_accuracy = estimateAccuracy(BBs[i], results_BBs[-1], dim=IoU_Space)

					Success_main.add_overlap(this_overlap)
					Precision_main.add_accuracy(this_accuracy)

					for num in range(3):
						if points_on_target < sparse_point[num + 1]:
							interval_Success[num].add_overlap(this_overlap)
							interval_Precision[num].add_accuracy(this_accuracy)
							break
					if points_on_target >= sparse_point[3]:
						interval_Success[3].add_overlap(this_overlap)
						interval_Precision[3].add_accuracy(this_accuracy)

					Success_batch.add_overlap(this_overlap)
					Precision_batch.add_accuracy(this_accuracy)

					if dataset.which_dataset == 'NUSCENES' and this_anno['is_key_frame']==1:
						Success_key_main.add_overlap(this_overlap)
						Precision_key_main.add_accuracy(this_accuracy)
					# measure elapsed time
					post_process_time.update(time.time() - end)
					end = time.time()

					t.update(1)

					if Success_main.count >= max_iter and max_iter >= 0:
						return Success_main.average, Precision_main.average

				t.set_description('Test {}: '.format(epoch) +
				                  'forward {:.3f}s '.format(batch_time.sum) +
                                  '(it:{:.3f}s) '.format(batch_time.avg) +
                                  'pre:{:.3f}s '.format(data_time.sum) +
                                  '(it:{:.3f}s), '.format(data_time.avg) +
                                  '(post:{:.3f}s), '.format(post_process_time.sum) +
                                  '(it:{:.3f}s), '.format(post_process_time.avg) +
				                  'Succ/Prec:' +
				                  '{:.1f}/'.format(Success_main.average) +
				                  '{:.1f} '.format(Precision_main.average)
				                  )
				logging.info('track_id:{} '.format(this_anno["track_id"]) + 'Succ/Prec:' +
				             '{:.1f}/'.format(Success_batch.average) +
				             '{:.1f}'.format(Precision_batch.average)+
				             ' tracklet_frames:{}'.format(len(PCs)))
				Success_batch.reset()
				Precision_batch.reset()
	for num in range(4):
		logging.info(
			"interval:{} frame:{} Succ/Prec:{}/{}".format(num + 1, interval_Success[num].count,
			                                              interval_Success[num].average,
			                                              interval_Precision[num].average))
	logging.info(
		"total_tracklet_nums:{} actual_tracklet_nums:{} total_track_anno_nums:{} actual_track_anno_nums:{}".format(len(loader.dataset.list_of_tracklet_anno),
		                                                                           tracklet_nums,
		                                                                           len(loader.dataset.list_of_anno),
		                                                                           track_anno_nums))
	logging.info(
		"key_frame mean Succ/Prec {}/{} key_frame_num {}".format(Success_key_main.average, Precision_key_main.average,
		                                                         Success_key_main.count))

	return Success_main.average, Precision_main.average


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--which_dataset', type=str, default='NUSCENES', help='datasets:NUSCENES,KITTI')
	parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
	parser.add_argument('--model_dir', type=str, default='nusc_model/Car', help='output folder')
	parser.add_argument('--data_dir', type=str, default='/opt/data/common/nuScenes/KITTI_style/val',
	                    help='dataset path')#/opt/data/common/nuScenes/KITTI_style/val /opt/data/common/kitti_tracking/kitti_t_o/training
	parser.add_argument('--model', type=str, default='netR_36.pth', help='model name for training resume')
	parser.add_argument('--category_name', type=str, default='Car', help='Object to Track (Car/Pedestrian/Van/Cyclist)')
	parser.add_argument('--shape_aggregation', required=False, type=str, default='firstandprevious',
	                    help='Aggregation of shapes (first/previous/firstandprevious/all)')
	parser.add_argument('--reference_BB', required=False, type=str, default="previous_result",
	                    help='previous_result/previous_gt/current_gt')
	parser.add_argument('--model_fusion', required=False, type=str, default="pointcloud",
	                    help='early or late fusion (pointcloud/latent/space)')
	parser.add_argument('--IoU_Space', required=False, type=int, default=3, help='IoUBox vs IoUBEV (2 vs 3)')

	args = parser.parse_args()

	interval = {'car': [0, 150, 1000, 2500], 'pedestrian': [0, 100, 500, 1000], 'van': [0, 150, 1000, 2500],
	            'cyclist': [0, 100, 500, 1000]}
	sparse_interval=interval[args.category_name.lower()]
	kitti_to_nusc_category = {'car': 'car', 'pedestrian': 'pedestrian', 'van': 'truck', 'cyclist': 'bicycle'}

	if args.which_dataset.upper() == 'NUSCENES':
		args.category_name = kitti_to_nusc_category[args.category_name.lower()]
	print(args)
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'
	logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
		                    filename=os.path.join(args.model_dir, datetime.now().strftime('%Y-%m-%d %H-%M-%S.log')),
		                    level=logging.INFO)
	logging.info(args.__repr__())
	logging.info('======================================================' + 'model:{}'.format(args.model))

	args.manualSeed = 1
	random.seed(args.manualSeed)
	torch.manual_seed(args.manualSeed)

	dataset_Test = SiameseTest(
			which_dataset=args.which_dataset,
			input_size=1024,
			path=args.data_dir,
			split='Test',
			category_name=args.category_name,
			offset_BB=0,
			scale_BB=1.25,
			voxel_size=[0.3, 0.3, 0.3],
			xy_size=[0.3, 0.3])

	test_loader = torch.utils.data.DataLoader(
			dataset_Test,
			collate_fn=lambda x: x,
			batch_size=1,
			shuffle=False,
			num_workers=16,
			pin_memory=True)

	netR = Pointnet_Tracking(input_channels=0, use_xyz=True,mode=False,
		                         voxel_size=torch.from_numpy(dataset_Test.voxel_size.copy()).float(),
		                         voxel_area=dataset_Test.voxel_grid_size,
		                         scene_ground=torch.from_numpy(dataset_Test.scene_ground.copy()).float())
	netR.voxelize.extra_repr()
	if args.ngpu > 1:
		netR = torch.nn.DataParallel(netR, range(args.ngpu))
	if args.model != '' and args.ngpu > 1:
		netR.load_state_dict(torch.load(os.path.join(args.model_dir, args.model)))
	elif args.model != '' and args.ngpu <= 1:
		state_dict_ = torch.load(os.path.join(args.model_dir, args.model),
			                         map_location=lambda storage, loc: storage)
		print('loaded {}'.format(os.path.join(args.model_dir, args.model)))
		state_dict = {}
		for k in state_dict_:
			if k.startswith('module') and not k.startswith('module_list'):
				state_dict[k[7:]] = state_dict_[k]
			else:
				state_dict[k] = state_dict_[k]
		netR.load_state_dict(state_dict,strict=True)
	netR.cuda()
	torch.cuda.synchronize()

	Success_run = AverageMeter()
	Precision_run = AverageMeter()

	if dataset_Test.isTiny():
		max_epoch = 2
	else:
		max_epoch = 1

	for epoch in range(max_epoch):
		Succ, Prec = test(
				test_loader,
				netR,
				shape_aggregation=args.shape_aggregation,
				reference_BB=args.reference_BB,
				model_fusion=args.model_fusion,
				IoU_Space=args.IoU_Space,
				model_dir=args.model_dir,
				sparse_point=sparse_interval,
			)
		Success_run.update(Succ)
		Precision_run.update(Prec)
		logging.info("mean Succ/Prec {}/{}".format(Success_run.avg, Precision_run.avg))
