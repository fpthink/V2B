from torch.utils.data import Dataset,DataLoader
from data_classes import PointCloud, Box
from pyquaternion import Quaternion
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import kitty_utils as utils
from kitty_utils import *
from searchspace import KalmanFiltering
import logging
import time
import pickle
from voxel.voxel_grid import VoxelGrid
logger =logging.getLogger('global')

class kittiDataset():

    def __init__(self, path,which_dataset):
        self.which_dataset=which_dataset
        self.KITTI_Folder = path
        self.KITTI_velo = os.path.join(self.KITTI_Folder, "velodyne")
        self.KITTI_label = os.path.join(self.KITTI_Folder, "label_02")

    def getSceneID(self, split):
        if self.which_dataset.upper() == 'NUSCENES':
            if "TRAIN" in split.upper():  # Training SET
                if "TINY" in split.upper():
                    sceneID = [0]
                else:
                    sceneID = list(range(0, 350))
            elif "VALID" in split.upper():  # Validation Set
                if "TINY" in split.upper():
                    sceneID = [0]
                else:
                    sceneID = list(range(0, 10))
            elif "TEST" in split.upper():  # Testing Set
                if "TINY" in split.upper():
                    sceneID = [0]
                else:
                    sceneID = list(range(0, 150))
        elif self.which_dataset.upper() == 'KITTI':
            if "TRAIN" in split.upper():  # Training SET
                if "TINY" in split.upper():
                    sceneID = [12]
                else:
                    sceneID = list(range(0, 17))
            elif "VALID" in split.upper():  # Validation Set
                if "TINY" in split.upper():
                    sceneID = [18]
                else:
                    sceneID = list(range(17, 19))
            elif "TEST" in split.upper():  # Testing Set
                if "TINY" in split.upper():
                    sceneID = [19]
                else:
                    sceneID = list(range(19, 21))

            else:  # Full Dataset
                sceneID = list(range(21))
        return sceneID
    def getBBandPC(self, anno):
        calib_path = os.path.join(self.KITTI_Folder, 'calib',
                                  anno['scene'] + ".txt")
        calib = self.read_calib_file(calib_path)
        transf_mat = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))
        PC, box = self.getPCandBBfromPandas(anno, transf_mat)
        return PC, box

    def getListOfAnno(self, sceneID, category_name="Car"):
        list_of_scene = [
            path for path in os.listdir(self.KITTI_velo)
            if os.path.isdir(os.path.join(self.KITTI_velo, path)) and
            int(path) in sceneID
        ]
        list_of_scene=sorted(list_of_scene)
        print(list_of_scene)
        list_of_tracklet_anno = []
        for scene in list_of_scene:

            label_file = os.path.join(self.KITTI_label, scene + ".txt")
            if self.which_dataset.upper()=='NUSCENES':
                df = pd.read_csv(
                    label_file,
                    sep=' ',
                    names=[
                        "frame", "track_id", "type", "truncated", "occluded",
                        "alpha", "bbox_left", "bbox_top", "bbox_right",
                        "bbox_bottom", "height", "width", "length", "x", "y", "z",
                        "rotation_y","score",'num_lidar_pts','is_key_frame'
                    ])
            else:
                df = pd.read_csv(
                    label_file,
                    sep=' ',
                    names=[
                        "frame", "track_id", "type", "truncated", "occluded",
                        "alpha", "bbox_left", "bbox_top", "bbox_right",
                        "bbox_bottom", "height", "width", "length", "x", "y", "z",
                        "rotation_y", "score"
                    ])
            df = df[df["type"] == category_name]
            df.insert(loc=0, column="scene", value=scene)
            for track_id in df.track_id.unique():
                df_tracklet = df[df["track_id"] == track_id]
                df_tracklet = df_tracklet.reset_index(drop=True)

                tracklet_anno = [anno for index, anno in df_tracklet.iterrows()]
                list_of_tracklet_anno.append(tracklet_anno)

        return list_of_tracklet_anno

    def getPCandBBfromPandas(self, box, calib):
        center = [box["x"], box["y"]- box["height"] / 2 , box["z"]]
        size = [box["width"], box["length"], box["height"]]

        orientation = Quaternion(
            axis=[0, 1, 0], radians=box["rotation_y"]) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)
        BB = Box(center, size, orientation)

        try:
            # VELODYNE PointCloud
            velodyne_path = os.path.join(self.KITTI_velo, box["scene"],
                                         '{:06}.bin'.format(box["frame"]))
            PC = PointCloud(
                np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
            PC.transform(calib)
        except :
            # in case the Point cloud is missing
            # (0001/[000177-000180].bin)
            PC = PointCloud(np.array([[0, 0, 0]]).T)

        return PC, BB

    def read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    ind = values[0].find(':')
                    if ind != -1:
                        data[values[0][:ind]] = np.array(
                            [float(x) for x in values[1:]]).reshape(3, 4)
                    else:
                        data[values[0]] = np.array(
                            [float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]).reshape(3, 3)
        return data


class SiameseDataset(Dataset):

    def __init__(self,
                 which_dataset,
                 input_size,
                 path,
                 split,
                 category_name="Car",
                 regress="GAUSSIAN",
                 offset_BB=0,
                 scale_BB=1.0,
                 ):

        self.dataset = kittiDataset(path=path,which_dataset=which_dataset)

        self.input_size = input_size
        self.split = split
        self.sceneID = self.dataset.getSceneID(split=split)
        self.getBBandPC = self.dataset.getBBandPC

        self.category_name = category_name
        self.regress = regress
        self.list_of_tracklet_anno = self.dataset.getListOfAnno(
            self.sceneID, category_name)
        self.list_of_anno = [
            anno for tracklet_anno in self.list_of_tracklet_anno
            for anno in tracklet_anno
        ]

    def isTiny(self):
        return ("TINY" in self.split.upper())

    def __getitem__(self, index):
        return self.getitem(index)


class SiameseTrain(SiameseDataset):

    def __init__(self,
                 which_dataset,
                 input_size,
                 path,
                 split="",
                 category_name="Car",
                 regress="GAUSSIAN",
                 sigma_Gaussian=1,
                 offset_BB=0,
                 scale_BB=1.0,
                 voxel_size=[0.2, 0.2, 0.2],
                 xy_size=[0.2, 0.2],
                 area_extents=[-5.6, 5.6, -3.6, 3.6, -2.4, 2.4],
                 xy_area_extents=[-5.6, 5.6, -3.6, 3.6],
                 downsample=1.0,
                 regress_radius=2,
                 num_T=10
                 ):
        super(SiameseTrain,self).__init__(
            which_dataset=which_dataset,
            input_size=input_size,
            path=path,
            split=split,
            category_name=category_name,
            regress=regress,
            offset_BB=offset_BB,
            scale_BB=scale_BB,
        )

        self.sigma_Gaussian = sigma_Gaussian
        self.offset_BB = offset_BB
        self.scale_BB = scale_BB
        self.voxel_size=np.array(voxel_size)
        self.xy_size=np.array(xy_size)*downsample
        self.area_extents = np.array(area_extents).reshape(3, 2)
        self.xy_area_extents = np.array(xy_area_extents).reshape(2, 2)
        self.regress_radius=regress_radius
        self.num_T=num_T
        voxel_extents_transpose = self.area_extents.copy().transpose()
        self.scene_ground = voxel_extents_transpose[0]
        self.voxel_grid_size = np.ceil(voxel_extents_transpose[1] / self.voxel_size) - np.floor(
            voxel_extents_transpose[0] / self.voxel_size)
        self.voxel_grid_size = self.voxel_grid_size.astype(np.int32)

        self.num_candidates_perframe = 4

        logging.info("preloading PC...")

        if which_dataset.upper()=="NUSCENES":
            dataset = 'val' if 'val' in self.split.lower() else 'train_track'
            preload_data_path = os.path.join('/opt/data/private_nas/nusc_kitti_all/', dataset,
                                         "preload_{}_{}_10.dat".format(self.category_name, self.split.lower()))
            print(preload_data_path)
        training_samples = []
        self.list_of_PCs = [None] * len(self.list_of_anno)
        self.list_of_BBs = [None] * len(self.list_of_anno)
        if which_dataset.upper() == "NUSCENES" and os.path.isfile(preload_data_path):
            print('loading from saved file {}.'.format(preload_data_path))
            with open(preload_data_path, 'rb') as f:
                training_samples = pickle.load(f)
            self.list_of_PCs = training_samples[0]
            self.list_of_BBs = training_samples[1]
        else:
            for index in tqdm(range(len(self.list_of_anno))):
                anno = self.list_of_anno[index]
                PC, box = self.getBBandPC(anno)
                new_PC = utils.cropPC(PC, box, offset=10)
                self.list_of_PCs[index] = new_PC
                self.list_of_BBs[index] = box

            if which_dataset.upper()=="NUSCENES":
                training_samples.append(self.list_of_PCs)
                training_samples.append(self.list_of_BBs)
                with open(preload_data_path, 'wb') as f:
                    print('saving loaded data to {}'.format(preload_data_path))
                    pickle.dump(training_samples, f)
        logging.info("PC preloaded!")



        logging.info("preloading Model..")
        self.model_PC = [None] * len(self.list_of_tracklet_anno)

        f=open('all_car_obeject_pc_num.txt','w')
        for i in tqdm(range(len(self.list_of_tracklet_anno))):
            list_of_anno = self.list_of_tracklet_anno[i]
            PCs = []
            BBs = []
            cnt = 0
            for anno in list_of_anno:
                this_PC, this_BB = self.getBBandPC(anno)

                PCs.append(this_PC)
                BBs.append(this_BB)

                anno["model_idx"] = i
                anno["relative_idx"] = cnt
                cnt += 1

            self.model_PC[i] = getModel(
                PCs, BBs, offset=self.offset_BB, scale=self.scale_BB)
            # pc=self.model_PC[i].points.copy()
            # pc_symmetry=pc.copy()
            # pc_symmetry[1,:]=-pc_symmetry[1,:]
            # pc=np.concatenate((pc,pc_symmetry),axis=1)
            # pc=self.unique(pc.T)
            # model_pc=PointCloud(pc)
            # self.model_PC_symmetry[i]=model_pc

        logging.info("Model preloaded!")
    #
    def unique_pc(self,pc):
        #pc:nx3
        order = np.lexsort(pc.T)
        pc = pc[order]
        diff = np.diff(pc, axis=0)
        ui = np.ones(len(pc), 'bool')
        ui[1:] = (diff != 0).any(axis=1)
        return pc[ui].T

    def __getitem__(self, index):
        return self.getitem(index)

    def getPCandBBfromIndex(self, anno_idx):
        this_PC = self.list_of_PCs[anno_idx]
        this_BB = self.list_of_BBs[anno_idx]
        return this_PC, this_BB

    def getitem(self, index):
        anno_idx = self.getAnnotationIndex(index)
        sample_idx = self.getSearchSpaceIndex(index)

        if sample_idx == 0:
            sample_offsets = np.zeros(4)
        else:
            gaussian = KalmanFiltering(bnd=[1, 1, 0.5, 5])
            sample_offsets = gaussian.sample(1)[0]

        this_anno = self.list_of_anno[anno_idx]
        model_idx=this_anno["model_idx"]
        this_PC, this_BB = self.getPCandBBfromIndex(anno_idx)
        sample_BB = utils.getOffsetBB(this_BB, sample_offsets)

        sample_PC, sample_label, sample_reg, new_box_gt,align_gt_PC = utils.cropAndCenterPC_label(
            this_PC,sample_BB, this_BB, sample_offsets, offset=self.offset_BB, scale=self.scale_BB,limit_area=self.area_extents)
        if sample_PC.nbr_points() <= 20:
            return self.getitem(np.random.randint(0, self.__len__()))

        sample_PC, sample_label, sample_reg,align_gt_PC,all_label = utils.regularizePCwithlabel(sample_PC,align_gt_PC,sample_label,sample_reg,self.input_size)

        if this_anno["relative_idx"] == 0:
            prev_idx = anno_idx
            fir_idx = anno_idx
        else:
            prev_idx = anno_idx - 1
            fir_idx = anno_idx - this_anno["relative_idx"]
        gt_PC_pre, gt_BB_pre = self.getPCandBBfromIndex(prev_idx)
        gt_PC_fir, gt_BB_fir = self.getPCandBBfromIndex(fir_idx)

        if sample_idx == 0:
            samplegt_offsets = np.zeros(4)
        else:
            samplegt_offsets = np.random.uniform(low=-0.3, high=0.3, size=4)
            samplegt_offsets[3] = samplegt_offsets[3]*5.0
        gt_BB_pre= utils.getOffsetBB(gt_BB_pre, samplegt_offsets)

        gt_PC = getModel([gt_PC_fir,gt_PC_pre], [gt_BB_fir,gt_BB_pre], offset=self.offset_BB, scale=self.scale_BB)

        if gt_PC.nbr_points() <= 20:
            return self.getitem(np.random.randint(0, self.__len__()))
        gt_PC = utils.regularizePC(gt_PC,self.input_size)

        completion_PC_part = utils.regularizePC(self.model_PC[model_idx],self.input_size*4)
        rot=Quaternion(axis=[0, 0, 1], angle=-sample_offsets[3] * np.pi / 180)
        #3 N
        completion_PC_part = completion_PC_part.transpose(0, 1).contiguous()
        completion_PC_part = completion_PC_part.numpy()
        completion_PC_part = np.dot(rot.rotation_matrix, completion_PC_part)
        completion_PC_part=torch.from_numpy(completion_PC_part).float()


        extents_transpose = self.xy_area_extents.transpose()
        if extents_transpose.shape != (2, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents_transpose.shape))

        # Set image grid extents
        self.min_img_coord = np.floor(extents_transpose[0] / self.xy_size)
        self.max_img_coord = np.ceil(extents_transpose[1] / self.xy_size) - 1
        self.img_size=((self.max_img_coord - self.min_img_coord) + 1).astype(np.int32)
        output_h=self.img_size[1]
        output_w=self.img_size[0]
        #hm
        corners=new_box_gt.corners()[:2].T
        corners=corners[[0,2,4,6]]
        corners_int=np.floor((corners / self.xy_size) - self.min_img_coord).astype(np.int32)
        corners_int_ul=np.min(corners_int,axis=0)
        corners_int_br=np.max(corners_int,axis=0)
        x = np.arange(corners_int_ul[0], corners_int_br[0]+1, 1)
        y = np.arange(corners_int_ul[1], corners_int_br[1]+1, 1)
        xx,yy=np.meshgrid(x,y)
        hm_grid=np.concatenate([xx[:,:,np.newaxis],yy[:,:,np.newaxis]],axis=2)
        ct=sample_reg[0,:2].numpy()
        ct_image = (ct/self.xy_size)-self.min_img_coord
        ct_image_int=np.floor(ct_image).astype(np.int32)
        # (1,h,w)
        hm_grid=np.sqrt(np.sum((hm_grid-ct_image_int)**2,axis=2))
        hm_grid[hm_grid==0]=1e-6
        hm_grid=1.0/hm_grid

        hm = np.zeros((1,output_h, output_w), dtype=np.float32)
        if hm[0,corners_int_ul[1]:corners_int_br[1]+1,corners_int_ul[0]:corners_int_br[0]+1].shape!=(corners_int_br[1]+1-corners_int_ul[1],corners_int_br[0]+1-corners_int_ul[0]):
            print(hm_grid.shape)
            return self.getitem(np.random.randint(0, self.__len__()))
        hm[0,corners_int_ul[1]:corners_int_br[1]+1,corners_int_ul[0]:corners_int_br[0]+1]=hm_grid
        hm[0,ct_image_int[1],ct_image_int[0]]=1.0
        hm[0,[ct_image_int[1],ct_image_int[1],ct_image_int[1]+1,ct_image_int[1]-1],[ct_image_int[0]-1,ct_image_int[0]+1,ct_image_int[0],ct_image_int[0]]]=0.8

        # (2r+1,3)x,y,ry
        loc_offsets = np.zeros(((2*self.regress_radius+1)**2, 3), dtype=np.float32)
        # (1,3)
        hwl = np.array([[this_anno["height"],this_anno["width"], this_anno["length"]]],dtype=np.float32)
        # (1,1)
        z_axis = np.array([[sample_reg[0,2]]],dtype=np.float32)
        #center ind
        ind_ct = np.array([ct_image_int[1]*output_w+ct_image_int[0]], dtype=np.int64)
        #loc offsets ind
        ind_offsets = np.zeros(((2*self.regress_radius+1)**2,), dtype=np.int64)

        for i in range(-self.regress_radius,self.regress_radius+1):
            for j in range(-self.regress_radius,self.regress_radius+1):
                offsets=np.zeros((3,),dtype=np.float32)
                offsets[:2]= ct_image - ct_image_int-np.array([i,j])
                #ry
                offsets[2]=sample_reg[0,3]
                loc_offsets[(i+self.regress_radius)*(2*self.regress_radius+1)+(j+self.regress_radius)] = offsets
                ind_int=ct_image_int+np.array([i,j])
                ind_offsets[(i+self.regress_radius)*(2*self.regress_radius+1)+(j+self.regress_radius)]=ind_int[1]*output_w+ind_int[0]


        hm=torch.from_numpy(hm)
        ind_ct=torch.from_numpy(ind_ct)
        hwl=torch.from_numpy(hwl)
        z_axis=torch.from_numpy(z_axis)
        ind_offsets=torch.from_numpy(ind_offsets)
        loc_offsets=torch.from_numpy(loc_offsets)

        ret={'search_pc':sample_PC,
             'model_pc':gt_PC,
             'search_seeds_label':sample_label,
             'search_all_label':all_label,
             'completion_PC_part':completion_PC_part,
             'hm':hm,
             'ind_ct':ind_ct,
             'hwl':hwl,
             'z_axis':z_axis,
             'ind_offsets':ind_offsets,
             'loc_reg':loc_offsets}
        return ret

    def __len__(self):
        nb_anno = len(self.list_of_anno)
        return nb_anno * self.num_candidates_perframe

    def getAnnotationIndex(self, index):
        return int(index / (self.num_candidates_perframe))

    def getSearchSpaceIndex(self, index):
        return int(index % self.num_candidates_perframe)

    def _gather_feat(self,feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat
    def _tranpose_and_gather_feat(self,feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat


class SiameseTest(SiameseDataset):

    def __init__(self,
                 which_dataset,
                 input_size,
                 path,
                 split="",
                 category_name="Car",
                 regress="GAUSSIAN",
                 offset_BB=0,
                 scale_BB=1.0,
                 voxel_size=[0.2, 0.2, 0.2],
                 xy_size=[0.2, 0.2],
                 # area_extents=[-2.0, 2.0, -2.0, 2.0, -1.5, 1.5],
                 # xy_area_extents=[-2.0, 2.0, -2.0, 2.0],
                 area_extents=[-5.6, 5.6, -3.6, 3.6, -2.4, 2.4],
                 xy_area_extents=[-5.6, 5.6, -3.6, 3.6],
                 downsample=1.0,
                 ):
        super(SiameseTest,self).__init__(
            which_dataset=which_dataset,
            input_size=input_size,
            path=path,
            split=split,
            category_name=category_name,
            regress=regress,
            offset_BB=offset_BB,
            scale_BB=scale_BB)
        self.which_dataset=which_dataset
        self.split = split
        self.offset_BB = offset_BB
        self.scale_BB = scale_BB
        self.voxel_size = np.array(voxel_size)
        self.area_extents = np.array(area_extents).reshape(3, 2)
        self.xy_size = np.array(xy_size) * downsample
        self.xy_area_extents = np.array(xy_area_extents).reshape(2, 2)
        extents_transpose = np.array(self.xy_area_extents).transpose()
        if extents_transpose.shape != (2, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents_transpose.shape))

        # Set image grid extents
        self.min_img_coord = np.floor(extents_transpose[0] / self.xy_size)
        voxel_extents_transpose = self.area_extents.transpose()
        self.scene_ground = voxel_extents_transpose[0]
        self.voxel_grid_size = np.ceil(voxel_extents_transpose[1] / self.voxel_size) - np.floor(
            voxel_extents_transpose[0] / self.voxel_size)
        self.voxel_grid_size = self.voxel_grid_size.astype(np.int32)

    def getitem(self, index):
        list_of_anno = self.list_of_tracklet_anno[index]
        PCs = []
        BBs = []
        for anno in list_of_anno:
            this_PC, this_BB = self.getBBandPC(anno)
            PCs.append(this_PC)
            BBs.append(this_BB)
        return PCs, BBs, list_of_anno

    def __len__(self):
        return len(self.list_of_tracklet_anno)




