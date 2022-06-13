import pdb
import torch
import torch.nn as nn
import etw_pytorch_utils as pt_utils
from lib.pointops.functions import pointops
from collections import namedtuple
import torch.nn.functional as F

from modules.pointnet2.utils.pointnet2_modules import PointNet2SAModule, PointNet2FPModule
from modules.voxel_utils.voxel.voxelnet import Conv_Middle_layers
from modules.voxel_utils.voxel.region_proposal_network import RPN
from modules.voxel_utils.voxelization import Voxelization

from modules.backbone_net import Pointnet2_Backbone
from modules.completion_net import ts_up_sampling

class V2B_Tracking(nn.Module):
    def __init__(self, opts):
        super(V2B_Tracking, self).__init__()
        self.opts = opts
        
        # voxel net
        self.voxel_size = opts.voxel_size
        self.voxel_area = opts.voxel_area
        self.scene_ground = opts.scene_ground
        self.min_img_coord = opts.min_img_coord
        self.xy_size = opts.xy_size
        
        self.mode = opts.mode
        self.feat_emb = opts.feat_emb

        self.backbone_net = Pointnet2_Backbone(opts.n_input_feats, use_xyz=opts.use_xyz)
        
        self.mask = nn.Sequential(
            nn.Conv1d(64,64,kernel_size=1,bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(64,1,kernel_size=1,bias=True),
            nn.Sigmoid()
        )
        self.cosine = nn.CosineSimilarity(dim=1)

        self.mlp = (
                pt_utils.Seq(4+32+32)
                .conv1d(32, bn=True)
                .conv1d(32, bn=True))
        self.global_weight = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=(1,1),bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32,32,kernel_size=(1,1),bias=True),
            nn.Sigmoid()
        )
        self.global_mlp = pt_utils.SharedMLP([3+32,32,32], bn=True)
        self.fea_layer = (pt_utils.Seq(64)
                .conv1d(64, bn=True)
                .conv1d(64, activation=None))

        self.completion_fc=ts_up_sampling(input_c=64,mid_c=32)

        self.voxelize = Voxelization(self.voxel_area[0], self.voxel_area[1], self.voxel_area[2], 
                                     scene_ground=self.scene_ground, mode=opts.mode, voxel_size=self.voxel_size)
        self.cml = Conv_Middle_layers(inplanes=3+64)
        self.RPN = RPN()

    def xcorr(self, x_label, x_object, template_xyz):

        B = x_object.size(0)
        f = x_object.size(1)
        n1 = x_object.size(2)
        n2 = x_label.size(2)
        final_out_cla = self.cosine(x_object.unsqueeze(-1).expand(B,f,n1,n2), x_label.unsqueeze(2).expand(B,f,n1,n2))
        final_out_cla_de = final_out_cla.detach()
        template_xyz_fea = torch.cat((template_xyz.transpose(1, 2).contiguous(),x_object),dim=1)
        max_ind = torch.argmax(final_out_cla_de,dim=1,keepdim=True).expand(-1,template_xyz_fea.size(1),-1)

        template_fea = template_xyz_fea.gather(dim=2,index=max_ind)
        max_cla = F.max_pool2d(final_out_cla.unsqueeze(dim=1),kernel_size=[final_out_cla.size(1), 1])
        max_cla = max_cla.squeeze(2)
        fusion_feature = torch.cat((max_cla,template_fea,x_label),dim=1)
        fusion_feature = self.mlp(fusion_feature)

        diff_fea = x_label.unsqueeze(2).expand(B,f,n1,n2)-x_object.unsqueeze(-1).expand(B, f, n1, n2)
        global_weight = self.global_weight(diff_fea)
        global_feature = (global_weight*x_object.unsqueeze(-1).expand(B, f, n1, n2))
        global_feature = torch.cat((template_xyz.transpose(1, 2).contiguous().unsqueeze(-1).expand(B,3,n1,n2),global_feature), dim=1)
        global_feature = self.global_mlp(global_feature)
        global_feature = F.max_pool2d(global_feature, kernel_size=[global_feature.size(2), 1])
        global_feature = global_feature.squeeze(2)
        fusion_feature = torch.cat((fusion_feature,global_feature),dim=1)
        fusion_feature = self.fea_layer(fusion_feature)

        return fusion_feature

    def forward(self, template, search):
        r"""
            template: B*512*3 or B*512*6
            search: B*1024*3 or B*1024*6
        """
        template_xyz, template_feature = self.backbone_net(template, [256, 128, 64])

        search_xyz, search_feature = self.backbone_net(search, [512, 256, 128])

        fusion_feature = self.xcorr(search_feature, template_feature,template_xyz)

        #b 1 1024
        search_mask = self.mask(fusion_feature)

        if self.mode:
            weighted_fusion_feature = search_mask * fusion_feature
            completion_points = self.completion_fc(weighted_fusion_feature)
            completion_points = completion_points.unsqueeze(-1)
        else:
            completion_points = None

        fusion_xyz_feature = torch.cat((search_xyz.transpose(1, 2).contiguous(),fusion_feature),dim = 1)
        voxel_features = self.voxelize(fusion_xyz_feature,search_xyz)
        voxel_features = voxel_features.permute(0, 1, 4, 3, 2).contiguous()
        cml_out = self.cml(voxel_features)
        # (b,1,36,56),(b,3,36,56),(b,3,36,56),(b,1,36,56),(b,3x1024,9)
        pred_hm, pred_loc, pred_z_axis = self.RPN(cml_out)

        return completion_points, pred_hm, pred_loc,pred_z_axis

