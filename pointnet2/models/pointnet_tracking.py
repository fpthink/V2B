from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import etw_pytorch_utils as pt_utils
from lib.pointops.functions import pointops
from collections import namedtuple
import torch.nn.functional as F
from pointnet2.utils.pointnet2_modules import PointNet2SAModule, PointNet2FPModule
from voxel.voxelnet import Conv_Middle_layers
from voxel.region_proposal_network import RPN
from modules.voxelization import Voxelization

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class ts_up_sampling(nn.Module):
    def __init__(self,input_c=128,mid_c=64):
        super(ts_up_sampling, self).__init__()
        self.input_c=input_c
        self.duplate=(pt_utils.Seq(input_c)
                .conv1d(2*input_c, bn=True))
        self.c1=(pt_utils.Seq(input_c)
                .conv1d(mid_c, bn=True))
        self.c2=pt_utils.SharedMLP([input_c,mid_c,mid_c,mid_c], bn=True)
        self.c3 = (pt_utils.Seq(mid_c)
                   .conv1d(input_c, bn=True)
                   .conv1d(input_c, bn=True))
        self.c4=(pt_utils.Seq(input_c+mid_c)
                .conv1d(2*input_c)
                .conv1d(3, activation=None))
    def forward(self,search_keypoint):
        B=search_keypoint.size(0)
        N=search_keypoint.size(2)

        up_sampling_fea=self.duplate(search_keypoint)
        up_sampling_fea=up_sampling_fea.view(B,self.input_c,int(2048/N),N)
        up_sampling_fea=up_sampling_fea.view(B,self.input_c,-1)
        up_sampling_fea=self.c1(up_sampling_fea)
        #b c 1
        global_fea=F.max_pool1d(up_sampling_fea,kernel_size=2048)
        #b 2c N K
        up_sampling_fea=get_graph_feature(up_sampling_fea,k=4)
        up_sampling_fea=self.c2(up_sampling_fea)
        up_sampling_fea=F.max_pool2d(up_sampling_fea, kernel_size=[1, up_sampling_fea.size(3)])

        global_fea=self.c3(global_fea)
        #b c N
        local_fea=up_sampling_fea.squeeze(-1)
        final_fea=torch.cat((local_fea,global_fea.expand(-1,-1,local_fea.size(2))),dim=1)
        points_coord=self.c4(final_fea)
        return points_coord

class Pointnet_Backbone(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, use_xyz=True):
        super(Pointnet_Backbone, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNet2SAModule(
                radius=0.3,
                nsample=32,
                mlp=[input_channels, 32, 32, 64],
                use_xyz=use_xyz,
                use_edge=False
            )
        )
        self.SA_modules.append(
            PointNet2SAModule(
                radius=None,#0.5
                nsample=48,
                mlp=[64, 64, 64, 128],
                use_xyz=use_xyz,
                use_edge=False
            )
        )
        self.SA_modules.append(
            PointNet2SAModule(
                radius=None,#0.7
                nsample=48,
                mlp=[128, 128, 128, 128],
                use_xyz=use_xyz,
                use_edge=False
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointNet2FPModule(mlp=[64,32,32]))
        self.FP_modules.append(PointNet2FPModule(mlp=[192,128,64]))
        self.FP_modules.append(PointNet2FPModule(mlp=[256,128,128]))
        self.cov_final = nn.Conv1d(32, 32, kernel_size=1)


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, numpoints):
        # type: (Pointnet2SSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i], numpoints[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in [2,1,0]:
            l_features[i]=self.FP_modules[i](l_xyz[i],l_xyz[i+1],l_features[i],l_features[i+1])
        # self.FP_modules[2](l_xyz[2],l_xyz[3],l_features[2],l_features[3])
        # self.FP_modules[1](l_xyz[1],l_xyz[2],l_features[1],l_features[2])
        # self.FP_modules[0](l_xyz[0],l_xyz[1],l_features[0],l_features[1])


        return l_xyz[0], self.cov_final(l_features[0])




class Pointnet_Tracking(nn.Module):
    r"""
        xorr the search and the template
    """
    def __init__(self, input_channels=3, use_xyz=True, objective = False,mode=True,voxel_size=torch.tensor([0.2,0.2,0.2]),voxel_area=None,scene_ground=None):
        super(Pointnet_Tracking, self).__init__()
        self.mode=mode
        self.backbone_net = Pointnet_Backbone(input_channels, use_xyz)

        # self.mask=(
        #         pt_utils.Seq(128)
        #         # .conv1d(128, bn=True)
        #         .conv1d(1, activation=None))
        self.mask=nn.Sequential(
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
        self.global_weight=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=(1,1),bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32,32,kernel_size=(1,1),bias=True),
            nn.Sigmoid()
        )
        self.global_mlp=pt_utils.SharedMLP([3+32,32,32], bn=True)
        self.fea_layer = (pt_utils.Seq(64)
                .conv1d(64, bn=True)
                .conv1d(64, activation=None))

        self.completion_fc=ts_up_sampling(input_c=64,mid_c=32)


        self.voxelize=Voxelization(voxel_area[0],voxel_area[1],voxel_area[2],scene_ground=scene_ground,mode=mode,voxel_size=voxel_size)
        self.cml=Conv_Middle_layers(inplanes=3+64)
        self.RPN=RPN()

    def xcorr(self, x_label, x_object, template_xyz):

        B = x_object.size(0)
        f = x_object.size(1)
        n1 = x_object.size(2)
        n2 = x_label.size(2)
        final_out_cla = self.cosine(x_object.unsqueeze(-1).expand(B,f,n1,n2), x_label.unsqueeze(2).expand(B,f,n1,n2))
        final_out_cla_de=final_out_cla.detach()
        template_xyz_fea=torch.cat((template_xyz.transpose(1, 2).contiguous(),x_object),dim=1)
        max_ind=torch.argmax(final_out_cla_de,dim=1,keepdim=True).expand(-1,template_xyz_fea.size(1),-1)

        template_fea=template_xyz_fea.gather(dim=2,index=max_ind)
        max_cla=F.max_pool2d(final_out_cla.unsqueeze(dim=1),kernel_size=[final_out_cla.size(1), 1])
        max_cla=max_cla.squeeze(2)
        fusion_feature=torch.cat((max_cla,template_fea,x_label),dim=1)
        fusion_feature = self.mlp(fusion_feature)

        diff_fea=x_label.unsqueeze(2).expand(B,f,n1,n2)-x_object.unsqueeze(-1).expand(B, f, n1, n2)
        global_weight=self.global_weight(diff_fea)
        global_feature=(global_weight*x_object.unsqueeze(-1).expand(B, f, n1, n2))
        global_feature = torch.cat((template_xyz.transpose(1, 2).contiguous().unsqueeze(-1).expand(B,3,n1,n2),global_feature), dim=1)
        global_feature = self.global_mlp(global_feature)
        global_feature = F.max_pool2d(global_feature, kernel_size=[global_feature.size(2), 1])
        global_feature = global_feature.squeeze(2)
        fusion_feature=torch.cat((fusion_feature,global_feature),dim=1)
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
        search_mask=self.mask(fusion_feature)

        if self.mode:
            weighted_fusion_feature = search_mask * fusion_feature
            completion_points = self.completion_fc(weighted_fusion_feature)
            completion_points = completion_points.unsqueeze(-1)
        else:
            completion_points = None

        fusion_xyz_feature = torch.cat((search_xyz.transpose(1, 2).contiguous(),fusion_feature),dim = 1)
        voxel_features=self.voxelize(fusion_xyz_feature,search_xyz)
        voxel_features = voxel_features.permute(0, 1, 4, 3, 2).contiguous()
        cml_out=self.cml(voxel_features)
        # (b,1,36,56),(b,3,36,56),(b,3,36,56),(b,1,36,56),(b,3x1024,9)
        pred_hm, pred_loc,pred_z_axis=self.RPN(cml_out)

        return completion_points,pred_hm, pred_loc,pred_z_axis

