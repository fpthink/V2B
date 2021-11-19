from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pointops.functions import pointops
import etw_pytorch_utils as pt_util
# from util import pt_util


class _PointNet2SAModuleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.groupers = None
        self.mlps = None
        self.use_edge = True

    def forward(self, xyz: torch.Tensor ,features: torch.Tensor = None, npoint:int =None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        self.npoint = npoint
        new_features_list = []
        xyz_trans = xyz.transpose(1, 2).contiguous()
        if self.npoint is not None:
            idx = torch.arange(self.npoint).repeat(xyz.size(0), 1).int().cuda()
            new_xyz= pointops.gathering(
                xyz_trans,
                idx
                # pointops.furthestsampling(xyz, self.npoint)
            ).transpose(1, 2).contiguous()
        else:
            new_xyz=None
            idx=None
        if self.use_edge and features is not None :
            # (B, C, npoint)
            idx=idx.long()
            center_features=features.transpose(1,2)[torch.arange(xyz.size(0)).repeat(self.npoint,1).transpose(0,1),idx,:].transpose(1,2).contiguous()
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)  # (B, 3+C, npoint, nsample)
            if self.use_edge and features is not None:
                #(B, C, npoint, nsample)
                center_features=center_features.unsqueeze(-1).expand(-1,-1,-1,self.groupers[i].nsample)
                # (B, 3+2c, npoint, nsample)
                new_features=torch.cat([new_features[:,0:3,:,:],new_features[:,3:,:,:]-center_features,center_features],dim=1)
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointNet2SAModuleMSG(_PointNet2SAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping
    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet_old before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True, use_xyz: bool = True,use_edge:bool=True):
        super().__init__()
        self.use_edge = use_edge
        assert len(radii) == len(nsamples) == len(mlps)
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        # if use_edge:
        #     self.edge=nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointops.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                # if npoint is not None else pointops.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_edge:
                # self.edge.append(
                #     nn.Sequential(*[nn.Conv2d(2 * mlp_spec[-1], mlp_spec[-1], kernel_size=(1, 1), bias=False),
                #                     nn.BatchNorm2d(mlp_spec[-1]),
                #                     nn.ReLU(),
                #                     nn.Conv2d(mlp_spec[-1], mlp_spec[-1], kernel_size=(1, 1), bias=False),
                #                     nn.BatchNorm2d(mlp_spec[-1]),
                #                     nn.ReLU()],
                #                   ))
                mlp_spec[0]*=2
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_util.SharedMLP(mlp_spec, bn=bn))


class PointNet2SAModule(PointNet2SAModuleMSG):
    r"""Pointnet set abstrction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet_old before the global max_pool
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], radius: float = None, nsample: int = None, bn: bool = True, use_xyz: bool = True,use_edge:bool=True):
        super().__init__(mlps=[mlp], radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz,use_edge=use_edge)


class PointnetProposalModule(PointNet2SAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
            self, mlp, radius=None, nsample=None, bn=True, use_xyz=True, vote=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetProposalModule, self).__init__(
            mlps=[mlp],
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
            vote=vote
        )

    def forward(self, xyz, features, npoint, score):
        # type: (_PointnetSAModuleBase, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        self.npoint = npoint
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()

        new_xyz = (
            pointops.gathering(
                # xyz_flipped, pointops.furthest_point_sample(xyz, self.npoint)
                xyz_flipped, torch.arange(self.npoint).repeat(xyz.size(0), 1).int().cuda()
            )
                .transpose(1, 2)
                .contiguous()
        )

        for i in range(len(self.groupers)):
            new_features, score_id = self.groupers[i](
                xyz, new_xyz, score, features
            )  # (B, C, npoint, nsample)
            # score_id = new_features[:,3,:,:].sum(dim = 2).argmax(dim = 1)

            # B
            # new_features_cpu = new_features.squeeze(0).detach().cpu().numpy()
            # np.savetxt('vote4.txt',new_features_cpu[0:4,i,:])
            idx = torch.arange(new_features.size(0))
            new_features = new_features[idx, :, score_id, :]
            # B*C*nsample
            new_features = new_features.unsqueeze(2)
            # B*C*1*nsample
            new_xyz = new_xyz[idx, score_id, :]
            # B*3

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1).squeeze(-1)  # (B, mlp[-1])

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)

class PointNet2FPModule(nn.Module):
    r"""Propigates the features of one set to another
    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = pt_util.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated
        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointops.nearestneighbor(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointops.interpolation(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats
        return self.mlp(new_features.unsqueeze(-1)).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = torch.randn(2, 9, 3, requires_grad=True).cuda()
    xyz_feats = torch.randn(2, 9, 6, requires_grad=True).cuda()

    test_module = PointNet2SAModuleMSG(npoint=2, radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]])
    test_module.cuda()
    print(test_module(xyz, xyz_feats))

    # test_module = PointNet2FPModule(mlp=[6, 6])
    # test_module.cuda()
    # from torch.autograd import gradcheck
    # inputs = (xyz, xyz, None, xyz_feats)
    # test = gradcheck(test_module, inputs, eps=1e-6, atol=1e-4)
    # print(test)

    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats)
        new_features.backward(torch.cuda.FloatTensor(*new_features.size()).fill_(1))
        print(new_features)
        print(xyz.grad)
