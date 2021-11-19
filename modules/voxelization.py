import torch
import torch.nn as nn
from modules.functional.voxelization import avg_voxelize,favg_voxelize

__all__ = ['Voxelization']


class Voxelization(nn.Module):
    def __init__(self, x,y,z,scene_ground=torch.tensor([-5.6, -3.6, -2.4]),voxel_size=torch.tensor([0.3, 0.3, 0.2]),mode=True):
        super().__init__()
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)
        self.scene_ground=scene_ground
        self.voxel_size=voxel_size
        self.min_voxel_coord=torch.floor(self.scene_ground/ self.voxel_size)
        self.resolution=(-2*self.min_voxel_coord).int()
        self.mode=mode

    def forward(self, features, coords):
        #(b,c,n)(b,n,3)
        coords_detach = coords.detach()
        discrete_pts = torch.floor(coords_detach / self.voxel_size.cuda())
        voxel_indices = (discrete_pts - self.min_voxel_coord.cuda()).int()
        voxel_indices=voxel_indices.transpose(1, 2).contiguous()
        if self.mode:
            return favg_voxelize(features, voxel_indices, self.x,self.y,self.z)
        else:
            return avg_voxelize(features, voxel_indices, self.x, self.y, self.z)

    def extra_repr(self):
        print('information:x {} y {} z {} min_voxel_coord {} voxel_size {} '.format(self.x,self.y,self.z,self.min_voxel_coord,self.voxel_size))
# if __name__ == '__main__':
#     conv=nn.Conv1d(128,128,1,1)
#     voxel=Voxelization(56,36,24)
#     voxel.cuda()
#     conv.cuda()
#     coords_x=torch.rand((1,2048,1),dtype=torch.float32)*5.6
#     coords_y=torch.rand((1,2048,1),dtype=torch.float32)*3.6
#     coords_z=torch.rand((1,2048,1),dtype=torch.float32)*2.4
#     # coord=torch.tensor([[0.1,0.1,0.1],[0,0,0],[0.05,0.05,0.05]],dtype=torch.float32).unsqueeze(dim=0).cuda()
#     coord=torch.cat([coords_x,coords_y,coords_z],dim=2).cuda()
#     output1 = [[], []]
#     for i in range(2):
#         import time
#         import random
#         ti1=time.time()
#         random.seed(0)
#         torch.manual_seed(0)
#         # torch.cuda.manual_seed(0)
#         features = torch.rand(1, 128, 2048, dtype=torch.float32).cuda().requires_grad_()
#         out = conv(features)
#         voxels=voxel(out,coord)
#         ti2 = time.time()
#         print(ti2-ti1)
#         output1[i].append(voxels.clone().detach())
#     for t1, t2 in zip(output1[0], output1[1]):
#         print(t1.equal(t2))
#     # print(voxels[0,:,28,18,12])
#     voxels=voxels.permute(0, 1, 4, 3, 2).contiguous()
#     voxels=voxels.sum()
#     voxels.backward()
#     print(features.grad,features.dtype,voxels.dtype)