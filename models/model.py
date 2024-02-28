import torch.nn

from .tensorBase import *
from utils import N_to_reso
import time
import numpy as np
import math
from .quaternion_utils import *
from einops import rearrange, repeat, reduce, einsum


class TensorDecomposition(torch.nn.Module):
    def __init__(self, grid_size, num_features, scale, device, core_num, reduce_sum=False):
        super(TensorDecomposition, self).__init__()
        self.core_num = core_num
        self.feature = num_features
        self.grid_size = torch.tensor(grid_size)
        self.num_voxels = grid_size[0] * grid_size[1] * grid_size[2]
        self.reduce_sum = reduce_sum

        X, Y, Z = grid_size
        self.plane_xy = torch.nn.Parameter(scale * torch.randn((1, num_features, Y, X), device=device))
        self.plane_yz = torch.nn.Parameter(scale * torch.randn((1, num_features, Z, Y), device=device))
        self.plane_xz = torch.nn.Parameter(scale * torch.randn((1, num_features, Z, X), device=device))
        self.core_tensor = torch.nn.Parameter(torch.randn((self.core_num, num_features), device=device))
        if not self.reduce_sum:
            self.linear_att = torch.nn.Linear(num_features, num_features, bias=False).to(device)


    def forward(self, coords_plane):
        feature_xy = F.grid_sample(self.plane_xy, coords_plane[0], mode='bilinear', align_corners=True)
        feature_yz = F.grid_sample(self.plane_yz, coords_plane[1], mode='bilinear', align_corners=True)
        feature_xz = F.grid_sample(self.plane_xz, coords_plane[2], mode='bilinear', align_corners=True)
        core_tensor = self.core_tensor.squeeze(0)

        out_xyz_prod = feature_xy * feature_yz * feature_xz
        out_xyz_prod = torch.sum(core_tensor.unsqueeze(-1).unsqueeze(-1) * out_xyz_prod, dim=0, keepdim=True)
        out_xyz = out_xyz_prod
        _, C, N, _ = out_xyz.size()
        output = out_xyz.view(-1, N).T

        return output

    def L1loss(self):
        loss = torch.abs(self.plane_xy).mean() + torch.abs(self.plane_yz).mean() + torch.abs(self.plane_xz).mean()
        loss = loss / 6

        return loss

    def TV_loss(self):
        loss = self.TV_loss_com(self.plane_xy)
        loss += self.TV_loss_com(self.plane_yz)
        loss += self.TV_loss_com(self.plane_xz)
        loss = loss / 6

        return loss

    def TV_loss_com(self, x):
        loss = (x[:, :, 1:] - x[:, :, :-1]).pow(2).mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).mean()
        return loss

    def shrink(self, bound):
        # bound [3, 2]
        x, y, z = bound[0], bound[1], bound[2]
        self.plane_xy = torch.nn.Parameter(self.plane_xy.data[:, :, y[0]:y[1], x[0]:x[1]])
        self.plane_yz = torch.nn.Parameter(self.plane_yz.data[:, :, z[0]:z[1], y[0]:y[1]])
        self.plane_xz = torch.nn.Parameter(self.plane_xz.data[:, :, z[0]:z[1], x[0]:x[1]])


        self.grid_size = bound[:, 1] - bound[:, 0]

    def upsample(self,  aabb):
        target_res = N_to_reso(self.num_voxels, aabb)

        self.grid_size = torch.tensor(target_res)

        self.plane_xy = torch.nn.Parameter(F.interpolate(self.plane_xy.data,
                                                         size=(target_res[1], target_res[0]), mode='bilinear',
                                                         align_corners=True))
        self.plane_yz = torch.nn.Parameter(F.interpolate(self.plane_yz.data,
                                                         size=(target_res[2], target_res[1]), mode='bilinear',
                                                         align_corners=True))
        self.plane_xz = torch.nn.Parameter(F.interpolate(self.plane_xz.data,
                                                         size=(target_res[2], target_res[0]), mode='bilinear',
                                                         align_corners=True))

class MultiscaleTensorDecom(torch.nn.Module):
    def __init__(self, num_levels, num_features, base_resolution, max_resolution, device, core_num, reduce_sum=False, scale=0.1):
        super(MultiscaleTensorDecom, self).__init__()
        self.reduce_sum = reduce_sum
        if not reduce_sum:
            self.linear_projection = torch.nn.Linear(num_features, num_features, bias=False).to(device)

        tensors = []

        if num_levels == 1:
            factor = 1
        else:
            factor = math.exp( (math.log(max_resolution) - math.log(base_resolution)) / (num_levels-1) )

        for i in range(num_levels):
            level_resolution = int(base_resolution * factor**i)
            level_grid = (level_resolution, level_resolution, level_resolution)
            tensors.append(TensorDecomposition(level_grid, num_features, scale, device, core_num, reduce_sum=reduce_sum))

        self.tensors = torch.nn.ModuleList(tensors)

    def coords_split(self, pts, dim=2, z_vals=None):
        N, D = pts.size()
        pts = pts.view(1, N, 1, D)

        out_plane = []
        if dim == 2:
            out_plane.append(pts[..., [0, 1]])
            out_plane.append(pts[..., [1, 2]])
            out_plane.append(pts[..., [0, 2]])
        elif dim == 3:
            out_plane.append(pts[..., [0, 1, 2]][:, :, None])
            out_plane.append(pts[..., [1, 2, 0]][:, :, None])
            out_plane.append(pts[..., [0, 2, 1]][:, :, None])


        return out_plane

    def L1loss(self):
        loss = 0.
        for tensor in self.tensors:
            loss += tensor.L1loss()

        return loss / len(self.tensors)

    def TVLoss(self):
        loss = 0.
        for tensor in self.tensors:
            loss += tensor.TV_loss()

        return loss / len(self.tensors)

    def shrink(self, aabb, new_aabb):
        aabb_size = aabb[1] - aabb[0]
        xyz_min, xyz_max = new_aabb

        for tensor in self.tensors:
            grid_size = tensor.grid_size
            units = aabb_size / (grid_size - 1)
            t_l, b_r = (xyz_min - aabb[0]) / units, (xyz_max - aabb[0]) / units

            t_l, b_r = torch.floor(t_l).long(), torch.ceil(b_r).long()
            b_r = torch.stack([b_r, grid_size]).amin(0)

            bound = torch.stack((t_l, b_r), dim=-1)
            tensor.shrink(bound)

    def upsample(self, aabb):
        for tensor in self.tensors:
            tensor.upsample(aabb)

    def forward(self, pts):
        coords_plane = self.coords_split(pts)

        if self.reduce_sum:
            output = pts.new_zeros(pts.size(0))
        else:
            output = []

        for level_tensor in self.tensors:
            out = level_tensor(coords_plane)
            if self.reduce_sum:
                output += torch.sum(out, dim=1)
            else:
                output += [self.linear_projection(out)]

        return output

class RenderingMLP(torch.nn.Module):
    def __init__(self, data_dim_color=32, viewpe=4, featureC=256, device='cpu', num_level=8):
        super(RenderingMLP, self).__init__()

        data_dim_color=4*num_level
        self.ch_cd = 3
        self.ch_s = 3
        self.ch_normal = 3
        self.ch_bottleneck = 128
        self.ch_normal_dot_viewdir = 1

        self.spatial_mlp = torch.nn.Sequential(
            torch.nn.Linear(data_dim_color, featureC),
            torch.nn.GELU(),
            torch.nn.Linear(featureC, featureC),
            torch.nn.GELU(),
            torch.nn.Linear(featureC, featureC),
            torch.nn.GELU(),
            torch.nn.Linear(featureC, self.ch_cd + self.ch_s + self.ch_bottleneck + self.ch_normal)).to(device)

        self.ch_normal_dot_viewdir = 1

        self.directional_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.ch_bottleneck + self.ch_normal_dot_viewdir + 3, featureC),
            torch.nn.GELU(),
            torch.nn.Linear(featureC, featureC),
            torch.nn.GELU(),
            torch.nn.Linear(featureC, featureC),
            torch.nn.GELU(),
            torch.nn.Linear(featureC, featureC),
            torch.nn.GELU(),
            torch.nn.Linear(featureC, 3)).to(device)

    def spatial_mlp_forward(self, x):
        out = self.spatial_mlp(x)
        sections = [self.ch_cd, self.ch_s, self.ch_normal, self.ch_bottleneck]
        diffuse_color, tint, normals, bottleneck= torch.split(out, sections, dim=-1)
        normals = -F.normalize(normals, dim=1)
        return diffuse_color, tint, normals, bottleneck, 0

    def directional_mlp_forward(self, x):
        out = self.directional_mlp(x)
        return out

    def reflect(self, viewdir, normal):
        out = 2 * (viewdir * normal).sum(dim=-1, keepdim=True) * normal - viewdir
        return out

    def positional_encoding(self, positions, freqs):
        freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1],))
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts


    def forward(self, viewdir, feature):
        diffuse_color, tint, normal, bottleneck, roughness = self.spatial_mlp_forward(feature)
        refdir = self.reflect(-viewdir, normal)

        normal_dot_viewdir = ((-viewdir) * normal).sum(dim=-1, keepdim=True)
        dir_mlp_input = torch.cat([bottleneck, normal_dot_viewdir, refdir], dim=-1)
        specular_color = self.directional_mlp(dir_mlp_input)

        raw_rgb = diffuse_color + tint * specular_color
        rgb = torch.sigmoid(raw_rgb)

        return rgb, normal

class BTDRF(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(BTDRF, self).__init__(aabb, gridSize, device, **kargs)

        self.rendering_net = RenderingMLP(device=device, num_level=kargs['num_level'])
        self.init_feature_field(device, kargs['core_num'], kargs['num_level'])


    def init_feature_field(self, device, core_num, num_level):
        self.density_field = MultiscaleTensorDecom(num_levels=num_level, num_features=2, base_resolution=16, max_resolution=512, device=device, core_num=core_num, reduce_sum=True)
        self.appearance_field = MultiscaleTensorDecom(num_levels=num_level, num_features=4, base_resolution=16, max_resolution=512, device=device, core_num=core_num)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_core=0.001, lr_init_network=0.001):
        grad_vars = []
        for tensor in self.density_field.tensors:
            grad_vars += [{'params': tensor.plane_xy, 'lr':lr_init_spatialxyz}]
            grad_vars += [{'params': tensor.plane_yz, 'lr': lr_init_spatialxyz}]
            grad_vars += [{'params': tensor.plane_xz, 'lr': lr_init_spatialxyz}]
            grad_vars += [{'params': tensor.core_tensor, 'lr': lr_core}]

        for tensor in self.appearance_field.tensors:
            grad_vars += [{'params': tensor.plane_xy, 'lr':lr_init_spatialxyz}]
            grad_vars += [{'params': tensor.plane_yz, 'lr': lr_init_spatialxyz}]
            grad_vars += [{'params': tensor.plane_xz, 'lr': lr_init_spatialxyz}]
            grad_vars += [{'params': tensor.core_tensor, 'lr': lr_core}]

        grad_vars += [{'params': self.appearance_field.linear_projection.parameters(), 'lr': lr_init_network}]
        grad_vars += [{'params': self.rendering_net.parameters(), 'lr': lr_init_network}]


        return grad_vars

    def density_L1(self):
        return self.density_field.L1loss()

    def TV_loss_density(self):
        return self.density_field.TVLoss()

    def TV_loss_app(self):
        return self.appearance_field.TVLoss()

    def compute_densityfeature(self, pts):
        output = self.density_field(pts)
        return output

    def compute_appfeature(self, pts):
        app_feature = self.appearance_field(pts)
        app_feature = torch.cat(app_feature, dim=-1)
        return app_feature

    @torch.no_grad()
    def shrink(self, new_aabb):
        self.train_aabb = new_aabb

        self.density_field.shrink(self.aabb.cpu(), new_aabb.cpu())
        self.appearance_field.shrink(self.aabb.cpu(), new_aabb.cpu())

        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.floor(t_l).long(), torch.ceil(b_r).long()
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb

        self.density_field.upsample(new_aabb.cpu())
        self.appearance_field.upsample(new_aabb.cpu())

        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.update_stepSize(res_target)