#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix,getProjectionMatrixCenterShift
from utils.general_utils import get_args, get_log_file
import utils.general_utils as utils
import time
import math

class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        cx, cy, fx, fy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        is_locally_loaded=True,
        dynamic_mask = None, pts_depth = None,timestamp=0.0,traversal_id=0, sky_mask=None,normal=None, depth=None
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.traversal_id = traversal_id
        self.sky_mask = sky_mask
        self.normal = normal
        self.depth = depth
        self.dynamic_mask = dynamic_mask
        self.pts_depth = pts_depth

        args = get_args()
        log_file = get_log_file()

        if args.time_image_loading:
            start_time = time.time()

        if is_locally_loaded:
            # load to cpu
            self.original_image_backup = image.contiguous()
            if args.preload_dataset_to_gpu:
                self.original_image_backup = self.original_image_backup.to("cuda")
            self.image_width = self.original_image_backup.shape[2]
            self.image_height = self.original_image_backup.shape[1]
        else:
            self.original_image_backup = None
            self.image_height, self.image_width = utils.get_img_size() # TODO: if images are of different sizes, this will fail.

        if args.time_image_loading:
            log_file.write(f"Image processing in {time.time() - start_time} seconds\n")

        if self.cx is None:
            self.zfar = 100.0
            self.znear = 0.01

            self.trans = trans
            self.scale = scale

            self.world_view_transform = (
                torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
            )
            self.world_view_transform_backup = self.world_view_transform.clone().detach()
            self.projection_matrix = (
                getProjectionMatrix(
                    znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
                )
                .transpose(0, 1)
                .cuda()
            )
            self.full_proj_transform = (
                self.world_view_transform.unsqueeze(0).bmm(
                    self.projection_matrix.unsqueeze(0)
                )
            ).squeeze(0)
            self.camera_center = self.world_view_transform.inverse()[3, :3]
        else:
            self.zfar = 1000.0 # GS 100 PVG1000
            self.znear = 0.01

            self.trans = trans
            self.scale = scale
            self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
            self.world_view_transform_backup = self.world_view_transform.clone().detach()
            self.FoVx = 2 * math.atan(0.5*self.image_width / fx)
            self.FoVy = 2 * math.atan(0.5*self.image_height / fy)
            self.projection_matrix = getProjectionMatrixCenterShift(self.znear, self.zfar, cx, cy, fx, fy,
                                                                    self.image_width, self.image_height).transpose(0, 1).cuda()
            self.full_proj_transform = (
                self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
            self.camera_center = self.world_view_transform.inverse()[3, :3]

    def get_camera2world(self):
        return self.world_view_transform_backup.t().inverse()

    def update(self, dx, dy, dz):
        # Update the position of this camera pose. TODO: support updating rotation of camera pose.
        with torch.no_grad():
            c2w = self.get_camera2world()
            c2w[0, 3] += dx
            c2w[1, 3] += dy
            c2w[2, 3] += dz

            t_prime = c2w[:3, 3]
            self.T = (-c2w[:3, :3].t() @ t_prime).cpu().numpy()
            # import pdb; pdb.set_trace()

            self.world_view_transform = (
                torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale))
                .transpose(0, 1)
                .cuda()
            )
            self.projection_matrix = (
                getProjectionMatrix(
                    znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
                )
                .transpose(0, 1)
                .cuda()
            )
            self.full_proj_transform = (
                self.world_view_transform.unsqueeze(0).bmm(
                    self.projection_matrix.unsqueeze(0)
                )
            ).squeeze(0)
            self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
