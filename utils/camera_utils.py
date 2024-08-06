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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, get_args, get_log_file
import utils.general_utils as utils
from tqdm import tqdm
from utils.graphics_utils import fov2focal
import time
import multiprocessing
import torch
from PIL import Image
import copy

def loadCam(args, id, cam_info, decompressed_image=None, return_image=False, is_locally_loaded=False):
    orig_w, orig_h = cam_info.width, cam_info.height
    assert (
        orig_w == utils.get_img_width() and orig_h == utils.get_img_height()
    ), "All images should have the same size. "

    args = get_args()
    log_file = get_log_file()
    resolution = orig_w, orig_h
    # NOTE: we do not support downsampling here.

    # may use cam_info.uid
    if is_locally_loaded:
        if args.time_image_loading:
            start_time = time.time()
        image = Image.open(cam_info.image_path)
        resized_image_rgb = PILtoTorch(
            image, resolution, args, log_file, decompressed_image=decompressed_image
        )
        if args.time_image_loading:
            log_file.write(f"PILtoTorch image in {time.time() - start_time} seconds\n")

        # assert resized_image_rgb.shape[0] == 3, "Image should have exactly 3 channels!"
        gt_image = resized_image_rgb[:3, ...].contiguous()
        loaded_mask = None
        if cam_info.sky_mask is not None:
            sky_mask = copy.deepcopy(cam_info.sky_mask)
            sky_mask = np.resize(sky_mask, (resolution[1], resolution[0]))
            sky_mask = torch.from_numpy(sky_mask).float()
        else:
            sky_mask = None
        if cam_info.dynamic_mask is not None:
            dynamic_mask = copy.deepcopy(cam_info.dynamic_mask)
            dynamic_mask = np.resize(dynamic_mask, (resolution[1], resolution[0],3))
            dynamic_mask = torch.from_numpy(np.array(dynamic_mask)).permute(2, 0, 1)
        else:
            dynamic_mask = None
        pts_depth = None
        if cam_info.pointcloud_camera is not None:
            h, w = gt_image.shape[1:]
            K = np.eye(4)
            cx = cam_info.cx 
            cy = cam_info.cy 
            fy = cam_info.fy 
            fx = cam_info.fx 
            if cam_info.cx:
                #print(scale)
                K[0, 0] = fx 
                K[1, 1] = fy 
                K[0, 2] = cx
                K[1, 2] = cy
            else:
                K[0, 0] = fov2focal(cam_info.FovX, w)
                K[1, 1] = fov2focal(cam_info.FovY, h)
                K[0, 2] = cam_info.width / 2
                K[1, 2] = cam_info.height / 2
            pts_depth = np.zeros([1, h, w])
            point_camera = cam_info.pointcloud_camera
            depth = point_camera[:, 2]
            uvz = point_camera[point_camera[:, 2] > 0]
            depth = depth[point_camera[:, 2] > 0]
            uvz = np.concatenate([uvz, np.ones((uvz.shape[0], 1))], axis=1)
            # viewpad = np.eye(4)
            # viewpad[:view.shape[0], :view.shape[1]] = view

            # nbr_points = points.shape[1]

            # # Do operation in homogenous coordinates.
            # points = np.concatenate((points, np.ones((1, nbr_points))))
            # points = np.dot(viewpad, points)
            # points = points[:3, :]

            # if normalize:
            #     points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

            # return points
            uvz = uvz @ K.T
            
            uvz = uvz[:, :3]
            #uvz = uvz / uvz[:, 2:3].repeat(3, 1).reshape(uvz.shape[0],3)
            uvz[:, :2] = uvz[:, :2] / uvz[:, 2:3].repeat(2, 1).reshape(uvz.shape[0], 2)
            #uvz[:, :2] /= (uvz[:, 2:])
            depth = depth[uvz[:, 1] >= 0]
            uvz = uvz[uvz[:, 1] >= 0]
            depth = depth[uvz[:, 1] < h]
            uvz = uvz[uvz[:, 1] < h]
            depth = depth[uvz[:, 0] >= 0]
            uvz = uvz[uvz[:, 0] >= 0]
            depth = depth[uvz[:, 0] < w]
            uvz = uvz[uvz[:, 0] < w]
            uv = uvz[:, :2]
            uv = uv.astype(int)
            #print(uv)
            # TODO: may need to consider overlap
            pts_depth[0, uv[:, 1], uv[:, 0]] = depth[:] #uvz[:, 2]
            pts_depth = torch.from_numpy(pts_depth).float()
        else:
            cx = None
            cy = None
            fx = None
            fy = None

        # Free the memory: because the PIL image has been converted to torch tensor, we don't need it anymore. And it takes up lots of cpu memory.
        image.close()
        image = None
    else:
        gt_image = None
        loaded_mask = None
        cx = None
        cy = None
        fx = None
        fy = None
        dynamic_mask = None
        pts_depth = None
        sky_mask = None

    if return_image:
        return gt_image

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        cx=cx, cy=cy, fx=fx, fy=fy,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        scale=args.scale,
        is_locally_loaded=is_locally_loaded,
        dynamic_mask=dynamic_mask, pts_depth=pts_depth,timestamp=cam_info.timestamp,
        traversal_id=cam_info.traversal_id,sky_mask=sky_mask
    )


def load_decompressed_image(params):
    args, id, cam_info, is_locally_loaded = params
    return loadCam(args, id, cam_info, decompressed_image=None, return_image=True, is_locally_loaded=is_locally_loaded)

def decompressed_images_from_camInfos_multiprocess(cam_infos, args, whether_cameras_locally_loaded):
    args = get_args()
    decompressed_images = []
    total_cameras = len(cam_infos)

    # Create a pool of processes
    with multiprocessing.Pool(processes=args.num_proc_for_dataloading) as pool:
        # Prepare data for processing
        tasks = [(args, id, cam_info, whether_cameras_locally_loaded[id]) for id, cam_info in enumerate(cam_infos)]

        # Map load_camera_data to the tasks
        # results = pool.map(load_decompressed_image, tasks)
        results = list(
            tqdm(
                pool.imap(load_decompressed_image, tasks),
                total=total_cameras,
                disable=(utils.LOCAL_RANK != 0),
            )
        )

        for id, result in enumerate(results):
            decompressed_images.append(result)

    return decompressed_images

def cameraList_from_camInfos(cam_infos, args, whether_cameras_locally_loaded):
    args = get_args()

    if args.multiprocesses_image_loading:
        decompressed_images = decompressed_images_from_camInfos_multiprocess(
            cam_infos, args, whether_cameras_locally_loaded
        )
    else:
        decompressed_images = [None for _ in cam_infos]

    camera_list = []
    for id, c in tqdm(
        enumerate(cam_infos), total=len(cam_infos), disable=(utils.LOCAL_RANK != 0)
    ):
        camera_list.append(
            loadCam(
                args,
                id,
                c,
                decompressed_image=decompressed_images[id],
                return_image=False,
                is_locally_loaded=whether_cameras_locally_loaded[id],
            )
        )

    if utils.DEFAULT_GROUP.size() > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
