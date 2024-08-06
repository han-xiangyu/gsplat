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

import os
import sys
import glob
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import utils.general_utils as utils
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch
from pyquaternion import Quaternion

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    cx: float = None
    cy: float = None
    fx: float = None
    fy: float = None
    dynamic_mask: np.array = None
    pointcloud_camera: np.array = None
    timestamp: float = None
    traversal_id: int = None
    sky_mask: np.array = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getCamCenter(cam_info):
    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    cam_centers = np.hstack(cam_centers)
    return np.transpose(cam_centers)


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    args = utils.get_args()
    cam_infos = []
    utils.print_rank_0("Loading cameras from disk...")
    for idx, key in tqdm(
        enumerate(cam_extrinsics),
        total=len(cam_extrinsics),
        disable=(utils.LOCAL_RANK != 0),
    ):

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            # we're ignoring the 4 distortion
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(
            image_path
        )  # this is a lazy load, the image is not loaded yet
        width, height = image.size

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=None,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )

        # release memory
        image.close()
        image = None

        cam_infos.append(cam_info)
    return cam_infos


def fetchPly(path):
    args = utils.get_args()
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T * args.scale
    try:
        colors = (
            np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
        )
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(
    path, images, eval, llffhold=10, init_type="sfm", num_pts=100000
):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if init_type == "sfm":
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            if utils.GLOBAL_RANK == 0:
                print(
                    "Converting point3d.bin to .ply, will happen only the first time you open the scene."
                )
                try:
                    xyz, rgb, _ = read_points3D_binary(bin_path)
                except:
                    xyz, rgb, _ = read_points3D_text(txt_path)
                storePly(ply_path, xyz, rgb)
            if utils.DEFAULT_GROUP.size() > 1:
                torch.distributed.barrier()
    else:
        ply_path = os.path.join(path, "random.ply")
        if utils.GLOBAL_RANK == 0:
            print(f"Generating random point cloud ({num_pts})...")
            if init_type == "random_cube":
                xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"] * 3 * 2 - (
                    nerf_normalization["radius"] * 3
                )
            elif init_type == "random_plane":
                cam_plane = utils.fit_plane(getCamCenter(train_cam_infos))
                nor_vec = np.hstack([cam_plane[0], cam_plane[1], -1])
                xy = (np.random.random((num_pts, 2)) * 2 - 1) * nerf_normalization["radius"]
                z = xy[:, 0] * cam_plane[0] + xy[:, 1] * cam_plane[1] + cam_plane[2]
                h = -0.7
                xyz = np.hstack([xy,z[:,np.newaxis]]) + h * nor_vec
            else:
                assert False, "init_type not supported"
            num_pts = xyz.shape[0]
            shs = np.random.random((num_pts, 3)) / 255.0
            storePly(ply_path, xyz, SH2RGB(shs) * 255)
        if utils.DEFAULT_GROUP.size() > 1:
            torch.distributed.barrier()

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransformsCity(
    path,
    transformsfile,
    random_background,
    white_background,
    extension=".png",
    undistorted=False,
    is_debug=False,
):
    cam_infos = []
    if undistorted:
        print("Undistortion the images!!!")
        # TODO: Support undistortion here. Please refer to octree-gs implementation.
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None

        frames = contents["frames"]
        # check if filename already contain postfix
        if frames[0]["file_path"].split(".")[-1] in ["jpg", "jpeg", "JPG", "png"]:
            extension = ""

        c2ws = np.array([frame["transform_matrix"] for frame in frames])

        Ts = c2ws[:, :3, 3]

        ct = 0

        progress_bar = tqdm(frames, desc="Loading dataset")

        for idx, frame in enumerate(frames):
            # cam_name = os.path.join(path, frame["file_path"] + extension)
            cam_name = frame["file_path"]
            if not os.path.exists(cam_name):
                print(f"File {cam_name} not found, skipping...")
                continue
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])

            if idx % 10 == 0:
                progress_bar.set_postfix({"num": f"{ct}/{len(frames)}"})
                progress_bar.update(10)
            if idx == len(frames) - 1:
                progress_bar.close()

            ct += 1
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = cam_name[-17:]  # Path(cam_name).stem
            image = Image.open(image_path)

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy
                FovX = fovx
            else:
                # given focal in pixel unit
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=None,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

            # release memory
            image.close()
            image = None

            if is_debug and idx > 50:
                break
    return cam_infos


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCityInfo(
    path,
    random_background,
    white_background,
    extension=".tif",
    llffhold=8,
    undistorted=False,
):

    train_json_path = os.path.join(path, f"transforms_train.json")
    test_json_path = os.path.join(path, f"transforms_test.json")
    print(
        "Reading Training Transforms from {} {}".format(train_json_path, test_json_path)
    )

    train_cam_infos = readCamerasFromTransformsCity(
        path,
        train_json_path,
        random_background,
        white_background,
        extension,
        undistorted,
    )
    test_cam_infos = readCamerasFromTransformsCity(
        path,
        test_json_path,
        random_background,
        white_background,
        extension,
        undistorted,
    )
    print("Load Cameras(train, test): ", len(train_cam_infos), len(test_cam_infos))

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = glob.glob(os.path.join(path, "*.ply"))[0]
    if os.path.exists(ply_path):
        try:
            pcd = fetchPly(ply_path)
        except:
            raise ValueError("must have tiepoints!")
    else:
        assert False, "No ply file found!"

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readIthacaSceneInfo(root_path, images, eval, llffhold=8,time_duration=[-0.5,0.5]):
    reading_dir = "images" if images == None else images 
    cam_infos= []
    points = []
    points_time=[]
    index =0
    last_traversal=None
    traversal_id=-1
    original_height = 1208
    original_width = 1928
    image_folder = os.path.join(root_path ,reading_dir)
    print("Loading images from: ", image_folder)
    img_list = sorted(os.listdir(image_folder))
    for idx, img_name in enumerate(tqdm(img_list)):
        # if not img_name[:4] in ['1119','1214','1215']:
        #     continue
        if not img_name[:4] in ['1119']:
            continue
        
        current_traversal = img_name[:4]
        if current_traversal != last_traversal:
            traversal_id+=1
            last_traversal = current_traversal
            
        image_path = os.path.join(image_folder, img_name)
        image = Image.open(image_path)
        width, height = image.size
        scale = original_height / height
        path  = image_folder.split("images")[0]
        image_name = img_name.split('.')[0]
        # print(image_name)
        lidar_name = image_name[:8] + image_name[-5:]
        
        # process camera data------------------------------------------------------------------------------
        with open(os.path.join(path, 'meta', 'camera_meta.json'), 'r') as file:
            camera_meta = json.load(file)
        #print('camera_timestamp: ', camera_meta[image_name]['timestamp'])
        camera2car = np.eye(4)
        camera2carR = Quaternion(camera_meta[image_name]['calib']['rotation_rect']).rotation_matrix
        camera2carT = np.array(camera_meta[image_name]['calib']['translation_rect'])
        camera2car[:3, :3] = camera2carR
        camera2car[:3, 3] = camera2carT
        
        car2world = np.eye(4)
        car2worldR = Quaternion(camera_meta[image_name]['camera_pose']['rotation']).rotation_matrix
        car2worldT = np.array(camera_meta[image_name]['camera_pose']['translation'])
        car2world[:3, :3] = car2worldR
        car2world[:3, 3] = car2worldT            
        
        
        camera2world = car2world @ camera2car
        world2camera = np.linalg.inv(camera2world)
        car2world1 = car2world
        R = world2camera[:3, :3].T  # WHETHER USE THIS T NEEDS CHECKS
        T = world2camera[:3, 3]
        
        # camera intrinsics-------------------------------------------------------------------------------
        intrinsic_matrix = camera_meta[image_name]['calib']['camera_matrix_rect']
        # here /2 is because Zhiheng resized the rectified image to be 1/2 resolution of original image.
        fx = intrinsic_matrix[0][0] / scale
        cx = intrinsic_matrix[0][2] / scale
        fy = intrinsic_matrix[1][1] / scale
        cy = intrinsic_matrix[1][2] / scale
        FovY = focal2fov(fy, height)
        FovX = focal2fov(fx, width)
        
        # handle mask--------------------------------------------------------------------------------------
        # mask_path = os.path.join(path, 'images_sweeps', image_name + '.npy')
        # mask = np.load(mask_path)
        # class_mapping = np.ones(len(CLASSES), dtype=np.int32)
        # class_mapping2 = np.ones(len(CLASSES), dtype=np.int32)
        # for i, cls in enumerate(CLASSES):
        #     if cls in dynamic_objects:
        #         class_mapping[i] = 0
        #     if cls in sky_objects:
        #         class_mapping2[i] = 0
        # dynamic_mask = class_mapping[mask][..., None]
        # dynamic_mask = np.repeat(dynamic_mask, repeats=3, axis=-1)
        # sky_mask = class_mapping2[mask][..., None]

        dynamic_mask_path = os.path.join(path, 'seg_mask', image_name + '.npy')
        sky_mask_path = os.path.join(path, 'sky_mask', image_name + '.npy')
        if os.path.exists(dynamic_mask_path):
            dynamic_mask = np.load(dynamic_mask_path)
            dynamic_mask = np.repeat(dynamic_mask[..., None], repeats=3, axis=-1)
        else:
            dynamic_mask = None
        if os.path.exists(sky_mask_path):
            sky_mask = np.load(sky_mask_path)
        else:
            sky_mask = None

        # process lidar data-------------------------------------------------------------------------------
        lidar_file = os.path.join(path, 'lidar', image_name[:8] + '0' + image_name[-5:] + '.bin')
        scan = np.fromfile(lidar_file, dtype=np.float32)
        pc0 = scan.reshape((-1, 4))[:,:3]
        pc = np.concatenate([pc0, np.ones((pc0.shape[0], 1))], axis=1)
        
        with open(os.path.join(path, 'meta', 'lidar_meta.json'), 'r') as file:
            lidar_meta = json.load(file)
        #print('lidar_timestamp: ', lidar_meta[lidar_name]['timestamp'])
        
        lidar2car = np.eye(4)
        lidar2carR = Quaternion(lidar_meta[lidar_name]['calib']['rotation']).rotation_matrix
        lidar2carT = np.array(lidar_meta[lidar_name]['calib']['translation'])
        lidar2car[:3, :3] = lidar2carR
        lidar2car[:3, 3] = lidar2carT
        
        car2world1 = np.eye(4)
        car2world1R = Quaternion(lidar_meta[lidar_name]['lidar_pose']['rotation']).rotation_matrix
        car2world1T = np.array(lidar_meta[lidar_name]['lidar_pose']['translation'])
        car2world1[:3, :3] = car2world1R
        car2world1[:3, 3] = car2world1T
        
        lidar2world = car2world1 @ lidar2car
        
        lidar_points = pc @ lidar2world.T  # this lidar_to_worlds
        points.append(lidar_points[:, :3])
        point_camera = lidar_points @ world2camera.T
        point_camera = point_camera[:, :3]
        timestamp= time_duration[0] + (time_duration[1] - time_duration[0]) * index / (len(img_list) - 1)
        point_time = np.full_like(point_camera[:, :1], timestamp)
        points_time.append(point_time)
        #------------------
        # print("First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep")
        # print(lidar2carR)
        # print(lidar2carT)
        # print("Second step: transform from ego to the global frame.")
        # print(car2world1R)
        # print(car2world1T)
        # print("Third step: transform from global into the ego vehicle frame for the timestamp of the image.")
        # print(car2worldR)
        # print(car2worldT)
        # print("Fourth step: transform from ego into the camera.")
        # print(camera2carR)
        # print(camera2carT)
        # print("view in the camera frame")
        # print(intrinsic_matrix)
        # print(point_camera[:10, :3])
        
        cam_info = CameraInfo(uid=index, R=R, T=T, FovY=FovY, FovX=FovX, cx=cx, cy=cy, fx=fx, fy=fy, image=image, dynamic_mask=dynamic_mask,
                            image_path=image_path, image_name=image_name, width=width, height=height, pointcloud_camera=point_camera,timestamp=timestamp,traversal_id=traversal_id,sky_mask=sky_mask)
        index += 1
        cam_infos.append(cam_info)
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Normalize pose
    # nerf_normalization['radius'] = 1/nerf_normalization['radius']
    # w2cs = np.zeros((len(cam_infos), 4, 4))
    # Rs = np.stack([c.R for c in cam_infos], axis=0)
    # Ts = np.stack([c.T for c in cam_infos], axis=0)
    # w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    # w2cs[:, :3, 3] = Ts
    # w2cs[:, 3, 3] = 1
    # c2ws = unpad_poses(np.linalg.inv(w2cs))
    # c2ws, transform, scale_factor = transform_poses_pca(c2ws, fix_radius=0.0)
    # c2ws = pad_poses(c2ws)

    for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data")):
        if cam_info.dynamic_mask is None:
            continue
        # c2w = c2ws[idx]
        # w2c = np.linalg.inv(c2w)
        # print(c2w)
        # print(w2c)
        # cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        # cam_info.T[:] = w2c[:3, 3]
        # cam_info.pointcloud_camera[:] *= scale_factor
        K = np.eye(3)
        K[0, 0] = cam_info.fx
        K[1, 1] = cam_info.fy
        K[0, 2] = cam_info.cx
        K[1, 2] = cam_info.cy
        pts_depth = np.zeros([1, cam_info.height, cam_info.width])
        uvz = cam_info.pointcloud_camera @ K.T
        uvz[:, :2] /= uvz[:, 2:]
        valid_mask = (uvz[:, 1] >= 0) & (uvz[:, 1] < cam_info.height) & (uvz[:, 0] >= 0) & (uvz[:, 0] < cam_info.width) & (uvz[:, 2] > 0)
        uvz = uvz.astype(int)
        pts_depth[0, uvz[valid_mask, 1], uvz[valid_mask,0]] = uvz[valid_mask, 2]
        pts_depth = torch.from_numpy(pts_depth).float()

        static_points_mask = cam_info.dynamic_mask[:,:,0][np.clip((uvz[:, 1]), 0, cam_info.height-1), np.clip((uvz[:, 0]), 0, cam_info.width-1)] == 1
        valid_static_mask = valid_mask & static_points_mask
        valid_dynamic_mask = valid_mask & ~static_points_mask
        pts_depth_static = np.zeros([1, cam_info.height, cam_info.width])
        pts_depth_static[0, uvz[valid_static_mask, 1], uvz[valid_static_mask,0]] = uvz[valid_static_mask, 2]
        pts_depth_static = torch.from_numpy(pts_depth_static).float()
        points[idx] = points[idx][valid_static_mask]
        points_time[idx] = points_time[idx][valid_static_mask]

    pointcloud = np.concatenate(points, axis=0)
    pointcloud_timestamp = np.concatenate(points_time, axis=0)
    # pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]

    # num_pts = pointcloud.shape[0] // 20
    num_pts = 1000000 
    pcd_path = os.path.join(root_path, f"points_total.npy")
    np.save(pcd_path, pointcloud)
    print("Saved total pointcloud to ", pcd_path)
    print("Out of {} points, using {} points".format(pointcloud.shape[0], num_pts))
    # np.random.seed(3407)
    indices = np.random.choice(pointcloud.shape[0], num_pts, replace=True)
    # np.random.seed(0)
    pointcloud = pointcloud[indices]
    pointcloud_timestamp = pointcloud_timestamp[indices]
    #Save the pointcloud for initialization
    pcd_path = os.path.join(root_path, f"points_subset_{num_pts}.npy")
    np.save(pcd_path, pointcloud)
    print("Saved pointcloud for initialization to ", pcd_path)
    
    
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    ply_path = os.path.join(path, "lidar_pc", "points3D.ply")
    os.makedirs(os.path.join(path, "lidar_pc"), exist_ok=True)

    if utils.GLOBAL_RANK == 0:
        rgbs = np.random.random((pointcloud.shape[0], 3))
        storePly(ply_path, pointcloud, rgbs)
        if utils.DEFAULT_GROUP.size() > 1:
            torch.distributed.barrier()
    else:
        if utils.DEFAULT_GROUP.size() > 1:
            torch.distributed.barrier()
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
        
    pcd = BasicPointCloud(pointcloud, colors=np.zeros([pointcloud.shape[0],3]), normals=None)


    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "City": readCityInfo,
    "Ithaca": readIthacaSceneInfo,
}