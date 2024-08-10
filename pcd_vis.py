import os
import pickle as pkl
import open3d as o3d

import matplotlib.pyplot as plt
from glob import glob
from typing import Tuple, List, Dict, Iterable
from PIL import Image
import numpy as np
from plyfile import PlyData, PlyElement

def load_raw_ply(path):
    print("Loading ", path)
    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    return xyz 

def vis_points_all(points):
    pcds = o3d.open3d.geometry.PointCloud()
    colors = np.ones((points.shape[0], 3)) * 58
    pcds.points = o3d.open3d.utility.Vector3dVector(points)
    # pcds.colors = o3d.open3d.utility.Vector3dVector(colors)
    need_vis = [pcds]
    vis = o3d.visualization.draw_geometries(need_vis, window_name='Open3D downSample', point_show_normal=False,
                                    mesh_show_wireframe=True,
                                    mesh_show_back_face=True, )
    del vis


folder = "mcmc_loc2450_pcd"
pcd_path1 = rf"D:\research\large_scale\trained_pcd\mcmc\{folder}\point_cloud_rk0_ws4.ply"
pcd_path2 = rf"D:\research\large_scale\trained_pcd\mcmc\{folder}\point_cloud_rk1_ws4.ply"
pcd_path3 = rf"D:\research\large_scale\trained_pcd\mcmc\{folder}\point_cloud_rk2_ws4.ply"
pcd_path4 = rf"D:\research\large_scale\trained_pcd\mcmc\{folder}\point_cloud_rk3_ws4.ply"
xyz1 = load_raw_ply(pcd_path1)
xyz2 = load_raw_ply(pcd_path2)
xyz3 = load_raw_ply(pcd_path3)
xyz4 = load_raw_ply(pcd_path4)

pcd_total = np.concatenate((xyz1, xyz2, xyz3, xyz4), axis=0)
print(pcd_total.shape)
# vis_points_all(xyz1)
# vis_points_all(xyz2)
vis_points_all(pcd_total)