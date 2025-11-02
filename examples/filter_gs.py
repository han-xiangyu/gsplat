#!/usr/bin/env python
"""
LiDAR-Guided Gaussian Splatting Point Cloud Filter (V2)

通过两阶段过滤清理3DGS PLY文件：
1. [几何过滤]: 使用LiDAR点云(points3D.txt)作为“保护膜”，剔除“浮游物”。
2. [属性过滤]: 剔除“毛刺”（高各向异性）和“巨型”GS（大尺度）。
"""

import os
import numpy as np
import argparse
from plyfile import PlyData, PlyElement
from tqdm import tqdm

# (load_lidar_points 和 build_voxel_hash_set 函数与你提供的V1版本完全相同)

def load_lidar_points(txt_path: str, cache_path: str) -> np.ndarray:
    """
    Load LiDAR points from COLMAP points3D.txt file with caching support.
    """
    
    # 1. First try to load from cache for faster execution
    if os.path.exists(cache_path):
        print(f"[数据加载] 发现缓存文件 {cache_path}，正在快速加载...")
        try:
            points = np.load(cache_path)
            print(f"[数据加载] 成功加载 {len(points)} 个LiDAR点。")
            return points
        except Exception as e:
            print(f"[数据加载] 缓存加载失败: {e}。将重新解析 .txt 文件。")

    # 2. If cache doesn't exist or loading fails, parse the .txt file
    print(f"[数据加载] 缓存未找到，正在解析 {txt_path} (可能需要几分钟)...")
    points = []
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f_count:
            total_lines = sum(1 for line in f_count if not line.startswith('#'))
    except Exception:
        total_lines = None  # Cannot estimate

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="解析 points3D.txt", total=total_lines, unit="lines"):
            if line.startswith('#'):
                continue
            
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                points.append([x, y, z])
            except ValueError:
                continue
                
    if not points:
        raise ValueError(f"No points parsed from {txt_path}.")

    pts_np = np.array(points, dtype=np.float32)
    print(f"[数据加载] 解析完成，加载了 {len(pts_np)} 个LiDAR点。")

    try:
        print(f"[数据加载] 正在将点云缓存到 {cache_path}...")
        np.save(cache_path, pts_np)
        print(f"[数据加载] 缓存保存成功。")
    except Exception as e:
        print(f"[数据加载] 缓存保存失败: {e}")

    return pts_np

def build_voxel_hash_set(lidar_points: np.ndarray, voxel_size: float, dilation_radius: int) -> set:
    """
    Build a dilated voxel hash set from LiDAR points (the 'protective mask').
    """
    
    print(f"[体素化] 正在将 {len(lidar_points)} 个LiDAR点体素化 (voxel_size={voxel_size}m)...")
    voxel_coords = np.floor(lidar_points / voxel_size).astype(np.int32)
    
    occupied_voxels = set(map(tuple, voxel_coords))
    del voxel_coords  # Free memory
    print(f"[体素化] 找到了 {len(occupied_voxels)} 个唯一的LiDAR体素。")

    if dilation_radius > 0:
        print(f"[体素化] 正在膨胀体素集 (半径={dilation_radius}, 邻域大小: {(2*dilation_radius+1)**3} voxels)...")
        dilated_voxels = set()
        r = dilation_radius
        
        for (i, j, k) in tqdm(occupied_voxels, desc="膨胀“保护膜”"):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    for dk in range(-r, r + 1):
                        dilated_voxels.add((i + di, j + dj, k + dk))
        
        print(f"[体素化] “保护膜”膨胀完成，总计 {len(dilated_voxels)} 个体素。")
        return dilated_voxels
    else:
        print("[体素化] 未进行膨胀 (dilation_radius=0)。")
        return occupied_voxels

def main(args):
    """Main execution function"""
    
    # --- 阶段 1: 几何过滤 (使用“保护膜”) ---
    
    # 1. 加载LiDAR点 (创建 .npy 缓存)
    lidar_cache_path = args.lidar_cache_path or (os.path.splitext(args.lidar_in)[0] + ".npy")
    lidar_points = load_lidar_points(args.lidar_in, lidar_cache_path)
    
    # 2. 构建“保护膜” (膨胀的体素哈希集)
    dilated_voxel_set = build_voxel_hash_set(
        lidar_points, 
        args.voxel_size, 
        args.dilation_radius
    )
    del lidar_points  # 释放内存
    
    # 3. 加载 GS PLY 文件
    print(f"[PLY加载] 正在加载待过滤的GS PLY文件: {args.ply_in}...")
    try:
        plydata_in = PlyData.read(args.ply_in)
        v_data = plydata_in['vertex'].data
    except Exception as e:
        print(f"Error loading {args.ply_in}: {e}")
        return

    gs_means = np.stack([v_data['x'], v_data['y'], v_data['z']], axis=1).astype(np.float32)
    N_before = len(gs_means)
    print(f"[PLY加载] 成功加载 {N_before} 个GS点。")
    
    # 4. 过滤GS点 (几何)
    print(f"[阶段1: 几何过滤] 开始过滤 {N_before} 个GS点...")
    gs_voxel_coords = np.floor(gs_means / args.voxel_size).astype(np.int32)
    mask_to_keep_geom = np.zeros(N_before, dtype=bool)
    
    for i in tqdm(range(N_before), desc="检查GS点是否在“膜”内"):
        if tuple(gs_voxel_coords[i]) in dilated_voxel_set:
            mask_to_keep_geom[i] = True
            
    # 5. 应用几何掩码
    filtered_v_data_geom = v_data[mask_to_keep_geom]
    N_after_geom = len(filtered_v_data_geom)
    N_removed_geom = N_before - N_after_geom
    print(f"[阶段1: 几何过滤] 完成。")
    print(f"[阶段1 结果] 保留: {N_after_geom} 个点 (在“保护膜”内)")
    print(f"[阶段1 结果] 剔除: {N_removed_geom} 个点 (在“保护膜”外)")
    
    if N_after_geom == 0:
        print("[错误] 几何过滤后没有剩下任何点！请检查体素参数或坐标系。")
        return

    # --- 阶段 2: 属性过滤 (清理“毛刺”和“巨型”GS) ---
    print(f"[阶段2: 属性过滤] 开始过滤 {N_after_geom} 个剩余的点...")
    
    # 提取属性 (在 log/logit 空间操作)
    scales_log = np.stack([
        filtered_v_data_geom['scale_0'], 
        filtered_v_data_geom['scale_1'], 
        filtered_v_data_geom['scale_2']
    ], axis=1).astype(np.float32)
    
    opacities_logit = filtered_v_data_geom['opacity'].astype(np.float32)

    # 1. 过滤“巨型”GS (解决“纯色区域”问题)
    # GS的尺度是以 log 空间存储的
    log_scale_threshold = np.log(args.max_scale)
    max_log_scale = np.max(scales_log, axis=1)
    mask_scale = max_log_scale < log_scale_threshold
    print(f"  ... [属性] 剔除 {np.sum(~mask_scale)} 个点 (尺度 > {args.max_scale:.2f}米)")

    # 2. 过滤“低透明度”GS
    # GS的透明度是以 logit 空间存储的: logit(p) = log(p / (1 - p))
    logit_opacity_threshold = np.log(args.min_opacity / (1.0 - args.min_opacity))
    mask_opacity = opacities_logit > logit_opacity_threshold
    print(f"  ... [属性] 剔除 {np.sum(~mask_opacity)} 个点 (透明度 < {args.min_opacity:.4f})")
    
    # 3. 过滤“极端各向异性”GS (毛刺)
    # Anisotropy = max_scale / min_scale
    # log(Anisotropy) = log(max_scale) - log(min_scale)
    log_anisotropy_threshold = np.log(args.max_anisotropy)
    min_log_scale = np.min(scales_log, axis=1)
    log_anisotropy = max_log_scale - min_log_scale
    mask_anisotropy = log_anisotropy < log_anisotropy_threshold
    print(f"  ... [属性] 剔除 {np.sum(~mask_anisotropy)} 个点 (各向异性 > {args.max_anisotropy:.1f})")

    # 4. 合并所有属性掩码
    mask_final_attr = mask_scale & mask_opacity & mask_anisotropy
    
    # 5. 应用最终掩码
    final_filtered_v_data = filtered_v_data_geom[mask_final_attr]
    N_after_attr = len(final_filtered_v_data)
    N_removed_attr = N_after_geom - N_after_attr
    
    print(f"[阶段2: 属性过滤] 完成。额外剔除了 {N_removed_attr} 个属性极端的点。")
    print(f"-"*30)
    print(f"[最终结果] 初始点数: {N_before}")
    print(f"[最终结果] 几何剔除: {N_removed_geom}")
    print(f"[最终结果] 属性剔除: {N_removed_attr}")
    print(f"[最终结果] 最终保留: {N_after_attr} 个点 ({(N_after_attr / N_before * 100):.2f}%)")
    print(f"-"*30)

    # 6. 保存新的PLY文件
    print(f"[保存] 正在将 {N_after_attr} 个干净的点云保存到: {args.ply_out} ...")
    el = PlyElement.describe(final_filtered_v_data, 'vertex')
    PlyData([el], text=False).write(args.ply_out)
    
    print(f"[完成] 干净的PLY文件已保存。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""(V2) 使用LiDAR“保护膜”和GS属性，两阶段过滤3DGS PLY文件。"""
    )
    
    # --- 几何过滤参数 (阶段1) ---
    geom_group = parser.add_argument_group("阶段1: 几何过滤 (保护膜)")
    geom_group.add_argument("--ply_in", type=str, required=True,
                            help="输入的、待清理的GS PLY文件 (例如 scene1.ply)")
    geom_group.add_argument("--lidar_in", type=str, required=True,
                            help="LiDAR点云文件 (例如 scene1/sparse/0/points3D.txt)")
    geom_group.add_argument("--ply_out", type=str, required=True,
                            help="输出的、清理干净的PLY文件 (例如 scene1_cleaned.ply)")
    geom_group.add_argument("--lidar_cache_path", type=str, default=None,
                            help="LiDAR点云NPY缓存路径 (默认: 'lidar_in' 同名的 .npy 文件)")
    geom_group.add_argument("--voxel_size", type=float, default=1.0,
                            help="“保护膜”的体素大小（米）。(默认: 1.0)")
    geom_group.add_argument("--dilation_radius", type=int, default=1,
                            help="“保护膜”的膨胀半径（体素）。(默认: 1, 即 3x3x3 邻域)")

    # --- 属性过滤参数 (阶段2) ---
    attr_group = parser.add_argument_group("阶段2: 属性过滤 (毛刺/巨型GS)")
    attr_group.add_argument("--max_scale", type=float, default=2.0,
                            help="[属性] 保留的GS最大物理尺度（米）。用于剔除“巨型”GS。(默认: 2.0)")
    attr_group.add_argument("--min_opacity", type=float, default=0.01,
                            help="[属性] 保留的GS最小物理透明度(0-1)。用于剔除“隐形”GS。(默认: 0.01)")
    attr_group.add_argument("--max_anisotropy", type=float, default=100.0,
                            help="[属性] 保留的GS最大各向异性(max_scale/min_scale)。用于剔除“毛刺”。(默认: 100.0)")

    args = parser.parse_args()
    
    main(args)