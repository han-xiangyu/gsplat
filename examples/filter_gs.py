#!/usr/bin/env python
"""
LiDAR-Guided Gaussian Splatting Point Cloud Filter
This script uses LiDAR point cloud as a 'protective mask' to filter out 
floating artifacts and noise from 3D Gaussian Splatting PLY files.
"""

import os
import numpy as np
import argparse
from plyfile import PlyData, PlyElement
from tqdm import tqdm

def load_lidar_points(txt_path: str, cache_path: str) -> np.ndarray:
    """
    Load LiDAR points from COLMAP points3D.txt file with caching support.
    
    Args:
        txt_path: Path to COLMAP points3D.txt file
        cache_path: Path to .npy cache file for faster loading
        
    Returns:
        numpy.ndarray: Array of LiDAR points with shape (N, 3)
    """
    
    # 1. First try to load from cache for faster execution
    if os.path.exists(cache_path):
        print(f"[Data Loading] Found cache file {cache_path}, loading quickly...")
        try:
            points = np.load(cache_path)
            print(f"[Data Loading] Successfully loaded {len(points)} LiDAR points.")
            return points
        except Exception as e:
            print(f"[Data Loading] Cache loading failed: {e}. Will re-parse .txt file.")

    # 2. If cache doesn't exist or loading fails, parse the .txt file
    print(f"[Data Loading] Cache not found, parsing {txt_path} (this may take several minutes)...")
    points = []
    
    # Estimate total lines for progress bar
    try:
        with open(txt_path, 'r', encoding='utf-8') as f_count:
            total_lines = sum(1 for line in f_count if not line.startswith('#'))
    except Exception:
        total_lines = None  # Cannot estimate

    # Parse the points3D.txt file
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Parsing points3D.txt", total=total_lines, unit="lines"):
            if line.startswith('#'):
                continue
            
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            
            # COLMAP format: POINT3D_ID, X, Y, Z, R, G, B, ...
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                points.append([x, y, z])
            except ValueError:
                # Skip malformed lines
                continue
                
    if not points:
        raise ValueError(f"No points parsed from {txt_path}.")

    pts_np = np.array(points, dtype=np.float32)
    print(f"[Data Loading] Parsing completed, loaded {len(pts_np)} LiDAR points.")

    # 3. Save to cache for future use
    try:
        print(f"[Data Loading] Caching point cloud to {cache_path}...")
        np.save(cache_path, pts_np)
        print(f"[Data Loading] Cache saved successfully.")
    except Exception as e:
        print(f"[Data Loading] Cache saving failed: {e}")

    return pts_np

def build_voxel_hash_set(lidar_points: np.ndarray, voxel_size: float, dilation_radius: int) -> set:
    """
    Build a dilated voxel hash set from LiDAR points (the 'protective mask').
    
    Args:
        lidar_points: Array of LiDAR points (N, 3)
        voxel_size: Size of each voxel in meters
        dilation_radius: Radius for voxel dilation (in voxels)
        
    Returns:
        set: Set of voxel coordinates representing the protective mask
    """
    
    # 1. Step 1: Voxelize LiDAR points
    print(f"[Voxelization] Voxelizing {len(lidar_points)} LiDAR points (voxel_size={voxel_size}m)...")
    voxel_coords = np.floor(lidar_points / voxel_size).astype(np.int32)
    
    # Convert NumPy array to Python set for fast lookup
    # Using map(tuple, ...) to efficiently convert (N, 3) array to set of (x,y,z) tuples
    occupied_voxels = set(map(tuple, voxel_coords))
    del voxel_coords  # Free memory
    print(f"[Voxelization] Found {len(occupied_voxels)} unique LiDAR voxels.")

    # 2. Step 2: Dilate voxels (thicken the "protective mask")
    if dilation_radius > 0:
        print(f"[Voxelization] Dilating voxel set (radius={dilation_radius}, neighborhood size: {(2*dilation_radius+1)**3} voxels)...")
        dilated_voxels = set()
        r = dilation_radius
        
        # This Python loop might be slow but is memory-efficient for sparse data
        for (i, j, k) in tqdm(occupied_voxels, desc="Dilating protective mask"):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    for dk in range(-r, r + 1):
                        dilated_voxels.add((i + di, j + dj, k + dk))
        
        print(f"[Voxelization] Protective mask dilation completed, total {len(dilated_voxels)} voxels.")
        return dilated_voxels
    else:
        print("[Voxelization] No dilation applied (dilation_radius=0).")
        return occupied_voxels

def main(args):
    """Main execution function"""
    
    # 1. Load LiDAR points (creates .npy cache if needed)
    lidar_cache_path = args.lidar_cache_path or (os.path.splitext(args.lidar_in)[0] + ".npy")
    lidar_points = load_lidar_points(args.lidar_in, lidar_cache_path)
    
    # 2. Build protective mask (dilated voxel hash set)
    dilated_voxel_set = build_voxel_hash_set(
        lidar_points, 
        args.voxel_size, 
        args.dilation_radius
    )
    del lidar_points  # Free memory
    
    # 3. Load GS PLY file to be filtered
    print(f"[PLY Loading] Loading GS PLY file to filter: {args.ply_in}...")
    try:
        plydata_in = PlyData.read(args.ply_in)
        v_data = plydata_in['vertex'].data
        v_properties = plydata_in['vertex'].properties
    except Exception as e:
        print(f"Error loading {args.ply_in}: {e}")
        return

    # Extract 'means' (coordinates)
    gs_means = np.stack([v_data['x'], v_data['y'], v_data['z']], axis=1).astype(np.float32)
    N_before = len(gs_means)
    print(f"[PLY Loading] Successfully loaded {N_before} GS points.")
    
    # 4. Filter GS points
    print(f"[Filtering] Starting to filter {N_before} GS points (this may take several minutes)...")
    
    # Convert GS coordinates to voxel indices
    gs_voxel_coords = np.floor(gs_means / args.voxel_size).astype(np.int32)
    
    # This is a pure Python loop, but set lookup is O(1) and very fast
    # The bottleneck is converting numpy rows to tuples
    mask_to_keep = np.zeros(N_before, dtype=bool)
    
    for i in tqdm(range(N_before), desc="Checking GS points against protective mask"):
        # Convert (i, j, k) coordinates to tuple for set lookup
        if tuple(gs_voxel_coords[i]) in dilated_voxel_set:
            mask_to_keep[i] = True
            
    # 5. Apply mask
    filtered_v_data = v_data[mask_to_keep]
    N_after = len(filtered_v_data)
    N_removed = N_before - N_after
    print(f"[Filtering] Filtering completed.")
    print(f"[Results] Kept: {N_after} points ({(N_after / N_before * 100):.2f}%)")
    print(f"[Results] Removed: {N_removed} points ({(N_removed / N_before * 100):.2f}%)")
    
    # 6. Save new PLY file
    print(f"[Saving] Saving cleaned point cloud to: {args.ply_out}...")
    
    # Create a new PlyElement with filtered data but preserving original structure
    el = PlyElement.describe(filtered_v_data, 'vertex')
    
    # Write file (binary=True for faster speed and smaller file size)
    PlyData([el], text=False).write(args.ply_out)
    
    print(f"[Completion] Clean PLY file saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Use LiDAR point cloud (points3D.txt) as a 'protective mask'
                       to filter out floating artifacts and noise from 3DGS PLY files."""
    )
    
    parser.add_argument("--ply_in", type=str, required=True,
                        help="Input GS PLY file path to clean (e.g., scene1.ply)")
                        
    parser.add_argument("--lidar_in", type=str, required=True,
                        help="LiDAR point cloud file defining the 'good' region (e.g., scene1/sparse/0/points3D.txt)")
                        
    parser.add_argument("--ply_out", type=str, required=True,
                        help="Output cleaned PLY file path (e.g., scene1_cleaned.ply)")
                        
    parser.add_argument("--lidar_cache_path", type=str, default=None,
                        help="Path for LiDAR point cloud NPY cache file. If empty, defaults to 'lidar_in' with .npy extension")
                        
    parser.add_argument("--voxel_size", type=float, default=1.0,
                        help="Voxel size in meters for building the protective mask (default: 1.0)")
                        
    parser.add_argument("--dilation_radius", type=int, default=1,
                        help="Dilation radius for the protective mask in voxels (default: 1, i.e., 3x3x3 neighborhood)")

    args = parser.parse_args()
    
    main(args)