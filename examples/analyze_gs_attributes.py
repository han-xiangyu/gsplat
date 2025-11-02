#!/usr/bin/env python
"""
3DGS 属性分析器 (analyze_gs_attributes.py)

加载一个 3DGS PLY 文件，将其属性 (scale, opacity) 从 log/logit 空间
转换到物理空间 (米, 0-1)，并打印它们的分布统计（百分位数），
以便为过滤脚本 (filter_ply_by_lidar_v2.py) 选择合理的阈值。
"""

import os
import numpy as np
import argparse
from plyfile import PlyData
from tqdm import tqdm
import torch # 使用 torch 进行快速的数学运算

def main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载 PLY 文件
    print(f"[PLY加载] 正在加载 {args.ply_in}...")
    try:
        plydata_in = PlyData.read(args.ply_in)
        v_data = plydata_in['vertex'].data
        
        scales_log = np.stack([
            v_data['scale_0'], 
            v_data['scale_1'], 
            v_data['scale_2']
        ], axis=1).astype(np.float32)
        
        opacities_logit = v_data['opacity'].astype(np.float32)
        
        N = len(scales_log)
        print(f"[PLY加载] 成功加载 {N} 个GS点。")
        
    except Exception as e:
        print(f"Error loading {args.ply_in}: {e}")
        return

    # 2. 将数据移动到 GPU/CPU 并进行物理转换
    print("[计算] 正在将属性转换为物理空间 (log->linear, logit->sigmoid)...")
    
    batch_size = args.batch_size
    all_max_scales = []
    all_anisotropies = []
    all_opacities = []

    for i in tqdm(range(0, N, batch_size), desc="转换属性"):
        s_idx = i
        e_idx = min(i + batch_size, N)
        
        scales_log_b = torch.from_numpy(scales_log[s_idx:e_idx]).to(device)
        opacities_logit_b = torch.from_numpy(opacities_logit[s_idx:e_idx]).to(device)

        scales_physical_b = torch.exp(scales_log_b)
        opacities_physical_b = torch.sigmoid(opacities_logit_b)
        
        max_scale_b = torch.max(scales_physical_b, dim=1).values
        min_scale_b = torch.min(scales_physical_b, dim=1).values
        
        anisotropy_b = max_scale_b / (min_scale_b + 1e-8)
        
        all_max_scales.append(max_scale_b.cpu())
        all_anisotropies.append(anisotropy_b.cpu())
        all_opacities.append(opacities_physical_b.cpu())
        
        del scales_log_b, opacities_logit_b, scales_physical_b
        del max_scale_b, min_scale_b, anisotropy_b, opacities_physical_b

    print("[计算] 属性转换完成。正在合并结果...")
    
    max_scales_cpu = torch.cat(all_max_scales)
    anisotropies_cpu = torch.cat(all_anisotropies)
    opacities_cpu = torch.cat(all_opacities)
    
    del all_max_scales, all_anisotropies, all_opacities
    print(f"总点数: {len(max_scales_cpu)}")

    # -----------------------------------------------------------------
    # --- [修改开始] ---
    # -----------------------------------------------------------------

    # --- [Plan B: 子采样] ---
    # 如果 Plan A (NumPy) 仍然内存不足，请取消注释下面的代码块。
    # 这将使用 1000 万个点的随机样本来 *近似* 统计，速度极快。
    # --------------------------------
    # sample_size = 10_000_000 # 1千万点
    # if N > sample_size:
    #     print(f"[分析] 张量过大, 使用 {sample_size} 个点的随机采样进行近似统计...")
    #     indices = torch.randperm(N)[:sample_size]
    #     max_scales_cpu = max_scales_cpu[indices]
    #     anisotropies_cpu = anisotropies_cpu[indices]
    #     opacities_cpu = opacities_cpu[indices]
    #     N = sample_size # 更新总点数
    # --------------------------------

    # 3. 计算统计数据 (Plan A: 使用 NumPy)
    print("[分析] 正在使用 NumPy (更内存高效) 计算百分位数统计...")

    # 定义百分位数 (现在是 numpy 数组)
    quantiles_high = np.array([0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0])
    q_names_high = ["50% (Median)", "90%", "95%", "99%", "99.5%", "99.9%", "100% (Max)"]
    
    quantiles_low = np.array([0.0, 0.001, 0.01, 0.05, 0.1, 0.5])
    q_names_low = ["0% (Min)", "0.1%", "1%", "5%", "10%", "50% (Median)"]

    # 将 Torch Tensors 转换为 NumPy arrays
    max_scales_np = max_scales_cpu.numpy()
    anisotropies_np = anisotropies_cpu.numpy()
    opacities_np = opacities_cpu.numpy()
    
    # 立即释放 Torch Tensors 的内存
    del max_scales_cpu, anisotropies_cpu, opacities_cpu

    # 使用 np.quantile (注意: PyTorch 2.x+ 版本的 np.quantile 也使用 'interpolation' 关键字)
    try:
        stats_max_scale = np.quantile(max_scales_np, quantiles_high, interpolation='linear')
        stats_anisotropy = np.quantile(anisotropies_np, quantiles_high, interpolation='linear')
        stats_opacity = np.quantile(opacities_np, quantiles_low, interpolation='linear')
    except TypeError:
        # 兼容旧版 NumPy (可能使用 'interpolation' 而不是 'interpolation')
        print("[警告] 'interpolation' 参数失败, 尝试旧版 'interpolation'...")
        stats_max_scale = np.quantile(max_scales_np, quantiles_high, interpolation='linear')
        stats_anisotropy = np.quantile(anisotropies_np, quantiles_high, interpolation='linear')
        stats_opacity = np.quantile(opacities_np, quantiles_low, interpolation='linear')

    # 释放 NumPy Arrays 的内存
    del max_scales_np, anisotropies_np, opacities_np

    # -----------------------------------------------------------------
    # --- [修改结束] ---
    # -----------------------------------------------------------------

    # 4. 打印报告
    print("\n" + "="*50)
    print(f" 属性分析报告: {os.path.basename(args.ply_in)}")
    print(f" (分析的点数: {N})")
    print("="*50)

    print("\n--- 1. 最大尺度 (Max Scale) (单位: 米) ---")
    print(" (用于设置 --max_scale，解决“纯色区域”问题)")
    print(f"{'百分位数':<15} | {'尺度 (米)':<15}")
    print(f"{'-'*15: <15} | {'-'*15: <15}")
    for name, val in zip(q_names_high, stats_max_scale):
        print(f"{name: <15} | {val.item():<15.4f}") # .item() 同样适用于 numpy 标量

    print("\n--- 2. 各向异性 (Anisotropy) (比例) ---")
    print(" (用于设置 --max_anisotropy，解决“毛刺”问题)")
    print(f"{'百分位数':<15} | {'比例 (max/min)':<15}")
    print(f"{'-'*15: <15} | {'-'*15: <15}")
    for name, val in zip(q_names_high, stats_anisotropy):
        print(f"{name: <15} | {val.item():<15.1f}")
        
    print("\n--- 3. 透明度 (Opacity) (0-1) ---")
    print(" (用于设置 --min_opacity，剔除“幽灵”点)")
    print(f"{'百分位数':<15} | {'透明度 (0-1)':<15}")
    print(f"{'-'*15: <15} | {'-'*15: <15}")
    for name, val in zip(q_names_low, stats_opacity):
        print(f"{name: <15} | {val.item():<15.6f}")

    print("\n" + "="*50)
    print(" [分析建议]")
    print(f" 1. 查看 '最大尺度' 的 99% 或 99.5% 的值。如果 99.9% 或 Max 突然变得非常大 (例如 > 1000)，")
    print(f"    则应选择 99% 或 99.5% 处的值 (例如 2.5) 作为 --max_scale 2.5。")
    print(f" 2. 查看 '各向异性'，同理，选择一个合理的上限 (例如 1000) 作为 --max_anisotropy 1000。")
    print(f" 3. 查看 '透明度'，选择一个你认为“不可见”的阈值 (例如 0.01) 作为 --min_opacity 0.01。")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""(分析工具) 分析 3DGS PLY 文件的物理属性分布，
                       以帮助为过滤脚本选择合适的阈值。"""
    )
    
    parser.add_argument("--ply_in", type=str, required=True,
                        help="输入的、待分析的GS PLY文件 (例如 scene1.ply)")
    parser.add_argument("--batch_size", type=int, default=10_000_000,
                        help="用于处理大文件的分块大小 (默认: 10,000,000)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="要使用的 GPU ID (默认: 0)")
    
    args = parser.parse_args()
    
    main(args)