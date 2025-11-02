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
    
    # 1. 加载 PLY 文件 (仅加载必要的属性)
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
    
    # 为超大张量分块处理
    batch_size = args.batch_size
    
    # 准备空的列表来收集物理值
    # 我们只收集我们需要的值，以节省内存
    all_max_scales = []
    all_anisotropies = []
    all_opacities = []

    for i in tqdm(range(0, N, batch_size), desc="转换属性"):
        s_idx = i
        e_idx = min(i + batch_size, N)
        
        # 加载分块
        scales_log_b = torch.from_numpy(scales_log[s_idx:e_idx]).to(device)
        opacities_logit_b = torch.from_numpy(opacities_logit[s_idx:e_idx]).to(device)

        # 转换
        scales_physical_b = torch.exp(scales_log_b)
        opacities_physical_b = torch.sigmoid(opacities_logit_b)
        
        # 计算尺度
        max_scale_b = torch.max(scales_physical_b, dim=1).values
        min_scale_b = torch.min(scales_physical_b, dim=1).values
        
        # 计算各向异性 (为防止除零，增加一个极小值)
        anisotropy_b = max_scale_b / (min_scale_b + 1e-8)
        
        # 收集结果 (移回CPU)
        all_max_scales.append(max_scale_b.cpu())
        all_anisotropies.append(anisotropy_b.cpu())
        all_opacities.append(opacities_physical_b.cpu())
        
        # (清理GPU内存)
        del scales_log_b, opacities_logit_b, scales_physical_b
        del max_scale_b, min_scale_b, anisotropy_b, opacities_physical_b

    print("[计算] 属性转换完成。正在合并结果...")
    
    # 合并所有分块的结果
    max_scales_cpu = torch.cat(all_max_scales)
    anisotropies_cpu = torch.cat(all_anisotropies)
    opacities_cpu = torch.cat(all_opacities)
    
    del all_max_scales, all_anisotropies, all_opacities # 释放内存
    print(f"总点数: {len(max_scales_cpu)}")

    # 3. 计算统计数据 (百分位数)
    print("[分析] 正在计算百分位数统计...")

    # 高百分位数 (用于检查“最大值”)
    quantiles_high = torch.tensor([0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0])
    q_names_high = ["50% (Median)", "90%", "95%", "99%", "99.5%", "99.9%", "100% (Max)"]
    
    # 低百分位数 (用于检查“最小值”)
    quantiles_low = torch.tensor([0.0, 0.001, 0.01, 0.05, 0.1, 0.5])
    q_names_low = ["0% (Min)", "0.1%", "1%", "5%", "10%", "50% (Median)"]

    # torch.quantile 在 CPU 上可以处理任意大小的张量
    stats_max_scale = torch.quantile(max_scales_cpu, quantiles_high, interpolation='linear')
    stats_anisotropy = torch.quantile(anisotropies_cpu, quantiles_high, interpolation='linear')
    stats_opacity = torch.quantile(opacities_cpu, quantiles_low, interpolation='linear')
    
    # 释放最后的大张量
    del max_scales_cpu, anisotropies_cpu, opacities_cpu

    # 4. 打印报告
    print("\n" + "="*50)
    print(f" 属性分析报告: {os.path.basename(args.ply_in)}")
    print(f" (总点数: {N})")
    print("="*50)

    print("\n--- 1. 最大尺度 (Max Scale) (单位: 米) ---")
    print(" (用于设置 --max_scale，解决“纯色区域”问题)")
    print(f"{'百分位数':<15} | {'尺度 (米)':<15}")
    print(f"{'-'*15: <15} | {'-'*15: <15}")
    for name, val in zip(q_names_high, stats_max_scale):
        print(f"{name: <15} | {val.item():<15.4f}")

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