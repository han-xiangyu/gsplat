#!/usr/bin/env python
"""
3DGS Attribute Analyzer (analyze_gs_attributes_v2.py)

Loads a 3DGS PLY file, converts attributes (scale, opacity) from log/logit 
space to physical space (meters, 0-1), prints their distribution 
statistics (quantiles), and optionally saves histogram plots.
"""

import os
import numpy as np
import argparse
from plyfile import PlyData
from tqdm import tqdm
import torch  # Use torch for fast math ops

# Try to import matplotlib. Raise error only if --save_plots is used.
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load PLY file (only necessary attributes)
    print(f"[PLY Loading] Loading {args.ply_in}...")
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
        print(f"[PLY Loading] Successfully loaded {N} GS points.")
        
    except Exception as e:
        print(f"Error loading {args.ply_in}: {e}")
        return

    # 2. Convert data to physical space (in batches)
    print("[Computing] Converting attributes to physical space (log->linear, logit->sigmoid)...")
    
    batch_size = args.batch_size
    all_max_scales = []
    all_anisotropies = []
    all_opacities = []

    for i in tqdm(range(0, N, batch_size), desc="Converting attributes"):
        s_idx = i
        e_idx = min(i + batch_size, N)
        
        # Load batch
        scales_log_b = torch.from_numpy(scales_log[s_idx:e_idx]).to(device)
        opacities_logit_b = torch.from_numpy(opacities_logit[s_idx:e_idx]).to(device)

        # Convert
        scales_physical_b = torch.exp(scales_log_b)
        opacities_physical_b = torch.sigmoid(opacities_logit_b)
        
        # Calculate derived metrics
        max_scale_b = torch.max(scales_physical_b, dim=1).values
        min_scale_b = torch.min(scales_physical_b, dim=1).values
        
        # Calculate anisotropy (ratio of max to min scale)
        # Add a small epsilon (1e-8) to prevent division by zero for points
        # where min_scale might be 0.
        anisotropy_b = max_scale_b / (min_scale_b + 1e-8)
        
        # Collect results (move back to CPU)
        all_max_scales.append(max_scale_b.cpu())
        all_anisotropies.append(anisotropy_b.cpu())
        all_opacities.append(opacities_physical_b.cpu())
        
        # (Clean up GPU memory)
        del scales_log_b, opacities_logit_b, scales_physical_b
        del max_scale_b, min_scale_b, anisotropy_b, opacities_physical_b

    print("[Computing] Attribute conversion complete. Concatenating results...")
    
    # Concatenate all batch results
    max_scales_cpu_torch = torch.cat(all_max_scales)
    anisotropies_cpu_torch = torch.cat(all_anisotropies)
    opacities_cpu_torch = torch.cat(all_opacities)
    
    del all_max_scales, all_anisotropies, all_opacities # Free memory
    print(f"Total points: {len(max_scales_cpu_torch)}")

    # 3. Calculate statistics (Using NumPy for memory efficiency)
    print("[Analyzing] Calculating quantile statistics using NumPy...")

    # Define quantiles
    quantiles_high = np.array([0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0])
    q_names_high = ["50% (Median)", "90%", "95%", "99%", "99.5%", "99.9%", "100% (Max)"]
    
    quantiles_low = np.array([0.0, 0.001, 0.01, 0.05, 0.1, 0.5])
    q_names_low = ["0% (Min)", "0.1%", "1%", "5%", "10%", "50% (Median)"]

    # Convert to NumPy arrays for quantile calculation
    max_scales_np = max_scales_cpu_torch.numpy()
    anisotropies_np = anisotropies_cpu_torch.numpy()
    opacities_np = opacities_cpu_torch.numpy()
    
    # Release torch tensor memory immediately
    del max_scales_cpu_torch, anisotropies_cpu_torch, opacities_cpu_torch

    # Use np.quantile
    try:
        # NumPy 1.22+
        stats_max_scale = np.quantile(max_scales_np, quantiles_high, method='linear')
        stats_anisotropy = np.quantile(anisotropies_np, quantiles_high, method='linear')
        stats_opacity = np.quantile(opacities_np, quantiles_low, method='linear')
    except (TypeError, AttributeError):
        # Fallback for older NumPy
        print("[Warning] Using fallback 'interpolation' for np.quantile. Please upgrade NumPy if possible.")
        stats_max_scale = np.quantile(max_scales_np, quantiles_high, interpolation='linear')
        stats_anisotropy = np.quantile(anisotropies_np, quantiles_high, interpolation='linear')
        stats_opacity = np.quantile(opacities_np, quantiles_low, interpolation='linear')

    # 4. Print Report
    print("\n" + "="*50)
    print(f" Attribute Analysis Report: {os.path.basename(args.ply_in)}")
    print(f" (Analyzed points: {N})")
    print("="*50)

    print("\n--- 1. Max Scale (meters) ---")
    print(" (Used for --max_scale, fixes 'solid color' regions)")
    print(f"{'Quantile':<15} | {'Scale (m)':<15}")
    print(f"{'-'*15: <15} | {'-'*15: <15}")
    for name, val in zip(q_names_high, stats_max_scale):
        print(f"{name: <15} | {val.item():<15.4f}")

    print("\n--- 2. Anisotropy (ratio) ---")
    print(" (Used for --max_anisotropy, fixes 'floaters/needles')")
    print(f"{'Quantile':<15} | {'Ratio (max/min)':<15}")
    print(f"{'-'*15: <15} | {'-'*15: <15}")
    for name, val in zip(q_names_high, stats_anisotropy):
        print(f"{name: <15} | {val.item():<15.1f}")
        
    print("\n--- 3. Opacity (0-1) ---")
    print(" (Used for --min_opacity, fixes 'ghost' points)")
    print(f"{'Quantile':<15} | {'Opacity (0-1)':<15}")
    print(f"{'-'*15: <15} | {'-'*15: <15}")
    for name, val in zip(q_names_low, stats_opacity):
        print(f"{name: <15} | {val.item():<15.6f}")

    print("\n" + "="*50)
    print(" [Analysis Advice]")
    print(f" 1. Look at 'Max Scale' 99% or 99.5%. If 99.9% or Max jumps to a huge value (e.g., > 1000),")
    print(f"    choose the 99% or 99.5% value (e.g., 2.5) as --max_scale 2.5.")
    print(f" 2. Look at 'Anisotropy', similarly, choose a reasonable upper bound (e.g., 1000) as --max_anisotropy 1000.")
    print(f" 3. Look at 'Opacity', choose a threshold you consider 'invisible' (e.g., 0.01) as --min_opacity 0.01.")
    print("="*50)

    # -----------------------------------------------------------------
    # --- [NEW FEATURE] Plotting ---
    # -----------------------------------------------------------------
    if args.save_plots:
        if not MATPLOTLIB_AVAILABLE:
            print("\n[Plotting Error] '--save_plots' was specified, but 'matplotlib' is not installed.")
            print("Please install it: pip install matplotlib")
            return
            
        print("\n[Plotting] Generating distribution plots...")
        output_dir = os.path.dirname(args.vis) or "."
        ply_basename = os.path.splitext(os.path.basename(args.ply_in))[0]

        # --- Plot 1: Max Scale ---
        try:
            plt.figure(figsize=(12, 7))
            
            # Filter out zeros/negatives for log scale
            scales_to_plot = max_scales_np[max_scales_np > 1e-6]
            if len(scales_to_plot) > 0:
                # Use log bins
                min_log = np.log10(scales_to_plot.min())
                max_log = np.log10(scales_to_plot.max())
                bins = np.logspace(min_log, max_log, 100)
                
                plt.hist(scales_to_plot, bins=bins, color='blue', alpha=0.7)
                plt.xscale('log') # Use log scale on x-axis
                
                # Add quantile lines
                q_vals = [stats_max_scale[1], stats_max_scale[3], stats_max_scale[5]]
                q_names = [q_names_high[1], q_names_high[3], q_names_high[5]]
                colors = ['green', 'orange', 'red']
                for qv, qn, c in zip(q_vals, q_names, colors):
                    plt.axvline(x=qv, color=c, linestyle='--', 
                                label=f'{qn} ({qv:.2f}m)')

                plt.title(f'Max Scale Distribution (Log Scale)\n{ply_basename}')
                plt.xlabel('Max Scale (meters) - Log Scale')
                plt.ylabel('Count')
                plt.legend()
                plt.grid(True, which="both", ls="--", alpha=0.5)
                
                plot_path = os.path.join(output_dir, f"{ply_basename}_scale_dist.png")
                plt.savefig(plot_path)
                plt.close()
                print(f"[Plotting] Saved Max Scale plot to {plot_path}")
            else:
                print("[Plotting] Skipped Max Scale plot (no data > 1e-6).")

        except Exception as e:
            print(f"[Plotting Error] Failed to generate Max Scale plot: {e}")

        # --- Plot 2: Anisotropy ---
        try:
            plt.figure(figsize=(12, 7))
            
            # Filter out values <= 1 for log scale (ratio < 1 is not meaningful)
            anisotropies_to_plot = anisotropies_np[anisotropies_np > 1]
            if len(anisotropies_to_plot) > 0:
                min_log_aniso = np.log10(anisotropies_to_plot.min())
                max_log_aniso = np.log10(anisotropies_to_plot.max())
                bins_aniso = np.logspace(min_log_aniso, max_log_aniso, 100)
                
                plt.hist(anisotropies_to_plot, bins=bins_aniso, color='purple', alpha=0.7)
                plt.xscale('log') # Use log scale on x-axis
                
                # Add quantile lines
                q_vals_aniso = [stats_anisotropy[1], stats_anisotropy[3], stats_anisotropy[5]]
                q_names_aniso = [q_names_high[1], q_names_high[3], q_names_high[5]]
                colors = ['green', 'orange', 'red']
                for qv, qn, c in zip(q_vals_aniso, q_names_aniso, colors):
                    plt.axvline(x=qv, color=c, linestyle='--', 
                                label=f'{qn} ({qv:.1f})')

                plt.title(f'Anisotropy Distribution (Log Scale)\n{ply_basename}')
                plt.xlabel('Anisotropy Ratio (max_scale / min_scale) - Log Scale')
                plt.ylabel('Count')
                plt.legend()
                plt.grid(True, which="both", ls="--", alpha=0.5)
                
                plot_path_aniso = os.path.join(output_dir, f"{ply_basename}_anisotropy_dist.png")
                plt.savefig(plot_path_aniso)
                plt.close()
                print(f"[Plotting] Saved Anisotropy plot to {plot_path_aniso}")
            else:
                print("[Plotting] Skipped Anisotropy plot (no data > 1).")
                
        except Exception as e:
            print(f"[Plotting Error] Failed to generate Anisotropy plot: {e}")

    # (Clean up large arrays explicitly)
    del max_scales_np, anisotropies_np, opacities_np
    print("[Done]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""(V2 w/ Plots) Analyze 3DGS PLY attributes and optionally plot distributions."""
    )
    
    parser.add_argument("--ply_in", type=str, required=True,
                        help="Input GS PLY file path to analyze (e.g., scene1.ply)")
    parser.add_argument("--batch_size", type=int, default=10_000_000,
                        help="Batch size for processing large files (default: 10,000,000)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use for conversion (default: 0)")
    
    # --- New Plotting Argument ---
    parser.add_argument("--save_plots", action='store_true',
                        help="[NEW] If specified, save histogram plots for scale and anisotropy.")

    parser.add_argument("--vis", type=str, default="/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/vis",
                        help="GPU ID to use for conversion (default: 0)")
    
    args = parser.parse_args()
    
    main(args)