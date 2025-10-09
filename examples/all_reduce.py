import torch
import torch.distributed as dist
import time
import os
from dataclasses import dataclass
import tyro
from gsplat.distributed import cli

@dataclass
class Config:
    """Configuration for the All-Reduce benchmark."""
    tensor_size_mb: int = 256
    """Size of the tensor to be reduced in Megabytes (MB)."""
    
    num_iterations: int = 10
    """Number of iterations to run the benchmark for timing."""
    
    warmup_iterations: int = 5
    """Number of warm-up iterations to run before timing."""

def main(local_rank: int, world_rank: int, world_size: int, cfg: Config):
    """
    Main function to run the distributed benchmark.
    This function is launched by the `gsplat.distributed.cli` helper.
    """
    if world_size < 2:
        if world_rank == 0:
            print("Error: This benchmark requires at least 2 processes (world_size >= 2).")
        return

    # 1. 设置当前进程使用的GPU
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    
    # 2. 计算张量大小
    # 1 MB = 1024 * 1024 bytes. torch.float32 is 4 bytes.
    num_elements = (cfg.tensor_size_mb * 1024 * 1024) // 4
    tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
    tensor_size_bytes = tensor.nelement() * tensor.element_size()

    if world_rank == 0:
        print("--------------------------------------------------")
        print(f"Starting All-Reduce Bandwidth Benchmark")
        print(f"World Size: {world_size} processes")
        print(f"Tensor Size: {cfg.tensor_size_mb} MB ({tensor_size_bytes / 1e9:.3f} GB)")
        print(f"Warm-up Iterations: {cfg.warmup_iterations}")
        print(f"Timed Iterations: {cfg.num_iterations}")
        print("--------------------------------------------------")

    # 3. 热身运动 (Warm-up)
    # 预热对于获得稳定的性能测量至关重要，它可以确保CUDA上下文已创建、缓存已预热等。
    if world_rank == 0:
        print("Running warm-up iterations...")
    for _ in range(cfg.warmup_iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # 确保所有GPU上的热身操作都已完成
    torch.cuda.synchronize()

    # 4. 精确计时
    # 使用 barrier 确保所有进程同时开始计时
    dist.barrier()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 开始记录时间
    start_event.record()

    for i in range(cfg.num_iterations):
        # 执行 all_reduce 操作
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # 结束记录时间
    end_event.record()

    # 等待所有GPU操作完成，确保时间的准确性
    torch.cuda.synchronize()
    
    # 5. 计算并报告结果 (仅在主进程上)
    if world_rank == 0:
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_s = elapsed_time_ms / 1000.0
        
        # 对于all_reduce, 数据传输量约等于 2 * 张量大小 * 迭代次数
        # (数据发送出去聚合，结果再接收回来)
        total_data_bytes = 2 * tensor_size_bytes * cfg.num_iterations
        total_data_gb = total_data_bytes / 1e9
        
        # 带宽 (GB/s) = 总数据量(GB) / 总时间(s)
        bandwidth_gb_s = total_data_gb / elapsed_time_s
        
        avg_time_per_iter_ms = elapsed_time_ms / cfg.num_iterations

        print("\n------------------ RESULTS ------------------")
        print(f"Total time for {cfg.num_iterations} iterations: {elapsed_time_s:.4f} seconds")
        print(f"Average time per iteration: {avg_time_per_iter_ms:.4f} ms")
        print(f"Total data transferred over the wire: {total_data_gb:.3f} GB")
        print(f"Estimated Inter-Node Bandwidth: {bandwidth_gb_s:.3f} GB/s")
        print("-------------------------------------------")
        print("Note: This is an approximation of the effective bandwidth for the all_reduce operation.")


if __name__ == "__main__":
    """
    Usage for 2-Node Benchmark:

    **On Node 0 (e.g., IP: 10.1.1.1):**
    ```bash
    torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank=0 \
        --master_addr="10.1.1.1" \
        --master_port=12345 \
        benchmark_all_reduce.py --tensor-size-mb 512
    ```

    **On Node 1 (e.g., IP: 10.1.1.2):**
    ```bash
    torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank=1 \
        --master_addr="10.1.1.1" \
        --master_port=12345 \
        benchmark_all_reduce.py --tensor-size-mb 512
    ```
    """
    cfg = tyro.cli(Config)
    # The gsplat cli helper handles the distributed environment setup.
    cli(main, cfg)