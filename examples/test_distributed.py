# test_distributed.py
import torch
import torch.distributed as dist
import os
import time

def setup_distributed():
    """初始化 PyTorch 分布式环境"""
    if not dist.is_available() or not torch.cuda.is_available():
        print("Distributed training or CUDA is not available.")
        return None, None, None, None

    # torchrun 会自动设置这些环境变量
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # 初始化进程组，NCCL 是 NVIDIA GPU 推荐的后端
    dist.init_process_group(backend='nccl')
    
    # 将当前进程绑定到指定的 GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    return rank, world_size, local_rank, device

def main():
    rank, world_size, local_rank, device = setup_distributed()

    if rank is None:
        return

    # 1. 基础信息打印 (每个进程都会打印)
    print(
        f"Hello from Global Rank {rank}/{world_size} on device cuda:{local_rank}. "
        f"Hostname: {os.uname()[1]}"
    )
    
    # 确保所有进程都完成了初始化打印
    dist.barrier()
    time.sleep(rank * 0.1) # 错开打印，避免日志混乱

    # --- 2. 测试 all_reduce ---
    # 每个进程创建一个值为自己 Global Rank 的张量
    tensor_to_reduce = torch.tensor([rank], dtype=torch.float32, device=device)
    
    if rank == 0:
        print("\n--- Testing all_reduce (SUM) ---")
    
    print(f"[Rank {rank}] Tensor before all_reduce: {tensor_to_reduce.item()}")
    
    # 执行 all_reduce 操作 (所有进程的张量相加)
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.SUM)
    
    dist.barrier()
    
    print(f"[Rank {rank}] Tensor after all_reduce: {tensor_to_reduce.item()}")
    
    if rank == 0:
        # 理论上的总和是 0 + 1 + 2 + ... + (world_size - 1)
        expected_sum = sum(range(world_size))
        print(f"--> Expected sum is {expected_sum}. All ranks should show this value.")

    # --- 3. 测试 all_gather ---
    dist.barrier()
    time.sleep(1)

    # 每个进程创建一个值为 Global Rank*10 的张量
    tensor_to_gather = torch.tensor([rank * 10], dtype=torch.float32, device=device)
    
    # 准备一个列表来存放从所有进程收集到的张量
    gather_list = [torch.zeros_like(tensor_to_gather) for _ in range(world_size)]
    
    if rank == 0:
        print("\n--- Testing all_gather ---")
        
    print(f"[Rank {rank}] Tensor before all_gather: {tensor_to_gather.item()}")

    # 执行 all_gather 操作
    dist.all_gather(gather_list, tensor_to_gather)
    
    dist.barrier()
    
    # 将 tensor 列表转换为普通 Python 列表以便打印
    gathered_data = [t.item() for t in gather_list]
    print(f"[Rank {rank}] List after all_gather: {gathered_data}")
    
    if rank == 0:
        expected_list = [i * 10 for i in range(world_size)]
        print(f"--> Expected list is {expected_list}. All ranks should show this list.")
        print("\n--- Distributed Test Finished Successfully! ---")

if __name__ == "__main__":
    main()