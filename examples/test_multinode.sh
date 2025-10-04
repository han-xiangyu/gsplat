#!/bin/bash
set -e

# 激活你的 Conda 环境
source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/bin/activate
conda activate mars_new

# 使用 torchrun 启动你的测试脚本
torchrun --nproc_per_node=8 \
         --nnodes=$SLURM_NNODES \
         --rdzv_id=$SLURM_JOB_ID \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
         test_distributed.py