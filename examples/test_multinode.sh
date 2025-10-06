#!/bin/bash
set -e

source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/bin/activate
conda activate mars_new

export PYTHONWARNINGS="ignore:The pynvml package is deprecated"

echo "Starting distributed training..."
torchrun --nproc_per_node=8 \
         --nnodes=$SLURM_NNODES \
         --rdzv_id=$SLURM_JOB_ID \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
         test_distributed.py