#!/bin/bash
gpus_per_node=1
nodes=2
account=nvr_av_foundations
PROJECT_DIR="/lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat/examples"

torchrun \
    --nproc_per_node=${gpus_per_node} \
    --nnodes=${nodes} \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ${PROJECT_DIR}/all_reduce.py \
    --tensor-size-mb 512