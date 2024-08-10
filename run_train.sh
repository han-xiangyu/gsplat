#!/bin/bash

# LOC=loc0_4
# LOC=loc1975_1
# LOC=loc1200_2
# LOC=loc1200_1
LOC=loc2450_1

GPU_NUM=8
CAP_MAX=5000000
NOISE_SCALE=500000
OPACITY_REG=0.001
SCALE_REG=0.001
DENSIFY_INTER=100

torchrun --standalone --nnodes=1 --nproc_per_node ${GPU_NUM} train.py --bsz ${GPU_NUM} -s /lustre/fs3/portfolios/nvr/users/ymingli/datasets/Ithaca-full/rectified/$LOC -m /lustre/fs3/portfolios/nvr/users/ymingli/experiments/mcmc/${LOC}_1mInit_Train200k_DensifyUntil100k_DensifyInterval${DENSIFY_INTER}_McmcNoiseScale${NOISE_SCALE}_capMax${CAP_MAX}_GPU${GPU_NUM}_OpacityReg${OPACITY_REG}_ScaleReg${SCALE_REG} \
 --iterations 200_000 --densify_until_iter 100_000 --mcmc --mcmc_noise_scale $NOISE_SCALE --cap_max $CAP_MAX --enable_timer --end2end_time --check_gpu_memory --check_cpu_memory --preload_dataset_to_gpu_threshold 0 --opacity_reg $OPACITY_REG --scale_reg $SCALE_REG --densification_interval $DENSIFY_INTER
