#!/bin/bash

# LOC=loc0_4
# LOC=loc1975_1
# LOC=loc1200_2
# LOC=loc1200_1
LOC=loc2450_1

GPU_NUM=4
torchrun --standalone --nnodes=1 --nproc_per_node ${GPU_NUM} train.py --bsz ${GPU_NUM} -s /lustre/fs3/portfolios/nvr/users/ymingli/datasets/Ithaca-full/rectified/$LOC -m /lustre/fs3/portfolios/nvr/users/ymingli/experiments/mcmc/${LOC}_1mInit_Train400k_McmcNoiseScale500k_capMas1m \
 --iterations 400_000 --densify_until_iter 200_000 --mcmc --mcmc_noise_scale 500000 --cap_max 1000000 --enable_timer --end2end_time --check_gpu_memory --check_cpu_memory --eval --preload_dataset_to_gpu_threshold 0
