#!/bin/bash
# LOC=loc0_2
# LOC=loc0_4
# LOC=loc1975_1
# LOC=loc1200_2
# LOC=loc1200_1
LOC=loc2450_1
# LOC=loc2450_100

GPU_NUM=8
CAP_MAX=2000000
NOISE_SCALE=500000
OPACITY_REG=0.01
SCALE_REG=0.01
ITER=400000
DENSIFY_FROM=2000
DENSIFY_UNTIL=200000
DENSIFY_INTER=100
# SCALE_LR=0.005
SCALE_LR=0.0005
OPACITY_RESET=3000
# POS_LR=0.00016
# POS_LR_FINAL=0.0000016
POS_LR=0.00002
POS_LR_FINAL=0.0000002
INIT_TYPE=random_cube

torchrun --standalone --nnodes=1 --nproc_per_node ${GPU_NUM} train.py --bsz ${GPU_NUM} -s /lustre/fs3/portfolios/nvr/users/ymingli/datasets/ithaca_colmap_aligned/$LOC -m /lustre/fs3/portfolios/nvr/users/ymingli/experiments/mcmc_colmap_aligned/${LOC}_ColmapInit_NoSpatialLrScale_Train${ITER}_DensifyFrom${DENSIFY_FROM}DensifyUntil${DENSIFY_UNTIL}_DensifyInterval${DENSIFY_INTER}_McmcNoiseScale${NOISE_SCALE}_capMax${CAP_MAX}_GPU${GPU_NUM}_OpacityReg${OPACITY_REG}_ScaleReg${SCALE_REG}_PosLr${PS_LR}Scheduler${ITER}_ScaleLR${SCALE_LR}_OpacityReset${OPACITY_RESET}_InitType${INIT_TYPE}  \
 --iterations $ITER  --densify_from_iter $DENSIFY_FROM --densify_until_iter $DENSIFY_UNTIL --mcmc --mcmc_noise_scale $NOISE_SCALE --cap_max $CAP_MAX --enable_timer --end2end_time --check_gpu_memory --check_cpu_memory --preload_dataset_to_gpu_threshold 0 --opacity_reg $OPACITY_REG --scale_reg $SCALE_REG --densification_interval $DENSIFY_INTER --position_lr_init $POS_LR --position_lr_final $POS_LR_FINAL --position_lr_max_steps $ITER --scaling_lr $SCALE_LR --opacity_reset_interval $OPACITY_RESET --init_type $INIT_TYPE
 
python render.py -s /lustre/fs3/portfolios/nvr/users/ymingli/datasets/ithaca_colmap_aligned/$LOC -m /lustre/fs3/portfolios/nvr/users/ymingli/experiments/mcmc_colmap_aligned/${LOC}_ColmapInit_NoSpatialLrScale_Train${ITER}_DensifyFrom${DENSIFY_FROM}DensifyUntil${DENSIFY_UNTIL}_DensifyInterval${DENSIFY_INTER}_McmcNoiseScale${NOISE_SCALE}_capMax${CAP_MAX}_GPU${GPU_NUM}_OpacityReg${OPACITY_REG}_ScaleReg${SCALE_REG}_PosLr${PS_LR}Scheduler${ITER}_ScaleLR${SCALE_LR}_OpacityReset${OPACITY_RESET}_InitType${INIT_TYPE}

/lustre/fsw/portfolios/nvr/users/ymingli/rclone copy /lustre/fs3/portfolios/nvr/users/ymingli/experiments/mcmc_colmap_aligned/${LOC}_ColmapInit_NoSpatialLrScale_Train${ITER}_DensifyFrom${DENSIFY_FROM}DensifyUntil${DENSIFY_UNTIL}_DensifyInterval${DENSIFY_INTER}_McmcNoiseScale${NOISE_SCALE}_capMax${CAP_MAX}_GPU${GPU_NUM}_OpacityReg${OPACITY_REG}_ScaleReg${SCALE_REG}_PosLr${PS_LR}Scheduler${ITER}_ScaleLR${SCALE_LR}_OpacityReset${OPACITY_RESET}_InitType${INIT_TYPE}/train/ours_399993/renders.mp4 gw2396:mcmc_colmap_aligned/${LOC}_ColmapInit_NoSpatialLrScale_Train${ITER}_DensifyFrom${DENSIFY_FROM}DensifyUntil${DENSIFY_UNTIL}_DensifyInterval${DENSIFY_INTER}_McmcNoiseScale${NOISE_SCALE}_capMax${CAP_MAX}_GPU${GPU_NUM}_OpacityReg${OPACITY_REG}_ScaleReg${SCALE_REG}_PosLr${PS_LR}Scheduler${ITER}_ScaleLR${SCALE_LR}_OpacityReset${OPACITY_RESET}_InitType${INIT_TYPE} -P