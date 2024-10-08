#!/bin/bash

# LOC=loc0_4
# LOC=loc1975_1
# LOC=loc1200_2
# LOC=loc1200_1
LOC=loc2450_1

GPU_NUM=8
CAP_MAX=2000000
NOISE_SCALE=500000
OPACITY_REG=0.01
SCALE_REG=0.01
ITER=400000
DENSIFY_FROM=2000
DENSIFY_UNTIL=200000
DENSIFY_INTER=100
SCALE_LR=0.0005
OPACITY_RESET=3000
POS_LR=0.00002
POS_LR_FINAL=0.0000002
INIT_TYPE=random_cube
RANDOM_INIT_NUM=1000000

torchrun --standalone --nnodes=1 --nproc_per_node ${GPU_NUM} train.py --bsz ${GPU_NUM} -s /lustre/fs3/portfolios/nvr/users/ymingli/datasets/Ithaca-full/rectified/$LOC -m /lustre/fs3/portfolios/nvr/users/ymingli/experiments/mcmc_gt/${LOC}_Train${ITER}_DensFrm${DENSIFY_FROM}DensUntl${DENSIFY_UNTIL}_DensIntvl${DENSIFY_INTER}_NoiseScale${NOISE_SCALE}_cap${CAP_MAX}_GPU${GPU_NUM}_OpaReg${OPACITY_REG}_ScalReg${SCALE_REG}_PosLr${PS_LR}Skjul${ITER}_SclLR${SCALE_LR}_OpaRist${OPACITY_RESET}_Init${INIT_TYPE}_InitNum${RANDOM_INIT_NUM}  \
 --iterations $ITER  --densify_from_iter $DENSIFY_FROM --densify_until_iter $DENSIFY_UNTIL --mcmc --mcmc_noise_scale $NOISE_SCALE --cap_max $CAP_MAX --enable_timer --end2end_time --check_gpu_memory --check_cpu_memory --preload_dataset_to_gpu_threshold 0 --opacity_reg $OPACITY_REG --scale_reg $SCALE_REG --densification_interval $DENSIFY_INTER --position_lr_init $POS_LR --position_lr_final $POS_LR_FINAL --position_lr_max_steps $ITER --scaling_lr $SCALE_LR --opacity_reset_interval $OPACITY_RESET --init_type $INIT_TYPE
 
python render.py -s /lustre/fs3/portfolios/nvr/users/ymingli/datasets/Ithaca-full/rectified/$LOC -m /lustre/fs3/portfolios/nvr/users/ymingli/experiments/mcmc_gt/${LOC}_Train${ITER}_DensFrm${DENSIFY_FROM}DensUntl${DENSIFY_UNTIL}_DensIntvl${DENSIFY_INTER}_NoiseScale${NOISE_SCALE}_cap${CAP_MAX}_GPU${GPU_NUM}_OpaReg${OPACITY_REG}_ScalReg${SCALE_REG}_PosLr${PS_LR}Skjul${ITER}_SclLR${SCALE_LR}_OpaRist${OPACITY_RESET}_Init${INIT_TYPE}_InitNum${RANDOM_INIT_NUM}

/lustre/fsw/portfolios/nvr/users/ymingli/rclone copy /lustre/fs3/portfolios/nvr/users/ymingli/experiments/mcmc_gt/${LOC}_Train${ITER}_DensFrm${DENSIFY_FROM}DensUntl${DENSIFY_UNTIL}_DensIntvl${DENSIFY_INTER}_NoiseScale${NOISE_SCALE}_cap${CAP_MAX}_GPU${GPU_NUM}_OpaReg${OPACITY_REG}_ScalReg${SCALE_REG}_PosLr${PS_LR}Skjul${ITER}_SclLR${SCALE_LR}_OpaRist${OPACITY_RESET}_Init${INIT_TYPE}_InitNum${RANDOM_INIT_NUM}/train/ours_399993/renders.mp4 gw2396:Visualization/LargeScale/LiDAR/MCMC/${LOC}_Train${ITER}_DensFrm${DENSIFY_FROM}DensUntl${DENSIFY_UNTIL}_DensIntvl${DENSIFY_INTER}_NoiseScale${NOISE_SCALE}_cap${CAP_MAX}_GPU${GPU_NUM}_OpaReg${OPACITY_REG}_ScalReg${SCALE_REG}_PosLr${PS_LR}Skjul${ITER}_SclLR${SCALE_LR}_OpaRist${OPACITY_RESET}_Init${INIT_TYPE}_InitNum${RANDOM_INIT_NUM} -P
