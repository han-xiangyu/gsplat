#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:a100:4
#SBATCH --time=12:00:00
#SBATCH --mem=64GB 
#SBATCH --job-name=CityGS
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --account=pr_116_tandon_priority
#SBATCH --mail-type=END
#SBATCH --mail-user=xh2967@nyu.edu


singularity exec --nv --overlay /home/xh2967/envs/citygs.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
/bin/bash -c "
    LOC=loc2450_1
    GPU_NUM=4
    CAP_MAX=10000000
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
    INIT_TYPE=sfm
    RANDOM_INIT_NUM=1000000

    cd /home/xh2967/code/citygs
    source /ext3/env.sh

    torchrun --standalone --nnodes=1 --nproc_per_node=${GPU_NUM} train.py \
        --bsz ${GPU_NUM} \
        -s /scratch/xh2967/data/ithaca_colmap_aligned/${LOC} \
        -m /scratch/xh2967/experiments/mcmc_colmap_aligned/${LOC}_ColmapInit_NoSpatialLrScale_Train${ITER}_DensifyFrom${DENSIFY_FROM}DensifyUntil${DENSIFY_UNTIL}_DensifyInterval${DENSIFY_INTER}_McmcNoiseScale${NOISE_SCALE}_capMax${CAP_MAX}_GPU${GPU_NUM}_OpacityReg${OPACITY_REG}_ScaleReg${SCALE_REG}_PosLr${POS_LR}Scheduler${ITER}_ScaleLR${SCALE_LR}_OpacityReset${OPACITY_RESET}_InitType${INIT_TYPE}_InitNum${RANDOM_INIT_NUM} \
        --iterations ${ITER} \
        --densify_from_iter ${DENSIFY_FROM} --densify_until_iter ${DENSIFY_UNTIL} --mcmc --mcmc_noise_scale ${NOISE_SCALE} --cap_max ${CAP_MAX} \
        --enable_timer --end2end_time --check_gpu_memory --check_cpu_memory \
        --preload_dataset_to_gpu_threshold 0 --opacity_reg ${OPACITY_REG} --scale_reg ${SCALE_REG} \
        --densification_interval ${DENSIFY_INTER} \
        --position_lr_init ${POS_LR} --position_lr_final ${POS_LR_FINAL} --position_lr_max_steps ${ITER} \
        --scaling_lr ${SCALE_LR} --opacity_reset_interval ${OPACITY_RESET} \
        --init_type ${INIT_TYPE} --init_num_pts ${RANDOM_INIT_NUM}

    python render.py \
        -s /scratch/xh2967/data/ithaca_colmap_aligned/${LOC} \
        -m /scratch/xh2967/experiments/mcmc_colmap_aligned/${LOC}_ColmapInit_NoSpatialLrScale_Train${ITER}_DensifyFrom${DENSIFY_FROM}DensifyUntil${DENSIFY_UNTIL}_DensifyInterval${DENSIFY_INTER}_McmcNoiseScale${NOISE_SCALE}_capMax${CAP_MAX}_GPU${GPU_NUM}_OpacityReg${OPACITY_REG}_ScaleReg${SCALE_REG}_PosLr${POS_LR}Scheduler${ITER}_ScaleLR${SCALE_LR}_OpacityReset${OPACITY_RESET}_InitType${INIT_TYPE}_InitNum${RANDOM_INIT_NUM}
"