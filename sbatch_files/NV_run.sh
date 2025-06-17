# NV cluster script

# source /home/ymingli/miniconda3/bin/activate
source /lustre/fs12/portfolios/nvr/users/ymingli/miniconda3/etc/profile.d/conda.sh
conda activate citygs

# Force to shield site-packages
export PYTHONNOUSERSITE=1
# Clear PYTHONPATH and  LD_LIBRARY_PATH
unset PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu

# Debug
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1 


GPU_NUM=8
CAP_MAX=2000000
NOISE_SCALE=500000
OPACITY_REG=0
SCALE_REG=0.01
ITER=100000
DENSIFY_FROM=500
DENSIFY_UNTIL=100000
DENSIFY_INTER=100
SCALE_LR=0.001
OPACITY_RESET=3000
POS_LR=2e-3
POS_LR_FINAL=2e-5
INIT_TYPE=sfm


PROJECT_NAME=CityGS_long_video
EXPERIENT_NAME=cap_max_8M_opacityREG0_scaleLR001_opacityLR005_posLR2e3_posLRfinal2e5_densifyFrom500Final100kIter100_frames12000_full_distributed8GPU

export WANDB_API_KEY=9700db021b335e724b1c96fef3f087b458aff31e
# export WANDB_MODE=disabled


cd /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/citygs


torchrun --standalone --nnodes=1 --nproc_per_node ${GPU_NUM} train.py --bsz ${GPU_NUM} \
            -s /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/data/long_video_processed_frames12000_pts_downsample \
            -m /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/models/long_video_frames12000_full_distributed8GPU \
            --iterations $ITER  \
            --densify_from_iter $DENSIFY_FROM \
            --densify_until_iter $DENSIFY_UNTIL \
            --mcmc --mcmc_noise_scale $NOISE_SCALE \
            --cap_max $CAP_MAX \
            --enable_timer --end2end_time --check_gpu_memory --check_cpu_memory --preload_dataset_to_gpu_threshold 0 \
            --opacity_reg $OPACITY_REG \
            --scale_reg $SCALE_REG \
            --densification_interval $DENSIFY_INTER \
            --position_lr_init $POS_LR \
            --position_lr_final $POS_LR_FINAL \
            --position_lr_max_steps $ITER \
            --scaling_lr $SCALE_LR \
            --opacity_reset_interval $OPACITY_RESET \
            --init_type $INIT_TYPE \
            --experiment_name  $EXPERIENT_NAME\
            --project_name $PROJECT_NAME \
            --auto_start_checkpoint
            # --resolution 4 \