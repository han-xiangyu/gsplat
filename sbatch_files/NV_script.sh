# NV cluster script

LOC=loc2450_1

GPU_NUM=1
CAP_MAX=2000000
NOISE_SCALE=500000
OPACITY_REG=0
SCALE_REG=0.01
ITER=400000
DENSIFY_FROM=500
DENSIFY_UNTIL=100000
DENSIFY_INTER=100
SCALE_LR=0.001
OPACITY_RESET=3000
POS_LR=2e-3
POS_LR_FINAL=2e-5
INIT_TYPE=sfm
RANDOM_INIT_NUM=1000000

export WANDB_API_KEY=9700db021b335e724b1c96fef3f087b458aff31e

torchrun --standalone --nnodes=1 --nproc_per_node ${GPU_NUM} train.py --bsz ${GPU_NUM} \
            -s /lustre/fsw/portfolios/nvr/users/ymingli/xiangyu/data/long_video_processed \
            -m /lustre/fsw/portfolios/nvr/users/ymingli/xiangyu/data/long_video_gs_model  \
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
            --init_num_pts $RANDOM_INIT_NUM \
            --experiment_name cap_max_8M_opacityREG0_scaleLR001_opacityLR005_posLR2e3_posLRfinal2e5_densifyFrom500Final100kIter100_masked \
            --project_name Grendel_MCMC_long_video 