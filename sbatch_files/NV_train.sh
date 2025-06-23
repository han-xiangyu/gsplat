# NV cluster script

cd /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/citygs

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


# Configuration
GPU_NUM=8
CAP_MAX=5000000
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


SOURCE_PATH=/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/data/spatial025_frames12000
MODEL_PATH=/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/model/spatial025_frames12000
source_name=$(basename "$SOURCE_PATH")
model_name=$(basename "$MODEL_PATH")

# wandb configuration
export WANDB_API_KEY=9700db021b335e724b1c96fef3f087b458aff31e
# export WANDB_MODE=disabled
PROJECT_NAME=CityGS_long_video
# EXPERIENT_NAME="${source_name}_cap${CAP_MAX}_opacityREG${OPACITY_REG}_scaleLR${SCALE_LR}_opacityLR005_posLR${POS_LR}_posLRfinal${POS_LR_FINAL}_densifyFrom${DENSIFY_FROM}Final${DENSIFY_UNTIL}Iter${ITER}"
EXPERIENT_NAME=$model_name
video_output_path="${MODEL_PATH}/${model_name}_train_set_video.mp4"
remote_video_name="${source_name}_$(date +%m%d_%H%M).mp4"

torchrun --standalone --nnodes=1 --nproc_per_node ${GPU_NUM} train.py --bsz ${GPU_NUM} \
            -s $SOURCE_PATH \
            -m $MODEL_PATH \
            --iterations $ITER  \
            --densify_from_iter $DENSIFY_FROM \
            --densify_until_iter $DENSIFY_UNTIL \
            --densification_interval $DENSIFY_INTER \
            --cap_max $CAP_MAX \
            --enable_timer --end2end_time --check_gpu_memory --check_cpu_memory --preload_dataset_to_gpu_threshold 0 \
            --opacity_reg $OPACITY_REG \
            --scale_reg $SCALE_REG \
            --position_lr_init $POS_LR \
            --position_lr_final $POS_LR_FINAL \
            --position_lr_max_steps $ITER \
            --scaling_lr $SCALE_LR \
            --opacity_reset_interval $OPACITY_RESET \
            --init_type $INIT_TYPE \
            --experiment_name  $EXPERIENT_NAME\
            --project_name $PROJECT_NAME \
            --auto_start_checkpoint \
            --mcmc --mcmc_noise_scale $NOISE_SCALE 
            # --backend gsplat \
            # --resolution 4 \


torchrun --nproc_per_node=${GPU_NUM} render.py --distributed_load -s $SOURCE_PATH  --model_path $MODEL_PATH

python render_video.py $MODEL_PATH --fps 15

rclone copy "${video_output_path}"  "xiangyuDrive:Research/CityGS/RenderVideos/${remote_video_name}" -P