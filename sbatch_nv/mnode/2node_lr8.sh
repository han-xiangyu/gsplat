#!/bin/bash
set -e
source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/bin/activate
conda activate mars_pytorh3d
cd /lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat
SOURCE_PATH="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/data/tra2_0to1000keyframes_fps_3cam"
MODEL_PATH="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/models_mnode/tra2_0to1000keyframes_fps_3cam_2node"
model_name=$(basename "$MODEL_PATH")
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat
export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat/pycolmap
export WANDB_DIR="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/wandb_logs/${model_name}"
export WANDB_API_KEY=42e7b9b31273e3a7a2bc3527a0784472e70848a2
export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_SILENT=true

PROJECT_NAME=gsplat_mnode
EXPERIENT_NAME=$model_name
video_output_path="${MODEL_PATH}/videos/traj_199999.mp4"
remote_video_name="${model_name}_$(date +%m%d_%H%M)"
max_steps=40_000
MEANS_LR=2e-3
MEAN_LR_FINAL_MULT=1e-4
densify_portion=0.001
depth_lambda=2e-3
pose_opt_start=1e5
# --use_bilateral_grid

export PYTHONWARNINGS="ignore:The pynvml package is deprecated"
echo "Starting distributed training..."
torchrun --nproc_per_node=8 \
     --nnodes=${SLURM_NNODES} \
     --rdzv_id=$SLURM_JOB_ID \
     --rdzv_backend=c10d \
     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
     examples/simple_trainer_origin_knn.py mcmc  \
     --data_factor 1 --data_dir $SOURCE_PATH --result_dir $MODEL_PATH \
     --resume \
     --resume_dir $MODEL_PATH \
     --wandb_project=$PROJECT_NAME \
     --wandb_group=gsplat \
     --wandb_name=$EXPERIENT_NAME \
     --wandb_mode='online' \
     --wandb_dir=$WANDB_DIR \
     --wandb_log_images_every=50000 \
     --means_lr $MEANS_LR \
     --mean_lr_final_mult $MEAN_LR_FINAL_MULT \
     --max_steps $max_steps \
     --depth_loss \
     --depth_lambda $depth_lambda \
     --strategy.cap-max 3000000 \
     --strategy.refine-start-iter 9000 \
     --strategy.refine-stop-iter 50000 \
     --strategy.refine-every 100 \
     --strategy.schedule-mode='original' \
     --strategy.densify_portion $densify_portion \

# if [ "$RANK" -eq 0 ]; then
#     echo "Training finished. Starting rendering on rank 0..."
#     python examples/render_from_ply.py \
#           --data_dir $SOURCE_PATH \
#           --ply_path $MODEL_PATH/ply/point_cloud_199999.ply \
#           --result_dir $MODEL_PATH \
#           --fps 15 \
#           --channels 2 1 3
# fi

echo "Script finished on rank $RANK."