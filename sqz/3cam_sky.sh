#!/bin/bash

export PATH="/root/envs/mars_new/bin:$PATH"

cd /root/cbw/gsplat-city
DATE="26_total1k_front_cams"
BASE_DIR="/root/datasets/citygs"

SOURCE_PATH="${BASE_DIR}/data/colmap_keyframe_start2k_total1k_front_cams"
MODEL_PATH="${BASE_DIR}/models/${DATE}"

model_name=$(basename "$0" .sh)

export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

export PYTHONPATH=$PYTHONPATH:/root/cbw/gsplat-city
export PYTHONPATH=$PYTHONPATH:/root/cbw/gsplat-city/pycolmap
export CPLUS_INCLUDE_PATH=/root/cbw/glm:$CPLUS_INCLUDE_PATH

export WANDB_DIR="${BASE_DIR}/wandb_logs/${model_name}"
export WANDB_API_KEY=42e7b9b31273e3a7a2bc3527a0784472e70848a2
export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_SILENT=true

PROJECT_NAME=citygs_newdata
EXPERIENT_NAME="${DATE}"

max_steps=150_000
MEANS_LR=2e-3
MEAN_LR_FINAL_MULT=1e-3
densify_portion=0.001
depth_lambda=2e-3

export PYTHONWARNINGS="ignore:The pynvml package is deprecated"

torchrun --standalone \
     --nproc_per_node=2 \
     --nnodes=1 \
     examples/simple_trainer.py mcmc \
     --data_factor 1 --data_dir $SOURCE_PATH --result_dir $MODEL_PATH \
     --wandb_project=$PROJECT_NAME \
     --wandb_group=gsplat \
     --wandb_name=$EXPERIENT_NAME \
     --wandb_mode='disabled' \
     --wandb_dir=$WANDB_DIR \
     --wandb_log_images_every=50000 \
     --means_lr $MEANS_LR \
     --mean_lr_final_mult $MEAN_LR_FINAL_MULT \
     --max_steps $max_steps \
     --depth_loss \
     --depth_lambda $depth_lambda \
     --strategy.cap-max 4000000 \
     --strategy.refine-start-iter 9000 \
     --strategy.refine-stop-iter 50000 \
     --strategy.refine-every 100 \
     --strategy.densify_portion $densify_portion \
     --ground_curriculum_steps 10000

echo "Training finished. Starting rendering ..."

python examples/render_from_ply_sky.py \
    --data-dir $SOURCE_PATH \
    --ply-path $MODEL_PATH/ply/point_cloud_149999.ply \
    --ckpt-path $MODEL_PATH/ckpts/ckpt_149999_rank0.pt \
    --use-sky \
    --result-dir $MODEL_PATH \
    --fps 15 \
    --channels 2 1 3
