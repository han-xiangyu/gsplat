#!/bin/bash
set -e

source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/bin/activate
conda activate mars_new

cd /lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city

BASE_DIR="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs"
ACCOUNT="foundations"

SOURCE_PATH="${BASE_DIR}/data/may/atlanta_1202_start1k_keyframes5k_downsampled_ground_all_cam"
MODEL_PATH="${BASE_DIR}/models/atlanta_1202_start1k_keyframes5k_downsampled_ground_all_cam_214"

export CUDA_HOME=/usr/local/cuda-12.1
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.0"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

unset PYTHONPATH_CONDA
unset PYTHONNOUSERSITE

export PYTHONPATH=/lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city:/lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/pycolmap
export TORCH_EXTENSIONS_DIR=/tmp/${USER}/torch_extensions/${SLURM_JOB_ID}
mkdir -p "$TORCH_EXTENSIONS_DIR"

model_name=$(basename "$0" .sh)
export WANDB_DIR="${BASE_DIR}/wandb_logs/${model_name}"
export WANDB_API_KEY=42e7b9b31273e3a7a2bc3527a0784472e70848a2
export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_SILENT=true
# export TORCH_EXTENSIONS_ROOT=/lustre/fs12/portfolios/nvr/projects/nvr_av_${ACCOUNT}/users/ymingli/cache/torch_extensions_${SLURM_NODEID}
# mkdir -p "$TORCH_EXTENSIONS_ROOT"

PROJECT_NAME=citygs_newdata
EXPERIENT_NAME="atlanta_1202_start1k_keyframes5k_downsampled_ground_all_cam"

export OMP_NUM_THREADS=1
export PYTHONWARNINGS="ignore:The pynvml package is deprecated"

max_steps=250_000
MEANS_LR=2e-3
MEAN_LR_FINAL_MULT=1e-3
densify_portion=0.001
depth_lambda=2e-3
pose_opt_start=1e5

torchrun --standalone \
     --nproc_per_node=8 \
     --nnodes=1 \
     examples/simple_trainer.py mcmc \
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
     --strategy.cap-max 2500000 \
     --strategy.refine-start-iter 9000 \
     --strategy.refine-stop-iter 50000 \
     --strategy.refine-every 100 \
     --strategy.densify_portion $densify_portion \
     --ground_curriculum_steps 10000 \
     --use_sky \

echo "Training finished. Starting rendering ..."
python examples/render_from_ply_sky.py \
    --data-dir $SOURCE_PATH \
    --ply-path $MODEL_PATH/ply/point_cloud_149999.ply \
    --ckpt-path $MODEL_PATH/ckpts/ckpt_149999_rank0.pt \
    --use-sky \
    --result-dir $MODEL_PATH \
    --fps 15 \
    --channels 2 1 3 \