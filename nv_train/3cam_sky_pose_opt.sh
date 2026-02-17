#!/bin/bash
set -e

source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/bin/activate
conda activate mars_new
cd /lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city
BASE_DIR="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs"
SOURCE_PATH="${BASE_DIR}/data/may/arlington_small"
MODEL_PATH="${BASE_DIR}/models/arlington_small_216"

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
export OMP_NUM_THREADS=1
export PYTHONWARNINGS="ignore:The pynvml package is deprecated"

PROJECT_NAME=citygs_newdata
EXPERIENT_NAME="arlington_small_216"

max_steps=300000
means_lr=2e-3
noise_lr=5e2
MEAN_LR_FINAL_MULT=1e-2
densify_portion=0.005
depth_lambda=1e-1
depth_mode="disparity"
pose_opt_start=1
pose_opt_reg=1e-5
pose_opt_lr=1e-3
cap_max=400_000
ground_curriculum_steps=30000

torchrun --standalone \
     --nproc_per_node=8 \
     --nnodes=1 \
     examples/simple_trainer.py mcmc \
     --data_factor 1 --data_dir $SOURCE_PATH --result_dir $MODEL_PATH \
     --resume \
     --crop_hfov_deg 90 \
     --crop_hfov_camera_ids 4 5 6 \
     --crop_vfov_deg 75 \
     --resume_dir $MODEL_PATH \
     --wandb_project=$PROJECT_NAME \
     --wandb_group=gsplat \
     --wandb_name=$EXPERIENT_NAME \
     --wandb_mode='online' \
     --wandb_dir=$WANDB_DIR \
     --wandb_log_images_every=50000 \
     --means_lr $means_lr \
     --mean_lr_final_mult $MEAN_LR_FINAL_MULT \
     --max_steps $max_steps \
     --depth_loss --depth_mode $depth_mode --depth_lambda $depth_lambda --depth_from_external \
     --strategy.noise_lr $noise_lr \
     --strategy.cap-max $cap_max \
     --strategy.refine-start-iter 30000 \
     --strategy.refine-stop-iter 60000 \
     --strategy.refine-every 100 \
     --strategy.densify_portion $densify_portion \
     --sh_degree 1 \
     --eval_steps 5000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 225000 250000 275000 $max_steps \
     --save_steps $max_steps \
     --ply_steps  $max_steps \
     --dynamic_mask \
     --ground_curriculum_steps $ground_curriculum_steps \
     --use_sky \
     --ground_distort_3d_lambda 5.0 \

echo "Training finished. Starting rendering ..."
python examples/render.py \
    --data_dir $SOURCE_PATH \
    --crop_hfov_deg 90 \
    --crop_hfov_camera_ids 4 5 6 \
    --crop_vfov_deg 75 \
    --ply_path $MODEL_PATH/ply/point_cloud_149999.ply \
    --channels 2 1 3 5 4 6\
    --result_dir $MODEL_PATH/video \
    --fps 15 \
    --ckpt_path $MODEL_PATH/ckpts/ckpt_149999_rank0.pt \
    --output_name ${model_name}ckpt29999_final.mp4 \
    --use_sky\