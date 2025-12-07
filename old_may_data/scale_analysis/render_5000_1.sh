#!/bin/bash
source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/bin/activate
conda activate mars_new
cd /lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat

NUM_CAMS=3
TRAVERSAL_ID=2
DOWNSAMPLE_TYPE="fps"
S=21
E=5021
BASE_DIR="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs"

BASE_PATH_NAME="tra${TRAVERSAL_ID}_${S}to${E}keyframes_${DOWNSAMPLE_TYPE}_${NUM_CAMS}cam"
PATH_SUFFIX=""
if [ "$1" == "difix" ]; then
    echo "--- 'difix' is on ---"
    echo "--- use '_with_newviews' path ---"
    PATH_SUFFIX="_with_newviews"
else
    echo "--- 'difix' is off ---"
    echo "--- will use standard path ---"
fi

SOURCE_PATH="${BASE_DIR}/data_scale/${BASE_PATH_NAME}${PATH_SUFFIX}"
MODEL_PATH="${BASE_DIR}/models_scale/${BASE_PATH_NAME}${PATH_SUFFIX}_densify_portion0.03_capmax_300w_old"

model_name=$(basename "$0" .sh)
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat
export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat/pycolmap
export WANDB_DIR="${BASE_DIR}/wandb_logs/${model_name}"
export WANDB_API_KEY=42e7b9b31273e3a7a2bc3527a0784472e70848a2
export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_SILENT=true
export TORCH_EXTENSIONS_ROOT=/lustre/fs12/portfolios/nvr/projects/nvr_av_end2endav/users/ymingli/cache/torch_extensions_${SLURM_NODEID}
mkdir -p "$TORCH_EXTENSIONS_ROOT"

PROJECT_NAME=gsplat_scale_resource_analysis
EXPERIENT_NAME="${S}to${E}_${SLURM_NNODES}node${PATH_SUFFIX}_lidar_capmax300w"
max_steps=150_000
MEANS_LR=2e-3
MEAN_LR_FINAL_MULT=1e-3
densify_portion=0.01
depth_lambda=2e-3
pose_opt_start=1e5
export PYTHONWARNINGS="ignore:The pynvml package is deprecated"

echo "Training finished. Starting rendering ..."
python examples/render_from_ply.py \
     --data_dir $SOURCE_PATH \
     --ply_path $MODEL_PATH/ply/point_cloud_139999.ply  \
     --result_dir $MODEL_PATH \
     --fps 15 \
     --channels 2 1 3 \
     --start $S \
     --end $E