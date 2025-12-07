#!/bin/bash
source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/bin/activate
conda activate mars_pytorh3d
cd /lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat
NUM_CAMS=3
TRAVERSAL_ID=2
DOWNSAMPLE_TYPE="fps"
S=2000
E=4200
BASE_DIR="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs"
SOURCE_PATH="${BASE_DIR}/data/tra${TRAVERSAL_ID}_${S}to${E}keyframes_${DOWNSAMPLE_TYPE}_${NUM_CAMS}cam"
MODEL_PATH="${BASE_DIR}/models/tra${TRAVERSAL_ID}_${S}to${E}keyframes_${DOWNSAMPLE_TYPE}_${NUM_CAMS}cam"
model_name=$(basename "$MODEL_PATH")
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat
export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat/pycolmap

pose_opt_start=1e5
# --use_bilateral_grid
export PYTHONWARNINGS="ignore:The pynvml package is deprecated"

echo "Training finished. Starting rendering ..."
python examples/render_from_ply.py \
     --data_dir $SOURCE_PATH \
     --ply_path $MODEL_PATH/ply/point_cloud_99999.ply  \
     --result_dir $MODEL_PATH \
     --fps 15 \
     --channels 2 1 3 \
     --start $S \
     --end $E
