#!/bin/bash
source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/bin/activate
conda activate mars_pytorh3d
cd /lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat
SOURCE_PATH="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/data/tra2_3000keyframes_fps_3cam"
MODEL_PATH="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/models/tra2_3000keyframes_fps_3cam_dynamicmask"
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
     --ply_path $MODEL_PATH/ply/point_cloud_149999.ply  \
     --result_dir $MODEL_PATH \
     --fps 15 \
     --channels 2 1 3

