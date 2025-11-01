#!/bin/bash
source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/bin/activate
conda activate mars_pytorh3d
cd /lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat
NUM_CAMS=3
TRAVERSAL_ID=2
DOWNSAMPLE_TYPE="fps"
S=13000
T1=14000
E=15000
BASE_DIR="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs"

SOURCE1="${BASE_DIR}/data/tra${TRAVERSAL_ID}_${S}to${T1}keyframes_${DOWNSAMPLE_TYPE}_${NUM_CAMS}cam"
SOURCE2="${BASE_DIR}/data/tra${TRAVERSAL_ID}_${T1}to${E}keyframes_${DOWNSAMPLE_TYPE}_${NUM_CAMS}cam"

MODEL1="${BASE_DIR}/models/tra${TRAVERSAL_ID}_${S}to${T1}keyframes_${DOWNSAMPLE_TYPE}_${NUM_CAMS}cam_with_newviews"
MODEL2="${BASE_DIR}/models/tra${TRAVERSAL_ID}_${T1}to${E}keyframes_${DOWNSAMPLE_TYPE}_${NUM_CAMS}cam_with_newviews"

MERGE_DIR="${BASE_DIR}/merge_models/tra${TRAVERSAL_ID}_${S}to${E}keyframes"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat
export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat/pycolmap

pose_opt_start=1e5
# --use_bilateral_grid
export PYTHONWARNINGS="ignore:The pynvml package is deprecated"

echo "Training finished. Starting rendering ..."
python examples/render_from_merge_ply.py \
     --data-dirs $SOURCE1 $SOURCE2 \
     --ply_path $MODEL1/ply/point_cloud_149999.ply $MODEL2/ply/point_cloud_149999.ply \
     --result_dir $MERGE_DIR \
     --fps 15 \
     --channels 2 1 3 \
     --start $S \
     --end $E
