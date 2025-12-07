#!/bin/bash
source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/bin/activate
conda activate mars_pytorh3d
NUM_CAMS=3
TRAVERSAL_ID=2
DOWNSAMPLE_TYPE="fps"
S=13000
T1=14000
E=15000
BASE_DIR="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs"
PROJECT_DIR="/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/gsplat"
SOURCE1="${BASE_DIR}/data/tra${TRAVERSAL_ID}_${S}to${T1}keyframes_${DOWNSAMPLE_TYPE}_${NUM_CAMS}cam_with_newviews"
SOURCE2="${BASE_DIR}/data/tra${TRAVERSAL_ID}_${T1}to${E}keyframes_${DOWNSAMPLE_TYPE}_${NUM_CAMS}cam_with_newviews"
MODEL1="${BASE_DIR}/models_block/tra${TRAVERSAL_ID}_${S}to${T1}keyframes_${DOWNSAMPLE_TYPE}_${NUM_CAMS}cam_with_newviews"
MODEL2="${BASE_DIR}/models_block/tra${TRAVERSAL_ID}_${T1}to${E}keyframes_${DOWNSAMPLE_TYPE}_${NUM_CAMS}cam_with_newviews"

python $PROJECT_DIR/examples/analyze_gs_attributes.py \
    --ply_in $MODEL1/ply/point_cloud_149999.ply \
    --save_plots

python $PROJECT_DIR/examples/analyze_gs_attributes.py \
    --ply_in $MODEL2/ply/point_cloud_149999.ply \
    --save_plots

