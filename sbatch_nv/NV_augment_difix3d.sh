# NV cluster script

cd /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/gsplat-city/submodules/gsplat/
source /lustre/fs12/portfolios/nvr/users/ymingli/miniconda3/etc/profile.d/conda.sh
conda activate gsplat

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu

# configs
NUM_CAMS=3
NUM_KEYFRAMES=1000
TRAVERSAL_ID=2
DOWNSAMPLE_TYPE="fps"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1 
new_traj_mode=sine
amplitude=1
BASE_PATH=/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs
SOURCE_PATH=${BASE_PATH}/data/tra${TRAVERSAL_ID}_${NUM_KEYFRAMES}keyframes_${DOWNSAMPLE_TYPE}_${NUM_CAMS}cam
MODEL_PATH=${BASE_PATH}/models/tra${TRAVERSAL_ID}_${NUM_KEYFRAMES}keyframes_${DOWNSAMPLE_TYPE}_${NUM_CAMS}cam

source_name=$(basename "$SOURCE_PATH")
model_name=$(basename "$MODEL_PATH")

# wandb configuration
export WANDB_DIR="$BASE_PATH/wandb_logs/${model_name}"
export WANDB_API_KEY=42e7b9b31273e3a7a2bc3527a0784472e70848a2
export WANDB_INSECURE_DISABLE_SSL=true
# export WANDB_MODE=disabled

PROJECT_NAME=gsplat_difix3d
EXPERIENT_NAME=$model_name
extrapolated_output_path="${MODEL_PATH}/extrapolated_renders/"

# Render new trajectory
CUDA_VISIBLE_DEVICES=0 python examples/render_extrapolated_from_ply.py \
  --data_dir $SOURCE_PATH \
  --ply_path $MODEL_PATH/ply/point_cloud_149999.ply  \
  --out_img_dir $extrapolated_output_path

# Difix3D repair
cd /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/Difix3D
conda activate difix3d

python batched_process_w_ref_dist_gsplat.py \
  --input_folder $MODEL_PATH/extrapolated_renders \
  --ref_folder $SOURCE_PATH/images \
  --output_folder $MODEL_PATH/extrapolated_difixed \
  --prompt "remove degradation"


# Register new views using GSplat
cd /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/gsplat-city/submodules/gsplat/
conda activate gsplat
python examples/register_new_views_gsplat.py \
  --data_dir $SOURCE_PATH \
  --output_sparse_dir_name new_sparse \
  --traj_type parallel \
  --amplitude 1.5

# Copy the original dataset to the new folder
NEW_SOURCE_PATH="${SOURCE_PATH}_with_newviews"
mkdir -p $NEW_SOURCE_PATH
rsync -av --progress $SOURCE_PATH/ $NEW_SOURCE_PATH/

# Rename the new sparse folder inside new dataset
mv $NEW_SOURCE_PATH/new_sparse/ $NEW_SOURCE_PATH/sparse/

# Copy the difixed images to the new dataset's image folder
cp $MODEL_PATH/extrapolated_difixed/* $NEW_SOURCE_PATH/images/