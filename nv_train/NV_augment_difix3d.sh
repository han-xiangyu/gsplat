# NV cluster script

cd /lustre/fsw/portfolios/nvr/users/ymingli/projects/citygs/code/gsplat
source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/bin/activate
conda activate gsplat

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export PYTHONHOME=""

export TORCH_LIB=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib
export LD_LIBRARY_PATH=$TORCH_LIB:/usr/local/cuda-11.8/lib64
export PYTHONPATH=/lustre/fsw/portfolios/nvr/users/ymingli/projects/citygs/code:$PYTHONPATH
export PYTHONPATH=/lustre/fsw/portfolios/nvr/users/ymingli/projects/citygs/code/gsplat/pycolmap:$PYTHONPATH
python -c "import torch; print('Using torch:', torch.__file__)"


export HF_ENDPOINT=https://hf-mirror.com
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1 

export TORCH_LIB=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib
export LD_LIBRARY_PATH=$TORCH_LIB:/usr/local/cuda-11.8/lib64
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH

export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
export FORCE_CUDA=1

rm -rf ~/.cache/torch_extensions/gsplat_cuda

DATE="131_3sky+ground"
new_traj_mode=sine
amplitude=1
CAM="front"
BASE_DIR=/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs
SOURCE_PATH="${BASE_DIR}/data/may/atlanta_1202_start1k_keyframes5k_downsampled_ground_${CAM}_cam"
MODEL_PATH="${BASE_DIR}/models/${DATE}"
extrapolated_output_path="${MODEL_PATH}/extrapolated_renders/"

# Render new trajectory
CUDA_VISIBLE_DEVICES=0 python examples/render_extrapolated_from_ply.py \
  --data_dir $SOURCE_PATH \
  --ply_path $MODEL_PATH/ply/point_cloud_149999.ply \
  --out_img_dir $extrapolated_output_path

# # Difix3D repair
# cd /lustre/fsw/portfolios/nvr/users/ymingli/projects/citygs/code/Difix3D
# conda activate difix3d

# python batched_process_w_ref_dist_gsplat.py \
#   --input_folder $MODEL_PATH/extrapolated_renders \
#   --ref_folder $SOURCE_PATH/images \
#   --output_folder $MODEL_PATH/extrapolated_difixed \
#   --prompt "remove degradation"


# # Register new views using GSplat
# cd /lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat/
# conda activate gsplat
# python examples/register_new_views_gsplat.py \
#   --data_dir $SOURCE_PATH \
#   --output_sparse_dir_name new_sparse \
#   --traj_type parallel \
#   --amplitude 1.5

# # Copy the original dataset to the new folder
# NEW_SOURCE_PATH="${SOURCE_PATH}_with_newviews"
# mkdir -p $NEW_SOURCE_PATH
# rsync -av --progress $SOURCE_PATH/ $NEW_SOURCE_PATH/

# # Rename the new sparse folder inside new dataset
# mv $NEW_SOURCE_PATH/new_sparse/ $NEW_SOURCE_PATH/sparse/

# # Copy the difixed images to the new dataset's image folder
# cp $MODEL_PATH/extrapolated_difixed/* $NEW_SOURCE_PATH/images/