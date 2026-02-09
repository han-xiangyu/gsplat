# NV cluster script

export PATH="/root/envs/mars_new/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

export PYTHONPATH=$PYTHONPATH:/root/cbw/gsplat-city
export PYTHONPATH=$PYTHONPATH:/root/cbw/gsplat-city/pycolmap
export CPLUS_INCLUDE_PATH=/root/cbw/glm:$CPLUS_INCLUDE_PATH

#rm -rf ~/.cache/torch_extensions/gsplat_cuda

DATE="26_total1k_front_cams"
new_traj_mode=sine
amplitude=1
BASE_DIR="/root/datasets/citygs"
SOURCE_PATH="${BASE_DIR}/data/colmap_keyframe_start2k_total1k_front_cams"
MODEL_PATH="${BASE_DIR}/models/${DATE}"
extrapolated_output_path="${MODEL_PATH}/extrapolated_renders/"
cd /root/cbw/gsplat-city
# Render new trajectory
CUDA_VISIBLE_DEVICES=0 python examples/render_extrapolated_from_ply.py \
  --data_dir $SOURCE_PATH \
  --ply_path $MODEL_PATH/ply/point_cloud_149999.ply \
  --out_img_dir $extrapolated_output_path

# Difix3D repair
# ====== Difix3D repair (switch env) ======
echo ">>> Switch to difix3d env"
cd /root/cbw/Difix3D
/root/miniconda3/envs/difix3d/bin/python batched_process_w_ref_dist_gsplat.py \
  --input_folder $MODEL_PATH/extrapolated_renders \
  --ref_folder $SOURCE_PATH/images \
  --output_folder $MODEL_PATH/extrapolated_difixed \
  --prompt "remove degradation"

# Register new views using GSplat

cd /root/cbw/gsplat-city
/root/miniconda3/envs/mars_new/bin/python examples/register_new_views.py \
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