#!/bin/bash
source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/bin/activate
conda activate mars_pytorh3d
cd /lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat

# --- 1. 配置模型和数据路径 (从你的训练脚本中复制) ---
NUM_CAMS=3
NUM_KEYFRAMES=3000
TRAVERSAL_ID=2
DOWNSAMPLE_TYPE="fps"

BASE_DIR="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs"
SOURCE_PATH="${BASE_DIR}/data/tra${TRAVERSAL_ID}_3000keyframes_${DOWNSAMPLE_TYPE}_${NUM_CAMS}cam"
MODEL_PATH="${BASE_DIR}/models/tra${TRAVERSAL_ID}_3000keyframes_${DOWNSAMPLE_TYPE}_${NUM_CAMS}cam_dmask"

# --- 2. 设置必要的环境变量 ---
export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat
export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat/pycolmap
export PYTHONWARNINGS="ignore:The pynvml package is deprecated"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# --- 3. 定义你想要评估的检查点步骤 ---
# (根据你的渲染脚本，你训练的最后一步是 149999)
CKPT_STEP=119999

# --- 4. 自动生成 8 个 DDP 检查点文件的路径列表 ---
CKPT_PATHS=""
for rank in {0..7}; do
  CKPT_PATHS+=" ${MODEL_PATH}/ckpts/ckpt_${CKPT_STEP}_rank${rank}.pt"
done

echo "Starting evaluation for ${MODEL_PATH}"
echo "Loading checkpoints: ${CKPT_PATHS}"

# --- 5. 运行评估命令 ---
# 我们仍然使用 torchrun 启动 8 个
# 进程来正确加载 8 个分片的模型检查点
torchrun --standalone \
     --nproc_per_node=8 \
     --nnodes=1 \
     examples/simple_trainer_origin_knn_dmask.py mcmc \
     --data_factor 1 --data_dir $SOURCE_PATH --result_dir $MODEL_PATH \
     --depth_loss \
     --depth_lambda 2e-3 \
     --strategy.cap-max 10000000 \
     --strategy.refine-start-iter 9000 \
     --strategy.refine-stop-iter 50000 \
     --strategy.refine-every 100 \
     --strategy.schedule-mode='original' \
     --strategy.densify_portion 0.001 \
     --ckpt $CKPT_PATHS

echo "Evaluation finished."