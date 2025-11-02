#!/bin/bash

# 1. 运行 "difix" 模式 (使用 _with_newviews 路径):
#MODE="difix"

# 2. 运行标准模式 (使用原始路径):
MODE=""

JOB_BASE_NAME="tra2_13000to14000_3cam"
LOG_BASE_PREFIX="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/log_block_merge"

JOB_SUFFIX=""
if [ "$MODE" == "difix" ]; then
    echo "--- 'difix' 模式已启用 ---"
    JOB_SUFFIX="_difix"
else
    echo "--- 标准模式已启用 ---"
fi

# 2. 组合成最终的 Job Name 和 Log Dir
job_name="${JOB_BASE_NAME}${JOB_SUFFIX}"
base_logdir="${LOG_BASE_PREFIX}/${job_name}" # Log Dir 路径现在也包含了后缀

echo "--- 正在提交任务 ---"
echo "Job Name: $job_name"
echo "Log Dir:  $base_logdir"
echo "Mode Arg: [${MODE:-standard}]"

for i in {1..3}; do
    submit_job --gpu 8 --cpu 16 --nodes 1 \
        --partition=grizzly,polar,polar3,polar4 \
        --account=nvr_av_foundations \
        --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/2304py3.sqsh \
        --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
        --duration 4 \
        --dependency=singleton \
        --name ${job_name} \
        --logdir ${base_logdir}/run_${i} \
        --notimestamp \
        --exclusive \
        --command "bash /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/gsplat/sbatch_block/NV_train_b6.sh $MODE"
done

echo "--- 3个任务已提交 ---"