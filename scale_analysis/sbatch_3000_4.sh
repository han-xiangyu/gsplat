#!/bin/bash

#MODE="difix"
MODE=""

JOB_BASE_NAME="tra2_0to3000_4node"
LOG_BASE_PREFIX="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/log_block"
gpus_per_node=8
nodes=4

JOB_SUFFIX=""
if [ "$MODE" == "difix" ]; then
    echo "--- 'difix' is on ---"
    JOB_SUFFIX="_difix"
else
    echo "--- 'standard' is on ---"
fi

job_name="${JOB_BASE_NAME}${JOB_SUFFIX}"
base_logdir="${LOG_BASE_PREFIX}/${job_name}"

echo "--- submit job ---"
echo "Job Name: $job_name"
echo "Log Dir:  $base_logdir"
echo "Mode Arg: [${MODE:-standard}]"

for i in {1..4}; do
    submit_job --more_srun_args=--gpus-per-node=$gpus_per_node --nodes $nodes \
        --partition=grizzly,polar,polar3,polar4 \
        --account=nvr_av_end2endav \
        --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/2304py3.sqsh \
        --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
        --duration 4 \
        --dependency=singleton \
        --name ${job_name} \
        --logdir ${base_logdir}/run_${i} \
        --notimestamp \
        --exclusive \
        --command "bash /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/gsplat/scale_analysis/3000_4.sh $MODE" \
        --email_mode never \
        --notification_mode never
done
echo "--- 4个任务已提交 ---"