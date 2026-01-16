#!/bin/bash

#MODE="difix"
MODE=""

JOB_BASE_NAME="116skydebug"
LOG_BASE_PREFIX="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/log"

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

for i in {2..3}; do
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
        --command "bash /lustre/fsw/portfolios/nvr/users/ymingli/projects/citygs/code/gsplat/nv_train/small_sky.sh $MODE"
done

echo "--- Tasks have been submited! ---"