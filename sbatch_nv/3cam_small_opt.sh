#!/bin/bash

JOB_BASE_NAME="yolo_exp"
LOG_BASE_PREFIX="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/log"
job_name="${JOB_BASE_NAME}"
base_logdir="${LOG_BASE_PREFIX}/${job_name}"

echo "--- submit job ---"
echo "Job Name: $job_name"
echo "Log Dir:  $base_logdir"

for i in {1..3}; do
    submit_job --gpu 8 --cpu 16 --nodes 1 \
        --partition=grizzly,polar,polar3,polar4 \
        --account=nvr_av_foundations \
        --email_mode never \
        --notification_mode never \
        --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/2304py3.sqsh \
        --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
        --duration 4 \
        --dependency=singleton \
        --name ${job_name} \
        --logdir ${base_logdir}/run_${i} \
        --notimestamp \
        --exclusive \
        --command "bash /lustre/fsw/portfolios/nvr/users/ymingli/projects/citygs/code/gsplat/nv_train/3cam_sky_pose_opt.sh"
done

echo "--- Tasks have been submited! ---"