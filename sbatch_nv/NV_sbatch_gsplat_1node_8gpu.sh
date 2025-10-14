job_name="gsplat_1node_8gpu_6cam_voxel"
base_logdir="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/log/gsplat_1node_8gpu_6cam_voxel"
account=nvr_av_foundations

submit_job --gpu 8 --cpu 16 --nodes 1 \
    --partition=grizzly,polar,polar3,polar4 \
    --account=$account \
    --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/2304py3.sqsh \
    --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
    --duration 4 \
    --dependency=singleton \
    --name $job_name \
    --logdir ${base_logdir} \
    --exclusive \
    --command "bash /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/gsplat/sbatch_nv/NV_train_origin_1node_8gpu.sh"

