job_name="tra2_1000keyframes_augment_difix3d"
base_logdir="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/log/tra2_1000keyframes_augment_difix3d"


submit_job --gpu 1 --cpu 16 --nodes 1 \
    --partition=grizzly,polar,polar3,polar4 \
    --account=nvr_av_end2endav \
    --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/2304py3.sqsh \
    --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
    --duration 4 \
    --dependency=singleton \
    --name ${job_name} \
    --logdir ${base_logdir} \
    --notimestamp \
    --exclusive \
    --command "bash /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/gsplat/sbatch_nv/NV_augment_difix3d.sh"
