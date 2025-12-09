job_name="altlanta10k_start_13763"
logdir="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/log/altlanta10k_start_13763"
gpus_per_node=8
nodes=4
account=nvr_av_foundations
submit_job --more_srun_args=--gpus-per-node=$gpus_per_node --nodes $nodes \
    --partition=grizzly,polar \
    --account=$account \
    --duration 4 \
    --exclusive \
    --logroot=$logdir \
    --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/2304py3.sqsh \
    --name=$job_name \
    --dependency=singleton \
    --command "bash /lustre/fsw/portfolios/nvr/users/ymingli/projects/citygs/code/gsplat/nv_train/10k_origin.sh" \
    --notimestamp \
    --email_mode never \
    --notification_mode never