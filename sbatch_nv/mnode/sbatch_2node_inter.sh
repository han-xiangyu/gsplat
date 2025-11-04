job_name="2node_8gpulr_citygs"
logdir="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/log_mnode/2node_8gpulr_citygs"
gpus_per_node=8
nodes=2
account=nvr_av_foundations
submit_job \
    --email_mode never \
    --notification_mode never \
    --more_srun_args=--gpus-per-node=$gpus_per_node \
    --nodes $nodes \
    --tasks_per_node 1 \
    --partition=grizzly,polar --account=$account \
    --duration 1 \
    --logroot=$logdir \
    --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/2304py3.sqsh \
    --name=$job_name \
    --dependency=singleton \
    --command "bash /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/gsplat/sbatch_nv/mnode/2node_lr8.sh" \
    --just_alloc \
    --notimestamp

