job_name="gsplat_wholesquence_2node_16gpu"
base_logdir="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/log/gsplat_wholesquence_2node_16gpu"
account=nvr_av_foundations
gpus_per_node=8
nodes=2

submit_job --more_srun_args=--gpus-per-node=$gpus_per_node --nodes $nodes \
    --partition=grizzly,polar \
    --account=$account \
    --duration 4 \
    --exclusive \
    --logdir ${base_logdir} \
    --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/2304py3.sqsh \
    --name $job_name \
    --dependency=singleton \
    --notimestamp \
    --command "bash /lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/sbatch_nv/NV_train_wholesquence.sh"

