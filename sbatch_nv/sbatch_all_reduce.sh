gpus_per_node=8
nodes=2
account=nvr_av_foundations

submit_job --more_srun_args=--gpus-per-node=$gpus_per_node --nodes $nodes \
    --partition=grizzly,polar \
    --account=$account \
    --duration 0.2 \
    --exclusive \
    --logroot=/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/log/allreduce \
    --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/2304py3.sqsh \
    --name=allreduce-2nodes \
    --command "/lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/sbatch_nv/all_reduce.sh" \
    --notimestamp