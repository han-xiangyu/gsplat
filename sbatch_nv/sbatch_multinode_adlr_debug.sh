gpus_per_node=8
nodes=2
#account=nvr_av_end2endav
account=nvr_av_foundations
submit_job --more_srun_args=--gpus-per-node=$gpus_per_node --nodes $nodes \
    --partition=grizzly,polar \
    --account=$account \
    --duration 0.5 \
    --exclusive \
    --logroot=/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/log/citygs_3000frames_difix_2nodes \
    --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/2304py3.sqsh \
    --name=citygs-difix-2nodes-8gpus_per_node-nsys \
    --dependency=singleton \
    --command "bash /lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/sbatch_nv/NV_train_multinode_debug.sh" \
    --notimestamp