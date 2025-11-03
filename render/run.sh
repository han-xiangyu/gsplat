job_name="rendercitygs"
base_logdir="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/log/rendercitygs"


submit_job --gpu 1 --cpu 16 --nodes 1 \
    --partition=grizzly,polar,polar3,polar4 \
    --account=nvr_av_foundations \
    --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/2304py3.sqsh \
    --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
    --duration 4 \
    --dependency=singleton \
    --name ${job_name} \
    --logdir ${base_logdir} \
    --notimestamp \
    --exclusive \
    --command "bash /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/gsplat/render/NV_only_render.sh"
