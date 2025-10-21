job_name="tra2_3000to6000keyframes_fps_3cam"
base_logdir="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/log/tra2_3000to6000keyframes_fps_3cam"

for i in {1..5}; do
    submit_job --gpu 8 --cpu 16 --nodes 1 \
        --partition=grizzly,polar,polar3,polar4 \
        --account=nvr_av_end2endav \
        --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/2304py3.sqsh \
        --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
        --duration 4 \
        --dependency=singleton \
        --name ${job_name} \
        --logdir ${base_logdir}/run_${i} \
        --notimestamp \
        --exclusive \
        --command "bash /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/gsplat/sbatch_nv/NV_train_b2.sh"
done
