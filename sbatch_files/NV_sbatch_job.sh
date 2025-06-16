
job_name="3DfoundationModel"
base_logdir="/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/logs/citygs_partial_6000_dspl_4_autoresume"


for i in {1..2}; do
    submit_job --gpu 1 --cpu 24 --nodes 1 --partition=grizzly,polar,polar3,polar4 --account=nvr_av_end2endav \
                    --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/cu118.sqsh  \
                    --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
                    --duration 4 \
                    --dependency=singleton \
                    --name $job_name \
                    --logdir ${base_logdir}/run_${i} \
                    --notimestamp \
                    --exclusive \
                    --command  "bash /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/citygs/sbatch_files/NV_run.sh"
done

# --copy_user_code /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/citygs:code:[*] \