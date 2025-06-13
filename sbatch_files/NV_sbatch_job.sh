
job_name="3DfoundationModel"
training_logs_dir="/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/logs/citygs_partial_12000"

submit_job --gpu 8 --cpu 24 --nodes 1 --partition=grizzly,polar,polar3,polar4 --account=nvr_av_end2endav \
                --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/cu118.sqsh  \
                --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
                --duration 4 \
                --dependency=singleton \
                --name $job_name \
                --logdir $training_logs_dir \
                --notimestamp \
                --command  "bash /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/citygs/sbatch_files/NV_run.sh"