
job_name="citygs_train"
training_logs_dir="/lustre/fsw/portfolios/nvr/users/ymingli/xiangyu/logs/citygs_train"

submit_job --gpu 1 --cpu 16 --nodes 1 --partition=grizzly,polar,polar3,polar4 --account=nvr_av_end2endav \
                --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/cu118.sqsh  \
                --container-mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
                --duration 4 \
                --dependency=singleton \
                --name $job_name \
                --logdir $training_logs_dir \
                --notimestamp \
                --command  “bash NV_run.sh”