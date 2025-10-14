#!/bin/bash
#SBATCH --job-name=multinode-citygs-3000frames-voxel0.2
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=32
#SBATCH --partition=grizzly,polar,polar3,polar4

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
PORT=$((RANDOM % 49152 + 10000))
echo "Node IP: $head_node_ip"
echo "Node Port: $PORT"

job_name="citygs_3000frames_original_step1.0_2nodes"
base_logdir="/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/log/citygs_3000frames_original_step1.0_2nodes"
code_dir="/lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city"

#account=nvr_av_foundations
 account=nvr_av_end2endav
# account=nvr_av_verifvalid

# Multi nodes Train
for i in {1..4}; do
    submit_job \
    --email_mode never \
    --notification_mode never \
    --nodes 2 \
    --mplaunch \
    --tasks_per_node 1 \
    --gpu 8 \
    --cpu 32 \
    --mem 50 \
    --partition=grizzly,polar,polar3,polar4 --account=$account \
    --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/2304py3.sqsh  \
    --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3 \
    --duration 4 \
    --dependency=singleton \
    --name $job_name \
    --logdir ${base_logdir}/run_${i} \
    --copy_user_code ${code_dir}:code:[*] \
    --command  "bash /lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/sbatch_nv/NV_train_difix0.5.sh"
    #--notimestamp \
    # --exclusive \
done
