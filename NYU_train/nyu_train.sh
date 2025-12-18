#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:h100:4
#SBATCH --time=6:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=CityGS
#SBATCH --output=/scratch/bc4211/slurm_logs/output_%x_%j.out
#SBATCH --error=/scratch/bc4211/slurm_logs/error_%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=bc4211@nyu.edu
#SBATCH --exclusive
#SBATCH --account=pr_116_tandon_advanced

singularity exec --nv --overlay /scratch/bc4211/envs/citygs.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
/bin/bash -c '

    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_USE_CUDA_DSA=1

    source /ext3/env.sh
    cd /scratch/bc4211/projects/gsplat-city/submodules/gsplat
    export TORCH_HOME=/scratch/bc4211/.cache/torch

    SOURCE_PATH="/scratch/bc4211/data/citygs/colmap_output"
    MODEL_PATH="/scratch/bc4211/models/small_sky_12181723"

    source_name=$(basename "$SOURCE_PATH")
    model_name=$(basename "$MODEL_PATH")

    # wandb configuration
    export WANDB_DIR="/scratch/bc4211/wandb_logs/${model_name}"
    export WANDB_API_KEY=42e7b9b31273e3a7a2bc3527a0784472e70848a2
    export WANDB_INSECURE_DISABLE_SSL=true
    # export WANDB_MODE=disabled

    PROJECT_NAME=gsplat_long_video
    EXPERIENT_NAME=$model_name
    video_output_path="${MODEL_PATH}/videos/traj_149999.mp4"
    remote_video_name="${model_name}_$(date +%m%d_%H%M)"

    max_steps=150_000
    MEANS_LR=2e-3
    MEAN_LR_FINAL_MULT=1e-3
    densify_portion=0.001
    depth_lambda=2e-3
    pose_opt_start=1e5

    CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/simple_trainer_sky.py mcmc  --data_factor 1 --data_dir $SOURCE_PATH --result_dir $MODEL_PATH \
        --wandb_project=$PROJECT_NAME \
        --wandb_group=gsplat \
        --wandb_name=$EXPERIENT_NAME \
        --wandb_mode='online' \
        --wandb_dir=$WANDB_DIR \
        --wandb_log_images_every=50000 \
        --means_lr $MEANS_LR \
        --mean_lr_final_mult $MEAN_LR_FINAL_MULT \
        --max_steps $max_steps \
        --depth_loss \
        --depth_lambda $depth_lambda \
        --strategy.cap-max 3000000 \
        --strategy.refine-start-iter 9000 \
        --strategy.refine-stop-iter 50000 \
        --strategy.refine-every 100 \
        --strategy.schedule-mode='original' \
        --strategy.densify_portion $densify_portion \

    CUDA_VISIBLE_DEVICES=0 python examples/render_from_ckpt_sky.py \
        --data_dir $SOURCE_PATH \
        --ply_path $MODEL_PATH/ply/point_cloud_149999.ply  \
        --result_dir $MODEL_PATH \
        --fps 15 \
        --channels 2 1 3
'
