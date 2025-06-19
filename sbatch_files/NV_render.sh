# NV cluster script

# source /home/ymingli/miniconda3/bin/activate
source /lustre/fs12/portfolios/nvr/users/ymingli/miniconda3/etc/profile.d/conda.sh
conda activate citygs

# Force to shield site-packages
export PYTHONNOUSERSITE=1
# Clear PYTHONPATH and  LD_LIBRARY_PATH
unset PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu

# Debug
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1 

# wandb configuration
export WANDB_API_KEY=9700db021b335e724b1c96fef3f087b458aff31e
export WANDB_MODE=disabled


cd /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/code/citygs


python render.py -s /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/data/long_video_processed_frames6000_pts_downsample  --model_path /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/models/long_video_frames6000_full_autoresume_distributed8GPU

python render_video.py

rclone copy /lustre/fsw/portfolios/nvr/users/ymingli/gaussian/models/long_video_frames6000_full_autoresume_distributed8GPU/train_set_video.mp4 xiangyuDrive:Research/CityGS/RenderVideos/long_video_partial6000_full.mp4 -P