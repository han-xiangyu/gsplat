import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap
import wandb
import torch.distributed as dist


from difix3d.pipeline_difix import DifixPipeline
from examples.utils import CameraPoseInterpolator
import random
from PIL import Image
from copy import deepcopy

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = True
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    resume: bool = True                         # 是否继续训练
    resume_ckpt: Optional[str] = None         # ckpt 路径；多卡可用模板，如 ".../ckpt_100000_rank{rank}.pt"
    resume_dir: Optional[str] = None          # 或者给目录，自动找当前 rank 最新的 ckpt
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "original"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 1
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 60
    use_val: bool = True
    eval_max_images: Optional[int] = 64   # 每次评估最多评多少张（None=全量）
    eval_stride: int = 50                  # 间隔抽样
    eval_save_images: bool = True         # 评估时是否落盘示例图
    eval_save_images_n: int = 2           # 最多保存多少张示例图
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = False
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 200_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 25_000, 50_000, 75_000, 100_000, 125_000, 150_000, 175_000, 200_000, 250_000, 300_000, 350_000, 400_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [100, 47_000, 94_000, 141_000, 188_000, 200_000, 300_000, 400_000])
    # # Steps to fix the artifacts
    # fix_steps: List[int] = field(default_factory=lambda: [300_000])
    # fix_mode: str = "extrapolate"
    # fix_distance: float = 1
    # # Weight for iterative 3d update
    # novel_data_lambda: float = 0.3
    # render_format: str = "jpg"   # png or jpg
    # render_batch: int = 4
    # fix_downsample_stride: int = 2


    # Whether to save ply file (storage size can be large)
    save_ply: bool = True
    # Steps to save the model as ply
    # ply_steps: List[int] = field(default_factory=lambda: [100_000, 200_000, 300_000, 400_000])
    ply_steps: List[int] = field(default_factory=lambda: [46_000, 92_000, 138_000, 184_000, 200_000])
    # Steps to save the model as ply
    video_render_steps: List[int] = field(default_factory=lambda: [])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 5000
    # Initial opacity of GS
    init_opa: float = 0.8
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2


    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1000

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = True
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = True

    # LR for 3D point positions
    means_lr: float = 2e-3
    # Final LR for 3D point positions as a multiple of the initial LR
    mean_lr_final_mult: float = 0.01
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-3
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.001
    # Scale regularization
    scale_reg: float = 0.01

    # Enable camera optimization.
    pose_opt: bool = False
    pose_opt_start: int = 100000
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = True
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False

    wandb_project: Optional[str] = None
    wandb_group: Optional[str] = None 
    wandb_name: Optional[str] = None 
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"
    wandb_dir: Optional[str] = None 
    wandb_log_images_every: int = 50000 

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.video_render_steps = [int(i * factor) for i in self.video_render_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val") if cfg.use_val else None
        # self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        self.scene_scale = cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        self.wb = None
        if self.world_rank == 0 and self.cfg.wandb_project:
            wandb_dir = self.cfg.wandb_dir or self.cfg.result_dir
            os.makedirs(wandb_dir, exist_ok=True)
            run_name = (
                self.cfg.wandb_name
                or f"{Path(self.cfg.data_dir).name}_{self.cfg.strategy.__class__.__name__}_bs{self.cfg.batch_size}"
            )
            self.wb = wandb.init(
                project=self.cfg.wandb_project,
                group=self.cfg.wandb_group,
                name=run_name,
                dir=wandb_dir,
                mode=self.cfg.wandb_mode,
                config=vars(self.cfg),
                resume="allow",
            )

                    
        # # Fixer trajectory 
        # self.interpolator = CameraPoseInterpolator(rotation_weight=1.0, translation_weight=1.0)

        # self.base_novel_poses = self.parser.camtoworlds[self.trainset.indices]
        # self.current_novel_poses = self.base_novel_poses
        # self.current_parser = self.parser

        # self.novelloaders = []
        # self.novelloaders_iter = []

        # # ---- Difix: load once per-rank ----
        # self.difix = DifixPipeline.from_pretrained(
        #     "nvidia/difix_ref", trust_remote_code=True
        # )
        # self.difix.set_progress_bar_config(disable=True)
        # self.difix.to(self.device if torch.cuda.is_available() else "cpu")

        


    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        distributed: Optional[bool] = True,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:

        if distributed is False:
            distributed_flag = distributed 
        else:
            distributed_flag = self.world_size > 1

        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=distributed_flag,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        if cfg.resume: 
            resume_path = None 
            if cfg.resume_ckpt: 
                resume_path = cfg.resume_ckpt.format(rank=self.world_rank) 
            elif cfg.resume_dir: # 自动在目录中查找最新的 checkpoint 
                ckpt_dir = Path(cfg.resume_dir) / "ckpts" 
                if ckpt_dir.exists(): # 查找格式为 ckpt_{step}_rank{rank}.pt 的文件 
                    pattern = f"ckpt_*_rank{self.world_rank}.pt" 
                    ckpts = sorted(ckpt_dir.glob(pattern), 
                                   key=lambda f: int(f.stem.split('_')[1]) # 按 step 排序 
                                   ) 
                    if ckpts: 
                        resume_path = ckpts[-1] # 取最新的一个

            if resume_path and os.path.exists(resume_path): 
                print(f"***** Resuming training from checkpoint: {resume_path} *****") 
                checkpoint = torch.load(resume_path, map_location=self.device) 
                # 恢复模型 (splats) 
                self.splats.load_state_dict(checkpoint['splats']) # 恢复优化器状态 (非常重要！) 
                # 注意：这里假设 checkpoint 保存了所有优化器的状态 
                # # 你的代码已经保存了 splats, 但没有保存 optimizers, 需要在保存时也加上 
                # # 为了向下兼容，我们先假设没有保存优化器，但强烈建议加上 
                if 'optimizers' in checkpoint: 
                    for name, opt in self.optimizers.items(): 
                        if name in checkpoint['optimizers']: 
                            opt.load_state_dict(checkpoint['optimizers'][name]) 
                # # 恢复相机姿态优化器 
                if cfg.pose_opt and 'pose_adjust' in checkpoint: 
                    pose_adjust_state = checkpoint['pose_adjust'] 
                    if world_size > 1: 
                        self.pose_adjust.module.load_state_dict(pose_adjust_state) 
                    else: 
                        self.pose_adjust.load_state_dict(pose_adjust_state) 
                # 恢复 appearance 优化器 
                if cfg.app_opt and 'app_module' in checkpoint: 
                    app_module_state = checkpoint['app_module'] 
                    if world_size > 1: 
                        self.app_module.module.load_state_dict(app_module_state) 
                    else: self.app_module.load_state_dict(app_module_state) 
                    
                # 更新起始步数 
                init_step = checkpoint['step'] + 1 
                print(f"***** Resumed from step {checkpoint['step']}. Starting next step at {init_step} *****") 
            else: 
                print("***** Resume enabled, but no checkpoint found. Starting from scratch. *****")


        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=cfg.mean_lr_final_mult ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )
        if init_step > 0: 
            print(f"Fast-forwarding schedulers to step {init_step}...") 
            # 手动将 scheduler 快进到正确的 step 
            for _ in range(init_step): 
                for scheduler in schedulers: 
                    scheduler.step()

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            if cfg.depth_loss and ("depths" in data):
                # points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt and step >= cfg.pose_opt_start:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss and ("depths" in data):

                #### Modified: Sample depth map at all valid pixels and use L1 loss
                depth_gt = data["depths"].to(device)[None, ..., None]  # [1, H, W, 1]
                valid = torch.isfinite(depth_gt) & (depth_gt > 0)
                valid = valid & (depths > 0) 
                # masked L1
                depthloss = torch.abs(depths - depth_gt)[valid].mean() * self.scene_scale
                loss += depthloss * cfg.depth_lambda


                #### Oringinal: Sample depth map at query points and use disparity loss
                # # query depths from depth map
                # points = torch.stack(
                #     [
                #         points[:, :, 0] / (width - 1) * 2 - 1,
                #         points[:, :, 1] / (height - 1) * 2 - 1,
                #     ],
                #     dim=-1,
                # )  # normalize to [-1, 1]
                # grid = points.unsqueeze(2)  # [1, M, 1, 2]
                # depths = F.grid_sample(
                #     depths.permute(0, 3, 1, 2), grid, align_corners=True
                # )  # [1, 1, M, 1]
                # depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # # calculate loss in disparity space
                # disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                # disp_gt = 1.0 / depths_gt  # [1, M]
                # depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                # loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )
            # if is_novel_data:
            #     loss = loss * cfg.novel_data_lambda
            # else:
            #     loss = loss * 1.5
            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss and ("depths" in data):
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                # self.writer.add_scalar("train/loss", loss.item(), step)
                # self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                # self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                # self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                # self.writer.add_scalar("train/mem", mem, step)
                # if cfg.depth_loss:
                #     self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                # if cfg.use_bilateral_grid:
                #     self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                # if cfg.tb_save_image:
                #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                #     canvas = canvas.reshape(-1, *canvas.shape[2:])
                #     self.writer.add_image("train/render", canvas, step)
                # self.writer.flush()

                # ---- W&B logging (rank0 only) ----
                if self.wb is not None:
                    logs = {
                        "train/loss": float(loss.item()),
                        "train/l1": float(l1loss.item()),
                        "train/ssimloss": float(ssimloss.item()),
                        "train/num_GS": int(len(self.splats["means"])),
                        "train/mem_gb": float(mem),
                        "train/sh_degree": int(sh_degree_to_use),
                    }
                    # depth loss
                    if cfg.depth_loss:
                        logs["train/depthloss"] = float(depthloss.item())
                    # means lr
                    if len(schedulers) > 0:
                        logs["lr/means"] = float(schedulers[0].get_last_lr()[0])


                    opa = torch.sigmoid(self.splats["opacities"])
                    sc  = torch.exp(self.splats["scales"])
                    vis = (info["radii"] > 0).all(-1).any(0).float().mean()

                    logs.update({
                    "stats/alpha_mean": float(opa.mean()),
                    "stats/scale_med":  float(sc.median()),
                    "stats/vis_frac":   float(vis)
                    })

                    # Rendering Images
                    if cfg.wandb_log_images_every > 0 and step % cfg.wandb_log_images_every == 0:
                        canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                        canvas = (canvas.reshape(-1, *canvas.shape[2:]) * 255).astype(np.uint8)
                        logs["train/render"] = wandb.Image(canvas, caption=f"step {step}")

                    self.wb.log(logs, step=step)


            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, 
                        "splats": self.splats.state_dict(),
                        #"optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()}
                }
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:

                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]


                @torch.no_grad()
                def gather_for_export(t: torch.Tensor) -> Optional[torch.Tensor]:
                    if not dist.is_initialized():
                        return t.detach().cpu()
                    # 1) 收集各 rank 的长度
                    local = t.detach()
                    n_local = torch.tensor([local.shape[0]], device=local.device, dtype=torch.long)
                    sizes = [torch.zeros_like(n_local) for _ in range(dist.get_world_size())]
                    dist.all_gather(sizes, n_local)
                    sizes = [int(s.item()) for s in sizes]
                    maxN = max(sizes)

                    # 2) pad 到同长后 all_gather
                    if local.shape[0] < maxN:
                        pad = torch.zeros((maxN - local.shape[0], *local.shape[1:]),
                                        device=local.device, dtype=local.dtype)
                        local = torch.cat([local, pad], 0)

                    bufs = [torch.empty_like(local) for _ in range(dist.get_world_size())]
                    dist.all_gather(bufs, local)

                    # 3) 去 pad + 拼接（只在 rank0 返回，其他 rank 返回 None）
                    if dist.get_rank() == 0:
                        chunks = [bufs[r][:sizes[r]] for r in range(dist.get_world_size())]
                        return torch.cat(chunks, 0).cpu()
                    else:
                        return None

                # 只在 rank0 写盘：
                if dist.is_initialized():
                    means     = gather_for_export(self.splats["means"])
                    scales    = gather_for_export(self.splats["scales"])
                    quats     = gather_for_export(self.splats["quats"])
                    opacities = gather_for_export(self.splats["opacities"])
                    sh0       = gather_for_export(self.splats["sh0"])
                    shN       = gather_for_export(self.splats["shN"])
                    if (means is not None):   # 仅 rank0
                        export_splats(means, scales, quats, opacities, sh0, shN,
                                    format="ply", save_to=f"{self.ply_dir}/point_cloud_{step}.ply")

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step, stage="train")
                if cfg.use_val:
                    self.eval(step, stage="val")
            if step in [i - 1 for i in cfg.video_render_steps]:
                self.render_traj(step)

            # # run fixer
            # if step in [i - 1 for i in cfg.fix_steps]:
            #     self.fix(step, fix_mode=cfg.fix_mode, distance=cfg.fix_distance, downsample_stride=cfg.fix_downsample_stride)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    # def fix(self, step: int, fix_mode: str = "extrapolate", distance: float = 0.5, downsample_stride: int = 8):

    #     print("Running fixer...")
    #     if len(self.cfg.fix_steps) == 1:
    #         novel_poses = self.parser.camtoworlds[self.valset.indices]
    #     elif fix_mode == "interpolate":
    #         novel_poses = self.interpolator.shift_poses(self.current_novel_poses, self.parser.camtoworlds[self.valset.indices], distance)
    #     elif fix_mode == "extrapolate":
    #         novel_poses = self.interpolator.horizontal_shift_poses(self.base_novel_poses, distance, downsample_stride)
        
    #     self.render_traj(step, novel_poses)
    #     image_paths = [f"{self.render_dir}/novel/{step}/Pred/{i:04d}.jpg" for i in range(len(novel_poses))]

    #     if len(self.novelloaders) == 0:
    #         ref_image_indices = self.interpolator.find_nearest_assignments(self.parser.camtoworlds[self.trainset.indices], novel_poses)
    #         ref_image_paths = [self.parser.image_paths[i] for i in np.array(self.trainset.indices)[ref_image_indices]]
    #     else:
    #         ref_image_indices = self.interpolator.find_nearest_assignments(self.parser.camtoworlds[self.trainset.indices], novel_poses)
    #         ref_image_paths = [self.parser.image_paths[i] for i in np.array(self.trainset.indices)[ref_image_indices]]
    #     assert len(image_paths) == len(ref_image_paths) == len(novel_poses)

    #     # for i in tqdm.trange(0, len(novel_poses), desc="Fixing artifacts..."):
    #     #     image = Image.open(image_paths[i]).convert("RGB")
    #     #     ref_image = Image.open(ref_image_paths[i]).convert("RGB")
    #     #     output_image = self.difix(prompt="remove degradation", image=image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
    #     #     output_image = output_image.resize(image.size, Image.LANCZOS)
    #     #     os.makedirs(f"{self.render_dir}/novel/{step}/Fixed", exist_ok=True)
    #     #     output_image.save(f"{self.render_dir}/novel/{step}/Fixed/{i:04d}.png")
    #     #     if ref_image is not None:
    #     #         os.makedirs(f"{self.render_dir}/novel/{step}/Ref", exist_ok=True)
    #     #         ref_image.save(f"{self.render_dir}/novel/{step}/Ref/{i:04d}.png")
    
    #     # parser = deepcopy(self.parser)
    #     # parser.test_every = 0
    #     # parser.image_paths = [f"{self.render_dir}/novel/{step}/Fixed/{i:04d}.png" for i in range(len(novel_poses))]
    #     # parser.image_names = [os.path.basename(p) for p in parser.image_paths]
    #     # parser.alpha_mask_paths = [f"{self.render_dir}/novel/{step}/Alpha/{i:04d}.png" for i in range(len(novel_poses))]
    #     # parser.camtoworlds = novel_poses
    #     # parser.camera_ids = [parser.camera_ids[0]] * len(novel_poses)
        
    #     # print(f"Adding {len(parser.image_paths)} fixed images to novel dataset...")
    #     # dataset = Dataset(parser, split="train")
    #     # dataloader = torch.utils.data.DataLoader(
    #     #     dataset,
    #     #     batch_size=self.cfg.batch_size,
    #     #     shuffle=True,
    #     #     num_workers=4,
    #     #     persistent_workers=True,
    #     #     pin_memory=True,
    #     #     load_depths=False
    #     # )
    #     # self.novelloaders = [dataloader]
    #     # self.novelloaders_iter = [iter(dataloader)]

    #     # self.current_novel_poses = novel_poses

    #     # 4) 分片跑 Difix
    #     N = len(novel_poses)
    #     if dist.is_initialized():
    #         rank = dist.get_rank()
    #         world_size = dist.get_world_size()
    #         my_ids = list(range(rank, N, world_size))
    #     else:
    #         my_ids = list(range(N))

    #     for i in tqdm.trange(0, len(my_ids), desc=f"Fixing artifacts (rank shard)"):
    #         idx = my_ids[i]
    #         pred_ext = "jpg"
    #         pred_p = f"{self.render_dir}/novel/{step}/Pred/{idx:04d}.{pred_ext}"
    #         ref_p  = ref_image_paths[idx]
    #         image = Image.open(pred_p).convert("RGB")
    #         ref_image = Image.open(ref_p).convert("RGB")
    #         out = self.difix(prompt="remove degradation", image=image, ref_image=ref_image,
    #                         num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
    #         out = out.resize(image.size, Image.LANCZOS)

    #         os.makedirs(f"{self.render_dir}/novel/{step}/Fixed", exist_ok=True)
    #         out.save(f"{self.render_dir}/novel/{step}/Fixed/{idx:04d}.jpg")
    #         # 可选：保存Ref方便查看
    #         os.makedirs(f"{self.render_dir}/novel/{step}/Ref", exist_ok=True)
    #         ref_image.save(f"{self.render_dir}/novel/{step}/Ref/{idx:04d}.jpg")

    #     if dist.is_initialized():
    #         dist.barrier() 

    #     from copy import deepcopy
    #     if dist.is_initialized():
    #         world_size = dist.get_world_size()
    #         rank = dist.get_rank()
    #     else:
    #         world_size, rank = 1, 0

    #     # 路径 & 扩展名（和 render_traj / fix 一致）
    #     pred_ext  = "jpg" if self.cfg.render_format == "jpg" else "png"
    #     alpha_ext = "jpg"

    #     N = len(novel_poses)
    #     parser = deepcopy(self.parser)
    #     parser.test_every = int(1e6)
    #     parser.image_paths = [f"{self.render_dir}/novel/{step}/Fixed/{i:04d}.{pred_ext}" for i in range(N)]
    #     parser.image_names = [os.path.basename(p) for p in parser.image_paths]
    #     parser.alpha_mask_paths = [f"{self.render_dir}/novel/{step}/Alpha/{i:04d}.{alpha_ext}" for i in range(N)]
    #     parser.camtoworlds = novel_poses
    #     parser.camera_ids  = [parser.camera_ids[0]] * N

    #     dataset = Dataset(parser, split="train", load_depths=False)

    #     # 标准分片
    #     sampler = None
    #     if dist.is_initialized():
    #         sampler = torch.utils.data.distributed.DistributedSampler(
    #             dataset,
    #             num_replicas=world_size,
    #             rank=rank,
    #             shuffle=True,
    #             drop_last=False,   # 如需各 rank 批次数完全一致，可设 True
    #         )

    #     # 注意：这个“临时”loader 不建议开多进程 worker，最稳妥
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=self.cfg.batch_size,
    #         shuffle=(sampler is None),
    #         sampler=sampler,
    #         num_workers=0,              # 动态创建/销毁的 loader：0 最不容易卡
    #         persistent_workers=False,   # 防止旧 worker 占着不退出
    #         pin_memory=True,
    #     )

    #     # 保存到实例里供训练循环用
    #     self.novelloaders = [dataloader]
    #     self.novelloaders_iter = [iter(dataloader)]
    #     self.novel_samplers = [sampler]   # 新增：保存 sampler，后面 set_epoch 用
    #     self.current_novel_poses = novel_poses

    #     torch.cuda.empty_cache()
    
    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Evaluate metrics on a split with subset sampling and limited image saving."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        # 选择数据集
        if stage == "train":
            dataset = self.trainset
        else:
            dataset = self.valset if getattr(self, "valset", None) else None
            if dataset is None or len(dataset) == 0:
                dataset = self.trainset  # fallback

        if dataset is None or len(dataset) == 0:
            if world_rank == 0:
                print(f"[eval] No data for stage='{stage}'. Skipped.")
            return

        # ---- 子集抽样：按 stride 抽样，再截断到 max_images ----
        N = len(dataset)
        idxs = list(range(0, N, max(1, cfg.eval_stride)))
        if cfg.eval_max_images is not None:
            idxs = idxs[: cfg.eval_max_images]
        N_eval = len(idxs)
        if N_eval == 0:
            if world_rank == 0:
                print(f"[eval] Nothing to evaluate for stage='{stage}'.")
            return

        ellipse_time = 0.0
        metrics = defaultdict(list)
        saved_cnt = 0
        last_canvas = None

        for i, idx in enumerate(idxs):
            # 直接索引 dataset（无需 DataLoader），注意补 batch 维
            data = dataset[idx]
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[0], pixels.shape[1]

            # 渲染
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[None] if camtoworlds.ndim == 2 else camtoworlds,
                Ks=Ks[None] if Ks.ndim == 2 else Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-6)

            colors = torch.clamp(colors, 0.0, 1.0)

            if world_rank == 0:
                # 指标
                pixels_p = pixels.unsqueeze(0).permute(0, 3, 1, 2)  # [1,3,H,W]
                colors_p = colors.permute(0, 3, 1, 2)               # [1,3,H,W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

                # 仅保存少量示例
                if cfg.eval_save_images and saved_cnt < cfg.eval_save_images_n:
                    canvas = torch.cat([pixels.unsqueeze(0), colors], dim=2).squeeze(0).cpu().numpy()
                    canvas = (canvas * 255).astype(np.uint8)
                    imageio.imwrite(f"{self.render_dir}/{stage}_step{step}_{i:04d}.png", canvas)
                    saved_cnt += 1
                    last_canvas = canvas

        if world_rank == 0:
            ellipse_time /= float(N_eval)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items() if len(v) > 0}
            stats.update({"ellipse_time": ellipse_time, "num_GS": len(self.splats["means"])})

            print(
                f"[{stage}] PSNR: {stats.get('psnr', float('nan')):.3f}, "
                f"SSIM: {stats.get('ssim', float('nan')):.4f}, "
                f"LPIPS: {stats.get('lpips', float('nan')):.3f} | "
                f"Time: {stats['ellipse_time']:.3f}s/img | "
                f"GS: {stats['num_GS']}"
            )

            # 保存统计
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)

            # W&B
            if self.wb is not None:
                log_dict = {
                    f"{stage}/psnr": float(stats.get("psnr", float("nan"))),
                    f"{stage}/ssim": float(stats.get("ssim", float("nan"))),
                    f"{stage}/lpips": float(stats.get("lpips", float("nan"))),
                    f"{stage}/num_GS": int(stats["num_GS"]),
                    f"{stage}/ellipse_time_s_per_img": float(stats["ellipse_time"]),
                }
                self.wb.log(log_dict, step=step)
                try:
                    if last_canvas is not None:
                        self.wb.log({f"{stage}/example": wandb.Image(last_canvas)}, step=step)
                except Exception:
                    pass

    @torch.no_grad()
    def render_traj(self, step: int, camtoworlds_all=None, tag="novel"):
        """Fast trajectory rendering with sharding + batching + RGB-only."""
        cfg = self.cfg
        device = self.device

        if camtoworlds_all is None and cfg.disable_video:
            return
        print("Running trajectory rendering...")

        # 1) 生成相机轨迹（与你原来一致，但不做深度相关操作）
        if camtoworlds_all is None:
            camtoworlds_all = self.parser.camtoworlds[5:-5]
            if cfg.render_traj_path == "interp":
                camtoworlds_all = generate_interpolated_path(camtoworlds_all, 1)
            elif cfg.render_traj_path == "ellipse":
                height = camtoworlds_all[:, 2, 3].mean()
                camtoworlds_all = generate_ellipse_path_z(camtoworlds_all, height=height)
            elif cfg.render_traj_path == "spiral":
                camtoworlds_all = generate_spiral_path(
                    camtoworlds_all,
                    bounds=self.parser.bounds * self.scene_scale,
                    spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
                )
            elif cfg.render_traj_path == "original":
                camtoworlds_all = self.parser.camtoworlds
            else:
                raise ValueError(f"Render trajectory type not supported: {cfg.render_traj_path}")

            # 补最后一行变成 [N,4,4]
            if cfg.render_traj_path != "original":
                camtoworlds_all = np.concatenate(
                    [camtoworlds_all,
                    np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0)],
                    axis=1
                )

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)

        # 2) 相机内参与分辨率
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # 4) 分片（真正避免多卡重复渲）
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        if world > 1:
            my_ids = list(range(rank, len(camtoworlds_all), world))
        else:
            my_ids = list(range(len(camtoworlds_all)))

        # 5) 输出目录（每个 rank 都写自己份，最终 barrier 保证完整）
        pred_dir  = f"{self.render_dir}/{tag}/{step}/Pred"
        alpha_dir = f"{self.render_dir}/{tag}/{step}/Alpha"
        if len(my_ids) > 0:
            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(alpha_dir, exist_ok=True)

        # 6) 批量渲染（RGB-only）
        B = max(1, int(cfg.render_batch))
        pbar = tqdm.trange(0, len(my_ids), B, desc=f"Rendering trajectory (rank {rank})",
                        disable=(world > 1 and rank != 0))

        for s in pbar:
            ids = my_ids[s:s+B]
            c2w = camtoworlds_all[ids]                 # [B,4,4]
            Ks  = K.expand(len(ids), -1, -1)           # [B,3,3]

            renders, alphas, _ = self.rasterize_splats(
                camtoworlds=c2w,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB",                     # 只算 RGB（关键加速）
                distributed=False
            )  # [B,H,W,3] and [B,H,W,1]

            # 转为 uint8（一次性批量搬运）
            colors_u8 = (torch.clamp(renders[..., :3], 0.0, 1.0) * 255).byte().cpu().numpy()
            alphas_u8 = (alphas[..., 0] * 255).byte().cpu().numpy()

            # 7) 快速写盘：Pred 用 JPEG/PNG（PNG 设置为 0 压缩），Alpha 用无压缩 PNG
            for j, fid in enumerate(ids):
                pred_path = f"{pred_dir}/{fid:04d}.{ 'jpg' if cfg.render_format=='jpg' else 'png' }"
                img = Image.fromarray(colors_u8[j])
                if cfg.render_format == "jpg":
                    # 高速 JPEG：关闭 optimize，质量 95
                    img.save(pred_path, quality=95, subsampling=0, optimize=False)
                else:
                    # 高速 PNG：0 压缩
                    img.save(pred_path, compress_level=0, optimize=False)

                aimg = Image.fromarray(alphas_u8[j], mode='L')
                alpha_path = f"{alpha_dir}/{fid:04d}.jpg"
                aimg.save(alpha_path, quality=95, subsampling=0, optimize=False)



    # @torch.no_grad()
    # def render_traj(self, step: int, camtoworlds_all=None, tag="novel"):
    #     """Entry for trajectory rendering."""
    #     if camtoworlds_all is None and self.cfg.disable_video:
    #         return
    #     print("Running trajectory rendering...")
    #     cfg = self.cfg
    #     device = self.device

    #     if camtoworlds_all is None:
    #         camtoworlds_all = self.parser.camtoworlds[5:-5]
    #         if cfg.render_traj_path == "interp":
    #             camtoworlds_all = generate_interpolated_path(
    #                 camtoworlds_all, 1
    #             )  # [N, 3, 4]
    #         elif cfg.render_traj_path == "ellipse":
    #             height = camtoworlds_all[:, 2, 3].mean()
    #             camtoworlds_all = generate_ellipse_path_z(
    #                 camtoworlds_all, height=height
    #             )  # [N, 3, 4]
    #         elif cfg.render_traj_path == "spiral":
    #             camtoworlds_all = generate_spiral_path(
    #                 camtoworlds_all,
    #                 bounds=self.parser.bounds * self.scene_scale,
    #                 spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
    #             )
    #         elif cfg.render_traj_path == "original":
    #             camtoworlds_all = self.parser.camtoworlds
    #         else:
    #             raise ValueError(
    #                 f"Render trajectory type not supported: {cfg.render_traj_path}"
    #             )
            
    #         if cfg.render_traj_path != "original":
    #             camtoworlds_all = np.concatenate(
    #                 [
    #                     camtoworlds_all,
    #                     np.repeat(
    #                         np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
    #                     ),
    #                 ],
    #                 axis=1,
    #             )  # [N, 4, 4]

    #     camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
    #     K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
    #     width, height = list(self.parser.imsize_dict.values())[0]



    #     rank = dist.get_rank() if (dist.is_initialized()) else 0

    #     # # save to video
    #     # video_dir = f"{cfg.result_dir}/videos"
    #     # os.makedirs(video_dir, exist_ok=True)
    #     # writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
    #     for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
    #         camtoworlds = camtoworlds_all[i : i + 1]
    #         Ks = K[None]

    #         renders, alphas, _ = self.rasterize_splats(
    #             camtoworlds=camtoworlds,
    #             Ks=Ks,
    #             width=width,
    #             height=height,
    #             sh_degree=cfg.sh_degree,
    #             near_plane=cfg.near_plane,
    #             far_plane=cfg.far_plane,
    #             render_mode="RGB+ED",
    #         )  # [1, H, W, 4]
    #         colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
    #         depths = renders[..., 3:4]  # [1, H, W, 1]
    #         depths = (depths - depths.min()) / (depths.max() - depths.min())
    #         canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

    #         # write images
    #         canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
    #         canvas = (canvas * 255).astype(np.uint8)
    #         # writer.append_data(canvas)
    #         if self.world_size == 1 or (self.world_size > 1 and rank==0):
    #             # Write for Difix3d fixer
    #             colors_path = f"{self.render_dir}/{tag}/{step}/Pred/{i:04d}.png"
    #             os.makedirs(os.path.dirname(colors_path), exist_ok=True)
    #             colors_canvas = colors.cpu().numpy()
    #             colors_canvas = (colors.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    #             imageio.imwrite(colors_path, colors_canvas)
                
    #             alphas_path = f"{self.render_dir}/{tag}/{step}/Alpha/{i:04d}.png"
    #             os.makedirs(os.path.dirname(alphas_path), exist_ok=True)
    #             alphas_canvas = alphas.squeeze(0).float().cpu().numpy()
    #             alphas_canvas = (alphas_canvas * 255).astype(np.uint8)
    #             Image.fromarray(alphas_canvas.squeeze(), mode='L').save(alphas_path)

    #     # writer.close()
    #     # print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.8,
                init_scale=0.25,
                opacity_reg=0.001,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
