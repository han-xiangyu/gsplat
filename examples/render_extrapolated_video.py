# render_from_ply_multich.py
import os, re, math, imageio
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Literal, Dict, List

import torch
from gsplat.rendering import rasterization
from datasets_utils.colmap import Parser
from tqdm import tqdm
import cv2


def horizontal_sine_poses(camtoworlds_np: np.ndarray, amplitude: float, period: int) -> np.ndarray:
    """
    基于原始 c2w，按帧索引 i 在相机局部 x 轴（右方向）做正弦摆动：
        center' = center + sin(2π * i / period) * amplitude * right_vector
    保持朝向不变，仅平移。
    输入:
        camtoworlds_np: [N,4,4] 原始相机位姿（c2w）
        amplitude: 摆动幅度
        period: 正弦周期（以帧索引 i 为自变量）
    返回:
        mod_c2w: [N,4,4] 调制后的位姿
    """
    assert camtoworlds_np.ndim == 3 and camtoworlds_np.shape[1:] == (4, 4)
    N = camtoworlds_np.shape[0]
    mod = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)
    mod[:, :, :] = camtoworlds_np
    for i in range(N):
        R = camtoworlds_np[i, :3, :3]
        t = camtoworlds_np[i, :3, 3]
        right_vec = R[:, 0]  # c2w 的 x 轴
        shift = amplitude * np.sin(2.0 * np.pi * (i / float(max(1, period))))
        mod[i, :3, 3] = t + shift * right_vec
    return mod

def horizontal_shift_poses(training_poses, distance):
    """
    Shift training poses horizontally.
    
    Args:
        training_poses: [N, 4, 4] array of training camera poses
        distance: float, the step size to move training pose toward testing pose
        
    Returns:
        novel_poses: [M, 4, 4] array of shifted poses
    """
    novel_poses = []

    for train_pose in training_poses:
        # Calculate horizontal shift
        right_vector = train_pose[:3, 0]
        center = train_pose[:3, 3]
        right_center = center + distance * right_vector
        left_center  = center - distance * right_vector

        # Construct shifted pose
        right_shifted_pose = np.eye(4)
        right_shifted_pose[:3, :3] = train_pose[:3, :3]
        right_shifted_pose[:3, 3] = right_center

        left_shifted_pose = np.eye(4)
        left_shifted_pose[:3, :3] = train_pose[:3, :3]
        left_shifted_pose[:3, 3] = left_center

        novel_poses.append(right_shifted_pose)
        novel_poses.append(left_shifted_pose)

    return np.array(novel_poses)


def apply_jet_cmap01(x01: np.ndarray, reverse: bool = True) -> np.ndarray:
    """
    x01: H×W, 已在 [0,1] 归一化的深度
    reverse=True: 0→红, 1→蓝（近红远蓝）；False 则相反
    return: H×W×3, float in [0,1]
    """
    x = np.clip(x01, 0.0, 1.0)
    if reverse:
        x = 1.0 - x  # 让 0(近) 走向红色，1(远) 走向蓝色

    # 经典 jet（近似实现）
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)

def _is_aug(name: str) -> bool:
    stem = os.path.splitext(os.path.basename(name))[0]
    return stem.endswith("_left_difix") or stem.endswith("_right_difix")

# 1) 正则表达式：匹配 loc, trav, ch, frame
PAT = re.compile(
    r"(?:loc_(?P<loc>\d+)_)?trav_(?P<trav>\d+)_channel_(?P<ch>\d+)_img_(?P<frame>\d+)\.(?:png|jpg|jpeg)$",
    re.IGNORECASE,
)

# 2) 解析函数：缺失 loc 时给默认值 0
def parse_meta(name: str):
    m = PAT.search(name)
    if not m:
        return None
    loc = m.group("loc")
    return {
        "loc": int(loc) if loc is not None else 0,   # <- 默认 0
        "trav": int(m.group("trav")),
        "ch": int(m.group("ch")),
        "frame": int(m.group("frame")),
    }

def ensure_4x4(c2w: np.ndarray) -> np.ndarray:
    assert c2w.ndim == 3 and c2w.shape[1] in (3, 4)
    if c2w.shape[1] == 4:
        return c2w
    N = c2w.shape[0]
    last = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (N, 1, 1))
    return np.concatenate([c2w, last], axis=1)

def pick_K_and_size(parser: Parser) -> Tuple[np.ndarray, Tuple[int, int]]:
    K = np.asarray(list(parser.Ks_dict.values())[0], dtype=np.float32)
    W, H = list(parser.imsize_dict.values())[0]
    return K, (W, H)

def sorted_props_by_prefix(names: List[str], prefix: str) -> List[str]:
    pat = re.compile(re.escape(prefix) + r"_(\d+)$")
    pairs = []
    for n in names:
        m = pat.match(n)
        if m: pairs.append((int(m.group(1)), n))
    pairs.sort(key=lambda x: x[0])
    return [n for _, n in pairs]

def load_splats_from_ply(ply_path: str, device: torch.device):
    """
    严格按 splat2ply_bytes 导出的字段读取：
      x,y,z
      f_dc_0..2
      f_rest_0..(K*3-1)
      opacity          # logit
      scale_0..2       # log-scale
      rot_0..3         # 与训练参数顺序一致，不重排
    返回 (splats: ParameterDict, meta: dict)
    """
    try:
        from plyfile import PlyData
    except Exception as e:
        raise ImportError("请先安装 plyfile：`pip install plyfile`") from e

    ply = PlyData.read(ply_path)
    v = next((el for el in ply.elements if el.name == "vertex"), None)
    if v is None:
        raise RuntimeError("PLY 缺少 'vertex' 元素")
    data, names = v.data, list(v.data.dtype.names)

    def col(k):
        if k not in names:
            raise KeyError(f"缺少字段 {k}")
        return np.asarray(data[k], dtype=np.float32)

    # positions
    means = np.stack([col("x"), col("y"), col("z")], 1)  # (N,3)

    # SH: f_dc_0..2
    for k in ["f_dc_0", "f_dc_1", "f_dc_2"]:
        if k not in names:
            raise KeyError("未找到 DC SH（f_dc_0..2）")
    sh0 = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], 1)  # (N,3)
    sh0 = sh0[:, None, :]  # (N,1,3)

    # SH: f_rest_0..(K*3-1)
    rest_keys = [k for k in names if k.startswith("f_rest_")]
    rest_keys.sort(key=lambda x: int(x.split("_")[-1]))

    rest_flat = np.stack([col(k) for k in rest_keys], 1)  # (N, K*3)，顺序是 c-major: [c0:k0..kK-1, c1:..., c2:...]
    N = rest_flat.shape[0]
    K = rest_flat.shape[1] // 3
    shN = rest_flat.reshape(N, 3, K).transpose(0, 2, 1)   # -> (N, K, 3)


    # opacity (logit)
    opacity = col("opacity")  # (N,)

    # scales (log-scale)
    scales = np.stack([col("scale_0"), col("scale_1"), col("scale_2")], 1)  # (N,3)

    # quats（按 rot_0..3 原样读取，不重排）
    quats = np.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], 1)  # (N,4)

    pd = torch.nn.ParameterDict({
        "means":     torch.nn.Parameter(torch.from_numpy(means).to(device), requires_grad=False),
        "scales":    torch.nn.Parameter(torch.from_numpy(scales).to(device), requires_grad=False),     # log
        "quats":     torch.nn.Parameter(torch.from_numpy(quats).to(device), requires_grad=False),      # as-is
        "opacities": torch.nn.Parameter(torch.from_numpy(opacity).to(device), requires_grad=False),    # logit
        "sh0":       torch.nn.Parameter(torch.from_numpy(sh0).to(device), requires_grad=False),
        "shN":       torch.nn.Parameter(torch.from_numpy(shN).to(device), requires_grad=False),
    })

    meta = {
        "scale_is_log": True,
        "opacity_is_logit": True,
        "quat_order": "as_is",
    }
    print(f"[PLY meta] 固定约定 -> scale_is_log=True, opacity_is_logit=True, quat_order=as_is")
    return pd, meta



# ============ 渲染配置 ============

@dataclass
class RenderConfig:
    data_dir: str
    ply_path: str
    result_dir: str = "results/render_from_ply_multich"
    fps: int = 15
    save_frames: bool = False
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    antialiased: bool = False
    packed: bool = False
    near_plane: float = 0.01
    far_plane: float = 1000.0
    device: str = "cuda"
    # 缺失通道的占位颜色 (RGB 0-255)
    placeholder_rgb: Tuple[int, int, int] = (0, 0, 0)
    channels: Tuple[int, ...] = (1,)
    out_img_dir: str = "extrapolated_render"


    # === 正弦摆动配置 ===
    traj: Literal["shifted_pairs", "horizontal_sine"] = "horizontal_sine"
    amplitude: float = 1.5     # 摆动幅度（世界坐标系，单位与场景一致）
    period: int = 60           # 每个正弦周期对应的“帧索引”步长
    sine_suffix: str = "sine"  # 输出文件名后缀


def get_gt_path(parser, image_names, i: int) -> Optional[str]:
    # 若 Parser 自带完整路径，优先用它（最稳）
    if hasattr(parser, "image_paths"):
        arr = getattr(parser, "image_paths")
        if isinstance(arr, (list, tuple)) and len(arr) == len(image_names):
            return str(arr[i])
    # 否则按常见目录名在 data_dir 下兜底
    name = image_names[i]
    for sub in ["images", "rgb", "imgs"]:
        p = Path(cfg.data_dir) / sub / name
        if p.exists():
            return str(p)
    return None


# ===== 2) 按 *_left / *_right 渲染落盘 =====
@torch.no_grad()
def render_shifted_pairs(cfg: RenderConfig, distance: float = 1.5):
    """
    对每个原始帧生成左右两个位姿并渲染到 png：
      <orig_name>_left.png / <orig_name>_right.png
    """
    os.makedirs(cfg.out_img_dir, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 读取数据与名字
    parser = Parser(data_dir=cfg.data_dir, factor=1, normalize=False, test_every=60)
    image_names = None
    for cand in ["image_paths", "image_names", "filenames", "images"]:
        if hasattr(parser, cand):
            arr = getattr(parser, cand)
            if isinstance(arr, (list, tuple)) and len(arr) > 0:
                image_names = [os.path.basename(str(x)) for x in arr]
                break
    if image_names is None:
        img_dir = Path(cfg.data_dir) / "images"
        image_names = sorted([p.name for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    # camtoworlds_np = ensure_4x4(parser.camtoworlds)   # [N,4,4]
    # N = camtoworlds_np.shape[0]
    # assert len(image_names) == N, "image_names 与 cam 数量不一致"


    keep_idx = [i for i, n in enumerate(image_names) if not _is_aug(n)]
    image_names = [image_names[i] for i in keep_idx]

    # 位姿与 camera_ids 同步子集化
    camtoworlds_all = ensure_4x4(parser.camtoworlds)
    camtoworlds_np = camtoworlds_all[keep_idx, :, :]
    camera_ids = [parser.camera_ids[i] for i in keep_idx]

    N = len(image_names)
    assert camtoworlds_np.shape[0] == N == len(camera_ids)

    K_global, (W, H) = pick_K_and_size(parser)

    # 加载 splats（与你现有逻辑保持一致）
    splats, meta = load_splats_from_ply(cfg.ply_path, device=device)
    K_total = 1 + splats["shN"].shape[1]
    sh_degree = int(math.isqrt(K_total)) - 1
    assert (sh_degree + 1) ** 2 == K_total

    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"]) if meta["scale_is_log"] else splats["scales"]
    opacities = torch.sigmoid(splats["opacities"]) if meta["opacity_is_logit"] else splats["opacities"]
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)


    videos_dir = os.path.join(cfg.result_dir, "extrapolated_videos")
    os.makedirs(videos_dir, exist_ok=True)
    video_path = os.path.join(videos_dir, f"extrapolated_{cfg.sine_suffix}.mp4")
    writer = imageio.get_writer(video_path, fps=cfg.fps)

    # --------------------------------- Sine Wave -------------------------------------
    if cfg.traj == 'horizontal_sine':
        mod_c2w_np = horizontal_sine_poses(camtoworlds_np, amplitude=cfg.amplitude, period=cfg.period)

        pbar = tqdm(range(N), desc="Horizontal-sine render")
        for i in pbar:
            base = image_names[i]
            stem, ext = os.path.splitext(base)
            if ext == "":
                ext = ".png"

            # 逐帧相机内参
            camera_id = camera_ids[i]
            K_np = np.asarray(parser.Ks_dict[camera_id], dtype=np.float32)
            Ks = torch.from_numpy(K_np).float().to(device)[None]

            # 该帧的正弦偏移后位姿
            c2w = torch.from_numpy(mod_c2w_np[i : i + 1]).float().to(device)
            viewmats = torch.linalg.inv(c2w)

            renders, alphas, info = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
                viewmats=viewmats, Ks=Ks, width=W, height=H,
                packed=cfg.packed, absgrad=False, sparse_grad=False,
                rasterize_mode="antialiased" if cfg.antialiased else "classic",
                distributed=False, camera_model=cfg.camera_model,
                with_ut=False, with_eval3d=False,
                render_mode="RGB+ED",
                near_plane=cfg.near_plane, far_plane=cfg.far_plane, sh_degree=sh_degree,
            )

            rgb = torch.clamp(renders[..., :3], 0.0, 1.0)
            rgb_u8 = (rgb[0].cpu().numpy() * 255).astype(np.uint8)
            out_path = os.path.join(cfg.out_img_dir, f"{stem}_{cfg.sine_suffix}{ext}")
            # imageio.imwrite(out_path, rgb_u8)
            writer.append_data(rgb_u8)

        writer.close()
        print(f"[Done] 已保存视频：{video_path}")
    else:
        raise ValueError(f"Unknown trajectory type {cfg.traj}")


# ============ CLI ============
if __name__ == "__main__":
    import tyro
    cfg = tyro.cli(RenderConfig)

    render_shifted_pairs(cfg, distance=1.5)