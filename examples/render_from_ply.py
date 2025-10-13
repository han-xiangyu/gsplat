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


# 1) 正则表达式：匹配 loc, trav, ch, frame
PAT = re.compile( r"(?:loc_(?P<loc>\d+)_)?trav_(?P<trav>\d+)_channel_(?P<ch>\d+)_img_(?P<frame>\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE, )


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
    fps: int = 30
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


# ============ 主流程：按 (loc, trav) 输出视频，帧内拼接(2|1|3) × (RGB|DEPTH) ============

@torch.no_grad()
def render_multichannel(cfg: RenderConfig):
    os.makedirs(cfg.result_dir, exist_ok=True)
    videos_dir = os.path.join(cfg.result_dir, "videos")
    frames_dir = os.path.join(cfg.result_dir, "frames")
    if cfg.save_frames:
        os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 1) 读相机与文件名（文件名用于解析 loc/trav/ch/frame）
    parser = Parser(data_dir=cfg.data_dir, factor=1, normalize=False, test_every=1000000)
    image_names = None
    for cand in ["image_paths", "image_names", "filenames", "images"]:
        if hasattr(parser, cand):
            arr = getattr(parser, cand)
            # 统一成 basename 列表
            if isinstance(arr, (list, tuple)) and len(arr) > 0:
                image_names = [os.path.basename(str(x)) for x in arr]
                break
    if image_names is None:
        img_dir = Path(cfg.data_dir) / "images"
        if not img_dir.exists():
            raise RuntimeError("无法从 Parser 获取图像文件名，且未找到 data_dir/images 兜底目录。")
        image_names = sorted([p.name for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    camtoworlds_np = ensure_4x4(parser.camtoworlds)  # [N,4,4]
    K_global, (W, H) = pick_K_and_size(parser)
    N = camtoworlds_np.shape[0]
    assert len(image_names) == N, f"文件名数量({len(image_names)})与相机数量({N})不一致"

    camtoworlds = torch.from_numpy(camtoworlds_np).float().to(device)
    K_tensor = torch.from_numpy(K_global).float().to(device)[None]  # 默认单内参

    # 2) 解析文件名 -> 建立索引: (loc, trav, frame, ch) -> image_idx
    index: Dict[Tuple[int, int, int, int], int] = {}
    groups: Dict[Tuple[int, int], set] = {}
    for i, name in enumerate(image_names):
        meta = parse_meta(name)
        if meta is None:
            #raise ValueError(f"文件名不符合规范: {name}")
            continue
        key = (meta["loc"], meta["trav"], meta["frame"], meta["ch"])
        index[key] = i
        lg = (meta["loc"], meta["trav"])
        groups.setdefault(lg, set()).add(meta["frame"])

    # 3) 载入 PLY
    splats, meta = load_splats_from_ply(cfg.ply_path, device=device)

    # 计算 sh_degree
    K_total = 1 + splats["shN"].shape[1]   # 1=DC, 其余为 f_rest
    sh_degree = int(math.isqrt(K_total)) - 1
    assert (sh_degree + 1) ** 2 == K_total, f"Invalid SH channels: {K_total}"


    # 4) 按 (loc, trav) 输出视频
    # order = [2, 1, 3]  # 左、中、右 的 channel 顺序
    order = list(cfg.channels)
    placeholder_rgb = np.full((H, W, 3), np.array(cfg.placeholder_rgb, dtype=np.uint8), dtype=np.uint8)
    placeholder_depth = np.zeros((H, W, 3), dtype=np.uint8)

    for (loc, trav), frame_set in sorted(groups.items()):
        frames_sorted = sorted(frame_set)
        out_name = f"loc_{loc}_trav_{trav}.mp4"
        video_path = os.path.join(videos_dir, out_name)
        writer = imageio.get_writer(video_path, fps=cfg.fps)
        print(f"[Render] (loc={loc}, trav={trav}) 共 {len(frames_sorted)} 帧 -> {video_path}")

        for fidx, frame_id in tqdm(enumerate(frames_sorted, 1), desc="Processing images", total=len(frames_sorted)):
            rgb_tiles: List[np.ndarray] = []
            gt_tiles: List[np.ndarray] = [] 
            depth_tensors: List[Optional[torch.Tensor]] = []  # 保存 raw depth，用于统一归一化
            rgb_raw_cache: List[Optional[torch.Tensor]] = []

            # 4.1 先渲染 3 个通道，收集每个通道的 depth（用于统一归一化）
            for ch in order:
                key = (loc, trav, frame_id, ch)
                if key not in index:
                    rgb_tiles.append(placeholder_rgb.copy())
                    gt_tiles.append(placeholder_rgb.copy())
                    depth_tensors.append(None)
                    rgb_raw_cache.append(None)
                    continue

                i = index[key]
                c2w = camtoworlds[i : i + 1]  # [1,4,4]
                
                # 逐帧K
                camera_id = parser.camera_ids[i]
                K_np = np.asarray(parser.Ks_dict[camera_id], dtype=np.float32)
                Ks = torch.from_numpy(K_np).float().to(device)[None]

                # 拿参数（与训练一致：exp/scales, sigmoid/opacity）
                means = splats["means"]
                quats = splats["quats"]

                # 按导出约定做参数域变换
                scales = torch.exp(splats["scales"]) if meta["scale_is_log"] else splats["scales"]
                opacities = torch.sigmoid(splats["opacities"]) if meta["opacity_is_logit"] else splats["opacities"]

                # 组 SH 颜色并显式给 sh_degree
                colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, (sh_degree+1)^2, 3]


                renders, alphas, info = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,                     # [N, (sh_degree+1)^2, 3]
                    viewmats=torch.linalg.inv(c2w),
                    Ks=Ks,
                    width=W,
                    height=H,
                    packed=cfg.packed,
                    absgrad=False,
                    sparse_grad=False,
                    rasterize_mode="antialiased" if cfg.antialiased else "classic",
                    distributed=False,
                    camera_model=cfg.camera_model,
                    with_ut=False,
                    with_eval3d=False,
                    render_mode="RGB+ED",
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    sh_degree=sh_degree,
                )
                rgb = torch.clamp(renders[..., :3], 0.0, 1.0)   # [1,H,W,3]
                depth = renders[..., 3:4]                        # [1,H,W,1]

                rgb_np = (rgb[0].cpu().numpy() * 255).astype(np.uint8)
                rgb_tiles.append(rgb_np)
                depth_tensors.append(depth)
                rgb_raw_cache.append(rgb)
                gt_path = get_gt_path(parser, image_names, i)
                if gt_path is not None:
                    gt_img = imageio.imread(gt_path)
                    # 统一到 H×W×3
                    if gt_img.shape[2] == 4:
                        gt_img = gt_img[..., :3]
                    gt_tiles.append(gt_img.astype(np.uint8))
                else:
                    gt_tiles.append(placeholder_rgb.copy())

            # 4.2 统一归一化深度（基于“原始 depth”的视差 1/depth；同一帧三列共享分位阈值）
            # 聚合全帧（3列）的有效原始 depth，得到稳健的分位阈值
            all_depths = []
            for d in depth_tensors:
                if d is None:
                    continue
                dv = d[0, ..., 0]
                mask = torch.isfinite(dv) & (dv > 0)
                if mask.any():
                    all_depths.append(dv[mask].cpu().numpy())
            if len(all_depths) > 0:
                all_depths_np = np.concatenate(all_depths, axis=0)
                disp_all = 1.0 / all_depths_np
                pmin, pmax = np.percentile(disp_all, (2.0, 98.0))
                if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
                    pmin, pmax = float(disp_all.min()), float(disp_all.max())
            else:
                pmin, pmax = 0.0, 1.0

            depth_tiles: List[np.ndarray] = []
            for d in depth_tensors:
                if d is None:
                    depth_tiles.append(placeholder_depth.copy())
                    continue
                dv = d[0, ..., 0].cpu().numpy()                  # 原始 depth
                valid = np.isfinite(dv) & (dv > 1e-6)
                disp = np.zeros_like(dv, dtype=np.float32)
                disp[valid] = 1.0 / dv[valid]                    # 视差：近大远小（近 -> 红）
                if pmax > pmin:
                    norm = (np.clip(disp, pmin, pmax) - pmin) / (pmax - pmin)
                else:
                    norm = np.zeros_like(disp, dtype=np.float32)
                depth_u8 = (norm * 255.0 + 0.5).astype(np.uint8)
                # Turbo 上色（OpenCV 为 BGR，需翻转为 RGB）
                depth_rgb = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)[..., ::-1]
                depth_tiles.append(depth_rgb)

            # 4.3 组装 2×3 拼图：上排 RGB, 下排 Depth；列顺序 2|1|3
            top = np.concatenate(gt_tiles, axis=1)
            mid = np.concatenate(rgb_tiles, axis=1)
            bottom = np.concatenate(depth_tiles, axis=1)
            canvas = np.concatenate([top, mid, bottom], axis=0)

            # 写帧
            writer.append_data(canvas)
            if cfg.save_frames:
                out_png = os.path.join(
                    frames_dir, f"loc_{loc}_trav_{trav}_frame_{frame_id:06d}.png"
                )
                imageio.imwrite(out_png, canvas)

            if fidx % 50 == 0 or fidx == len(frames_sorted):
                print(f"  - frame {fidx}/{len(frames_sorted)} 完成")

        writer.close()
        print(f"[Done] 保存视频：{video_path}")


# ============ CLI ============

if __name__ == "__main__":
    try:
        import tyro
    except:
        raise ImportError("请 `pip install tyro` 以使用命令行参数")
    cfg = tyro.cli(RenderConfig)
    render_multichannel(cfg)
    print(f"渲染完成，结果保存在 {cfg.result_dir}")