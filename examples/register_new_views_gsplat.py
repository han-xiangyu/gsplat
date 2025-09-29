# register_new_views_from_parser.py
# Register new camera views into a NEW COLMAP sparse directory,
# using the same IO/parsing style as your "render_from_ply" pipeline.
import os, re, math, shutil
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Literal
from argparse import ArgumentParser
import torch
from tqdm import tqdm
import tyro


# ==== 与 render_from_ply 保持一致的命名提取 ====
PAT = re.compile(
    r"(?:loc_(?P<loc>\d+)_)?trav_(?P<trav>\d+)_channel_(?P<ch>\d+)_img_(?P<frame>\d+)\.(?:png|jpg|jpeg)$",
    re.IGNORECASE,
)

def parse_meta(name: str):
    m = PAT.search(name)
    if not m:
        return None
    loc = m.group("loc")
    return {
        "loc": int(loc) if loc is not None else 0,
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



def rotmat_to_qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

@dataclass
class Args:
    data_dir: str
    output_sparse_dir_name: str = "new_sparse"
    traj_type: Literal["parallel", "horizontal_sine"] = "parallel"
    amplitude: float = 1.5
    left_suffix: str = "_left_difix3d"
    right_suffix: str = "_right_difix3d"
    sine_suffix: str = "_sine"

@dataclass
class BaseCam:
    idx: int                # 在 Parser 里的索引
    image_name: str         # 原 basename
    camera_id: int          # COLMAP cameras.txt 的 camera_id
    R_c2w: np.ndarray       # 3x3
    C: np.ndarray           # 3, camera center in world


def _with_suffix(name: str, suffix: str) -> str:
    stem, ext = os.path.splitext(name)
    return f"{stem}{suffix}{ext if ext else '.jpg'}"

def register_to_new_sparse(data_dir: str, new_sparse_dir: str,
                           image_entries: List[Tuple[str, int, np.ndarray, np.ndarray]]):
    """
    image_entries: list of (image_name, camera_id, R_w2c(3x3), t_w2c(3,))
    """
    original_sparse = os.path.join(data_dir, "sparse", "0")
    new_sparse_path = os.path.join(data_dir, new_sparse_dir, "0")
    os.makedirs(new_sparse_path, exist_ok=True)
    print(f"🚀 Creating new sparse model at: {new_sparse_path}")

    # 1) 复制 cameras.txt / points3D.txt
    for fname in ["cameras.txt", "points3D.txt"]:
        src = os.path.join(original_sparse, fname)
        if os.path.exists(src):
            shutil.copy2(src, new_sparse_path)
            print(f"✅ Copied {fname}")
        else:
            print(f"⚠️ {fname} not found in {original_sparse} (skipped)")

    # 2) 读取原 images.txt，确定最后的 image_id
    orig_images = os.path.join(original_sparse, "images.txt")
    new_images  = os.path.join(new_sparse_path, "images.txt")
    if not os.path.exists(orig_images):
        raise FileNotFoundError(f"Original images.txt not found at {orig_images}")

    with open(orig_images, "r") as f:
        lines = f.readlines()

    with open(new_images, "w") as f:
        f.writelines(lines)
        last_id = 0
        # COLMAP images.txt: 每两行一组，第1行以 image_id 开头
        for i in range(4, len(lines), 2):
            parts = lines[i].split()
            if parts:
                try:
                    last_id = max(last_id, int(parts[0]))
                except:
                    pass
        print(f"✅ Last original IMAGE_ID is {last_id}. New start from {last_id + 1}.")

        # 3) 追加新图像
        for k, (img_name, cam_id, R_w2c, t_w2c) in enumerate(tqdm(image_entries, desc="Appending new views")):
            new_id = last_id + 1 + k
            qvec = rotmat_to_qvec(R_w2c)
            q_str = " ".join(map(lambda x: f"{x:.10f}", qvec.tolist()))
            t_str = " ".join(map(lambda x: f"{x:.10f}", t_w2c.tolist()))
            # 确保换行
            if len(lines) > 0 and not lines[-1].endswith("\n"):
                f.write("\n")
            f.write(f"{new_id} {q_str} {t_str} {cam_id} {img_name}\n")
            f.write("\n")  # 第二行（2D-3D 对应关系）留空
    print(f"🎉 Successfully wrote {len(image_entries)} new entries into {new_images}")



# ========== 主流程 ==========
def main(args: Args):
    # 1) 用 datasets.colmap.Parser 读取（与你 render_from_ply 同步）
    from datasets_utils.colmap import Parser
    parser = Parser(data_dir=args.data_dir, factor=1, normalize=False, test_every=60)

    # 获得 basename 列表
    image_names = None
    for cand in ["image_paths", "image_names", "filenames", "images"]:
        if hasattr(parser, cand):
            arr = getattr(parser, cand)
            if isinstance(arr, (list, tuple)) and len(arr) > 0:
                image_names = [os.path.basename(str(x)) for x in arr]
                break
    if image_names is None:
        img_dir = Path(args.data_dir) / "images"
        if not img_dir.exists():
            raise RuntimeError("无法从 Parser 获取图像文件名，且未找到 data_dir/images 兜底目录。")
        image_names = sorted([p.name for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    camtoworlds_np = ensure_4x4(parser.camtoworlds).astype(np.float32)  # [N,4,4]
    camera_ids = parser.camera_ids                                      # len N
    N = camtoworlds_np.shape[0]
    assert len(image_names) == N == len(camera_ids)

    # 2) 基础相机集合
    base_cams: List[BaseCam] = []
    for i in range(N):
        c2w = camtoworlds_np[i]
        R = c2w[:3, :3]
        C = c2w[:3, 3]
        base_cams.append(BaseCam(
            idx=i,
            image_name=image_names[i],
            camera_id=int(camera_ids[i]),
            R_c2w=R.copy(),
            C=C.copy()
        ))

    # 3) 生成新位姿
    new_entries = []  # 记录：[(image_name, camera_id, R_w2c, t_w2c)]
    if args.traj_type == "parallel":
        for cam in tqdm(base_cams, desc="Generating parallel views"):
            right = cam.R_c2w[:, 0]
            leftC  = cam.C - args.amplitude * right
            rightC = cam.C + args.amplitude * right
            R_w2c = cam.R_c2w.T
            t_left  = -R_w2c @ leftC
            t_right = -R_w2c @ rightC

            left_name  = _with_suffix(cam.image_name, args.left_suffix)
            right_name = _with_suffix(cam.image_name, args.right_suffix)

            new_entries.append((left_name,  cam.camera_id, R_w2c.copy(), t_left.copy()))
            new_entries.append((right_name, cam.camera_id, R_w2c.copy(), t_right.copy()))

    elif args.traj_type == "horizontal_sine":
        period = 60
        amp = args.amplitude
        for i, cam in tqdm(list(enumerate(base_cams)), desc="Generating sine views"):
            right = cam.R_c2w[:, 0]
            shift = amp * math.sin(2.0 * math.pi * (i / period))
            C_new = cam.C + shift * right
            R_w2c = cam.R_c2w.T
            t_new = -R_w2c @ C_new
            sine_name = _with_suffix(cam.image_name, args.sine_suffix)
            new_entries.append((sine_name, cam.camera_id, R_w2c.copy(), t_new.copy()))
    else:
        raise ValueError(f"Unknown traj_type: {args.traj_type}")

    # 4) 写入新的 sparse 目录
    register_to_new_sparse(
        data_dir=args.data_dir,
        new_sparse_dir=args.output_sparse_dir_name,
        image_entries=new_entries
    )
    print("✨ All done.")

if __name__ == "__main__":

    args = tyro.cli(Args)

    main(args)
