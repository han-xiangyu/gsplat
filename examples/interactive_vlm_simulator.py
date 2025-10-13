import base64
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from datasets_utils.colmap import Parser
from openai import OpenAI

from render_from_ply import load_splats_from_ply, pick_K_and_size


def _deg2rad(deg: Sequence[float]) -> np.ndarray:
    return np.deg2rad(np.asarray(deg, dtype=np.float32))


def yaw_pitch_roll_to_matrix(deg: Sequence[float]) -> np.ndarray:
    """Returns rotation matrix from yaw(Z), pitch(Y), roll(X) in degrees."""
    yaw, pitch, roll = _deg2rad(deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


@dataclass
class Pose:
    translation: np.ndarray
    rotation_deg: np.ndarray

    @classmethod
    def from_lists(cls, translation: Sequence[float], rotation_deg: Sequence[float]) -> "Pose":
        return cls(
            translation=np.asarray(translation, dtype=np.float32),
            rotation_deg=np.asarray(rotation_deg, dtype=np.float32),
        )

    def as_matrix(self) -> np.ndarray:
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = yaw_pitch_roll_to_matrix(self.rotation_deg)
        mat[:3, 3] = self.translation
        return mat

    def describe(self) -> str:
        t = ", ".join(f"{v:.3f}" for v in self.translation)
        r = ", ".join(f"{v:.2f}" for v in self.rotation_deg)
        return f"translation[{t}] / rotation_deg[{r}]"


@dataclass
class StepLog:
    index: int
    pose_before: Pose
    reasoning: str
    action: Optional[str]
    pose_after: Pose
    done: bool

    def as_text(self) -> str:
        act = self.action if self.action else "unspecified"
        return (
            f"Step {self.index}: pose {self.pose_before.describe()} -> {self.pose_after.describe()} | "
            f"action={act} | reasoning={self.reasoning}"
        )


@dataclass
class SimulatorConfig:
    data_dir: str
    ply_path: str
    task: str
    result_dir: str = "results/interactive_vlm"
    model: str = "gpt-4.1-mini"
    api_base: Optional[str] = None
    initial_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_rotation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_steps: int = 12
    temperature: float = 0.2
    top_p: float = 1.0
    save_frames: bool = True
    camera_model: str = "pinhole"
    device: str = "cuda"
    near_plane: float = 0.01
    far_plane: float = 1000.0


class StreetRenderer:
    def __init__(self, cfg: SimulatorConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        parser = Parser(data_dir=cfg.data_dir, factor=1, normalize=False, test_every=60)
        self.parser = parser
        self.K_np, (self.width, self.height) = pick_K_and_size(parser)
        self.K_tensor = torch.from_numpy(self.K_np).float().to(self.device)[None]
        self.splats, self.meta = load_splats_from_ply(cfg.ply_path, device=self.device)
        self.sh_degree = math.isqrt(1 + self.splats["shN"].shape[1]) - 1
        self.means = self.splats["means"]
        self.quats = self.splats["quats"]
        self.scales = torch.exp(self.splats["scales"]) if self.meta["scale_is_log"] else self.splats["scales"]
        self.opacities = (
            torch.sigmoid(self.splats["opacities"]) if self.meta["opacity_is_logit"] else self.splats["opacities"]
        )
        self.colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)

    def render_rgb(self, pose: Pose) -> np.ndarray:
        from gsplat.rendering import rasterization

        c2w = torch.from_numpy(pose.as_matrix()[None, ...]).to(self.device)
        renders, _, _ = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            viewmats=torch.linalg.inv(c2w),
            Ks=self.K_tensor,
            width=self.width,
            height=self.height,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode="classic",
            distributed=False,
            camera_model=self.cfg.camera_model,
            with_ut=False,
            with_eval3d=False,
            render_mode="RGB",
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            sh_degree=self.sh_degree,
        )
        rgb = torch.clamp(renders[0, ..., :3], 0.0, 1.0).cpu().numpy()
        return (rgb * 255.0).astype(np.uint8)


class VLMNavigator:
    def __init__(self, cfg: SimulatorConfig):
        client_kwargs = {}
        if cfg.api_base:
            client_kwargs["base_url"] = cfg.api_base
        self.client = OpenAI(**client_kwargs)
        self.cfg = cfg
        self.system_prompt = (
            "You are an embodied agent navigating a city-scale 3D Gaussian splat simulator. "
            "You must complete the user task by selecting camera poses. "
            "Each step you will receive the current RGB observation, your current pose, and the history. "
            "Reply strictly in JSON with keys: reasoning (string), action (string), done (boolean), "
            "next_pose (object with translation [x,y,z] in meters and rotation_deg [yaw,pitch,roll]). "
            "Return done=true only when the task is complete or impossible."
        )

    def decide(
        self,
        image_b64: str,
        pose: Pose,
        history: List[StepLog],
        step_index: int,
        task: str,
    ) -> StepLog:
        history_text = "\n".join(log.as_text() for log in history) if history else "None yet."
        user_text = (
            f"Task: {task}\n"
            f"Step: {step_index}\n"
            f"Current pose: {pose.describe()}\n"
            f"History:\n{history_text}\n"
            "Respond with a JSON object."
        )
        response = self.client.responses.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            input=[
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "input_image", "image_base64": image_b64},
                    ],
                },
            ],
        )
        reply_text = ""
        for output in response.output:
            if output.type == "output_text":
                for item in output.content:
                    reply_text += item.text
        data = json.loads(reply_text)
        reasoning = data.get("reasoning", "")
        action = data.get("action")
        done = bool(data.get("done", False))
        next_pose_data = data.get("next_pose")
        if not next_pose_data:
            raise ValueError("Model response missing 'next_pose'.")
        next_translation = next_pose_data.get("translation")
        next_rotation = next_pose_data.get("rotation_deg")
        if next_translation is None or next_rotation is None:
            raise ValueError("Model response missing pose fields.")
        next_pose = Pose.from_lists(next_translation, next_rotation)
        return StepLog(
            index=step_index,
            pose_before=pose,
            reasoning=reasoning,
            action=action,
            pose_after=next_pose,
            done=done,
        )


def encode_png_base64(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Failed to encode rendered image.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def run_simulation(cfg: SimulatorConfig) -> List[StepLog]:
    renderer = StreetRenderer(cfg)
    navigator = VLMNavigator(cfg)
    history: List[StepLog] = []
    current_pose = Pose.from_lists(cfg.initial_translation, cfg.initial_rotation_deg)
    for step_index in range(cfg.max_steps):
        frame = renderer.render_rgb(current_pose)
        frame_b64 = encode_png_base64(frame)
        step_log = navigator.decide(
            image_b64=frame_b64,
            pose=current_pose,
            history=history,
            step_index=step_index,
            task=cfg.task,
        )
        history.append(step_log)
        current_pose = step_log.pose_after
        if cfg.save_frames:
            _maybe_save_frame(cfg, step_index, frame)
        if step_log.done:
            break
    return history


def _maybe_save_frame(cfg: SimulatorConfig, step_index: int, frame: np.ndarray) -> None:
    import pathlib

    out_dir = pathlib.Path(cfg.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step_{step_index:03d}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _format_summary(history: List[StepLog]) -> str:
    lines = ["Simulation summary:"]
    for log in history:
        status = "DONE" if log.done else "CONTINUE"
        lines.append(f"{log.index:02d} [{status}] {log.as_text()}")
    if history:
        final_pose = history[-1].pose_after.describe()
        lines.append(f"Final pose: {final_pose}")
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        import tyro
    except ImportError as exc:
        raise ImportError("Please install tyro (pip install tyro) to run CLI entrypoints.") from exc

    cfg = tyro.cli(SimulatorConfig)
    logs = run_simulation(cfg)
    print(_format_summary(logs))
