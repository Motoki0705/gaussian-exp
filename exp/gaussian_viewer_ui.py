#!/usr/bin/env python3
import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, RadioButtons, Slider

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix
from utils.system_utils import searchForMaxIteration


@dataclass
class RenderPipe:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False
    antialiasing: bool = False


class CameraPositionBank:
    def __init__(self, positions_xyz: np.ndarray):
        if positions_xyz.ndim != 2 or positions_xyz.shape[1] != 3:
            raise ValueError(f"positions must be [N,3], got {positions_xyz.shape}")
        self.positions = positions_xyz.astype(np.float32)

    @classmethod
    def from_model_path(cls, model_path: str | Path):
        cam_json_path = Path(model_path) / "cameras.json"
        if not cam_json_path.exists():
            raise FileNotFoundError(f"cameras.json not found: {cam_json_path}")
        with open(cam_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            raise RuntimeError(f"Invalid cameras.json: {cam_json_path}")
        pos = []
        for item in data:
            p = item.get("position", None)
            if p is None or len(p) != 3:
                continue
            pos.append([float(p[0]), float(p[1]), float(p[2])])
        if not pos:
            raise RuntimeError(f"No valid camera positions in: {cam_json_path}")
        return cls(np.asarray(pos, dtype=np.float32))

    def size(self) -> int:
        return int(self.positions.shape[0])

    def get(self, idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, self.size() - 1))
        return self.positions[idx].copy()


class GaussianSceneRenderer:
    def __init__(self, model_path: str, sh_degree: int, ply_path: str | None, iteration: int, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            raise RuntimeError("This viewer currently requires CUDA.")

        self.iteration = self._resolve_iteration(iteration)
        self.ply_path = Path(ply_path) if ply_path else self._resolve_ply_path(self.iteration)

        self.gaussians = GaussianModel(sh_degree)
        self.gaussians.load_ply(str(self.ply_path), use_train_test_exp=False)
        self.pipe = RenderPipe()
        self.bg = torch.zeros(3, dtype=torch.float32, device=self.device)

        xyz = self.gaussians.get_xyz.detach().cpu().numpy()
        self.center = xyz.mean(axis=0)
        bbox_min = xyz.min(axis=0)
        bbox_max = xyz.max(axis=0)
        self.scene_diag = float(np.linalg.norm(bbox_max - bbox_min))

    def _resolve_iteration(self, iteration: int) -> int:
        if iteration >= 0:
            return iteration
        point_cloud_dir = self.model_path / "point_cloud"
        return int(searchForMaxIteration(str(point_cloud_dir)))

    def _resolve_ply_path(self, iteration: int) -> Path:
        return self.model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"

    @staticmethod
    def _rgb_tensor_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
        x = img_chw.detach().clamp(0.0, 1.0).float().cpu().permute(1, 2, 0).numpy()
        return np.clip(x * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _gray_tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
        x = img.detach().float().cpu().numpy()
        if x.ndim == 3:
            x = x[0]
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        den = max(x_max - x_min, 1e-6)
        x = (x - x_min) / den
        x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        return np.stack([x, x, x], axis=-1)

    @staticmethod
    def _latent_tensor_to_uint8(latent_chw: torch.Tensor) -> np.ndarray:
        x = latent_chw.detach().float().cpu().numpy()  # [3,H,W]
        if x.shape[0] != 3:
            raise ValueError(f"Expected latent with 3 channels, got {x.shape}")
        flat = x.reshape(3, -1)
        x_min = flat.min(axis=1, keepdims=True)
        x_max = flat.max(axis=1, keepdims=True)
        den = np.maximum(x_max - x_min, 1e-6)
        norm = (flat - x_min) / den
        rgb = np.clip(norm.reshape(3, x.shape[1], x.shape[2]).transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        return rgb

    def render_frame(self, camera: MiniCam, mode: str) -> np.ndarray:
        pkg = render(
            viewpoint_camera=camera,
            pc=self.gaussians,
            pipe=self.pipe,
            bg_color=self.bg,
            use_trained_exp=False,
            separate_sh=False,
        )
        if mode == "rgb":
            return self._rgb_tensor_to_uint8(pkg["render"])
        if mode == "depth":
            return self._gray_tensor_to_uint8(pkg["depth"])
        if mode == "latent":
            latent = pkg.get("latent", None)
            if latent is None:
                h = int(camera.image_height)
                w = int(camera.image_width)
                return np.zeros((h, w, 3), dtype=np.uint8)
            return self._latent_tensor_to_uint8(latent)
        raise ValueError(f"Unknown mode: {mode}")


class OrbitCameraController:
    def __init__(
        self,
        width: int,
        height: int,
        fovx_deg: float,
        fovy_deg: float,
        position_xyz: np.ndarray,
        look_at_xyz: np.ndarray,
        znear: float = 0.01,
        zfar: float = 100.0,
        device: str = "cuda",
    ):
        self.width = int(width)
        self.height = int(height)
        self.fovx = math.radians(float(fovx_deg))
        self.fovy = math.radians(float(fovy_deg))
        self.znear = float(znear)
        self.zfar = float(zfar)
        self.device = device

        self.position = position_xyz.astype(np.float32)
        init_dir = look_at_xyz.astype(np.float32) - self.position
        init_dir /= max(np.linalg.norm(init_dir), 1e-8)
        self.yaw_deg, self.pitch_deg = self._dir_to_angles(init_dir)
        self._init_yaw_deg = self.yaw_deg
        self._init_pitch_deg = self.pitch_deg
        self._init_position = self.position.copy()

    @staticmethod
    def _dir_to_angles(direction: np.ndarray) -> tuple[float, float]:
        x, y, z = direction
        yaw = math.degrees(math.atan2(y, x))
        pitch = math.degrees(math.asin(np.clip(z, -1.0, 1.0)))
        return yaw, pitch

    def set_angles(self, yaw_deg: float, pitch_deg: float) -> None:
        self.yaw_deg = float(yaw_deg)
        self.pitch_deg = float(np.clip(pitch_deg, -89.0, 89.0))

    def set_position(self, position_xyz: np.ndarray) -> None:
        self.position = position_xyz.astype(np.float32)

    def reset(self) -> None:
        self.yaw_deg = float(self._init_yaw_deg)
        self.pitch_deg = float(self._init_pitch_deg)
        self.position = self._init_position.copy()

    def _direction(self) -> np.ndarray:
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        d = np.array(
            [
                math.cos(pitch) * math.cos(yaw),
                math.cos(pitch) * math.sin(yaw),
                math.sin(pitch),
            ],
            dtype=np.float32,
        )
        d /= max(np.linalg.norm(d), 1e-8)
        return d

    def _world_to_view(self) -> np.ndarray:
        eye = self.position
        forward = self._direction()
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        right = np.cross(forward, up_hint)
        if np.linalg.norm(right) < 1e-6:
            up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(forward, up_hint)
        right /= max(np.linalg.norm(right), 1e-8)
        up = np.cross(right, forward)
        up /= max(np.linalg.norm(up), 1e-8)

        r = np.stack([right, up, forward], axis=0)
        t = -r @ eye

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = r
        w2c[:3, 3] = t
        return w2c

    def to_viewpoint_camera(self) -> MiniCam:
        w2c = self._world_to_view()
        world_view_transform = torch.tensor(w2c, dtype=torch.float32, device=self.device).transpose(0, 1)
        projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.fovx, self.fovy).to(self.device).transpose(0, 1)
        full_proj_transform = world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0)).squeeze(0)

        return MiniCam(
            width=self.width,
            height=self.height,
            fovy=self.fovy,
            fovx=self.fovx,
            znear=self.znear,
            zfar=self.zfar,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform,
        )


class PointUI:
    def __init__(
        self,
        scene_renderer: GaussianSceneRenderer,
        camera_controller: OrbitCameraController,
        position_bank: CameraPositionBank,
        init_position_idx: int = 0,
    ):
        self.scene_renderer = scene_renderer
        self.camera_controller = camera_controller
        self.position_bank = position_bank
        self.init_position_idx = int(np.clip(init_position_idx, 0, self.position_bank.size() - 1))
        self.current_position_idx = self.init_position_idx
        self.mode = "rgb"

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(left=0.07, right=0.78, bottom=0.28)

        init_image = self._render_current()
        self.image_artist = self.ax.imshow(init_image)
        self.ax.set_title(self._title_text())
        self.ax.axis("off")

        self._build_widgets()

    def _title_text(self) -> str:
        return (
            f"Mode={self.mode} | "
            f"PosIdx={self.current_position_idx} | "
            f"Yaw={self.camera_controller.yaw_deg:.1f} Pitch={self.camera_controller.pitch_deg:.1f} | "
            f"Pos=({self.camera_controller.position[0]:.2f}, {self.camera_controller.position[1]:.2f}, {self.camera_controller.position[2]:.2f})"
        )

    def _build_widgets(self) -> None:
        yaw_ax = self.fig.add_axes([0.07, 0.17, 0.58, 0.03])
        pitch_ax = self.fig.add_axes([0.07, 0.12, 0.58, 0.03])
        pos_idx_ax = self.fig.add_axes([0.07, 0.07, 0.58, 0.03])
        mode_ax = self.fig.add_axes([0.80, 0.42, 0.17, 0.20])
        reset_ax = self.fig.add_axes([0.80, 0.32, 0.17, 0.06])

        self.yaw_slider = Slider(yaw_ax, "Yaw", -180.0, 180.0, valinit=self.camera_controller.yaw_deg)
        self.pitch_slider = Slider(pitch_ax, "Pitch", -89.0, 89.0, valinit=self.camera_controller.pitch_deg)
        self.position_idx_slider = Slider(
            pos_idx_ax,
            "Position Idx",
            0,
            max(self.position_bank.size() - 1, 0),
            valinit=float(self.current_position_idx),
            valstep=1,
        )

        self.mode_radio = RadioButtons(mode_ax, ("rgb", "depth", "latent"), active=0)
        self.reset_btn = Button(reset_ax, "Reset")

        self.yaw_slider.on_changed(self._on_camera_change)
        self.pitch_slider.on_changed(self._on_camera_change)
        self.position_idx_slider.on_changed(self._on_position_idx_change)
        self.mode_radio.on_clicked(self._on_mode_change)
        self.reset_btn.on_clicked(self._on_reset)

    def _render_current(self) -> np.ndarray:
        cam = self.camera_controller.to_viewpoint_camera()
        return self.scene_renderer.render_frame(cam, self.mode)

    def _on_camera_change(self, _val) -> None:
        self.camera_controller.set_angles(self.yaw_slider.val, self.pitch_slider.val)
        self._redraw()

    def _on_mode_change(self, label: str) -> None:
        self.mode = label
        self._redraw()

    def _on_position_idx_change(self, val) -> None:
        idx = int(round(float(val)))
        idx = int(np.clip(idx, 0, self.position_bank.size() - 1))
        self.current_position_idx = idx
        self.camera_controller.set_position(self.position_bank.get(idx))
        self._redraw()

    def _on_reset(self, _event) -> None:
        self.current_position_idx = self.init_position_idx
        self.camera_controller.set_position(self.position_bank.get(self.init_position_idx))
        self.position_idx_slider.set_val(self.init_position_idx)
        self.camera_controller.reset()
        self.yaw_slider.set_val(self.camera_controller.yaw_deg)
        self.pitch_slider.set_val(self.camera_controller.pitch_deg)
        self.mode = "rgb"
        self.mode_radio.set_active(0)
        self._redraw()

    def _redraw(self) -> None:
        img = self._render_current()
        self.image_artist.set_data(img)
        self.ax.set_title(self._title_text())
        self.fig.canvas.draw_idle()

    def run(self) -> None:
        plt.show()


def _parse_vec3(text: str) -> np.ndarray:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected vec3 as 'x,y,z', got: {text}")
    return np.asarray(parts, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 3DGS viewer (fixed position + rotation-only camera)")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--ply-path", type=str, default="")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--width", type=int, default=1264)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--fovx-deg", type=float, default=60.0)
    parser.add_argument("--fovy-deg", type=float, default=45.0)
    parser.add_argument("--camera-position", type=str, default="")
    parser.add_argument("--look-at", type=str, default="")
    parser.add_argument("--init-position-idx", type=int, default=0)
    args = parser.parse_args()

    renderer = GaussianSceneRenderer(
        model_path=args.model_path,
        sh_degree=args.sh_degree,
        ply_path=args.ply_path if args.ply_path else None,
        iteration=args.iteration,
    )

    center = renderer.center.astype(np.float32)
    radius = max(renderer.scene_diag * 0.6, 0.5)

    position_bank = CameraPositionBank.from_model_path(args.model_path)
    init_position_idx = int(np.clip(args.init_position_idx, 0, position_bank.size() - 1))

    if args.camera_position:
        cam_pos = _parse_vec3(args.camera_position)
    else:
        cam_pos = position_bank.get(init_position_idx)

    if args.look_at:
        look_at = _parse_vec3(args.look_at)
    else:
        look_at = center

    camera = OrbitCameraController(
        width=args.width,
        height=args.height,
        fovx_deg=args.fovx_deg,
        fovy_deg=args.fovy_deg,
        position_xyz=cam_pos,
        look_at_xyz=look_at,
        device=renderer.device,
    )

    ui = PointUI(
        scene_renderer=renderer,
        camera_controller=camera,
        position_bank=position_bank,
        init_position_idx=init_position_idx,
    )
    ui.run()


if __name__ == "__main__":
    main()
