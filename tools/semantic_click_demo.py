#!/usr/bin/env python3
import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from scene.cameras import MiniCam
from utils.general_utils import safe_state
from utils.graphics_utils import geom_transform_points


LEVELS = ("whole", "part", "subpart")


def latest_semantic_ckpt(model_path: str):
    sem_dir = os.path.join(model_path, "semantic")
    if not os.path.isdir(sem_dir):
        return None, None
    pat = re.compile(r"semantic_ckpt_(\d+)\.pth$")
    cands = []
    for name in os.listdir(sem_dir):
        m = pat.match(name)
        if m:
            cands.append((int(m.group(1)), os.path.join(sem_dir, name)))
    if not cands:
        return None, None
    cands.sort(key=lambda x: x[0])
    return cands[-1]


def ensure_semantics_loaded(dataset, scene, gaussians):
    if gaussians.has_semantic_features:
        return scene, gaussians

    ckpt_iter, ckpt_path = latest_semantic_ckpt(dataset.model_path)
    if ckpt_path is None:
        raise RuntimeError("No semantic tensors found in point cloud or semantic/*.pth")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sem_whole = ckpt.get("sem_whole")
    sem_part = ckpt.get("sem_part")
    sem_subpart = ckpt.get("sem_subpart")
    if sem_whole is None or sem_part is None or sem_subpart is None:
        raise RuntimeError(f"Invalid semantic checkpoint: {ckpt_path}")

    n_points = gaussians.get_xyz.shape[0]
    if sem_whole.shape[0] != n_points:
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=ckpt_iter, shuffle=False)
        n_points = gaussians.get_xyz.shape[0]
        if sem_whole.shape[0] != n_points:
            raise RuntimeError("Semantic checkpoint does not match any loaded point cloud point count.")

    gaussians._sem_whole = torch.nn.Parameter(sem_whole.to("cuda").float(), requires_grad=False)
    gaussians._sem_part = torch.nn.Parameter(sem_part.to("cuda").float(), requires_grad=False)
    gaussians._sem_subpart = torch.nn.Parameter(sem_subpart.to("cuda").float(), requires_grad=False)
    print(f"[semantic_click_demo] Loaded semantic tensors from {ckpt_path}")
    return scene, gaussians


def make_row_rotation(yaw_rad: float, pitch_rad: float, device="cuda"):
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)

    # Column-vector rotations.
    ry_col = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx_col = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    r_col = ry_col @ rx_col
    r_row = torch.from_numpy(r_col.T).to(device=device)

    out = torch.eye(4, dtype=torch.float32, device=device)
    out[:3, :3] = r_row
    return out


def build_virtual_camera(base_cam, yaw_rad: float, pitch_rad: float):
    rot_row = make_row_rotation(yaw_rad, pitch_rad, device=base_cam.world_view_transform.device)
    wv = base_cam.world_view_transform @ rot_row
    full_proj = wv @ base_cam.projection_matrix
    return MiniCam(
        width=base_cam.image_width,
        height=base_cam.image_height,
        fovy=base_cam.FoVy,
        fovx=base_cam.FoVx,
        znear=base_cam.znear,
        zfar=base_cam.zfar,
        world_view_transform=wv,
        full_proj_transform=full_proj,
    )


def torch_rgb_to_bgr_u8(img_chw: torch.Tensor):
    x = img_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    x = (x * 255.0).astype(np.uint8)
    return x[:, :, ::-1].copy()


def semantic_mask_from_click(latent_chw: torch.Tensor, x: int, y: int, cos_thresh: float, score_ratio: float):
    c, h, w = latent_chw.shape
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    feat = latent_chw.permute(1, 2, 0).detach().cpu().numpy()
    seed = feat[y, x]
    seed_norm = np.linalg.norm(seed) + 1e-8
    feat_norm = np.linalg.norm(feat, axis=2) + 1e-8
    cos = (feat @ seed) / (feat_norm * seed_norm)
    score = feat_norm
    seed_score = score[y, x]

    cand = (cos >= float(cos_thresh)) & (score >= float(seed_score) * float(score_ratio))
    if not cand[y, x]:
        cand[y, x] = True

    # Keep only connected component containing the clicked pixel.
    num_labels, labels = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return cand
    target = labels[y, x]
    return labels == target


def project_points_to_pixels(xyz: torch.Tensor, cam):
    # xyz: [N,3] row-vector convention in this codebase
    ndc = geom_transform_points(xyz, cam.full_proj_transform)
    view_pts = geom_transform_points(xyz, cam.world_view_transform)
    h = int(cam.image_height)
    w = int(cam.image_width)

    finite = torch.isfinite(ndc).all(dim=1)
    in_ndc = (
        (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) &
        (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
    )
    in_front = view_pts[:, 2] > 0
    keep = finite & in_ndc & in_front

    px = ((ndc[:, 0] + 1.0) * 0.5 * (w - 1)).round().long()
    py = ((1.0 - (ndc[:, 1] + 1.0) * 0.5) * (h - 1)).round().long()
    px = torch.clamp(px, 0, w - 1)
    py = torch.clamp(py, 0, h - 1)
    return px, py, view_pts[:, 2], keep


def build_selected_gaussians_from_click(
    click_xy,
    latent_chw: torch.Tensor,
    cam,
    gaussians,
    radii: torch.Tensor,
    level: str,
    cos_thresh: float,
    score_ratio: float,
    seed_radius_px: float,
    min_opacity: float,
    max_track_points: int,
):
    # 1) 2D mask from semantic latent around click.
    click_mask = semantic_mask_from_click(latent_chw, click_xy[0], click_xy[1], cos_thresh, score_ratio)
    h, w = click_mask.shape

    # 2) Project all gaussians and keep those inside click mask + visible.
    xyz = gaussians.get_xyz.detach()
    px, py, _z, keep = project_points_to_pixels(xyz, cam)

    valid = keep & (radii > 0) & (gaussians.get_opacity.squeeze(1) >= float(min_opacity))
    mask_hit = torch.from_numpy(click_mask.astype(np.uint8)).to(device=xyz.device)[py, px] > 0
    selected = valid & mask_hit

    # 3) If too few points, fallback to local screen-neighborhood seeds.
    if int(selected.sum().item()) < 64:
        dx = px.float() - float(click_xy[0])
        dy = py.float() - float(click_xy[1])
        d2 = dx * dx + dy * dy
        near = valid & (d2 <= float(seed_radius_px) * float(seed_radius_px))
        selected = near

    idx = selected.nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=xyz.device), click_mask

    # 4) Semantic refinement by prototype similarity.
    feat = gaussians.get_semantic_features(level).detach()
    feat_sel = feat[idx]
    proto = feat_sel.mean(dim=0, keepdim=True)
    proto = proto / (proto.norm(dim=1, keepdim=True) + 1e-8)
    feat_n = feat / (feat.norm(dim=1, keepdim=True) + 1e-8)
    sim = torch.sum(feat_n * proto, dim=1)

    refined = selected & (sim >= float(cos_thresh))
    ridx = refined.nonzero(as_tuple=False).squeeze(1)
    if ridx.numel() == 0:
        ridx = idx

    # 5) Limit tracked point count by confidence (semantic sim * opacity).
    if ridx.numel() > int(max_track_points):
        score = sim[ridx] * gaussians.get_opacity.squeeze(1)[ridx]
        topk = torch.topk(score, k=int(max_track_points), largest=True).indices
        ridx = ridx[topk]

    return ridx, click_mask


def overlay_mask(bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.45):
    out = bgr.copy()
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    m = mask.astype(bool)
    out_f = out.astype(np.float32)
    out_f[m] = (1.0 - alpha) * out_f[m] + alpha * color_arr
    return out_f.astype(np.uint8)


def render_frame(cam, gaussians, pipe, bg, level: str):
    rgb = render(cam, gaussians, pipe, bg, clamp_output=True, use_trained_exp=False)["render"]
    latent = render(
        cam,
        gaussians,
        pipe,
        bg,
        override_color=gaussians.get_semantic_features(level),
        clamp_output=False,
        use_trained_exp=False,
    )["render"]
    return rgb, latent


def selected_ids_to_mask(selected_ids, cam, gaussians, radii, max_disk_radius: int):
    h = int(cam.image_height)
    w = int(cam.image_width)
    if selected_ids is None or selected_ids.numel() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    xyz_sel = gaussians.get_xyz.detach()[selected_ids]
    px, py, _z, keep = project_points_to_pixels(xyz_sel, cam)
    rad = radii[selected_ids].detach()

    px = px[keep].cpu().numpy()
    py = py[keep].cpu().numpy()
    rr = rad[keep].cpu().numpy()

    mask = np.zeros((h, w), dtype=np.uint8)
    for x, y, r in zip(px, py, rr):
        rad_i = int(np.clip(np.ceil(float(r)), 1, int(max_disk_radius)))
        cv2.circle(mask, (int(x), int(y)), rad_i, 255, -1)
    return mask


def draw_hud(img, level, yaw_deg, pitch_deg, n_selected):
    text = (
        f"level={level} | yaw={yaw_deg:.1f} pitch={pitch_deg:.1f} | selected={n_selected} | "
        "WASD rotate | 1/2/3 level | click select | r clear | q quit"
    )
    cv2.putText(img, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main():
    parser = ArgumentParser(description="Interactive semantic click demo with orientation-only camera control.")
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--base_index", type=int, default=0)
    parser.add_argument("--level", choices=list(LEVELS), default="part")
    parser.add_argument("--yaw_step_deg", type=float, default=3.0)
    parser.add_argument("--pitch_step_deg", type=float, default=3.0)
    parser.add_argument("--cos_thresh", type=float, default=0.9)
    parser.add_argument("--score_ratio", type=float, default=0.5)
    parser.add_argument("--seed_radius_px", type=float, default=10.0)
    parser.add_argument("--min_opacity", type=float, default=0.01)
    parser.add_argument("--max_track_points", type=int, default=25000)
    parser.add_argument("--max_disk_radius", type=int, default=10)
    parser.add_argument("--window_name", type=str, default="semantic-click-demo")
    parser.add_argument("--headless_frames", type=int, default=0, help="If >0, run headless and save N frames.")
    parser.add_argument("--save_dir", type=str, default="output/semantic_click_demo")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = mp.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    scene, gaussians = ensure_semantics_loaded(dataset, scene, gaussians)
    views = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
    if len(views) == 0:
        raise RuntimeError(f"No cameras in split={args.split}.")
    base_idx = int(np.clip(args.base_index, 0, len(views) - 1))
    base_cam = views[base_idx]

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    level = args.level

    yaw_deg = 0.0
    pitch_deg = 0.0
    click_xy = None

    if args.headless_frames > 0:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        selected_ids = None
        for i in range(args.headless_frames):
            yaw = np.deg2rad(yaw_deg)
            pitch = np.deg2rad(pitch_deg)
            cam = build_virtual_camera(base_cam, yaw, pitch)
            out_rgb = render(cam, gaussians, pipe, bg, clamp_output=True, use_trained_exp=False)
            rgb = out_rgb["render"]
            radii = out_rgb["radii"]
            latent = render(
                cam,
                gaussians,
                pipe,
                bg,
                override_color=gaussians.get_semantic_features(level),
                clamp_output=False,
                use_trained_exp=False,
            )["render"]
            bgr = torch_rgb_to_bgr_u8(rgb)

            if click_xy is None:
                click_xy = (bgr.shape[1] // 2, bgr.shape[0] // 2)
                selected_ids, _ = build_selected_gaussians_from_click(
                    click_xy=click_xy,
                    latent_chw=latent,
                    cam=cam,
                    gaussians=gaussians,
                    radii=radii,
                    level=level,
                    cos_thresh=args.cos_thresh,
                    score_ratio=args.score_ratio,
                    seed_radius_px=args.seed_radius_px,
                    min_opacity=args.min_opacity,
                    max_track_points=args.max_track_points,
                )

            mask = selected_ids_to_mask(selected_ids, cam, gaussians, radii, args.max_disk_radius)
            vis = overlay_mask(bgr, mask > 0)
            vis = draw_hud(vis, level, yaw_deg, pitch_deg, int(selected_ids.numel()) if selected_ids is not None else 0)
            cv2.imwrite(str(save_dir / f"frame_{i:03d}.png"), vis)
            yaw_deg += args.yaw_step_deg
        print(f"Saved headless demo frames to {save_dir}")
        return

    state = {"click_xy": None, "selected_ids": None}

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click_xy"] = (int(x), int(y))

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(args.window_name, on_mouse)

    while True:
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        cam = build_virtual_camera(base_cam, yaw, pitch)
        out_rgb = render(cam, gaussians, pipe, bg, clamp_output=True, use_trained_exp=False)
        rgb = out_rgb["render"]
        radii = out_rgb["radii"]
        latent = render(
            cam,
            gaussians,
            pipe,
            bg,
            override_color=gaussians.get_semantic_features(level),
            clamp_output=False,
            use_trained_exp=False,
        )["render"]
        bgr = torch_rgb_to_bgr_u8(rgb)

        if state["click_xy"] is not None:
            state["selected_ids"], _ = build_selected_gaussians_from_click(
                click_xy=state["click_xy"],
                latent_chw=latent,
                cam=cam,
                gaussians=gaussians,
                radii=radii,
                level=level,
                cos_thresh=args.cos_thresh,
                score_ratio=args.score_ratio,
                seed_radius_px=args.seed_radius_px,
                min_opacity=args.min_opacity,
                max_track_points=args.max_track_points,
            )
            state["click_xy"] = None

        if state["selected_ids"] is not None and state["selected_ids"].numel() > 0:
            mask = selected_ids_to_mask(state["selected_ids"], cam, gaussians, radii, args.max_disk_radius)
            bgr = overlay_mask(bgr, mask > 0)

        bgr = draw_hud(
            bgr,
            level,
            yaw_deg,
            pitch_deg,
            int(state["selected_ids"].numel()) if state["selected_ids"] is not None else 0,
        )
        cv2.imshow(args.window_name, bgr)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("a"):
            yaw_deg -= args.yaw_step_deg
        elif key == ord("d"):
            yaw_deg += args.yaw_step_deg
        elif key == ord("w"):
            pitch_deg -= args.pitch_step_deg
        elif key == ord("s"):
            pitch_deg += args.pitch_step_deg
        elif key == ord("1"):
            level = "whole"
        elif key == ord("2"):
            level = "part"
        elif key == ord("3"):
            level = "subpart"
        elif key == ord("r"):
            state["click_xy"] = None
            state["selected_ids"] = None

        pitch_deg = float(np.clip(pitch_deg, -85.0, 85.0))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
