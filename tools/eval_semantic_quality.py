#!/usr/bin/env python3
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.graphics_utils import geom_transform_points


LEVELS = ("whole", "part", "subpart")
LEVEL_TO_LABEL = {"whole": 1, "part": 2, "subpart": 3}
LABEL_TO_LEVEL = {1: "whole", 2: "part", 3: "subpart"}


def otsu_threshold(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return 0.0
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax <= vmin:
        return vmax
    hist, bin_edges = np.histogram(vals, bins=256, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(prob)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0.0] = np.nan
    sigma_b2 = ((mu_t * omega - mu) ** 2) / denom
    idx = int(np.nanargmax(sigma_b2))
    return float(bin_centers[idx])


def load_level_mask(masks_root: Path, level: str, image_name: str, target_hw):
    h, w = target_hw
    stem = Path(image_name).stem
    candidates = [
        masks_root / level / f"{image_name}.png",
        masks_root / level / f"{stem}.png",
    ]
    for p in candidates:
        if p.exists():
            arr = np.array(Image.open(p).convert("L"))
            if arr.shape != (h, w):
                arr = np.array(Image.fromarray(arr).resize((w, h), resample=Image.NEAREST))
            return arr > 127
    return np.zeros((h, w), dtype=bool)


def build_gt_label_map(masks_by_level):
    # Priority: finer level overwrites coarser level.
    h, w = masks_by_level["whole"].shape
    label_map = np.zeros((h, w), dtype=np.uint8)
    label_map[masks_by_level["whole"]] = LEVEL_TO_LABEL["whole"]
    label_map[masks_by_level["part"]] = LEVEL_TO_LABEL["part"]
    label_map[masks_by_level["subpart"]] = LEVEL_TO_LABEL["subpart"]
    return label_map


def iou_stats(pred: np.ndarray, gt: np.ndarray):
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    iou = float(inter / union) if union > 0 else 1.0
    return inter, union, iou


def evaluate_2d_iou(scene, gaussians, pipe, bg, masks_root: Path, views, threshold_mode: str, fixed_thr: float):
    level_acc = {lvl: {"inter": 0, "union": 0, "view_iou": []} for lvl in LEVELS}
    gt_label_maps = {}

    with torch.no_grad():
        for view in views:
            h = int(view.image_height)
            w = int(view.image_width)
            gt_masks = {lvl: load_level_mask(masks_root, lvl, view.image_name, (h, w)) for lvl in LEVELS}
            gt_label_maps[view.image_name] = build_gt_label_map(gt_masks)

            for lvl in LEVELS:
                latent = render(
                    view,
                    gaussians,
                    pipe,
                    bg,
                    override_color=gaussians.get_semantic_features(lvl),
                    clamp_output=False,
                    use_trained_exp=False,
                )["render"]
                score = torch.linalg.norm(latent, dim=0).detach().cpu().numpy()
                thr = fixed_thr if threshold_mode == "fixed" else otsu_threshold(score)
                pred = score >= thr

                inter, union, iou = iou_stats(pred, gt_masks[lvl])
                level_acc[lvl]["inter"] += inter
                level_acc[lvl]["union"] += union
                level_acc[lvl]["view_iou"].append(iou)

    out = {}
    for lvl in LEVELS:
        inter = level_acc[lvl]["inter"]
        union = level_acc[lvl]["union"]
        out[lvl] = {
            "global_iou": float(inter / union) if union > 0 else 1.0,
            "mean_iou": float(np.mean(level_acc[lvl]["view_iou"])) if level_acc[lvl]["view_iou"] else 0.0,
            "num_views": len(level_acc[lvl]["view_iou"]),
        }
    return out, gt_label_maps


def evaluate_3d_label_consistency(scene, gaussians, gt_label_maps, views, min_3d_conf: float, proj_chunk: int):
    xyz = gaussians.get_xyz.detach()
    sem_norms = torch.stack(
        [
            torch.linalg.norm(gaussians.get_semantic_features("whole"), dim=1),
            torch.linalg.norm(gaussians.get_semantic_features("part"), dim=1),
            torch.linalg.norm(gaussians.get_semantic_features("subpart"), dim=1),
        ],
        dim=1,
    )
    pred_labels = torch.argmax(sem_norms, dim=1) + 1
    pred_conf = torch.max(sem_norms, dim=1).values
    valid_3d = pred_conf > float(min_3d_conf)

    totals = {
        "projected_points": 0,
        "points_on_labeled_pixels": 0,
        "label_matches": 0,
        "per_class": {lvl: {"hits": 0, "total": 0} for lvl in LEVELS},
    }

    with torch.no_grad():
        for view in views:
            h = int(view.image_height)
            w = int(view.image_width)
            gt_label = gt_label_maps[view.image_name]
            n = xyz.shape[0]

            for st in range(0, n, proj_chunk):
                ed = min(st + proj_chunk, n)
                xyz_chunk = xyz[st:ed]
                valid_chunk = valid_3d[st:ed]
                if not torch.any(valid_chunk):
                    continue

                ndc = geom_transform_points(xyz_chunk, view.full_proj_transform)
                view_pts = geom_transform_points(xyz_chunk, view.world_view_transform)
                in_front = view_pts[:, 2] > 0
                finite = torch.isfinite(ndc).all(dim=1)
                in_ndc = (
                    (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) &
                    (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
                )
                keep = valid_chunk & in_front & finite & in_ndc
                if not torch.any(keep):
                    continue

                ndc_k = ndc[keep]
                px = ((ndc_k[:, 0] + 1.0) * 0.5 * (w - 1)).round().long().cpu().numpy()
                py = ((1.0 - (ndc_k[:, 1] + 1.0) * 0.5) * (h - 1)).round().long().cpu().numpy()

                lbl3d = pred_labels[st:ed][keep].long().cpu().numpy()
                lbl2d = gt_label[py, px]
                labeled = lbl2d > 0

                totals["projected_points"] += int(lbl3d.shape[0])
                totals["points_on_labeled_pixels"] += int(labeled.sum())
                if labeled.any():
                    matches = (lbl3d[labeled] == lbl2d[labeled])
                    totals["label_matches"] += int(matches.sum())

                    for c in (1, 2, 3):
                        cls_mask = labeled & (lbl2d == c)
                        if cls_mask.any():
                            lvl = LABEL_TO_LEVEL[c]
                            totals["per_class"][lvl]["total"] += int(cls_mask.sum())
                            totals["per_class"][lvl]["hits"] += int((lbl3d[cls_mask] == c).sum())

    covered = totals["points_on_labeled_pixels"]
    consistency = float(totals["label_matches"] / covered) if covered > 0 else 0.0
    class_acc = {}
    for lvl in LEVELS:
        t = totals["per_class"][lvl]["total"]
        h = totals["per_class"][lvl]["hits"]
        class_acc[lvl] = float(h / t) if t > 0 else 0.0

    return {
        "overall_consistency": consistency,
        "points_projected": totals["projected_points"],
        "points_on_labeled_pixels": covered,
        "label_matches": totals["label_matches"],
        "labeled_coverage_ratio": float(covered / totals["projected_points"]) if totals["projected_points"] > 0 else 0.0,
        "per_class_consistency": class_acc,
    }


def main():
    parser = ArgumentParser(description="Evaluate semantic quality with 2D IoU and 3D label consistency.")
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--masks_root", type=str, required=True, help="Root folder with whole/part/subpart masks.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--threshold_mode", type=str, default="otsu", choices=["otsu", "fixed"])
    parser.add_argument("--fixed_threshold", type=float, default=0.2)
    parser.add_argument("--min_3d_conf", type=float, default=1e-4)
    parser.add_argument("--proj_chunk", type=int, default=200000)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = mp.extract(args)
    pipe = pp.extract(args)
    masks_root = Path(args.masks_root).resolve()
    if not masks_root.exists():
        raise FileNotFoundError(f"masks_root not found: {masks_root}")

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        if not gaussians.has_semantic_features:
            raise RuntimeError("Loaded point cloud has no semantic features.")

        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        views = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
        if len(views) == 0:
            raise RuntimeError(f"No views in split={args.split}. Try setting --eval for test split.")

        iou_res, gt_label_maps = evaluate_2d_iou(
            scene=scene,
            gaussians=gaussians,
            pipe=pipe,
            bg=bg,
            masks_root=masks_root,
            views=views,
            threshold_mode=args.threshold_mode,
            fixed_thr=args.fixed_threshold,
        )
        c3d_res = evaluate_3d_label_consistency(
            scene=scene,
            gaussians=gaussians,
            gt_label_maps=gt_label_maps,
            views=views,
            min_3d_conf=args.min_3d_conf,
            proj_chunk=int(args.proj_chunk),
        )

    result = {
        "model_path": dataset.model_path,
        "iteration": scene.loaded_iter,
        "split": args.split,
        "threshold_mode": args.threshold_mode,
        "fixed_threshold": args.fixed_threshold,
        "masks_root": str(masks_root),
        "iou_2d": iou_res,
        "consistency_3d": c3d_res,
    }

    out_json = Path(args.output_json) if args.output_json else Path(dataset.model_path) / f"semantic_quality_{args.split}_iter{scene.loaded_iter}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
