#!/usr/bin/env python3
"""Interactive viewer for offline NMS tuning from stored SAM3 candidates."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from PIL import Image


LEVELS = ("whole", "part", "subpart")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune NMS/min_area from stored SAM3 candidates.")
    parser.add_argument("--storage-dir", type=Path, default=Path("exp/sam3_hier_tmp"))
    parser.add_argument("--original-image", type=Path, default=Path("exp/sam3_hier_demo/original.png"))
    parser.add_argument("--init-nms", type=float, default=0.9)
    parser.add_argument("--init-min-area", type=int, default=64)
    return parser.parse_args()


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def nms_by_iou(masks: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    if masks.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores)
    keep = []
    for idx in order:
        duplicate = False
        for kept_idx in keep:
            if mask_iou(masks[idx], masks[kept_idx]) >= iou_thresh:
                duplicate = True
                break
        if not duplicate:
            keep.append(int(idx))
    return np.array(keep, dtype=np.int64)


def compose_instance_map(kept_masks: np.ndarray, kept_scores: np.ndarray) -> np.ndarray:
    if kept_masks.shape[0] == 0:
        # fallback shape inferred by caller; caller avoids this path with template shape
        raise ValueError("compose_instance_map requires at least one mask")

    _, h, w = kept_masks.shape
    out = np.zeros((h, w), dtype=np.int32)
    best_score = np.full((h, w), -np.inf, dtype=np.float32)

    order = np.argsort(-kept_scores)
    for rank, idx in enumerate(order, start=1):
        mask = kept_masks[idx]
        score = float(kept_scores[idx])
        update = np.logical_and(mask, score > best_score)
        out[update] = rank
        best_score[update] = score
    return out


def colorize_instance_map(instance_map: np.ndarray) -> np.ndarray:
    h, w = instance_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for idx in np.unique(instance_map):
        i = int(idx)
        if i <= 0:
            continue
        colored[instance_map == i] = np.array(
            [
                (37 * i + 17) % 256,
                (73 * i + 29) % 256,
                (109 * i + 53) % 256,
            ],
            dtype=np.uint8,
        )
    return colored


class StoredNMSViewer:
    def __init__(self, storage_dir: Path, original_image: Path, init_nms: float, init_min_area: int) -> None:
        self.storage_dir = Path(storage_dir).resolve()
        self.original_image = Path(original_image).resolve()
        self.init_nms = float(init_nms)
        self.init_min_area = int(init_min_area)

        if not self.storage_dir.exists():
            raise FileNotFoundError(f"Storage dir not found: {self.storage_dir}")
        if not self.original_image.exists():
            raise FileNotFoundError(f"Original image not found: {self.original_image}")

        self.image_np = np.array(Image.open(self.original_image).convert("RGB"))
        self.h, self.w = self.image_np.shape[:2]

        self.level_masks = {}
        self.level_scores = {}
        self.level_areas = {}
        self._load_candidates()

        self.fig = None
        self.ax = {}
        self.img_artists = {}
        self.info_text = None
        self.slider_nms = None
        self.slider_min_area = None

    def _load_candidates(self) -> None:
        for level in LEVELS:
            level_dir = self.storage_dir / level
            files = sorted(level_dir.glob("chunk_*.npz"))

            if not files:
                self.level_masks[level] = np.zeros((0, self.h, self.w), dtype=bool)
                self.level_scores[level] = np.zeros((0,), dtype=np.float32)
                self.level_areas[level] = np.zeros((0,), dtype=np.int32)
                continue

            masks_list = []
            scores_list = []
            for f in files:
                data = np.load(f)
                masks = np.asarray(data["masks"]).astype(bool)
                ious = np.asarray(data["ious"]).astype(np.float32)
                if masks.ndim != 3 or ious.ndim != 1 or masks.shape[0] != ious.shape[0]:
                    raise ValueError(f"Invalid chunk file: {f}")
                masks_list.append(masks)
                scores_list.append(ious)

            all_masks = np.concatenate(masks_list, axis=0)
            all_scores = np.concatenate(scores_list, axis=0)
            all_areas = all_masks.reshape(all_masks.shape[0], -1).sum(axis=1).astype(np.int32)

            self.level_masks[level] = all_masks
            self.level_scores[level] = all_scores
            self.level_areas[level] = all_areas

    def _compute_level_map(self, level: str, nms_iou_thresh: float, min_mask_area: int):
        masks = self.level_masks[level]
        scores = self.level_scores[level]
        areas = self.level_areas[level]

        if masks.shape[0] == 0:
            return np.zeros((self.h, self.w), dtype=np.int32), 0, 0

        keep_area = areas >= int(min_mask_area)
        cand_masks = masks[keep_area]
        cand_scores = scores[keep_area]

        if cand_masks.shape[0] == 0:
            return np.zeros((self.h, self.w), dtype=np.int32), 0, 0

        keep_idx = nms_by_iou(cand_masks, cand_scores, float(nms_iou_thresh))
        final_masks = cand_masks[keep_idx]
        final_scores = cand_scores[keep_idx]

        if final_masks.shape[0] == 0:
            return np.zeros((self.h, self.w), dtype=np.int32), int(cand_masks.shape[0]), 0

        inst = compose_instance_map(final_masks, final_scores)
        return inst, int(cand_masks.shape[0]), int(final_masks.shape[0])

    def _refresh(self, _val=None) -> None:
        nms = float(self.slider_nms.val)
        min_area = int(round(self.slider_min_area.val))

        lines = [f"nms_iou_thresh={nms:.3f}, min_mask_area={min_area}"]

        for level in LEVELS:
            inst, n_cand, n_final = self._compute_level_map(level, nms, min_area)
            colored = colorize_instance_map(inst)
            self.img_artists[level].set_data(colored)
            lines.append(f"{level}: candidates={n_cand}, kept={n_final}")

        self.info_text.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    def run(self) -> None:
        self.fig = plt.figure(figsize=(14, 9))

        self.ax["original"] = self.fig.add_axes([0.04, 0.52, 0.42, 0.44])
        self.ax["whole"] = self.fig.add_axes([0.54, 0.72, 0.42, 0.24])
        self.ax["part"] = self.fig.add_axes([0.54, 0.42, 0.42, 0.24])
        self.ax["subpart"] = self.fig.add_axes([0.54, 0.12, 0.42, 0.24])

        self.ax["original"].set_title("Original")
        self.ax["whole"].set_title("Whole")
        self.ax["part"].set_title("Part")
        self.ax["subpart"].set_title("Subpart")

        for k in ("original", "whole", "part", "subpart"):
            self.ax[k].axis("off")

        self.img_artists["original"] = self.ax["original"].imshow(self.image_np)
        self.img_artists["whole"] = self.ax["whole"].imshow(np.zeros((self.h, self.w, 3), dtype=np.uint8))
        self.img_artists["part"] = self.ax["part"].imshow(np.zeros((self.h, self.w, 3), dtype=np.uint8))
        self.img_artists["subpart"] = self.ax["subpart"].imshow(np.zeros((self.h, self.w, 3), dtype=np.uint8))

        ax_s_nms = self.fig.add_axes([0.08, 0.05, 0.36, 0.03])
        ax_s_area = self.fig.add_axes([0.08, 0.01, 0.36, 0.03])
        self.slider_nms = Slider(ax=ax_s_nms, label="nms_iou_thresh", valmin=0.5, valmax=0.99, valinit=self.init_nms)
        self.slider_min_area = Slider(ax=ax_s_area, label="min_mask_area", valmin=0, valmax=5000, valinit=self.init_min_area, valstep=1)

        self.info_text = self.fig.text(0.04, 0.43, "", fontsize=10, va="top")

        self.slider_nms.on_changed(self._refresh)
        self.slider_min_area.on_changed(self._refresh)
        self._refresh()
        plt.show()


def main() -> None:
    args = parse_args()
    viewer = StoredNMSViewer(
        storage_dir=args.storage_dir,
        original_image=args.original_image,
        init_nms=args.init_nms,
        init_min_area=args.init_min_area,
    )
    viewer.run()


if __name__ == "__main__":
    main()
