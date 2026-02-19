#!/usr/bin/env python3
"""SAM3 hierarchical map generator (whole/part/subpart) from point-grid prompts."""

from pathlib import Path
import shutil
from urllib.request import urlretrieve

import numpy as np
from exp.util.io import ensure_dir, load_rgb_image, save_rgb_image
from exp.util.vis import colorize_instance_map

from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model_builder import build_sam3_video_model


class SAM3HierarchicalMapGenerator:
    LEVELS = ("whole", "part", "subpart")

    def __init__(
        self,
        checkpoint_path: Path = Path("checkpoints/sam3/sam3.pt"),
        bpe_path: Path = Path("checkpoints/sam3/bpe_simple_vocab_16e6.txt.gz"),
        device: str = "cuda",
        grid_size: int = 32,
        point_batch_size: int = 128,
        iou_thresholds: float | dict[str, float] = 0.0,
        nms_iou_thresh: float = 0.9,
        min_mask_area: int = 64,
        storage_dir: Path = Path("exp/sam3_hier_tmp"),
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.bpe_path = Path(bpe_path).resolve()
        self.device = device

        self.grid_size = int(grid_size)
        self.point_batch_size = int(point_batch_size)
        self.nms_iou_thresh = float(nms_iou_thresh)
        self.min_mask_area = int(min_mask_area)
        self.storage_dir = Path(storage_dir).resolve()

        if isinstance(iou_thresholds, dict):
            self.iou_thresholds = {lvl: float(iou_thresholds.get(lvl, 0.0)) for lvl in self.LEVELS}
        else:
            v = float(iou_thresholds)
            self.iou_thresholds = {lvl: v for lvl in self.LEVELS}

        self._load_from_hf = False
        self.predictor = None
        self._grid_cache: dict[tuple[int, int, int], np.ndarray] = {}

        self.masks = None
        self.ious = None

        self._ensure_assets()
        self.predictor = self._build_predictor()

    def _ensure_assets(self) -> None:
        self.bpe_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.bpe_path.exists():
            urlretrieve(
                "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz",
                str(self.bpe_path),
            )

        self._load_from_hf = not self.checkpoint_path.exists()

    def _build_predictor(self) -> SAM3InteractiveImagePredictor:
        model = build_sam3_video_model(
            checkpoint_path=str(self.checkpoint_path) if not self._load_from_hf else None,
            bpe_path=str(self.bpe_path),
            load_from_HF=self._load_from_hf,
            device=self.device,
        )
        model.tracker.backbone = model.detector.backbone
        return SAM3InteractiveImagePredictor(model.tracker)

    def reset_state(self) -> None:
        self.masks = None
        self.ious = None

    def _create_point_grid(self, height: int, width: int) -> np.ndarray:
        key = (height, width, self.grid_size)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached

        ys = np.linspace(0, height - 1, self.grid_size, dtype=np.float32)
        xs = np.linspace(0, width - 1, self.grid_size, dtype=np.float32)
        points = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
        self._grid_cache[key] = points
        return points

    def _predict_points_batched(self, image_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError(f"image_np must be HxWx3 RGB, got {image_np.shape}")

        h, w = image_np.shape[:2]
        points = self._create_point_grid(h, w)
        n_points = points.shape[0]
        batch_size = max(1, int(self.point_batch_size))

        self.predictor.set_image(image_np)

        masks_batches = []
        ious_batches = []

        for start in range(0, n_points, batch_size):
            end = min(start + batch_size, n_points)
            batch_points = points[start:end]  # [B,2]

            point_coords = batch_points[:, None, :].astype(np.float32)  # [B,1,2]
            point_labels = np.ones((batch_points.shape[0], 1), dtype=np.int32)  # [B,1]

            masks, ious, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
                normalize_coords=False,
            )

            masks = np.asarray(masks)
            ious = np.asarray(ious)

            # SAM3InteractiveImagePredictor may squeeze batch dim when B==1.
            if masks.ndim == 3:
                masks = masks[None, ...]
            if ious.ndim == 1:
                ious = ious[None, ...]

            if masks.ndim != 4:
                raise ValueError(f"Unexpected masks shape from predictor: {masks.shape}")
            if ious.ndim != 2:
                raise ValueError(f"Unexpected ious shape from predictor: {ious.shape}")

            masks_batches.append(masks)  # [B,3,H,W]
            ious_batches.append(ious)  # [B,3]

        all_masks = np.concatenate(masks_batches, axis=0)  # [P,3,H,W]
        all_ious = np.concatenate(ious_batches, axis=0)  # [P,3]

        if all_masks.shape[0] != n_points:
            raise ValueError(f"Unexpected number of outputs: {all_masks.shape[0]} vs points {n_points}")
        if all_masks.shape[1] != 3 or all_ious.shape[1] != 3:
            raise ValueError(
                f"Expected 3 mask tracks from SAM3, got masks {all_masks.shape}, ious {all_ious.shape}"
            )

        return all_masks.astype(bool), all_ious.astype(np.float32)

    def _generate_iou_reject_index(self, ious: np.ndarray, thr: float) -> np.ndarray:
        if ious.ndim != 1:
            raise ValueError(f"ious must be 1D, got {ious.shape}")
        return ious >= float(thr)

    def _prepare_storage(self) -> None:
        if self.storage_dir.exists():
            shutil.rmtree(self.storage_dir)
        for level in self.LEVELS:
            (self.storage_dir / level).mkdir(parents=True, exist_ok=True)

    def _save_level_chunk(self, level_name: str, chunk_idx: int, masks: np.ndarray, ious: np.ndarray) -> None:
        if masks.ndim != 3 or ious.ndim != 1:
            raise ValueError(f"Invalid chunk shape for {level_name}: masks={masks.shape}, ious={ious.shape}")
        out_path = self.storage_dir / level_name / f"chunk_{chunk_idx:06d}.npz"
        np.savez_compressed(out_path, masks=masks.astype(np.bool_), ious=ious.astype(np.float32))

    def _load_level_candidates(self, level_name: str, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        level_dir = self.storage_dir / level_name
        files = sorted(level_dir.glob("chunk_*.npz"))
        if not files:
            return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=np.float32)

        mask_list = []
        iou_list = []
        for f in files:
            d = np.load(f)
            m = np.asarray(d["masks"]).astype(bool)
            s = np.asarray(d["ious"]).astype(np.float32)
            if m.ndim != 3 or s.ndim != 1:
                raise ValueError(f"Invalid stored chunk: {f} -> masks={m.shape}, ious={s.shape}")
            if m.shape[0] != s.shape[0]:
                raise ValueError(f"Stored chunk size mismatch: {f} -> masks={m.shape}, ious={s.shape}")
            mask_list.append(m)
            iou_list.append(s)

        return np.concatenate(mask_list, axis=0), np.concatenate(iou_list, axis=0)

    @staticmethod
    def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        if union == 0:
            return 0.0
        return float(inter) / float(union)

    def _iou_nms(self, masks: np.ndarray, ious: np.ndarray) -> np.ndarray:
        if masks.ndim != 3:
            raise ValueError(f"masks must be [N,H,W], got {masks.shape}")
        if ious.ndim != 1:
            raise ValueError(f"ious must be [N], got {ious.shape}")
        if masks.shape[0] != ious.shape[0]:
            raise ValueError(f"N mismatch: masks {masks.shape} vs ious {ious.shape}")

        if masks.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)

        order = np.argsort(-ious)
        keep = []

        for idx in order:
            candidate = masks[idx]
            duplicate = False
            for kept_idx in keep:
                if self._mask_iou(candidate, masks[kept_idx]) >= self.nms_iou_thresh:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(int(idx))

        return np.array(keep, dtype=np.int64)

    def _compose_map(self, kept_masks: np.ndarray, kept_scores: np.ndarray, mode: str = "instance") -> np.ndarray:
        if kept_masks.ndim != 3:
            raise ValueError(f"kept_masks must be [K,H,W], got {kept_masks.shape}")
        if kept_scores.ndim != 1:
            raise ValueError(f"kept_scores must be [K], got {kept_scores.shape}")
        if kept_masks.shape[0] != kept_scores.shape[0]:
            raise ValueError(f"K mismatch: masks {kept_masks.shape} vs scores {kept_scores.shape}")

        k, h, w = kept_masks.shape
        if mode == "union":
            if k == 0:
                return np.zeros((h, w), dtype=np.uint8)
            return (np.any(kept_masks, axis=0).astype(np.uint8) * 255)

        if mode != "instance":
            raise ValueError(f"Unsupported mode: {mode}")

        # Winner-takes-all on overlap: higher score mask wins per pixel.
        instance_map = np.zeros((h, w), dtype=np.int32)
        best_score = np.full((h, w), -np.inf, dtype=np.float32)

        order = np.argsort(-kept_scores)
        for rank, idx in enumerate(order, start=1):
            mask = kept_masks[idx]
            score = float(kept_scores[idx])
            update = np.logical_and(mask, score > best_score)
            instance_map[update] = rank
            best_score[update] = score

        return instance_map

    def generate_hier_maps(self, image_np: np.ndarray, map_mode: str = "instance") -> dict:
        """Generate hierarchical maps for one image.

        Returns dict with keys: whole/part/subpart and meta.
        """
        self.reset_state()
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError(f"image_np must be HxWx3 RGB, got {image_np.shape}")

        h, w = image_np.shape[:2]
        points = self._create_point_grid(h, w)
        n_points = int(points.shape[0])
        batch_size = max(1, int(self.point_batch_size))

        self._prepare_storage()
        chunk_counter = {lvl: 0 for lvl in self.LEVELS}
        num_after_reject = {lvl: 0 for lvl in self.LEVELS}

        self.predictor.set_image(image_np)
        for start in range(0, n_points, batch_size):
            end = min(start + batch_size, n_points)
            batch_points = points[start:end]  # [B,2]
            point_coords = batch_points[:, None, :].astype(np.float32)  # [B,1,2]
            point_labels = np.ones((batch_points.shape[0], 1), dtype=np.int32)  # [B,1]

            masks, ious, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
                normalize_coords=False,
            )
            masks = np.asarray(masks)
            ious = np.asarray(ious)
            if masks.ndim == 3:
                masks = masks[None, ...]  # [1,3,H,W]
            if ious.ndim == 1:
                ious = ious[None, ...]  # [1,3]

            if masks.ndim != 4 or ious.ndim != 2 or masks.shape[1] != 3 or ious.shape[1] != 3:
                raise ValueError(f"Unexpected predictor outputs: masks={masks.shape}, ious={ious.shape}")

            batch_n = masks.shape[0]
            for b in range(batch_n):
                for level_idx, level_name in enumerate(self.LEVELS):
                    score = float(ious[b, level_idx])
                    if score < self.iou_thresholds[level_name]:
                        continue
                    mask = masks[b, level_idx].astype(bool)
                    if int(mask.sum()) < self.min_mask_area:
                        continue
                    num_after_reject[level_name] += 1
                    self._save_level_chunk(
                        level_name=level_name,
                        chunk_idx=chunk_counter[level_name],
                        masks=mask[None, ...],
                        ious=np.array([score], dtype=np.float32),
                    )
                    chunk_counter[level_name] += 1

        results: dict[str, dict] = {}
        self.masks = {}
        self.ious = {}
        for level_name in self.LEVELS:
            cand_masks, cand_ious = self._load_level_candidates(level_name, h, w)
            keep_idx = self._iou_nms(cand_masks, cand_ious)
            final_masks = cand_masks[keep_idx]
            final_ious = cand_ious[keep_idx]

            self.masks[level_name] = final_masks
            self.ious[level_name] = final_ious

            level_map = self._compose_map(final_masks, final_ious, mode=map_mode)
            union_map = self._compose_map(final_masks, final_ious, mode="union")

            results[level_name] = {
                "map": level_map,
                "union": union_map,
                "kept_masks": final_masks,
                "kept_ious": final_ious,
                "num_points": n_points,
                "num_after_reject": int(num_after_reject[level_name]),
                "num_candidates_saved": int(cand_masks.shape[0]),
                "num_after_nms": int(final_masks.shape[0]),
            }

        results["meta"] = {
            "grid_size": self.grid_size,
            "point_batch_size": self.point_batch_size,
            "iou_thresholds": dict(self.iou_thresholds),
            "nms_iou_thresh": self.nms_iou_thresh,
            "min_mask_area": self.min_mask_area,
            "storage_dir": str(self.storage_dir),
            "raw_masks_shape": (n_points, 3, h, w),
            "raw_ious_shape": (n_points, 3),
            "levels": list(self.LEVELS),
            "map_mode": map_mode,
        }

        return results


def main() -> None:
    image_path = Path("data/tandt_db/db/playroom/images/DSC05574.jpg")
    output_dir = Path("exp/sam3_hier_demo")
    ensure_dir(output_dir)

    image_np = load_rgb_image(image_path)
    save_rgb_image(output_dir / "original.png", image_np)

    generator = SAM3HierarchicalMapGenerator(
        checkpoint_path=Path("checkpoints/sam3/sam3.pt"),
        bpe_path=Path("checkpoints/sam3/bpe_simple_vocab_16e6.txt.gz"),
        device="cuda",
        grid_size=32,
        point_batch_size=32,
        iou_thresholds={"whole": 0.75, "part": 0.75, "subpart": 0.75},
        nms_iou_thresh=0.7,
        min_mask_area=64,
        storage_dir=Path("exp/sam3_hier_tmp"),
    )

    out = generator.generate_hier_maps(image_np=image_np, map_mode="instance")
    for level in generator.LEVELS:
        level_map = out[level]["map"]
        colored = colorize_instance_map(level_map)
        save_rgb_image(output_dir / f"{level}_map.png", colored)

    print(f"Saved original + 3 maps to: {output_dir.resolve()}")
    for level in generator.LEVELS:
        print(
            f"{level}: points={out[level]['num_points']}, "
            f"after_reject={out[level]['num_after_reject']}, "
            f"saved={out[level]['num_candidates_saved']}, after_nms={out[level]['num_after_nms']}"
        )


if __name__ == "__main__":
    main()
