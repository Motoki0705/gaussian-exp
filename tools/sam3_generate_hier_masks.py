#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from tqdm import tqdm

from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model_builder import build_sam3_video_model


LEVELS = ("whole", "part", "subpart")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate hierarchical SAM3 masks from point-grid prompts. "
            "This follows the LangSplat-style point-prompt setting."
        )
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=32, help="Point grid resolution (default: 32 for 32x32).")
    parser.add_argument(
        "--point-batch-size",
        type=int,
        default=128,
        help="Number of point prompts processed per SAM forward pass.",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/sam3/sam3.pt"))
    parser.add_argument("--bpe-path", type=Path, default=Path("checkpoints/sam3/bpe_simple_vocab_16e6.txt.gz"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--dedup-iou-thresh", type=float, default=0.9)
    parser.add_argument("--overlap-thresh", type=float, default=0.95)
    parser.add_argument("--save-label-maps", action="store_true", help="Save per-level instance-id maps (.npy).")
    return parser.parse_args()


def list_images(images_dir: Path) -> List[Path]:
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        images.extend(images_dir.glob(ext))
    images = sorted(images)
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return images


def ensure_assets(checkpoint_path: Path, bpe_path: Path) -> Tuple[Path, Path]:
    checkpoint_path = checkpoint_path.resolve()
    bpe_path = bpe_path.resolve()

    bpe_path.parent.mkdir(parents=True, exist_ok=True)
    if not bpe_path.exists():
        urlretrieve("https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz", str(bpe_path))

    if not checkpoint_path.exists():
        # fallback to HF download inside build function by passing checkpoint_path=None
        checkpoint_path = None

    return checkpoint_path, bpe_path


def create_point_grid(height: int, width: int, grid_size: int) -> np.ndarray:
    ys = np.linspace(0, height - 1, grid_size)
    xs = np.linspace(0, width - 1, grid_size)
    points = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
    return points


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def overlap_ratio(candidate: np.ndarray, covered_union: np.ndarray) -> float:
    area = candidate.sum()
    if area == 0:
        return 1.0
    overlap = np.logical_and(candidate, covered_union).sum()
    return float(overlap) / float(area)


def deduplicate_masks(
    candidates: List[Tuple[np.ndarray, float]],
    min_area: int,
    dedup_iou_thresh: float,
    overlap_thresh: float,
) -> List[np.ndarray]:
    filtered = [(m, s) for m, s in candidates if int(m.sum()) >= min_area]
    filtered.sort(key=lambda x: x[1], reverse=True)

    kept: List[np.ndarray] = []
    covered_union = None

    for mask, _score in filtered:
        if covered_union is not None:
            if overlap_ratio(mask, covered_union) >= overlap_thresh:
                continue
        duplicate = False
        for prev in kept:
            if mask_iou(mask, prev) >= dedup_iou_thresh:
                duplicate = True
                break
        if duplicate:
            continue

        kept.append(mask)
        if covered_union is None:
            covered_union = mask.copy()
        else:
            covered_union = np.logical_or(covered_union, mask)

    return kept


def to_union_mask(masks: List[np.ndarray], height: int, width: int) -> np.ndarray:
    if not masks:
        return np.zeros((height, width), dtype=np.uint8)
    union = np.zeros((height, width), dtype=bool)
    for m in masks:
        union |= m
    return union.astype(np.uint8) * 255


def to_instance_map(masks: List[np.ndarray], height: int, width: int) -> np.ndarray:
    # Higher-score masks should come first in input list.
    inst = np.zeros((height, width), dtype=np.int32)
    for idx, m in enumerate(masks, start=1):
        inst[np.logical_and(m, inst == 0)] = idx
    return inst


def infer_hier_masks_for_image(
    predictor: SAM3InteractiveImagePredictor,
    image_np: np.ndarray,
    grid_size: int,
    min_mask_area: int,
    dedup_iou_thresh: float,
    overlap_thresh: float,
    point_batch_size: int,
) -> Dict[str, np.ndarray]:
    h, w = image_np.shape[:2]
    predictor.set_image(image_np)

    points = create_point_grid(h, w, grid_size)
    candidates = {"whole": [], "part": [], "subpart": []}
    n_points = points.shape[0]
    point_batch_size = max(1, int(point_batch_size))

    for start in range(0, n_points, point_batch_size):
        end = min(start + point_batch_size, n_points)
        batch_points = points[start:end]  # [B, 2]

        # Batched point prompts for one image: point_coords [B,1,2], point_labels [B,1]
        point_coords = batch_points[:, None, :].astype(np.float32)
        point_labels = np.ones((batch_points.shape[0], 1), dtype=np.int32)

        masks, ious, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            normalize_coords=False,
        )

        # Expected shapes:
        # masks: [B, C, H, W], ious: [B, C]
        mask_bool = masks.astype(bool)
        if mask_bool.ndim == 3:
            mask_bool = mask_bool[None]
        if ious.ndim == 1:
            ious = ious[None]

        batch_size = mask_bool.shape[0]
        for b in range(batch_size):
            per_point_masks = mask_bool[b]
            per_point_ious = ious[b] if ious.shape[0] > b else np.zeros((per_point_masks.shape[0],), dtype=np.float32)

            if per_point_masks.ndim == 2:
                per_point_masks = per_point_masks[None]
            if per_point_masks.shape[0] == 0:
                continue

            # SAM returns multiple masks for ambiguous point prompts.
            # We define local hierarchy per point by mask area order: small->subpart, medium->part, large->whole.
            areas = per_point_masks.reshape(per_point_masks.shape[0], -1).sum(axis=1)
            order = np.argsort(areas)

            small_idx = order[0]
            med_idx = order[len(order) // 2]
            large_idx = order[-1]

            candidates["subpart"].append(
                (per_point_masks[small_idx], float(per_point_ious[small_idx]) if per_point_ious.size else 0.0)
            )
            candidates["part"].append(
                (per_point_masks[med_idx], float(per_point_ious[med_idx]) if per_point_ious.size else 0.0)
            )
            candidates["whole"].append(
                (per_point_masks[large_idx], float(per_point_ious[large_idx]) if per_point_ious.size else 0.0)
            )

    outputs = {}
    for level in LEVELS:
        kept = deduplicate_masks(
            candidates[level],
            min_area=min_mask_area,
            dedup_iou_thresh=dedup_iou_thresh,
            overlap_thresh=overlap_thresh,
        )
        outputs[level] = {
            "masks": kept,
            "union": to_union_mask(kept, h, w),
            "instance_map": to_instance_map(kept, h, w),
        }

    return outputs


def main() -> None:
    args = parse_args()

    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()
    for level in LEVELS:
        (output_dir / level).mkdir(parents=True, exist_ok=True)
    if args.save_label_maps:
        for level in LEVELS:
            (output_dir / f"{level}_instances").mkdir(parents=True, exist_ok=True)

    checkpoint_path, bpe_path = ensure_assets(args.checkpoint_path, args.bpe_path)

    model = build_sam3_video_model(
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        bpe_path=str(bpe_path),
        load_from_HF=checkpoint_path is None,
        device=args.device,
    )
    # SAM3InteractiveImagePredictor expects a tracker with an image backbone.
    # The video model tracker ships with `backbone=None`, so we reuse detector backbone.
    model.tracker.backbone = model.detector.backbone
    predictor = SAM3InteractiveImagePredictor(model.tracker)

    images = list_images(images_dir)
    if args.max_images is not None:
        images = images[: args.max_images]

    for img_path in tqdm(images, desc="SAM3 point-grid masks"):
        image_np = np.array(Image.open(img_path).convert("RGB"))
        out = infer_hier_masks_for_image(
            predictor,
            image_np,
            grid_size=args.grid_size,
            min_mask_area=args.min_mask_area,
            dedup_iou_thresh=args.dedup_iou_thresh,
            overlap_thresh=args.overlap_thresh,
            point_batch_size=args.point_batch_size,
        )

        for level in LEVELS:
            union = out[level]["union"]
            Image.fromarray(union, mode="L").save(output_dir / level / f"{img_path.stem}.png")
            if args.save_label_maps:
                np.save(output_dir / f"{level}_instances" / f"{img_path.stem}.npy", out[level]["instance_map"])

    print(f"Saved hierarchical masks to {output_dir}")
    print(f"Processed images: {len(images)}")


if __name__ == "__main__":
    main()
