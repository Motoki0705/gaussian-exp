#!/usr/bin/env python3
"""Minimal SAM3 point-prompt example with hardcoded paths."""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model_builder import build_sam3_video_model


# ---- Hardcoded settings ----
IMAGE_PATH = Path("data/tandt_db/db/playroom/images/DSC05572.jpg")
OUTPUT_MASK_DIR = Path("exp/sam3_points_masks")
CHECKPOINT_PATH = Path("checkpoints/sam3/sam3.pt")
BPE_PATH = Path("checkpoints/sam3/bpe_simple_vocab_16e6.txt.gz")
DEVICE = "cuda"

# (x, y) in pixel coordinates
POINTS = np.array([
    [400.0, 300.0],
    [420.0, 320.0],
], dtype=np.float32)

# 1: positive point, 0: negative point
POINT_LABELS = np.array([1, 1], dtype=np.int32)


def ensure_assets(checkpoint_path: Path, bpe_path: Path):
    checkpoint_path = checkpoint_path.resolve()
    bpe_path = bpe_path.resolve()

    bpe_path.parent.mkdir(parents=True, exist_ok=True)
    if not bpe_path.exists():
        urlretrieve("https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz", str(bpe_path))

    if not checkpoint_path.exists():
        checkpoint_path = None

    return checkpoint_path, bpe_path


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    checkpoint_path, bpe_path = ensure_assets(CHECKPOINT_PATH, BPE_PATH)

    model = build_sam3_video_model(
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        bpe_path=str(bpe_path),
        load_from_HF=checkpoint_path is None,
        device=DEVICE,
    )
    model.tracker.backbone = model.detector.backbone
    predictor = SAM3InteractiveImagePredictor(model.tracker)

    image_np = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    predictor.set_image(image_np)

    point_coords = POINTS[None, :, :]  # [1, N, 2]
    point_labels = POINT_LABELS[None, :]  # [1, N]

    masks, ious, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
        normalize_coords=False,
    )
    masks = np.asarray(masks)  # [C, H, W]
    ious = np.asarray(ious)    # [C]
    assert masks.ndim == 3, f"Unexpected masks shape: {masks.shape}"
    assert ious.ndim == 1, f"Unexpected ious shape: {ious.shape}"
    assert masks.shape[0] == ious.shape[0], (
        f"Mismatch between masks and ious: {masks.shape} vs {ious.shape}"
    )
    n = int(masks.shape[0])

    OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_np, mode="RGB").save(OUTPUT_MASK_DIR / "original.png")
    for i in range(n):
        mask = masks[i].astype(np.uint8) * 255
        out_path = OUTPUT_MASK_DIR / f"mask_{i:02d}_iou_{float(ious[i]):.4f}.png"
        Image.fromarray(mask, mode="L").save(out_path)

    print(f"image: {IMAGE_PATH}")
    print(f"points: {POINTS.tolist()}")
    print(f"labels: {POINT_LABELS.tolist()}")
    print(f"saved dir: {OUTPUT_MASK_DIR}")
    print(f"num masks: {n}")


if __name__ == "__main__":
    main()
