#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_video_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate binary masks from image sequence using SAM3 video predictor."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory that contains input frames (jpg/jpeg/png).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save output binary masks (.png).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="wall and ceiling",
        help='SAM3 text prompt. Default: "wall and ceiling".',
    )
    parser.add_argument(
        "--anchor-frames",
        type=str,
        default="0",
        help='Comma-separated frame indices for text prompts (example: "0,112,224").',
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help='Comma-separated GPU ids (example: "0" or "0,1").',
    )
    parser.add_argument(
        "--bpe-path",
        type=Path,
        default=Path("checkpoints/sam3/bpe_simple_vocab_16e6.txt.gz"),
        help="Path to CLIP BPE vocabulary file.",
    )
    return parser.parse_args()


def list_frames(images_dir: Path) -> List[Path]:
    frames = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        frames.extend(images_dir.glob(ext))
    frames = sorted(frames)
    if not frames:
        raise FileNotFoundError(f"No image frames found in {images_dir}")
    return frames


def outputs_to_union_mask(outputs: Dict, height: int, width: int) -> np.ndarray:
    masks = outputs.get("out_binary_masks", None)
    if masks is None:
        return np.zeros((height, width), dtype=np.uint8)

    masks = np.asarray(masks)
    if masks.size == 0:
        return np.zeros((height, width), dtype=np.uint8)

    if masks.ndim == 2:
        union = masks.astype(bool)
    elif masks.ndim == 3:
        union = np.any(masks.astype(bool), axis=0)
    else:
        raise ValueError(f"Unexpected mask shape: {masks.shape}")

    return (union.astype(np.uint8) * 255)


def main() -> None:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = list_frames(images_dir)
    anchor_frames = []
    for token in args.anchor_frames.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        idx = max(0, min(idx, len(frames) - 1))
        anchor_frames.append(idx)
    if not anchor_frames:
        anchor_frames = [0]
    anchor_frames = sorted(set(anchor_frames))

    first_img = Image.open(frames[0]).convert("RGB")
    width, height = first_img.size

    bpe_path = args.bpe_path.resolve()
    if not bpe_path.exists():
        bpe_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(
            "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz",
            str(bpe_path),
        )

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    predictor = build_sam3_video_predictor(
        bpe_path=str(bpe_path),
        gpus_to_use=gpu_ids,
    )

    response = predictor.handle_request(
        {"type": "start_session", "resource_path": str(images_dir)}
    )
    session_id = response["session_id"]

    frame_outputs: Dict[int, Dict] = {}
    try:
        for anchor in anchor_frames:
            first = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": anchor,
                    "text": args.prompt,
                }
            )
            frame_outputs[int(first["frame_index"])] = first["outputs"]

        for step in predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        ):
            frame_outputs[int(step["frame_index"])] = step["outputs"]
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})

    for idx, frame_path in enumerate(frames):
        outputs = frame_outputs.get(idx, {"out_binary_masks": np.zeros((0, height, width))})
        mask = outputs_to_union_mask(outputs, height=height, width=width)
        out_path = output_dir / f"{frame_path.stem}.png"
        Image.fromarray(mask, mode="L").save(out_path)

    print(f"Saved {len(frames)} mask files to {output_dir}")
    print(f"Prompt: {args.prompt}")
    print(f"Anchor frames: {anchor_frames}")
    print(f"Frames with SAM3 outputs: {len(frame_outputs)} / {len(frames)}")


if __name__ == "__main__":
    main()
