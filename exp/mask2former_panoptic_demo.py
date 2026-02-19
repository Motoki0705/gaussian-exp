#!/usr/bin/env python3
"""Mask2Former panoptic segmentation one-step demo."""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.util.io import ensure_dir, load_rgb_image, save_npy, save_rgb_image, save_text_lines
from exp.util.vis import colorize_instance_map, overlay_rgb

MODEL_VARIANTS = {
    "tiny": "facebook/mask2former-swin-tiny-coco-panoptic",
    "small": "facebook/mask2former-swin-small-coco-panoptic",
    "base": "facebook/mask2former-swin-base-coco-panoptic",
    "large": "facebook/mask2former-swin-large-coco-panoptic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mask2Former panoptic segmentation demo.")
    parser.add_argument("--image-path", type=Path, default=Path("data/tandt_db/db/playroom/images/DSC05572.jpg"))
    parser.add_argument("--output-dir", type=Path, default=Path("exp/mask2former_panoptic_demo"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--variant",
        type=str,
        choices=sorted(MODEL_VARIANTS.keys()),
        default="tiny",
        help="Predefined model size variant.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional full HF model name. If set, this overrides --variant.",
    )
    return parser.parse_args()


class Mask2FormerPanoptic:
    def __init__(
        self,
        model_name: str = "facebook/mask2former-swin-tiny-coco-panoptic",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(self.device).eval()
        self.id2label = dict(self.model.config.id2label)

    def predict_panoptic(self, image_np: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError(f"image must be HxWx3 RGB, got {image_np.shape}")

        h, w = image_np.shape[:2]
        image_pil = Image.fromarray(image_np, mode="RGB")
        inputs = self.processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        processed = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[(h, w)],
        )[0]

        seg = processed["segmentation"].detach().cpu().numpy().astype(np.int32)
        segments_info = processed["segments_info"]
        return seg, segments_info


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    image_np = load_rgb_image(image_path)
    model_name = args.model_name if args.model_name is not None else MODEL_VARIANTS[args.variant]

    model = Mask2FormerPanoptic(
        model_name=model_name,
        device=args.device,
    )
    instance_map, segments_info = model.predict_panoptic(image_np)

    color_map = colorize_instance_map(instance_map)
    overlay = overlay_rgb(image_np, color_map, alpha=0.45)

    save_rgb_image(output_dir / "original.png", image_np)
    save_rgb_image(output_dir / "panoptic_instance_map.png", color_map)
    save_rgb_image(output_dir / "panoptic_overlay.png", overlay)
    save_npy(output_dir / "panoptic_segmentation_ids.npy", instance_map)

    serializable = []
    for x in segments_info:
        d = {
            "id": int(x.get("id", -1)),
            "label_id": int(x.get("label_id", -1)),
            "score": float(x.get("score", 0.0)),
            "was_fused": bool(x.get("was_fused", False)),
        }
        d["label_name"] = model.id2label.get(d["label_id"], f"class_{d['label_id']}")
        serializable.append(d)

    with (output_dir / "segments_info.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    present = sorted({int(x["label_id"]) for x in serializable})
    lines = []
    for i in present:
        lines.append(f"{i}\t{model.id2label.get(i, f'class_{i}')}")
    save_text_lines(output_dir / "present_classes.txt", lines)

    print(f"Saved demo to: {output_dir.resolve()}")
    print(f"model: {model.model_name}")
    print(f"segments: {len(serializable)}")


if __name__ == "__main__":
    main()
