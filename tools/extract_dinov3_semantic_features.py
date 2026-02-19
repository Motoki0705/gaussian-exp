#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from third_party.dinov3.api.dinov3_loader import get_dinov3_vits16


LEVELS = ("whole", "part", "subpart")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract DINOv3 semantic features from SAM masks.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dinov3-ckpt", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bbox-margin", type=int, default=8)
    parser.add_argument(
        "--resolution-divisor",
        type=int,
        default=1,
        help="Downscale factor for output feature maps. 1 keeps original size.",
    )
    return parser.parse_args()


def list_images(images_dir: Path):
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        images.extend(images_dir.glob(ext))
    images = sorted(images)
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return images


def load_mask(mask_path: Path, h: int, w: int):
    if not mask_path.exists():
        return np.zeros((h, w), dtype=np.uint8)
    return (np.array(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)


def crop_with_mask(image_np, mask_np, margin=8):
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return None

    x0 = max(0, xs.min() - margin)
    x1 = min(image_np.shape[1], xs.max() + 1 + margin)
    y0 = max(0, ys.min() - margin)
    y1 = min(image_np.shape[0], ys.max() + 1 + margin)
    return image_np[y0:y1, x0:x1]


def main():
    args = parse_args()
    images_dir = args.images_dir.resolve()
    masks_dir = args.masks_dir.resolve()
    output_dir = args.output_dir.resolve()

    for level in LEVELS:
        (output_dir / level).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = get_dinov3_vits16(str(args.dinov3_ckpt.resolve())).eval().to(device)

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    images = list_images(images_dir)
    divisor = max(1, int(args.resolution_divisor))

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        image_np = np.array(img)
        h, w = image_np.shape[:2]
        out_h = max(1, h // divisor)
        out_w = max(1, w // divisor)

        if divisor > 1:
            work_img_np = np.array(img.resize((out_w, out_h), Image.BILINEAR))
        else:
            work_img_np = image_np

        for level in LEVELS:
            mask_path = masks_dir / level / f"{img_path.stem}.png"
            mask = load_mask(mask_path, h, w)
            if divisor > 1:
                mask = np.array(Image.fromarray(mask * 255, mode="L").resize((out_w, out_h), Image.NEAREST)) > 0
                mask = mask.astype(np.uint8)
            feat_map = np.zeros((out_h, out_w, 384), dtype=np.float16)

            if mask.sum() > 0:
                crop = crop_with_mask(work_img_np, mask, margin=args.bbox_margin)
                if crop is not None:
                    crop_pil = Image.fromarray(crop)
                    x = tfm(crop_pil).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        feat = model(x).squeeze(0).float().cpu().numpy()
                    feat_map[mask > 0] = feat.astype(np.float16)

            np.save(output_dir / level / f"{img_path.stem}.npy", feat_map)

    print(f"Saved semantic DINOv3 feature maps to {output_dir}")


if __name__ == "__main__":
    main()
