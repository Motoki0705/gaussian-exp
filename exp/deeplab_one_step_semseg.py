#!/usr/bin/env python3
"""One-step semantic segmentation demo using torchvision DeepLabV3."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

from exp.util.io import ensure_dir, load_rgb_image, save_npy, save_rgb_image, save_text_lines
from exp.util.vis import colorize_instance_map, overlay_rgb

class OneStepSemanticSegmentation:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.weights = DeepLabV3_ResNet101_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.categories = list(self.weights.meta["categories"])

        self.model = deeplabv3_resnet101(weights=self.weights).to(self.device).eval()

    def predict(self, image_np: np.ndarray) -> np.ndarray:
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError(f"image must be HxWx3 RGB, got {image_np.shape}")

        h, w = image_np.shape[:2]
        image_pil = Image.fromarray(image_np, mode="RGB")
        x = self.transforms(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)["out"]  # [1,C,h,w]
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        pred = logits[0].argmax(0).cpu().numpy().astype(np.int32)  # [H,W]
        return pred


def main() -> None:
    image_path = Path("data/tandt_db/db/playroom/images/DSC05572.jpg")
    output_dir = Path("exp/deeplab_hier_demo")
    ensure_dir(output_dir)

    image_np = load_rgb_image(image_path)
    model = OneStepSemanticSegmentation(device="cuda")
    label_map = model.predict(image_np)
    color_map = colorize_instance_map(label_map)
    overlay = overlay_rgb(image_np, color_map, alpha=0.45)

    save_rgb_image(output_dir / "original.png", image_np)
    save_rgb_image(output_dir / "semantic_map.png", color_map)
    save_rgb_image(output_dir / "semantic_overlay.png", overlay)
    save_npy(output_dir / "semantic_labels.npy", label_map)

    present_ids = sorted(int(x) for x in np.unique(label_map))
    lines = []
    for i in present_ids:
        name = model.categories[i] if 0 <= i < len(model.categories) else f"class_{i}"
        lines.append(f"{i}\t{name}")
    save_text_lines(output_dir / "present_classes.txt", lines)

    print(f"Saved demo to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
