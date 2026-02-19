from __future__ import annotations

import numpy as np


def colorize_instance_map(instance_map: np.ndarray, background_color=(0, 0, 0)) -> np.ndarray:
    if instance_map.ndim != 2:
        raise ValueError(f"instance_map must be [H,W], got {instance_map.shape}")

    h, w = instance_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[:, :] = np.array(background_color, dtype=np.uint8)

    unique_ids = np.unique(instance_map)
    for idx in unique_ids:
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


def overlay_rgb(base_rgb: np.ndarray, overlay_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if base_rgb.shape != overlay_rgb.shape:
        raise ValueError(f"shape mismatch: base={base_rgb.shape}, overlay={overlay_rgb.shape}")
    if base_rgb.ndim != 3 or base_rgb.shape[2] != 3:
        raise ValueError(f"base_rgb must be HxWx3, got {base_rgb.shape}")

    base = base_rgb.astype(np.float32)
    over = overlay_rgb.astype(np.float32)
    out = ((1.0 - float(alpha)) * base + float(alpha) * over).clip(0, 255).astype(np.uint8)
    return out
