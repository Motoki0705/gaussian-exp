from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def ensure_dir(path: Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_rgb_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    Image.fromarray(image_rgb, mode="RGB").save(path)


def save_npy(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, array)


def save_text_lines(path: Path, lines: list[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
