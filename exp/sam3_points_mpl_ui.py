#!/usr/bin/env python3
"""Matplotlib UI for SAM3 point-based mask prediction."""

from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from PIL import Image

from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model_builder import build_sam3_video_model


class SAM3PointMask:
    def __init__(
        self,
        image_path: Path,
        checkpoint_path: Path,
        bpe_path: Path,
        device: str = "cuda",
    ) -> None:
        self.image_path = Path(image_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.bpe_path = Path(bpe_path).resolve()
        self.device = device

        self.masks = None
        self.ious = None

        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        self.image_np = np.array(Image.open(self.image_path).convert("RGB"))

        self._load_from_hf = False
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

    def predict(self, points_xy: np.ndarray, labels: np.ndarray):
        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError(f"points_xy must be [N,2], got {points_xy.shape}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be [N], got {labels.shape}")
        if points_xy.shape[0] != labels.shape[0]:
            raise ValueError("points and labels count mismatch")

        self.predictor.set_image(self.image_np)

        point_coords = points_xy[None, :, :].astype(np.float32)  # [1, N, 2]
        point_labels = labels[None, :].astype(np.int32)  # [1, N]

        masks, ious, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            normalize_coords=False,
        )

        self.masks = np.asarray(masks)  # [C, H, W]
        self.ious = np.asarray(ious)  # [C]

        assert self.masks.ndim == 3, f"Unexpected masks shape: {self.masks.shape}"
        assert self.ious.ndim == 1, f"Unexpected ious shape: {self.ious.shape}"
        assert self.masks.shape[0] == self.ious.shape[0], (
            f"Mismatch between masks and ious: {self.masks.shape} vs {self.ious.shape}"
        )

        return self.masks, self.ious

    def reset_predictions(self) -> None:
        self.masks = None
        self.ious = None


class PointUI:
    def __init__(self, sam: SAM3PointMask, alpha: float = 0.45) -> None:
        self.sam = sam
        self.alpha = float(alpha)

        self.points = []
        self.labels = []
        self.current_label = 1
        self.selected_mask_idx = 0

        self.fig = None
        self.ax_image = None
        self.im_artist = None
        self.status_text = None
        self.iou_text = None

        self.btn_predict = None
        self.btn_reset = None
        self.btn_fg = None
        self.btn_bg = None
        self.mask_buttons = []
        self.mask_button_axes = []

        self._build_figure()
        self._connect_events()
        self._render()

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(14, 8))

        self.ax_image = self.fig.add_axes([0.04, 0.08, 0.64, 0.86])
        self.ax_image.set_title("Click to add points (red=FG, blue=BG)")
        self.ax_image.axis("off")
        self.im_artist = self.ax_image.imshow(self.sam.image_np)

        ax_predict = self.fig.add_axes([0.72, 0.88, 0.10, 0.06])
        ax_reset = self.fig.add_axes([0.84, 0.88, 0.10, 0.06])
        ax_fg = self.fig.add_axes([0.72, 0.80, 0.10, 0.06])
        ax_bg = self.fig.add_axes([0.84, 0.80, 0.10, 0.06])

        self.btn_predict = Button(ax_predict, "Predict")
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_fg = Button(ax_fg, "FG Point")
        self.btn_bg = Button(ax_bg, "BG Point")

        self.status_text = self.fig.text(0.72, 0.74, "Status: Ready", fontsize=10)
        self.iou_text = self.fig.text(0.72, 0.50, "IoUs:\n-", fontsize=10, va="top")

    def _connect_events(self) -> None:
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.btn_predict.on_clicked(self._on_predict)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_fg.on_clicked(self._on_fg)
        self.btn_bg.on_clicked(self._on_bg)

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax_image:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        h, w = self.sam.image_np.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return

        self.points.append([float(x), float(y)])
        self.labels.append(int(self.current_label))
        self._set_status(
            f"Added {'FG' if self.current_label == 1 else 'BG'} point at ({x}, {y}); total={len(self.points)}"
        )
        self._render()

    def _on_predict(self, _event) -> None:
        if len(self.points) == 0:
            self._set_status("Add at least one point before Predict")
            self._render()
            return

        points_xy = np.array(self.points, dtype=np.float32)
        labels = np.array(self.labels, dtype=np.int32)

        self.sam.predict(points_xy, labels)
        self.selected_mask_idx = 0
        self._rebuild_mask_buttons()
        self._set_status(f"Predicted {self.sam.masks.shape[0]} masks")
        self._render()

    def _on_reset(self, _event) -> None:
        self.reset()

    def _on_fg(self, _event) -> None:
        self.current_label = 1
        self._set_status("Point mode: FG")
        self._render()

    def _on_bg(self, _event) -> None:
        self.current_label = 0
        self._set_status("Point mode: BG")
        self._render()

    def _set_status(self, msg: str) -> None:
        self.status_text.set_text(f"Status: {msg}")

    def _mask_button_callback(self, idx: int):
        def _cb(_event):
            if self.sam.masks is None:
                return
            if 0 <= idx < self.sam.masks.shape[0]:
                self.selected_mask_idx = idx
                self._set_status(f"Selected mask {idx}")
                self._render()

        return _cb

    def _clear_mask_buttons(self) -> None:
        for ax in self.mask_button_axes:
            ax.remove()
        self.mask_button_axes = []
        self.mask_buttons = []

    def _rebuild_mask_buttons(self) -> None:
        self._clear_mask_buttons()

        if self.sam.masks is None or self.sam.ious is None:
            return

        c = int(self.sam.masks.shape[0])
        y_top = 0.44
        row_h = 0.045
        for i in range(c):
            y = y_top - i * row_h
            if y < 0.05:
                break
            ax_btn = self.fig.add_axes([0.72, y, 0.22, 0.036])
            label = f"Mask {i} (IoU {float(self.sam.ious[i]):.4f})"
            btn = Button(ax_btn, label)
            btn.on_clicked(self._mask_button_callback(i))
            self.mask_button_axes.append(ax_btn)
            self.mask_buttons.append(btn)

    def _format_iou_text(self) -> str:
        if self.sam.ious is None:
            return "IoUs:\n-"

        lines = ["IoUs:"]
        for i, score in enumerate(self.sam.ious):
            mark = "*" if i == self.selected_mask_idx else " "
            lines.append(f"{mark} [{i}] {float(score):.6f}")
        return "\n".join(lines)

    def _compose_display_image(self) -> np.ndarray:
        img = self.sam.image_np.astype(np.float32).copy()

        if self.sam.masks is not None and self.sam.ious is not None:
            mask = self.sam.masks[self.selected_mask_idx].astype(bool)
            overlay_color = np.array([0.0, 255.0, 0.0], dtype=np.float32)
            img[mask] = (1.0 - self.alpha) * img[mask] + self.alpha * overlay_color

        out = img.astype(np.uint8)

        for (x, y), label in zip(self.points, self.labels):
            xi = int(round(x))
            yi = int(round(y))
            r = 5
            color = np.array([255, 0, 0], dtype=np.uint8) if label == 1 else np.array([0, 128, 255], dtype=np.uint8)
            y0 = max(0, yi - r)
            y1 = min(out.shape[0], yi + r + 1)
            x0 = max(0, xi - r)
            x1 = min(out.shape[1], xi + r + 1)
            out[y0:y1, x0:x1] = color

        return out

    def _render(self) -> None:
        self.im_artist.set_data(self._compose_display_image())
        mode = "FG" if self.current_label == 1 else "BG"
        self.ax_image.set_title(f"Click to add points (red=FG, blue=BG) | Mode: {mode}")
        self.iou_text.set_text(self._format_iou_text())
        self.fig.canvas.draw_idle()

    def reset(self) -> None:
        self.points = []
        self.labels = []
        self.selected_mask_idx = 0
        self.sam.reset_predictions()
        self._clear_mask_buttons()
        self._set_status("Reset done")
        self._render()

    def run(self) -> None:
        plt.show()


def main() -> None:
    image_path = Path("data/PPW_utatanewosuruneko.jpg")
    checkpoint_path = Path("checkpoints/sam3/sam3.pt")
    bpe_path = Path("checkpoints/sam3/bpe_simple_vocab_16e6.txt.gz")
    device = "cuda"

    sam = SAM3PointMask(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        device=device,
    )
    ui = PointUI(sam)
    ui.run()


if __name__ == "__main__":
    main()
