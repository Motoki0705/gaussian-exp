from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class SemanticFeatureCache:
    def __init__(self, root, levels=None):
        self.root = Path(root)
        self.levels = levels or ["whole", "part", "subpart"]
        if not self.root.exists():
            raise FileNotFoundError(f"semantic feature root not found: {self.root}")

    def _candidate_paths(self, image_name, level):
        name = Path(image_name).stem
        base = self.root / level
        return [
            base / f"{image_name}.npy",
            base / f"{name}.npy",
        ]

    def _load_level(self, image_name, level):
        for p in self._candidate_paths(image_name, level):
            if p.exists():
                arr = np.load(p)
                return arr
        raise FileNotFoundError(f"feature map not found for image={image_name}, level={level}")

    def get_maps_for_camera(self, camera, device="cuda"):
        outputs = {}
        h = int(camera.image_height)
        w = int(camera.image_width)

        for level in self.levels:
            arr = self._load_level(camera.image_name, level)
            if arr.ndim != 3:
                raise ValueError(f"Expected HxWxC feature map, got shape={arr.shape} for {camera.image_name}")
            feat = torch.from_numpy(arr).float().to(device)
            # HWC -> CHW
            feat = feat.permute(2, 0, 1).contiguous()
            if feat.shape[1] != h or feat.shape[2] != w:
                feat = F.interpolate(feat.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
            outputs[level] = feat

        return outputs

    @staticmethod
    def valid_mask(feature_chw):
        # Pixels with non-zero target features are considered supervised.
        return feature_chw.abs().sum(dim=0, keepdim=True) > 0
