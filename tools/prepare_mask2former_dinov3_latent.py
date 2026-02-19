#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class SegmentFeatureCache:
    image_stem: str
    panoptic_ids_path: Path
    segment_ids: np.ndarray  # [K]
    cls_features: np.ndarray  # [K, 384]


class FeatureAutoEncoder(nn.Module):
    def __init__(self, in_dim: int = 384, latent_dim: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, in_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Phase0TeacherPreparer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.scene_root = Path(args.scene_root)
        self.images_dir = self.scene_root / "images"

        self.relative_depth_dir = self.scene_root / "relative_depth"
        self.latent_map_dir = self.scene_root / "latent_map"
        self.latent_valid_mask_dir = self.scene_root / "latent_valid_mask"
        self.cache_dir = self.scene_root / "_phase0_cache"
        self.ae_dir = self.scene_root / "_phase0_ae"

        self.device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self._ensure_dirs()
        self._build_models()

    def _ensure_dirs(self) -> None:
        for d in [
            self.relative_depth_dir,
            self.latent_map_dir,
            self.latent_valid_mask_dir,
            self.cache_dir,
            self.ae_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def _build_models(self) -> None:
        self.mask_processor = AutoImageProcessor.from_pretrained(self.args.mask2former_model)
        self.mask_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.args.mask2former_model
        ).to(self.device).eval()

        self.depth_pipe = pipeline(
            task="depth-estimation",
            model=self.args.depth_model,
            device=0 if self.device == "cuda" else -1,
        )

        self.dino = torch.hub.load(
            str(REPO_ROOT / "third_party" / "dinov3"),
            "dinov3_vits16",
            source="local",
            weights=self.args.dinov3_ckpt,
        ).to(self.device).eval()

    def _list_images(self) -> List[Path]:
        files = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not files:
            raise RuntimeError(f"No images found in {self.images_dir}")
        return files

    @staticmethod
    def _save_depth_png(depth_np: np.ndarray, out_path: Path) -> None:
        depth_u8 = np.clip(depth_np, 0, 255).astype(np.uint8)
        Image.fromarray(depth_u8, mode="L").save(out_path)

    def _run_depth(self, image_pil: Image.Image) -> np.ndarray:
        out = self.depth_pipe(image_pil)
        depth = np.asarray(out["depth"]).astype(np.float32)
        return depth

    def _run_panoptic(self, image_pil: Image.Image) -> Tuple[np.ndarray, List[Dict]]:
        inputs = self.mask_processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                outputs = self.mask_model(**inputs)
        processed = self.mask_processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[(image_pil.height, image_pil.width)],
        )[0]
        panoptic_ids = processed["segmentation"].cpu().numpy().astype(np.int32)
        segments_info = processed["segments_info"]
        return panoptic_ids, segments_info

    def _extract_segment_cls_features(
        self, image_np: np.ndarray, panoptic_ids: np.ndarray, segments_info: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        segment_ids: List[int] = []
        cls_feats: List[np.ndarray] = []

        for seg in segments_info:
            seg_id = int(seg["id"])
            mask = panoptic_ids == seg_id
            if mask.sum() < self.args.min_segment_pixels:
                continue

            ys, xs = np.where(mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1

            crop_img = image_np[y0:y1, x0:x1].copy()
            crop_mask = mask[y0:y1, x0:x1]
            crop_img[~crop_mask] = 0

            crop = Image.fromarray(crop_img)
            feat = self._extract_cls_feature(crop)

            segment_ids.append(seg_id)
            cls_feats.append(feat)

        if not segment_ids:
            return np.zeros((0,), dtype=np.int32), np.zeros((0, 384), dtype=np.float32)

        return np.asarray(segment_ids, dtype=np.int32), np.stack(cls_feats, axis=0).astype(np.float32)

    def _extract_cls_feature(self, image_pil: Image.Image) -> np.ndarray:
        x = image_pil.convert("RGB").resize((224, 224), Image.BILINEAR)
        x_np = np.asarray(x).astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        x_np = (x_np - mean) / std
        x_t = torch.from_numpy(x_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                feats = self.dino.forward_features(x_t)
        cls = feats["x_norm_clstoken"][0].detach().float().cpu().numpy()
        return cls

    def first_pass_collect(self, image_paths: List[Path]) -> Tuple[List[SegmentFeatureCache], np.ndarray]:
        caches: List[SegmentFeatureCache] = []
        all_feats: List[np.ndarray] = []

        for image_path in tqdm(image_paths, desc="Phase0 pass1"):
            stem = image_path.stem
            image_pil = Image.open(image_path).convert("RGB")
            image_np = np.asarray(image_pil)

            depth = self._run_depth(image_pil)
            self._save_depth_png(depth, self.relative_depth_dir / f"{stem}.png")

            panoptic_ids, segments_info = self._run_panoptic(image_pil)
            seg_ids, cls_feats = self._extract_segment_cls_features(image_np, panoptic_ids, segments_info)

            panoptic_path = self.cache_dir / f"{stem}_panoptic_ids.npy"
            np.save(panoptic_path, panoptic_ids)
            np.savez_compressed(
                self.cache_dir / f"{stem}_segments.npz",
                segment_ids=seg_ids,
                cls_features=cls_feats,
            )

            cache = SegmentFeatureCache(
                image_stem=stem,
                panoptic_ids_path=panoptic_path,
                segment_ids=seg_ids,
                cls_features=cls_feats,
            )
            caches.append(cache)

            if cls_feats.shape[0] > 0:
                all_feats.append(cls_feats)

        if all_feats:
            feats = np.concatenate(all_feats, axis=0).astype(np.float32)
        else:
            feats = np.zeros((0, 384), dtype=np.float32)

        return caches, feats

    def train_ae(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        if features.shape[0] == 0:
            raise RuntimeError("No segment CLS features extracted. Cannot train AE.")

        feat_mean = features.mean(axis=0, keepdims=True).astype(np.float32)
        feat_std = features.std(axis=0, keepdims=True).astype(np.float32)
        feat_std = np.maximum(feat_std, 1e-6)
        feat_norm = (features - feat_mean) / feat_std

        x = torch.from_numpy(feat_norm).to(self.device)
        dataset = torch.utils.data.TensorDataset(x)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.ae_batch_size,
            shuffle=True,
            drop_last=False,
        )

        ae = FeatureAutoEncoder(in_dim=384, latent_dim=3).to(self.device)
        opt = torch.optim.AdamW(ae.parameters(), lr=self.args.ae_lr, weight_decay=1e-4)

        ae.train()
        for _ in tqdm(range(self.args.ae_epochs), desc="AE pretrain"):
            for (xb,) in loader:
                z, rec = ae(xb)
                loss = F.mse_loss(rec, xb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        torch.save(ae.state_dict(), self.ae_dir / "ae_384_3_384.pt")
        np.save(self.ae_dir / "feat_mean.npy", feat_mean)
        np.save(self.ae_dir / "feat_std.npy", feat_std)

        ae.eval()
        with torch.no_grad():
            z_all = ae.encode(x).detach().cpu().numpy().astype(np.float32)

        return {
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "z_all": z_all,
            "ae_model": ae,
        }

    def second_pass_build_latent_maps(
        self,
        image_paths: List[Path],
        ae: FeatureAutoEncoder,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
    ) -> None:
        ae.eval()

        for image_path in tqdm(image_paths, desc="Phase0 pass2"):
            stem = image_path.stem
            seg_npz = np.load(self.cache_dir / f"{stem}_segments.npz")
            segment_ids = seg_npz["segment_ids"].astype(np.int32)
            cls_features = seg_npz["cls_features"].astype(np.float32)
            panoptic_ids = np.load(self.cache_dir / f"{stem}_panoptic_ids.npy").astype(np.int32)

            h, w = panoptic_ids.shape
            latent_map = np.zeros((h, w, 3), dtype=np.float32)
            latent_valid_mask = np.zeros((h, w), dtype=np.uint8)

            if cls_features.shape[0] > 0:
                f_norm = (cls_features - feat_mean) / feat_std
                f_t = torch.from_numpy(f_norm).to(self.device)
                with torch.no_grad():
                    z = ae.encode(f_t).detach().cpu().numpy().astype(np.float32)

                for i, seg_id in enumerate(segment_ids.tolist()):
                    mask = panoptic_ids == int(seg_id)
                    if mask.any():
                        latent_map[mask] = z[i]
                        latent_valid_mask[mask] = 255

            np.save(self.latent_map_dir / f"{stem}.npy", latent_map)
            Image.fromarray(latent_valid_mask, mode="L").save(
                self.latent_valid_mask_dir / f"{stem}.png"
            )

    def verify_outputs(self, image_paths: List[Path]) -> Dict[str, object]:
        stems = [p.stem for p in image_paths]

        miss_depth = [s for s in stems if not (self.relative_depth_dir / f"{s}.png").exists()]
        miss_latent = [s for s in stems if not (self.latent_map_dir / f"{s}.npy").exists()]
        miss_mask = [s for s in stems if not (self.latent_valid_mask_dir / f"{s}.png").exists()]

        mask_coverages = []
        latent_stats = []

        for s in stems[: min(len(stems), self.args.verify_samples)]:
            lmap = np.load(self.latent_map_dir / f"{s}.npy")
            lmask = np.asarray(Image.open(self.latent_valid_mask_dir / f"{s}.png").convert("L"))
            dmap = np.asarray(Image.open(self.relative_depth_dir / f"{s}.png").convert("L"))

            if lmap.ndim != 3 or lmap.shape[2] != 3:
                raise RuntimeError(f"latent map shape invalid: {s} -> {lmap.shape}")
            if lmask.ndim != 2:
                raise RuntimeError(f"latent valid mask shape invalid: {s} -> {lmask.shape}")
            if dmap.ndim != 2:
                raise RuntimeError(f"relative depth shape invalid: {s} -> {dmap.shape}")
            if lmap.shape[:2] != lmask.shape or lmap.shape[:2] != dmap.shape:
                raise RuntimeError(
                    f"shape mismatch: {s}, latent={lmap.shape}, mask={lmask.shape}, depth={dmap.shape}"
                )

            mask_coverages.append(float((lmask > 0).mean()))
            latent_stats.append((float(lmap.min()), float(lmap.max())))

        result = {
            "num_images": len(stems),
            "missing": {
                "relative_depth": len(miss_depth),
                "latent_map": len(miss_latent),
                "latent_valid_mask": len(miss_mask),
            },
            "sample_mask_coverage_mean": float(np.mean(mask_coverages)) if mask_coverages else 0.0,
            "sample_latent_min": float(np.min([x[0] for x in latent_stats])) if latent_stats else 0.0,
            "sample_latent_max": float(np.max([x[1] for x in latent_stats])) if latent_stats else 0.0,
            "verify_samples": min(len(stems), self.args.verify_samples),
        }

        with open(self.scene_root / "phase0_summary.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    def run(self) -> None:
        image_paths = self._list_images()
        print(f"[INFO] images={len(image_paths)} device={self.device}")
        print(f"[INFO] mask2former={self.args.mask2former_model}")
        print(f"[INFO] depth={self.args.depth_model}")

        caches, features = self.first_pass_collect(image_paths)
        print(f"[INFO] segment caches={len(caches)} total segment features={features.shape[0]}")

        ae_out = self.train_ae(features)
        self.second_pass_build_latent_maps(
            image_paths=image_paths,
            ae=ae_out["ae_model"],
            feat_mean=ae_out["feat_mean"],
            feat_std=ae_out["feat_std"],
        )

        summary = self.verify_outputs(image_paths)
        print("[INFO] Phase0 summary:")
        print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase0 teacher data generator for playroom")
    parser.add_argument(
        "--scene-root",
        type=str,
        default="data/tandt_db/db/playroom",
        help="Scene root that contains images/",
    )
    parser.add_argument(
        "--mask2former-model",
        type=str,
        default="facebook/mask2former-swin-large-coco-panoptic",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default="depth-anything/Depth-Anything-V2-Large-hf",
    )
    parser.add_argument(
        "--dinov3-ckpt",
        type=str,
        default="checkpoints/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    )
    parser.add_argument("--ae-epochs", type=int, default=25)
    parser.add_argument("--ae-batch-size", type=int, default=256)
    parser.add_argument("--ae-lr", type=float, default=1e-3)
    parser.add_argument("--min-segment-pixels", type=int, default=64)
    parser.add_argument("--verify-samples", type=int, default=20)
    parser.add_argument("--cpu", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preparer = Phase0TeacherPreparer(args)
    preparer.run()


if __name__ == "__main__":
    main()
