import os
import re
from argparse import ArgumentParser

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, SemanticParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from semantic import FeatureAutoEncoder
from utils.general_utils import safe_state


LEVELS = ("whole", "part", "subpart")


def normalize_latent(x):
    # x: C,H,W
    x_min = x.amin(dim=(1, 2), keepdim=True)
    x_max = x.amax(dim=(1, 2), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-8)


def _latest_semantic_ckpt(model_path):
    sem_dir = os.path.join(model_path, "semantic")
    if not os.path.isdir(sem_dir):
        return None, None
    pat = re.compile(r"semantic_ckpt_(\d+)\.pth$")
    cands = []
    for name in os.listdir(sem_dir):
        m = pat.match(name)
        if m:
            cands.append((int(m.group(1)), os.path.join(sem_dir, name)))
    if not cands:
        return None, None
    cands.sort(key=lambda x: x[0])
    return cands[-1]


def _try_load_semantic_from_ckpt(dataset, scene, gaussians):
    if gaussians.has_semantic_features:
        return scene, gaussians

    ckpt_iter, ckpt_path = _latest_semantic_ckpt(dataset.model_path)
    if ckpt_path is None:
        raise RuntimeError(
            "No semantic attributes in point cloud and no semantic checkpoint found at "
            f"{os.path.join(dataset.model_path, 'semantic')}"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sem_whole = ckpt.get("sem_whole")
    sem_part = ckpt.get("sem_part")
    sem_subpart = ckpt.get("sem_subpart")
    if sem_whole is None or sem_part is None or sem_subpart is None:
        raise RuntimeError(f"Invalid semantic checkpoint (missing tensors): {ckpt_path}")

    n_points = gaussians.get_xyz.shape[0]
    if sem_whole.shape[0] != n_points:
        # Geometry and semantic tensors are out of sync; reload scene at semantic checkpoint iteration.
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=ckpt_iter, shuffle=False)
        n_points = gaussians.get_xyz.shape[0]
        if sem_whole.shape[0] != n_points:
            raise RuntimeError(
                f"Semantic checkpoint point count ({sem_whole.shape[0]}) does not match "
                f"point cloud point count ({n_points}) even at iteration {ckpt_iter}."
            )

    gaussians._sem_whole = torch.nn.Parameter(sem_whole.to("cuda").float(), requires_grad=False)
    gaussians._sem_part = torch.nn.Parameter(sem_part.to("cuda").float(), requires_grad=False)
    gaussians._sem_subpart = torch.nn.Parameter(sem_subpart.to("cuda").float(), requires_grad=False)
    print(f"[render_semantic] Loaded semantic tensors from {ckpt_path}")
    return scene, gaussians


def render_semantic_sets(dataset, pipeline, sem_args, iteration, skip_train, skip_test):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scene, gaussians = _try_load_semantic_from_ckpt(dataset, scene, gaussians)

        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

        ae = FeatureAutoEncoder(in_dim=384, latent_dim=sem_args.latent_dim).cuda().eval()
        ae_ckpt = os.path.join(dataset.model_path, "semantic", "feat_ae.pt")
        if os.path.exists(ae_ckpt):
            ae.load_state_dict(torch.load(ae_ckpt, map_location="cpu"))

        targets = []
        if not skip_train:
            targets.append(("train", scene.getTrainCameras()))
        if not skip_test:
            targets.append(("test", scene.getTestCameras()))

        for split_name, views in targets:
            base_dir = os.path.join(dataset.model_path, split_name, f"semantic_{scene.loaded_iter}")
            for level in LEVELS:
                os.makedirs(os.path.join(base_dir, level, "latent"), exist_ok=True)
                os.makedirs(os.path.join(base_dir, level, "decoded_l2"), exist_ok=True)

            for idx, view in enumerate(tqdm(views, desc=f"Render semantic {split_name}")):
                for level in LEVELS:
                    sem = gaussians.get_semantic_features(level)
                    latent = render(
                        view,
                        gaussians,
                        pipeline,
                        bg,
                        override_color=sem,
                        clamp_output=False,
                        use_trained_exp=False,
                    )["render"]

                    latent_vis = normalize_latent(latent)
                    latent_path = os.path.join(base_dir, level, "latent", f"{idx:05d}.png")
                    torchvision.utils.save_image(latent_vis, latent_path)

                    dec = ae.decode(latent.permute(1, 2, 0)).permute(2, 0, 1)
                    dec_norm = torch.linalg.norm(dec, dim=0, keepdim=True)
                    dec_norm = dec_norm / (dec_norm.max() + 1e-8)
                    dec_path = os.path.join(base_dir, level, "decoded_l2", f"{idx:05d}.png")
                    torchvision.utils.save_image(dec_norm, dec_path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render semantic feature maps")
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    sp = SemanticParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = mp.extract(args)
    pipeline = pp.extract(args)
    sem_args = sp.extract(args)

    render_semantic_sets(dataset, pipeline, sem_args, args.iteration, args.skip_train, args.skip_test)
