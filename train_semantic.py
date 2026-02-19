import os
from argparse import ArgumentParser
from random import randint

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, SemanticParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from semantic import FeatureAutoEncoder, SemanticFeatureCache
from utils.general_utils import safe_state


def freeze_rgb_parameters(gaussians: GaussianModel):
    for tensor in [
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity,
    ]:
        tensor.requires_grad_(False)


def masked_l1(pred, target, valid_mask):
    # pred/target: C,H,W, valid_mask: 1,H,W
    diff = (pred - target).abs() * valid_mask
    denom = valid_mask.sum() * pred.shape[0]
    if denom.item() == 0:
        return pred.new_tensor(0.0)
    return diff.sum() / denom


def masked_cosine(pred, target, valid_mask):
    # cosine per-pixel along channels
    cos = F.cosine_similarity(pred, target, dim=0)
    mask = valid_mask.squeeze(0)
    denom = mask.sum()
    if denom.item() == 0:
        return pred.new_tensor(0.0)
    return ((1.0 - cos) * mask).sum() / denom


def sample_valid_features(feature_chw, max_samples=8192):
    # feature_chw: C,H,W
    c, h, w = feature_chw.shape
    feature_hwc = feature_chw.permute(1, 2, 0).reshape(h * w, c)
    valid = (feature_hwc.abs().sum(dim=1) > 0)
    idx = valid.nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return None
    if idx.numel() > max_samples:
        perm = torch.randperm(idx.numel(), device=idx.device)[:max_samples]
        idx = idx[perm]
    return feature_hwc[idx]


def train_semantic(dataset, pipe, sem_args, load_iteration):
    levels = [x.strip() for x in sem_args.semantic_levels.split(",") if x.strip()]
    if not levels:
        raise ValueError("semantic_levels is empty")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=True)
    freeze_rgb_parameters(gaussians)

    if not gaussians.has_semantic_features or gaussians.get_semantic_features("whole").shape[1] != sem_args.latent_dim:
        gaussians.initialize_semantic_features(latent_dim=sem_args.latent_dim)

    gaussians.semantic_training_setup(sem_args.semantic_lr)
    ae = FeatureAutoEncoder(in_dim=384, latent_dim=sem_args.latent_dim).cuda()
    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=sem_args.ae_lr)

    if sem_args.semantic_resume:
        resume_path = sem_args.semantic_resume
        ckpt = torch.load(resume_path, map_location="cpu")
        ae.load_state_dict(ckpt["ae"])
        if "sem_whole" in ckpt:
            gaussians._sem_whole.data.copy_(ckpt["sem_whole"].to("cuda"))
            gaussians._sem_part.data.copy_(ckpt["sem_part"].to("cuda"))
            gaussians._sem_subpart.data.copy_(ckpt["sem_subpart"].to("cuda"))

    cache = SemanticFeatureCache(sem_args.semantic_feature_root, levels=levels)

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    os.makedirs(os.path.join(scene.model_path, "semantic"), exist_ok=True)

    progress = tqdm(range(1, sem_args.semantic_iterations + 1), desc="Semantic training")
    ema_loss = 0.0

    for iteration in progress:
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        target_maps = cache.get_maps_for_camera(viewpoint_cam, device="cuda")

        gaussians.semantic_optimizer.zero_grad(set_to_none=True)
        ae_optimizer.zero_grad(set_to_none=True)

        sem_losses = []
        ae_samples = []

        for level in levels:
            target_map = target_maps[level]
            with torch.set_grad_enabled(True):
                target_latent = ae.encode(target_map.permute(1, 2, 0)).permute(2, 0, 1).contiguous()

            rendered = render(
                viewpoint_cam,
                gaussians,
                pipe,
                bg,
                override_color=gaussians.get_semantic_features(level),
                clamp_output=not sem_args.no_clamp_semantic,
                use_trained_exp=False,
            )["render"]

            valid_mask = SemanticFeatureCache.valid_mask(target_map)
            sem_loss = masked_l1(rendered, target_latent, valid_mask) + sem_args.lambda_cosine * masked_cosine(rendered, target_latent, valid_mask)
            sem_losses.append(sem_loss)

            sampled = sample_valid_features(target_map)
            if sampled is not None:
                ae_samples.append(sampled)

        if ae_samples:
            ae_batch = torch.cat(ae_samples, dim=0)
            recon, _ = ae(ae_batch)
            ae_loss = (recon - ae_batch).abs().mean() + sem_args.lambda_cosine * (1.0 - F.cosine_similarity(recon, ae_batch, dim=-1)).mean()
        else:
            ae_loss = torch.tensor(0.0, device="cuda")

        sem_total = torch.stack(sem_losses).mean() if sem_losses else torch.tensor(0.0, device="cuda")

        if iteration <= sem_args.warmup_ae_iters:
            total_loss = sem_args.lambda_ae * ae_loss
        else:
            total_loss = sem_args.lambda_semantic * sem_total + sem_args.lambda_ae * ae_loss

        total_loss.backward()

        ae_optimizer.step()
        if iteration > sem_args.warmup_ae_iters:
            gaussians.semantic_optimizer.step()

        ema_loss = 0.4 * total_loss.item() + 0.6 * ema_loss
        progress.set_postfix({"loss": f"{ema_loss:.6f}", "sem": f"{sem_total.item():.6f}", "ae": f"{ae_loss.item():.6f}"})

        if iteration % sem_args.semantic_save_interval == 0 or iteration == sem_args.semantic_iterations:
            scene.save(iteration)
            torch.save(
                {
                    "iteration": iteration,
                    "ae": ae.state_dict(),
                    "sem_whole": gaussians._sem_whole.detach().cpu(),
                    "sem_part": gaussians._sem_part.detach().cpu(),
                    "sem_subpart": gaussians._sem_subpart.detach().cpu(),
                },
                os.path.join(scene.model_path, "semantic", f"semantic_ckpt_{iteration}.pth"),
            )
            torch.save(
                ae.state_dict(),
                os.path.join(scene.model_path, "semantic", "feat_ae.pt"),
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Semantic feature distillation training")
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    sp = SemanticParams(parser)
    parser.add_argument("--iteration", type=int, default=-1, help="3DGS point cloud iteration to load")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = mp.extract(args)
    pipe = pp.extract(args)
    sem_args = sp.extract(args)

    if not sem_args.semantic_feature_root:
        raise ValueError("--semantic_feature_root is required")

    train_semantic(dataset, pipe, sem_args, args.iteration)
