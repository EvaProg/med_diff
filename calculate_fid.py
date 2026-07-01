"""
End-to-end FID pipeline for SiT models.

Generates images with a pre-trained SiT, saves them to disk, computes FID
using pytorch-fid's InceptionV3 (the FID-specific port of the original
TensorFlow Inception weights), and optionally deletes the samples afterwards.

IMPORTANT: This script computes FID using the same Inception weights that
were used to build OpenAI's ADM reference statistics files
(VIRTUAL_imagenet{size}_labeled.npz). Using torchvision's stock Inception-v3
instead (as an earlier version of this script did) gives numbers that are
NOT comparable to published FID scores, since it's a different feature
space entirely.

Usage (single GPU):
    python fid_pipeline.py ODE \
        --ckpt path/to/sit.pt \
        --num-samples 10000 \
        --save-path fid_samples \
        --delete-after

Requirements:
    pip install pytorch-fid tqdm
"""

import argparse
import os
import sys
import shutil
import math

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers.models import AutoencoderKL

from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def generate_images(mode, args, save_dir):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # Existing images — allow resuming a partial run
    existing = {
        f for f in os.listdir(save_dir) if f.endswith(".png")
    }
    already_done = len(existing)
    if already_done >= args.num_samples:
        print(f"Found {already_done} existing images — skipping generation.")
        return

    if already_done:
        print(f"Resuming: {already_done} images already on disk, "
              f"need {args.num_samples - already_done} more.")

    # Load model
    ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)

    # find_model unwraps "ema" but not "model" — handle both
    if "model" in state_dict and "final_layer.linear.weight" not in state_dict:
        state_dict = state_dict["model"]

    # Infer learn_sigma from the checkpoint output size
    # output = patch_size² × in_channels × (2 if learn_sigma else 1)
    out_channels = state_dict["final_layer.linear.weight"].shape[0]
    patch_size = int(args.model.split("/")[1])
    in_channels = 4  # VAE latent channels
    learn_sigma = out_channels == patch_size ** 2 * in_channels * 2

    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
    )
    sampler = Sampler(transport)

    if mode == "ODE":
        sample_fn = sampler.sample_ode(
            sampling_method=args.sampling_method,
            num_steps=args.num_sampling_steps,
            atol=args.atol,
            rtol=args.rtol,
            reverse=args.reverse,
        )
    else:
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )

    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{args.vae}"
    ).to(device)

    using_cfg = args.cfg_scale > 1.0
    batch_size = args.batch_size
    total_needed = args.num_samples - already_done
    num_batches = math.ceil(total_needed / batch_size)
    global_index = already_done

    with tqdm(total=total_needed, desc="Generating images") as pbar:
        for _ in range(num_batches):
            this_batch = min(batch_size, args.num_samples - global_index)
            if this_batch <= 0:
                break

            z = torch.randn(
                this_batch, 4, latent_size, latent_size, device=device
            )
            y = torch.randint(0, args.num_classes, (this_batch,), device=device)

            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([args.num_classes] * this_batch, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                model_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                model_fn = model.forward

            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)

            samples = vae.decode(samples / 0.18215).sample
            samples = (
                torch.clamp(127.5 * samples + 128.0, 0, 255)
                .permute(0, 2, 3, 1)
                .to("cpu", dtype=torch.uint8)
                .numpy()
            )

            for img_np in samples:
                Image.fromarray(img_np).save(
                    os.path.join(save_dir, f"{global_index:06d}.png")
                )
                global_index += 1

            pbar.update(this_batch)

    # Free the generation-time models before we load Inception for scoring.
    del model, vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Saved {global_index} images to {save_dir}")


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def _load_fid_inception(device):
    """Load pytorch-fid's InceptionV3 (FID-specific weights).

    This is a PyTorch port of the exact TensorFlow Inception checkpoint
    used to build the ADM reference statistics files. It is NOT the same
    as torchvision.models.inception_v3 — using the wrong one silently
    produces FID scores that are off by orders of magnitude.

    Weights are downloaded automatically on first use from:
        https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth
    and cached under ~/.cache/torch/hub/checkpoints/ afterwards — no manual
    download needed, just internet access the first time this runs.
    """
    from pytorch_fid.inception import InceptionV3

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx], resize_input=True, normalize_input=True)
    return inception.to(device).eval()


def get_inception_features(image_dir, batch_size, device, inception=None):
    """Extract FID-Inception pool3 features (2048-d) from all PNGs in image_dir."""
    import torchvision.transforms as T

    if inception is None:
        inception = _load_fid_inception(device)

    # pytorch-fid's InceptionV3 handles resizing internally (resize_input=True)
    # and expects images as float tensors in [0, 1] — no manual Resize/CenterCrop
    # or ImageNet-style normalization here.
    transform = T.ToTensor()

    paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir) if f.endswith(".png")
    ])

    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size), desc="Extracting Inception features"):
            batch_paths = paths[i:i + batch_size]
            imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch_paths])
            feats = inception(imgs.to(device))[0]
            feats = feats.squeeze(-1).squeeze(-1)  # (B, 2048, 1, 1) -> (B, 2048)
            all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


def compute_stats(feats):
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def _get_imagenet_stats(image_size):
    """Download ADM's precomputed ImageNet stats npz if not already cached."""
    import urllib.request
    stats_dir = os.path.join(os.path.expanduser("~"), ".cache", "sit_fid_stats")
    os.makedirs(stats_dir, exist_ok=True)
    fname = f"VIRTUAL_imagenet{image_size}_labeled.npz"
    local_path = os.path.join(stats_dir, fname)
    if not os.path.exists(local_path):
        url = (
            "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
            f"ref_batches/imagenet/{image_size}/{fname}"
        )
        print(f"Downloading ImageNet-{image_size} reference stats (~1.4GB, first run only)...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Saved to {local_path}")
    return local_path


def compute_fid(save_dir, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception = _load_fid_inception(device)

    # Generated image features
    gen_feats = get_inception_features(save_dir, args.fid_batch_size, device, inception=inception)
    m1, s1 = compute_stats(gen_feats)

    if args.dataset_path:
        print(f"\nComputing FID vs real images in {args.dataset_path}...")
        ref_feats = get_inception_features(args.dataset_path, args.fid_batch_size, device, inception=inception)
        m2, s2 = compute_stats(ref_feats)
    else:
        print(f"\nComputing FID vs precomputed ImageNet-{args.image_size} stats...")
        stats_path = _get_imagenet_stats(args.image_size)
        ref = np.load(stats_path)
        m2, s2 = ref["mu"], ref["sigma"]

    return frechet_distance(m1, s1, m2, s2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(mode, args):
    os.makedirs(args.save_path, exist_ok=True)

    # Step 1: generate
    generate_images(mode, args, args.save_path)

    # Step 2: FID
    fid_score = compute_fid(args.save_path, args)
    print(f"\nFID: {fid_score:.4f}")

    # Step 3: optionally clean up
    if args.delete_after:
        shutil.rmtree(args.save_path)
        print(f"Deleted sample folder: {args.save_path}")

    return fid_score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fid_pipeline.py <ODE|SDE> [options]")
        sys.exit(1)

    mode = sys.argv[1]
    assert mode[:2] != "--", "Usage: python fid_pipeline.py <ODE|SDE> [options]"
    assert mode in ["ODE", "SDE"], "Mode must be 'ODE' or 'SDE'"

    parser = argparse.ArgumentParser(description="SiT FID evaluation pipeline")

    # Core pipeline args
    parser.add_argument("--num-samples", type=int, default=10_000,
                        help="Number of images to generate for FID (default: 10000)")
    parser.add_argument("--save-path", type=str, default="fid_samples",
                        help="Directory to save generated images")
    parser.add_argument("--delete-after", action="store_true",
                        help="Delete generated images after FID is computed")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for image generation")
    parser.add_argument("--fid-batch-size", type=int, default=64,
                        help="Batch size for FID feature extraction")
    parser.add_argument("--seed", type=int, default=0)

    # Dataset args (mutually exclusive)
    ds_group = parser.add_mutually_exclusive_group()
    ds_group.add_argument("--dataset", type=str, default="imagenet_256",
                          help="Label only; imagenet_256 uses the built-in precomputed "
                               "reference stats (default: imagenet_256)")
    ds_group.add_argument("--dataset-path", type=str, default=None,
                          help="Path to a local folder of real images (overrides --dataset)")

    # Model args
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()),
                        default="SiT-XL/2")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
    elif mode == "SDE":
        parse_sde_args(parser)

    args = parser.parse_known_args()[0]
    main(mode, args)