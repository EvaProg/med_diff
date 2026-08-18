# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VAE round-trip fidelity check: encodes and immediately decodes a handful of
real CheXpert images through the same frozen VAE (stabilityai/sd-vae-ft-*)
that SiT generates into/out of, with no diffusion model involved at all.

Motivation: if generated samples stay soft/blurry/artifact-heavy no matter
how long SiT trains, one explanation is that SiT itself hasn't converged;
another is that the VAE — trained on natural RGB photos, not grayscale
medical X-rays — simply can't faithfully encode/decode this image domain. In
the second case, no amount of additional SiT training can produce sharper
output, since SiT only ever sees/produces the VAE's latent space and depends
on the VAE's decoder for the last mile of image formation.

This script isolates that question: run real X-rays through encode->decode
with no generation involved, and look at what comes out. If real images come
out blurry too, the VAE is the ceiling. If they come out sharp, the
bottleneck is more likely SiT's training progress or capacity, not the VAE.

Usage:
    python vae_roundtrip_check.py \
        --data-path /path/to/CheXpert-v1.0 --csv-name eval_split.csv \
        --num-images 8 --output-dir vae_roundtrip_check
"""

import argparse
import os

import torch
from diffusers.models import AutoencoderKL
from torchvision import transforms
from torchvision.utils import save_image

from datasets.chexpert_dataset import CheXpertDataset
from train import center_crop_arr


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # No RandomHorizontalFlip here (unlike train.py's training transform) —
    # this is a deterministic fidelity check, not training-time augmentation.
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = CheXpertDataset(
        args.data_path, csv_name=args.csv_name, transform=transform, frontal_only=True
    )

    torch.manual_seed(args.seed)
    indices = torch.randperm(len(dataset))[:args.num_images].tolist()

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    originals = []
    roundtrips = []
    mae_per_image = []
    with torch.no_grad():
        for idx in indices:
            image, _ = dataset[idx]
            x = image.unsqueeze(0).to(device)
            # Same encode/decode call train.py's training loop and sample_ddp.py's
            # decode step use — .sample() (not .mode()), so this faithfully mirrors
            # what SiT is actually trained to reconstruct from.
            latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
            recon = vae.decode(latent / 0.18215).sample
            mae_per_image.append((recon - x).abs().mean().item())
            originals.append(image)
            roundtrips.append(recon.squeeze(0).cpu())

    # Side-by-side grid: each row is [original, round-tripped].
    pairs = []
    for orig, recon in zip(originals, roundtrips):
        pairs.append(orig)
        pairs.append(recon)
    grid_path = os.path.join(args.output_dir, "roundtrip_comparison.png")
    save_image(pairs, grid_path, nrow=2, normalize=True, value_range=(-1, 1))

    print(f"Per-image MAE (original vs. round-tripped, [-1, 1] scale): "
          f"{[round(m, 4) for m in mae_per_image]}")
    print(f"Mean MAE: {sum(mae_per_image) / len(mae_per_image):.4f}")
    print(f"Saved side-by-side comparison grid (left=original, right=round-tripped) to {grid_path}")
    print("If the round-tripped (right) images look blurry/artifact-heavy even though the "
          "originals (left) are sharp real X-rays, the VAE itself is a likely quality ceiling "
          "for SiT's output, independent of how long SiT trains.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE encode/decode round-trip fidelity check for chest X-rays")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Folder containing --csv-name (e.g. the extracted CheXpert-v1.0 directory)")
    parser.add_argument("--csv-name", type=str, default="eval_split.csv")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--output-dir", type=str, default="vae_roundtrip_check")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)