# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Conditional-accuracy eval for the CheXpert-conditional SiT model.

FID/FRD/etc. (calculate_fid.py, evaluate.sh) answer "do generated images look
realistic overall" — they say nothing about whether a sample conditioned on
"Cardiomegaly" actually looks like Cardiomegaly rather than, say, Pneumothorax
or a generic-but-realistic chest X-ray. This script answers that question
using an external pathology classifier as a proxy judge:

For each of CheXpert's 14 classes, sample --samples-per-class images from the
SiT checkpoint conditioned on that class, run them through a pretrained
CheXpert-labeling classifier (torchxrayvision), and check whether the
classifier agrees the target pathology is present. This is the generative
analogue of the "Inception accuracy" check used for class-conditional
ImageNet models.

Caveats baked into the design (see the README section this script is
documented alongside for the full discussion):
  - torchxrayvision is a multi-label sigmoid classifier, not a softmax over
    mutually-exclusive classes, so "accuracy" here means "the target
    pathology's predicted probability crossed --threshold", i.e. a per-class
    hit rate, not argmax accuracy.
  - Not every CHEXPERT_CLASSES entry has a torchxrayvision analogue (e.g.
    "Support Devices" isn't a radiological finding, "Pleural Other" has no
    matching xrv output). Those classes are still *sampled* from (so you can
    eyeball them) but excluded from the scored hit-rate table — the script
    prints which classes got skipped and why.
  - This only tells you whether the classifier is fooled into agreeing with
    the label, not whether the image is otherwise realistic — pair with
    per-class FID/FRD (see README), not instead of it.

Requirements:
    pip install torchxrayvision

Usage (single GPU):
    python evaluate_conditional_accuracy.py ODE \
        --ckpt path/to/sit_chexpert_conditional.pt \
        --samples-per-class 50 \
        --cfg-scale 4.0 \
        --output-dir eval_conditional_accuracy

Optionally compare against real images of each class (e.g. the held-out
split from utils/split_chexpert_trainval.py) as a reference ceiling:
    python evaluate_conditional_accuracy.py ODE \
        --ckpt path/to/sit_chexpert_conditional.pt \
        --real-data-path /path/to/CheXpert-v1.0 --real-csv-name eval_split.csv
"""

import argparse
import csv
import json
import math
import os
import sys

import torch
from diffusers.models import AutoencoderKL

from datasets.chexpert_dataset import CHEXPERT_CLASSES, CheXpertDataset
from download import find_model
from models import SiT_models
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from transport import Sampler, create_transport

# torchxrayvision's pathology names don't always match CheXpert's column
# names verbatim; map the differences here. Anything not listed is looked up
# by exact name.
XRV_NAME_ALIASES = {
    "Pleural Effusion": "Effusion",
}

# CHEXPERT_CLASSES entries with no torchxrayvision analogue at all — these
# get sampled (so you can still look at them) but never scored.
NO_CLASSIFIER_ANALOGUE = {"Support Devices", "Pleural Other"}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _load_classifier(device):
    """
    Loads torchxrayvision's DenseNet121 trained directly on CheXpert's own
    label space (as opposed to the "all"-dataset-merged weights), so its
    pathology vocabulary lines up with CHEXPERT_CLASSES as closely as
    possible.
    """
    try:
        import torchxrayvision as xrv
    except ImportError as e:
        raise ImportError(
            "This script requires torchxrayvision: pip install torchxrayvision"
        ) from e

    model = xrv.models.DenseNet(weights="densenet121-res224-chex").to(device)
    model.eval()
    return model, xrv


def _build_class_mapping(classifier_pathologies):
    """
    Maps each CHEXPERT_CLASSES index (except "No Finding", handled specially)
    to its index in the classifier's pathology list, if one exists. Returns
    (mapping: {chexpert_idx: xrv_idx}, skipped: [class names with no analogue]).
    """
    mapping = {}
    skipped = []
    for idx, name in enumerate(CHEXPERT_CLASSES):
        if name == "No Finding":
            continue
        xrv_name = XRV_NAME_ALIASES.get(name, name)
        if xrv_name in classifier_pathologies:
            mapping[idx] = classifier_pathologies.index(xrv_name)
        else:
            skipped.append(name)
    return mapping, skipped


def _preprocess_for_classifier(images, xrv):
    """
    images: (B, 3, H, W) tensor in the VAE's decoded range (~[-1, 1]),
    as produced by vae.decode(...).sample.
    Returns a (B, 1, 224, 224) tensor ready for the xrv classifier.
    """
    import numpy as np
    import torchvision.transforms

    gray = images.mean(dim=1, keepdim=True)  # (B, 1, H, W); RGB channels are replicated grayscale
    gray_255 = torch.clamp(127.5 * gray + 128.0, 0, 255).cpu().numpy()

    resizer = xrv.datasets.XRayResizer(224)
    out = np.zeros((gray_255.shape[0], 1, 224, 224), dtype=np.float32)
    for i in range(gray_255.shape[0]):
        img = xrv.datasets.normalize(gray_255[i, 0], maxval=255)  # -> xrv's expected [-1024, 1024]-ish range
        out[i, 0] = resizer(img[None, ...])[0]
    return torch.from_numpy(out)


# ---------------------------------------------------------------------------
# Sampling + scoring generated images
# ---------------------------------------------------------------------------

def score_generated(mode, args, device):
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=len(CHEXPERT_CLASSES),
        class_dropout_prob=args.class_dropout_prob,
    ).to(device)
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    transport = create_transport(
        args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps
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
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    classifier, xrv = _load_classifier(device)
    mapping, skipped = _build_class_mapping(list(classifier.pathologies))
    if skipped:
        print(f"No torchxrayvision analogue for: {skipped} — sampled but not scored.")

    using_cfg = args.cfg_scale > 1.0
    # rows: conditioned CHEXPERT_CLASSES index -> {xrv pathology name: mean predicted prob}
    mean_probs = {}
    hit_rates = {}

    for class_idx, class_name in enumerate(CHEXPERT_CLASSES):
        num_batches = math.ceil(args.samples_per_class / args.batch_size)
        collected = []
        remaining = args.samples_per_class
        for _ in range(num_batches):
            n = min(args.batch_size, remaining)
            remaining -= n
            z = torch.randn(n, 4, latent_size, latent_size, device=device)
            y = torch.full((n,), class_idx, device=device, dtype=torch.long)

            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.full((n,), len(CHEXPERT_CLASSES), device=device, dtype=torch.long)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                model_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                model_fn = model.forward

            with torch.no_grad():
                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                if using_cfg:
                    samples, _ = samples.chunk(2, dim=0)
                images = vae.decode(samples / 0.18215).sample
                clf_input = _preprocess_for_classifier(images, xrv).to(device)
                probs = torch.sigmoid(classifier(clf_input)).cpu()
            collected.append(probs)

        probs = torch.cat(collected, dim=0)  # (samples_per_class, num_xrv_pathologies)

        if class_name == "No Finding":
            # "Clean" iff every pathology we can actually score stays below threshold.
            scorable_xrv_idx = list(mapping.values())
            below = probs[:, scorable_xrv_idx] < args.threshold
            hit = below.all(dim=1).float().mean().item()
            hit_rates[class_name] = hit
            mean_probs[class_name] = {
                classifier.pathologies[i]: probs[:, i].mean().item() for i in scorable_xrv_idx
            }
        elif class_idx in mapping:
            xrv_idx = mapping[class_idx]
            hit = (probs[:, xrv_idx] > args.threshold).float().mean().item()
            hit_rates[class_name] = hit
            mean_probs[class_name] = {classifier.pathologies[xrv_idx]: probs[:, xrv_idx].mean().item()}
        else:
            hit_rates[class_name] = None  # not scorable

        status = f"{hit_rates[class_name]:.3f}" if hit_rates[class_name] is not None else "n/a (no classifier analogue)"
        print(f"[{class_idx:2d}] {class_name:<28s} hit rate: {status}")

    del model, vae, classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return hit_rates, mean_probs


# ---------------------------------------------------------------------------
# Optional: same scoring on real images, as a reference ceiling
# ---------------------------------------------------------------------------

def score_real(args, device):
    from torchvision import transforms

    from train import center_crop_arr  # reuse the exact preprocessing train.py uses

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = CheXpertDataset(
        args.real_data_path, csv_name=args.real_csv_name, transform=transform, frontal_only=True
    )

    classifier, xrv = _load_classifier(device)
    mapping, _ = _build_class_mapping(list(classifier.pathologies))

    per_class_probs = {name: [] for name in CHEXPERT_CLASSES}
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    with torch.no_grad():
        for images, labels in loader:
            clf_input = _preprocess_for_classifier(images.to(device), xrv).to(device)
            probs = torch.sigmoid(classifier(clf_input)).cpu()
            for i, label in enumerate(labels.tolist()):
                per_class_probs[CHEXPERT_CLASSES[label]].append(probs[i])

    hit_rates = {}
    for class_idx, class_name in enumerate(CHEXPERT_CLASSES):
        if not per_class_probs[class_name]:
            hit_rates[class_name] = None
            continue
        probs = torch.stack(per_class_probs[class_name])
        if class_name == "No Finding":
            scorable_xrv_idx = list(mapping.values())
            below = probs[:, scorable_xrv_idx] < args.threshold
            hit_rates[class_name] = below.all(dim=1).float().mean().item()
        elif class_idx in mapping:
            xrv_idx = mapping[class_idx]
            hit_rates[class_name] = (probs[:, xrv_idx] > args.threshold).float().mean().item()
        else:
            hit_rates[class_name] = None

    del classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return hit_rates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(mode, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    generated_hit_rates, mean_probs = score_generated(mode, args, device)

    real_hit_rates = None
    if args.real_data_path is not None:
        print("\nScoring real images for reference...")
        real_hit_rates = score_real(args, device)

    report_path = os.path.join(args.output_dir, "conditional_accuracy.csv")
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["class", "generated_hit_rate"]
        if real_hit_rates is not None:
            header.append("real_hit_rate")
        writer.writerow(header)
        for class_name in CHEXPERT_CLASSES:
            row = [class_name, generated_hit_rates[class_name]]
            if real_hit_rates is not None:
                row.append(real_hit_rates[class_name])
            writer.writerow(row)
    print(f"\nWrote {report_path}")

    with open(os.path.join(args.output_dir, "mean_probs.json"), "w") as f:
        json.dump(mean_probs, f, indent=2)

    scorable = [h for h in generated_hit_rates.values() if h is not None]
    if scorable:
        print(f"\nMean hit rate across {len(scorable)} scorable classes: {sum(scorable) / len(scorable):.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_conditional_accuracy.py <ODE|SDE> [options]")
        sys.exit(1)

    mode = sys.argv[1]
    assert mode[:2] != "--", "Usage: python evaluate_conditional_accuracy.py <ODE|SDE> [options]"
    assert mode in ["ODE", "SDE"], "Mode must be 'ODE' or 'SDE'"

    parser = argparse.ArgumentParser(description="Classifier-based conditional-accuracy eval for CheXpert-conditional SiT")

    parser.add_argument("--ckpt", type=str, required=True, help="Path to a CheXpert-conditional SiT checkpoint")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-S/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--class-dropout-prob", type=float, default=0.1,
                        help="Must match the value the checkpoint was trained with (train.py's default "
                             "for --dataset chexpert is 0.1) — determines the label-embedding table size.")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classifier probability above which a pathology counts as predicted-present")
    parser.add_argument("--output-dir", type=str, default="eval_conditional_accuracy")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--real-data-path", type=str, default=None,
                         help="Optional: also score real CheXpert images (e.g. the held-out split from "
                              "utils/split_chexpert_trainval.py) as a reference ceiling for the hit rates above")
    parser.add_argument("--real-csv-name", type=str, default="eval_split.csv")

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
    elif mode == "SDE":
        parse_sde_args(parser)

    args = parser.parse_known_args()[0]
    main(mode, args)