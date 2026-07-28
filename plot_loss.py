# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Parses an experiment's log.txt (written by train.py's logger) and plots the
training loss curve: the raw per-step loss plus a rolling-average smoothed
curve on top. Saves loss_curve.png into the same experiment directory.
"""
import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt

LOG_LINE_RE = re.compile(r"step=(\d+)\).*?Train Loss: ([\d.]+)")


def parse_log(log_path):
    steps, losses = [], []
    with open(log_path, "r") as f:
        for line in f:
            match = LOG_LINE_RE.search(line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    if not steps:
        raise RuntimeError(f"No 'Train Loss' lines found in {log_path}.")
    return steps, losses


def smooth(losses, window):
    if window <= 1:
        return losses
    kernel = np.ones(window) / window
    padded = np.pad(losses, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main(args):
    experiment_dir = args.experiment_dir.rstrip("/\\")
    log_path = os.path.join(experiment_dir, "log.txt")
    steps, losses = parse_log(log_path)
    losses = np.array(losses)
    smoothed = smooth(losses, args.smooth_window)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, linewidth=0.5, color="#a8c6e8", alpha=0.8, label="Loss")
    ax.plot(steps, smoothed, linewidth=2, color="#2b6cb0", label=f"Smoothed (window={args.smooth_window})")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Train loss")
    ax.set_title(os.path.basename(experiment_dir))
    ax.grid(alpha=0.3)
    ax.legend()

    out_path = os.path.join(experiment_dir, "loss_curve.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved loss curve ({len(steps)} points) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=str,
                        help="Path to the experiment directory (contains log.txt), "
                             "e.g. results/000-SiT-S-2-Linear-velocity-None")
    parser.add_argument("--smooth-window", type=int, default=50,
                        help="Rolling-average window size (in log entries) for the smoothed curve.")
    args = parser.parse_args()
    main(args)