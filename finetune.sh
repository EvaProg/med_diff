#!/bin/bash
set -euo pipefail

LOGFILE=$(mktemp)
trap 'rm -f "$LOGFILE"' EXIT

torchrun --nproc_per_node=2 train.py \
    --model SiT-S/2 \
    --dataset chexpert \
    --data-path /data/evan/CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0 \
    --csv-name train_split.csv \
    --frontal-only \
    --ckpt /home/evan/SiT/results/007-SiT-S-2-Linear-velocity-None/checkpoints/0003000.pt \
    --finetune \
    --exp-name chexpert-conditional-run1 \
    --global-batch-size 128 \
    --lr 2e-5 \
    --epochs 5 \
    --ckpt-every 3000 \
    --sample-every 1500 \
    2>&1 | tee "$LOGFILE"

EXPERIMENT_DIR=$(grep -oE "Experiment directory created at .*" "$LOGFILE" | tail -n1 | sed -E 's/^Experiment directory created at //')

if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Could not determine experiment directory from training output; skipping loss plot."
    exit 1
fi

echo "Plotting loss curve for $EXPERIMENT_DIR"
python3 plot_loss.py "$EXPERIMENT_DIR"
