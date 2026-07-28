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
    --ckpt /home/evan/SiT/pretrained_models/SiT-S-2-256.pt \
    --finetune \
    --global-batch-size 128 \
    --epochs 10 \
    --ckpt-every 2000 \
    --sample-every 2000 \
    2>&1 | tee "$LOGFILE"

EXPERIMENT_DIR=$(grep -oE "Experiment directory created at .*" "$LOGFILE" | tail -n1 | sed -E 's/^Experiment directory created at //')

if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Could not determine experiment directory from training output; skipping loss plot."
    exit 1
fi

echo "Plotting loss curve for $EXPERIMENT_DIR"
python3 plot_loss.py "$EXPERIMENT_DIR"
