#!/bin/bash
# Samples images from a finetuned SiT checkpoint, then evaluates the generated
# set against a folder of real images using
# medical-image-similarity-metrics/compute_allmetrics.sh.
#
# Run this from the SiT repo root (same place you'd normally invoke
# sample_ddp.py / train.py from).
set -euo pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

usage() {
    cat <<EOF
Usage: $0 --ckpt PATH --real-images DIR [options]

Required:
  --ckpt PATH              Path to a SiT checkpoint saved by train.py
  --real-images DIR        Folder of real reference images to compare against

Options:
  --metrics LIST           Comma-separated metrics, e.g. FRD,FID,KID, or ALL (default: ALL)
  --num-samples N          Number of images to generate (default: 5000)
  --model NAME             SiT model variant (default: SiT-S/2)
  --num-classes N          Must match training (default: 1, unconditional)
  --class-dropout-prob P   Must match training (default: 0.0, unconditional)
  --cfg-scale S            Classifier-free guidance scale (default: 1.0)
  --per-proc-batch-size N  Sampling batch size per GPU (default: 32)
  --nproc-per-node N       Number of GPUs to sample with (default: 1)
  --sample-dir DIR         Where sample_ddp.py writes generated images, relative
                            to this repo root (default: samples)
  --metrics-repo DIR       Path to the medical-image-similarity-metrics checkout
                            (default: ./medical-image-similarity-metrics)
  --subset N               Randomly sample N images per folder for FRD's radiomics
                            computation instead of using the whole folder (requires
                            the compute_allmetrics.sh patch that forwards this to
                            analyze_radiomics.py --subset). Default: use everything.
EOF
    exit 1
}

# Defaults
METRICS="ALL"
NUM_SAMPLES=5000
MODEL="SiT-S/2"
NUM_CLASSES=1
CLASS_DROPOUT_PROB=0.0
CFG_SCALE=1.0
PER_PROC_BATCH_SIZE=32
NPROC_PER_NODE=1
SAMPLE_DIR="samples"
METRICS_REPO="medical-image-similarity-metrics"
SUBSET=""
CKPT=""
REAL_IMAGES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt) CKPT="$2"; shift 2 ;;
        --real-images) REAL_IMAGES="$2"; shift 2 ;;
        --metrics) METRICS="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --num-classes) NUM_CLASSES="$2"; shift 2 ;;
        --class-dropout-prob) CLASS_DROPOUT_PROB="$2"; shift 2 ;;
        --cfg-scale) CFG_SCALE="$2"; shift 2 ;;
        --per-proc-batch-size) PER_PROC_BATCH_SIZE="$2"; shift 2 ;;
        --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
        --sample-dir) SAMPLE_DIR="$2"; shift 2 ;;
        --metrics-repo) METRICS_REPO="$2"; shift 2 ;;
        --subset) SUBSET="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ -z "$CKPT" || -z "$REAL_IMAGES" ]]; then
    echo "Error: --ckpt and --real-images are required."
    usage
fi
if [[ ! -f "sample_ddp.py" ]]; then
    echo "Error: sample_ddp.py not found in the current directory. Run this script from the SiT repo root."
    exit 1
fi
if [[ ! -f "$CKPT" ]]; then
    echo "Error: checkpoint not found at $CKPT"
    exit 1
fi
if [[ ! -d "$REAL_IMAGES" ]]; then
    echo "Error: real image folder not found at $REAL_IMAGES"
    exit 1
fi
if [[ ! -d "$METRICS_REPO" ]]; then
    echo "Error: medical-image-similarity-metrics checkout not found at $METRICS_REPO (set --metrics-repo)"
    exit 1
fi

# Resolve to absolute paths now, before we cd anywhere for the metrics step.
REAL_IMAGES_ABS="$(readlink -f "$REAL_IMAGES")"

echo "=== Step 1/2: sampling $NUM_SAMPLES images from $CKPT ==="
SAMPLE_LOG="$(mktemp)"
torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" sample_ddp.py ODE \
    --model "$MODEL" \
    --ckpt "$CKPT" \
    --num-classes "$NUM_CLASSES" \
    --class-dropout-prob "$CLASS_DROPOUT_PROB" \
    --cfg-scale "$CFG_SCALE" \
    --num-fid-samples "$NUM_SAMPLES" \
    --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
    --sample-dir "$SAMPLE_DIR" \
    2>&1 | tee "$SAMPLE_LOG"

GENERATED_DIR="$(grep "Saving .png samples at" "$SAMPLE_LOG" | tail -1 | sed 's/.*Saving \.png samples at //')"
rm -f "$SAMPLE_LOG"

if [[ -z "$GENERATED_DIR" ]]; then
    echo "Error: could not determine the generated samples folder from sample_ddp.py's output."
    exit 1
fi
GENERATED_DIR_ABS="$(readlink -f "$GENERATED_DIR")"
echo "Generated images saved to: $GENERATED_DIR_ABS"

echo "=== Step 2/2: computing metrics ($METRICS) against $REAL_IMAGES_ABS ==="
SUBSET_ARGS=()
if [[ -n "$SUBSET" ]]; then
    SUBSET_ARGS=(--subset "$SUBSET")
fi
python3 "$SCRIPT_DIR/run_metrics.py" \
    --generated-images "$GENERATED_DIR_ABS" \
    --real-images "$REAL_IMAGES_ABS" \
    --metrics "$METRICS" \
    --metrics-repo "$METRICS_REPO" \
    --ckpt "$CKPT" \
    --num-samples "$NUM_SAMPLES" \
    "${SUBSET_ARGS[@]}"