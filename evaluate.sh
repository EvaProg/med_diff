#!/bin/bash
# Computes FRD/FID/RadFID/KID/CMMD between two image folders and writes a
# filtered summary of just the metric values to an output file.
#
# This is the only script you need -- it replaces compute_allmetrics.sh +
# metrics_summary.sh + the old evaluate.sh with one self-contained file.
# Lives at the SiT repo root, next to the medical-image-similarity-metrics/
# subdirectory. Callable from anywhere:
#   bash evaluate.sh <REAL> <GENERATED> [METRICS] [OUTPUT_FILE]
#
# REAL / GENERATED: image folders (relative or absolute; resolved before
#   this script changes directory, so relative paths are relative to
#   wherever you ran the command from, not this script's location).
# METRICS: comma-separated (FRD,FID,RadFID,KID,CMMD) or ALL (default: ALL).
# OUTPUT_FILE: default: results_summary.txt in the current directory. A
#   temporary raw log (progress bars, warnings, tracebacks) is kept only
#   for the duration of the run so a crashed metric can still be diagnosed
#   from the terminal output, then deleted before the script exits -- only
#   OUTPUT_FILE is left on disk.

# Deliberately not `set -e`: if one metric fails (e.g. CMMD), the rest should
# still run and get recorded, matching the original compute_allmetrics.sh.
set -uo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <REAL_IMAGES> <GENERATED_IMAGES> [METRICS] [OUTPUT_FILE]"
    exit 1
fi

# This script's own directory, regardless of where it's called from, plus
# the metrics repo as a subdirectory of it. The underlying python scripts
# (analyze_radiomics.py, fid_score.py, etc.) need METRICS_REPO as their cwd
# for their own relative imports/model paths.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METRICS_REPO="$PROJECT_ROOT/medical-image-similarity-metrics"

# Resolve everything to absolute paths *before* changing directory below.
REAL_IMAGES="$(realpath "$1")"
GENERATED_IMAGES="$(realpath "$2")"
METRICS="${3:-ALL}"
METRICS="$(echo "$METRICS" | tr '[:lower:]' '[:upper:]' | tr -d ' ')"
OUTPUT_FILE="${4:-results_summary.txt}"
case "$OUTPUT_FILE" in
    /*) : ;;  # already absolute
    *) OUTPUT_FILE="$(pwd)/$OUTPUT_FILE" ;;
esac
RAW_LOG="${OUTPUT_FILE}.full.log"

if [ ! -d "$METRICS_REPO" ]; then
    echo "Error: expected medical-image-similarity-metrics/ next to this script at $METRICS_REPO"
    exit 1
fi

contains_metric() {
    [[ "$METRICS" == "ALL" || "$METRICS" == *"$1"* ]]
}

cd "$METRICS_REPO" || exit 1

{
    if contains_metric "FRD"; then
        echo "FRD:"
        /usr/bin/time -f "Time to compute FRD: %E" python3 analyze_radiomics.py \
            --image_folder1 "$REAL_IMAGES" \
            --image_folder2 "$GENERATED_IMAGES"
    fi

    if contains_metric "FID" || contains_metric "RADFID" || contains_metric "KID"; then
        cd "$METRICS_REPO/src/gan-metrics-pytorch" || exit 1
    fi

    if contains_metric "FID"; then
        echo "FID:"
        /usr/bin/time -f "Time to compute FID: %E" python3 fid_score.py \
            --true "$REAL_IMAGES" \
            --fake "$GENERATED_IMAGES"
    fi

    if contains_metric "RADFID"; then
        echo "RadFID:"
        /usr/bin/time -f "Time to compute RadFID: %E" python3 fid_score.py \
            --true "$REAL_IMAGES" \
            --fake "$GENERATED_IMAGES" \
            --use-rad-imagenet-features
    fi

    if contains_metric "KID"; then
        echo "KID:"
        /usr/bin/time -f "Time to compute KID: %E" python3 kid_score.py \
            --true "$REAL_IMAGES" \
            --fake "$GENERATED_IMAGES" \
            --img-size 256
        cd "$METRICS_REPO" || exit 1
    fi

    if contains_metric "CMMD"; then
        cd "$METRICS_REPO/src/cmmd-pytorch" || exit 1
        echo "CMMD:"
        /usr/bin/time -f "Time to compute CMMD: %E" python3 main_cmmd.py \
            "$REAL_IMAGES" \
            "$GENERATED_IMAGES" \
            --batch_size=32 \
            --max_count=30000
        cd "$METRICS_REPO" || exit 1
    fi
} 2>&1 | tee "$RAW_LOG"

{
    echo "Real images: $REAL_IMAGES"
    echo "Generated images: $GENERATED_IMAGES"
    echo "Metrics requested: $METRICS"
    echo "Date: $(date)"
    echo ""

    # FRD: value is printed on the line right after its "FRD results..." label.
    if grep -q "^FRD results" "$RAW_LOG"; then
        frd_val=$(grep -A1 "^FRD results" "$RAW_LOG" | tail -n1 | tr -d '[:space:]')
        echo "FRD: $frd_val"
    fi

    # FID: line looks like "FID: 27.79 (0.224)". Anchored so it won't match "RadFID:" lines.
    fid_val=$(grep -E "^FID: [0-9]" "$RAW_LOG" | tail -n1 | grep -oE "[0-9]+\.?[0-9]* \([0-9]+\.?[0-9]*\)")
    [ -n "$fid_val" ] && echo "FID: $fid_val"

    # RadFID: line looks like "RadFID: 0.03 (0.001)".
    radfid_val=$(grep -E "^RadFID: [0-9]" "$RAW_LOG" | tail -n1 | grep -oE "[0-9]+\.?[0-9]* \([0-9]+\.?[0-9]*\)")
    [ -n "$radfid_val" ] && echo "RadFID: $radfid_val"

    # KID: line looks like "KID (path): 0.002 (0.001)" or "KID (RadImageNet) (path): 0.002 (0.001)".
    kid_val=$(grep -E "^KID.*: [0-9]" "$RAW_LOG" | tail -n1 | grep -oE "[0-9]+\.?[0-9]* \([0-9]+\.?[0-9]*\)$")
    [ -n "$kid_val" ] && echo "KID: $kid_val"

    # CMMD: line looks like "The CMMD value is:  27.123".
    cmmd_val=$(grep -E "^The CMMD value is:" "$RAW_LOG" | tail -n1 | grep -oE "[0-9]+\.[0-9]+$")
    [ -n "$cmmd_val" ] && echo "CMMD: $cmmd_val"
} > "$OUTPUT_FILE"

# Raw log was only needed to parse the summary above and for live terminal
# output via tee; delete it now so only OUTPUT_FILE remains on disk.
rm -f "$RAW_LOG"

echo ""
echo "Summary written to $OUTPUT_FILE:"
cat "$OUTPUT_FILE"