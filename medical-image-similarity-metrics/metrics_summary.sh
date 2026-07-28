#!/bin/bash

# Wrapper around compute_allmetrics.sh that also writes just the final
# metric values (FRD, FID, RadFID, KID, CMMD) to a plain text summary file.
#
# Usage:
#   bash compute_allmetrics_summary.sh <IMAGE_FOLDER1> <IMAGE_FOLDER2> [metrics: all|FRD,FID,RadFID,KID,CMMD] [output_txt]
#
# The full, unfiltered output of compute_allmetrics.sh still prints to the
# terminal as usual. Only the summary file is new.

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <IMAGE_FOLDER1> <IMAGE_FOLDER2> [metrics: all|FRD,FID,RadFID,KID,CMMD] [output_txt]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_FOLDER1="$1"
IMAGE_FOLDER2="$2"
METRICS="${3:-all}"
OUTPUT_TXT="${4:-results_summary.txt}"

LOGFILE=$(mktemp)
trap 'rm -f "$LOGFILE"' EXIT

# Run the real script, showing output live while also saving it for parsing.
bash "$SCRIPT_DIR/compute_allmetrics.sh" "$IMAGE_FOLDER1" "$IMAGE_FOLDER2" "$METRICS" 2>&1 | tee "$LOGFILE"

{
    echo "Image folder 1: $IMAGE_FOLDER1"
    echo "Image folder 2: $IMAGE_FOLDER2"
    echo "Date: $(date)"
    echo ""

    # FRD: value is printed on the line right after the "FRD results (with logarithm)..." header.
    if grep -q "^FRD results (with logarithm)" "$LOGFILE"; then
        frd_val=$(grep -A1 "^FRD results (with logarithm)" "$LOGFILE" | tail -n1 | tr -d '[:space:]')
        echo "FRD: $frd_val"
    fi

    # FID: line looks like "FID: 27.79 (0.224)". Anchored so it won't match "RadFID:" lines.
    fid_val=$(grep -E "^FID: [0-9]" "$LOGFILE" | tail -n1 | grep -oE "[0-9]+\.?[0-9]* \([0-9]+\.?[0-9]*\)")
    [ -n "$fid_val" ] && echo "FID: $fid_val"

    # RadFID: line looks like "RadFID: 0.03 (0.001)".
    radfid_val=$(grep -E "^RadFID: [0-9]" "$LOGFILE" | tail -n1 | grep -oE "[0-9]+\.?[0-9]* \([0-9]+\.?[0-9]*\)")
    [ -n "$radfid_val" ] && echo "RadFID: $radfid_val"

    # KID: line looks like "KID (path): 0.002 (0.001)" or "KID (RadImageNet) (path): 0.002 (0.001)".
    kid_val=$(grep -E "^KID.*: [0-9]" "$LOGFILE" | tail -n1 | grep -oE "[0-9]+\.?[0-9]* \([0-9]+\.?[0-9]*\)$")
    [ -n "$kid_val" ] && echo "KID: $kid_val"

    # CMMD: line looks like "The CMMD value is:  27.123".
    cmmd_val=$(grep -E "^The CMMD value is:" "$LOGFILE" | tail -n1 | grep -oE "[0-9]+\.[0-9]+$")
    [ -n "$cmmd_val" ] && echo "CMMD: $cmmd_val"

} > "$OUTPUT_TXT"

echo ""
echo "Summary written to $OUTPUT_TXT:"
cat "$OUTPUT_TXT"