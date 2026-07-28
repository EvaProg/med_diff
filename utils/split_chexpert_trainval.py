# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Splits CheXpert's train.csv into a training subset and a held-out evaluation
subset, at the *patient* level, and builds a flat folder of symlinks to the eval subset's frontal
images for use as a real-image comparison set.

This exists because CheXpert's official valid.csv is too small (~202
frontal images) for stable FID/FRD/KID/CMMD estimates

Usage:
    python3 split_chexpert_train_eval.py \
        --root /path/to/CheXpert-v1.0 \
        --train-csv train.csv \
        --eval-size 3000 \
        --out-dir /path/to/chexpert_frontal_eval \
        --seed 0
"""
import argparse
import csv
import os
import random
import re

PATIENT_RE = re.compile(r"(patient\d+)")


def find_patient_id(path):
    match = PATIENT_RE.search(path)
    return match.group(1) if match else None


def is_frontal(path):
    return "frontal" in path.lower()


def resolve_image_path(root, rel_path):
    candidate = os.path.join(root, rel_path)
    if os.path.isfile(candidate):
        return candidate
    # CSV paths include the dataset's own top-level folder name; strip it if
    # `root` already points inside that folder.
    return os.path.join(root, *rel_path.split("/")[1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                         help="CheXpert root folder (contains --train-csv).")
    parser.add_argument("--train-csv", default="train.csv",
                         help="CSV to split (default: train.csv).")
    parser.add_argument("--eval-size", type=int, default=3000,
                         help="Target number of frontal images to hold out for eval (default: 3000).")
    parser.add_argument("--out-dir", required=True,
                         help="Flat output folder to fill with symlinks to the eval subset's frontal images.")
    parser.add_argument("--seed", type=int, default=0,
                         help="Random seed for the patient split, for reproducibility (default: 0).")
    args = parser.parse_args()

    root_abs = os.path.realpath(args.root)
    csv_path = os.path.join(root_abs, args.train_csv)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Could not find {args.train_csv} at {csv_path}.")

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Group rows by patient, and count each patient's frontal images -- the
    # split target is measured in frontal images since that's what actually
    # gets used for evaluation and (optionally) training.
    rows_by_patient = {}
    frontal_count_by_patient = {}
    for row in rows:
        patient_id = find_patient_id(row["Path"])
        if patient_id is None:
            continue
        rows_by_patient.setdefault(patient_id, []).append(row)
        if is_frontal(row["Path"]):
            frontal_count_by_patient[patient_id] = frontal_count_by_patient.get(patient_id, 0) + 1

    patients = list(rows_by_patient.keys())
    random.Random(args.seed).shuffle(patients)

    eval_patients = set()
    eval_frontal_total = 0
    for patient_id in patients:
        if eval_frontal_total >= args.eval_size:
            break
        eval_patients.add(patient_id)
        eval_frontal_total += frontal_count_by_patient.get(patient_id, 0)

    train_patients = [p for p in patients if p not in eval_patients]

    train_split_path = os.path.join(root_abs, "train_split.csv")
    eval_split_path = os.path.join(root_abs, "eval_split.csv")

    with open(train_split_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for patient_id in train_patients:
            writer.writerows(rows_by_patient[patient_id])

    with open(eval_split_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for patient_id in sorted(eval_patients):
            writer.writerows(rows_by_patient[patient_id])

    print(f"Split {len(patients)} patients: {len(train_patients)} patients -> {train_split_path}, "
          f"{len(eval_patients)} patients -> {eval_split_path} ({eval_frontal_total} frontal images).")

    # Build the flat eval folder directly from the eval patients' frontal images.
    os.makedirs(args.out_dir, exist_ok=True)
    out_dir_abs = os.path.realpath(args.out_dir)
    linked, missing = 0, 0
    for patient_id in eval_patients:
        for row in rows_by_patient[patient_id]:
            if not is_frontal(row["Path"]):
                continue
            src = resolve_image_path(root_abs, row["Path"])
            if not os.path.isfile(src):
                print(f"Warning: missing file, skipping: {src}")
                missing += 1
                continue
            dest_name = os.path.relpath(src, root_abs).replace(os.sep, "_").replace("/", "_")
            dest = os.path.join(out_dir_abs, dest_name)
            if not os.path.exists(dest):
                os.symlink(src, dest)
                linked += 1

    print(f"Linked {linked} frontal eval images into {out_dir_abs} ({missing} missing files).")
    print(f"\nTo train on the reduced set: --csv-name train_split.csv")


if __name__ == "__main__":
    main()