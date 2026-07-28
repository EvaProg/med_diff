# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Builds a flat folder of symlinks to frontal-view images from CheXpert's
"valid" directory, for use as a real-image comparison set for evaluation.

Rather than trusting valid.csv's list (which only contains the smaller,
expert-relabeled subset of the official validation-designated patients),
this walks the physical valid/ folder directly for every frontal image, then
cross-checks each one's *patient ID* against every patient referenced in
train.csv -- excluding any patient that shows up in both. This is a
patient-level check, not just a file-path check, since data leakage in
medical imaging is a patient-level concern: the same patient's anatomy
shouldn't appear in both the training set and the evaluation set, even if
the exact image file differs.

Also handles the filename-collision issue from before: every patient's
frontal image is literally named "view1_frontal.jpg" regardless of patient,
so symlinks are named after their full relative path instead of the bare
filename.

Usage:
    python3 flatten_chexpert_frontal.py \
        --root /path/to/CheXpert-v1.0 \
        --valid-dir valid \
        --train-csv train.csv \
        --out-dir /path/to/chexpert_frontal_eval
"""
import argparse
import csv
import os
import re

PATIENT_RE = re.compile(r"(patient\d+)")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def find_patient_id(path):
    match = PATIENT_RE.search(path)
    return match.group(1) if match else None


def train_patient_ids(root, train_csv_name):
    csv_path = os.path.join(root, train_csv_name)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Could not find {train_csv_name} at {csv_path}.")
    patient_ids = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            patient_id = find_patient_id(row["Path"])
            if patient_id:
                patient_ids.add(patient_id)
    return patient_ids


def find_valid_images(root, valid_dir, frontal_only):
    valid_root = os.path.join(root, valid_dir)
    if not os.path.isdir(valid_root):
        raise FileNotFoundError(f"Could not find valid directory at {valid_root}.")
    paths = []
    for dirpath, _dirnames, filenames in os.walk(valid_root):
        for fname in filenames:
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            if frontal_only and "frontal" not in fname.lower():
                continue
            paths.append(os.path.join(dirpath, fname))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                         help="CheXpert root folder (contains --train-csv and --valid-dir).")
    parser.add_argument("--valid-dir", default="valid",
                         help="Subdirectory under --root to walk for real images (default: valid).")
    parser.add_argument("--train-csv", default="train.csv",
                         help="CSV to check for patient-level overlap (default: train.csv).")
    parser.add_argument("--out-dir", required=True,
                         help="Flat output folder to create and fill with symlinks.")
    parser.add_argument("--frontal-only", dest="frontal_only", action="store_true", default=True,
                         help="Only include frontal-view images (default: on, matching --frontal-only training).")
    parser.add_argument("--no-frontal-only", dest="frontal_only", action="store_false",
                         help="Include lateral views too.")
    args = parser.parse_args()

    root_abs = os.path.realpath(args.root)

    print(f"Reading patient IDs from {args.train_csv}...")
    train_patients = train_patient_ids(root_abs, args.train_csv)
    print(f"Found {len(train_patients)} distinct patients in {args.train_csv}.")

    print(f"Walking {os.path.join(root_abs, args.valid_dir)} for frontal images...")
    valid_images = find_valid_images(root_abs, args.valid_dir, args.frontal_only)
    if not valid_images:
        raise SystemExit(f"No images found under {os.path.join(root_abs, args.valid_dir)} "
                          f"(frontal_only={args.frontal_only}).")
    print(f"Found {len(valid_images)} candidate images.")

    os.makedirs(args.out_dir, exist_ok=True)
    out_dir_abs = os.path.realpath(args.out_dir)

    linked, contaminated, skipped, no_patient_id = 0, 0, 0, 0
    for src in valid_images:
        patient_id = find_patient_id(src)
        if patient_id is None:
            print(f"Warning: could not determine patient ID for {src}, skipping.")
            no_patient_id += 1
            continue
        if patient_id in train_patients:
            contaminated += 1
            continue
        rel = os.path.relpath(src, root_abs).replace(os.sep, "_").replace("/", "_")
        dest = os.path.join(out_dir_abs, rel)
        if os.path.exists(dest):
            skipped += 1
            continue
        os.symlink(src, dest)
        linked += 1

    print(f"Linked {linked} images into {out_dir_abs}.")
    print(f"  Excluded {contaminated} images whose patient also appears in {args.train_csv} "
          f"(train/eval leakage guard).")
    if skipped:
        print(f"  Skipped {skipped} already present in {out_dir_abs}.")
    if no_patient_id:
        print(f"  Skipped {no_patient_id} with no detectable patient ID in their path.")


if __name__ == "__main__":
    main()