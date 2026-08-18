# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset loader for the official CheXpert release, used for single-label
pathology-conditional image generation.
"""
import csv
import os

from PIL import Image
from torch.utils.data import Dataset

# CheXpert's 14 pathology-label columns, in the same order they appear in the
# official train.csv/valid.csv. Index into this list is the class label fed
# to the model's LabelEmbedder.
CHEXPERT_CLASSES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# Checked in this order, first positive ("1.0") wins. "No Finding" is checked
# last/separately since CheXpert's own labeling convention already sets it to
# 1 only when none of the other 13 columns are positive or uncertain, so an
# explicit pathology positive should take priority over it if the two ever
# disagree.
#
# CheXpert's labeler has a hierarchy where "Lung Opacity" is the *parent* of
# Lung Lesion/Edema/Consolidation/Pneumonia/Atelectasis (an image with
# Pneumonia is very often also marked Lung Opacity=1), and "Enlarged
# Cardiomediastinum" is the parent of Cardiomegaly. Most-specific-first
# ordering below so a leaf finding wins over its own umbrella category
# instead of the umbrella swallowing it.
_PATHOLOGY_PRIORITY = [
    "Pneumonia", "Consolidation", "Edema", "Atelectasis", "Lung Lesion", "Lung Opacity",
    "Cardiomegaly", "Enlarged Cardiomediastinum",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
assert set(_PATHOLOGY_PRIORITY) == set(CHEXPERT_CLASSES[1:])


def _single_label(row):
    """
    Collapses CheXpert's multi-label row (each column 1.0/0.0/-1.0/blank) down
    to a single class index for single-label conditioning. Uncertain (-1.0)
    and blank values are treated as "not present". Returns None if no column
    has a confident positive (i.e. no usable label).
    """
    for name in _PATHOLOGY_PRIORITY:
        if row.get(name) in ("1.0", "1"):
            return CHEXPERT_CLASSES.index(name)
    if row.get("No Finding") in ("1.0", "1"):
        return 0
    return None


class CheXpertDataset(Dataset):
    """
    Reads image paths and pathology labels out of CheXpert's own
    train.csv/valid.csv. Each image's 14 pathology columns are collapsed to a
    single class label (see `_single_label`) so they're drop-in compatible
    with the (x, y) loop in train.py for class-conditional training; images
    with no confident positive label are dropped.

    `root` should be the directory that directly contains `csv_name`
    (e.g. the extracted "CheXpert-v1.0-small" folder). The CSV's Path column is
    rooted at that same folder name, so it's stripped if necessary.
    """
    def __init__(self, root, csv_name="train.csv", transform=None, frontal_only=True):
        self.transform = transform
        csv_path = os.path.join(root, csv_name)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Could not find {csv_name} at {csv_path}. Point --data-path at the folder "
                f"that directly contains {csv_name} (e.g. the extracted CheXpert-v1.0-small directory)."
            )

        self.samples = []
        num_dropped = 0
        class_counts = [0] * len(CHEXPERT_CLASSES)
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                rel_path = row["Path"]
                if frontal_only and "frontal" not in rel_path:
                    continue

                label = _single_label(row)
                if label is None:
                    num_dropped += 1
                    continue

                candidate = os.path.join(root, rel_path)
                if not os.path.isfile(candidate):
                    # CSV paths include the dataset's own top-level folder name;
                    # strip it if `root` already points inside that folder.
                    candidate = os.path.join(root, *rel_path.split("/")[1:])
                self.samples.append((candidate, label))
                class_counts[label] += 1

        if not self.samples:
            raise RuntimeError(f"No usable (image, label) pairs found in {csv_path}")

        print(f"CheXpertDataset: kept {len(self.samples)} images, dropped {num_dropped} with no "
              f"confident single label. Class counts: "
              f"{dict(zip(CHEXPERT_CLASSES, class_counts))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # CheXpert images are grayscale; replicate to 3 channels for the RGB VAE.
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label