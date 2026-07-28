# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset loader for the official CheXpert release, used for unconditional
(no pathology-label conditioning) image generation.
"""
import csv
import os

from PIL import Image
from torch.utils.data import Dataset


class CheXpertDataset(Dataset):
    """
    Reads image paths out of CheXpert's own train.csv/valid.csv (ignoring the
    14 pathology-label columns) and serves them as (image, 0) pairs so they're
    drop-in compatible with the (x, y) loop in train.py for unconditional training.

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
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                rel_path = row["Path"]
                if frontal_only and "frontal" not in rel_path:
                    continue
                candidate = os.path.join(root, rel_path)
                if not os.path.isfile(candidate):
                    # CSV paths include the dataset's own top-level folder name;
                    # strip it if `root` already points inside that folder.
                    candidate = os.path.join(root, *rel_path.split("/")[1:])
                self.samples.append(candidate)

        if not self.samples:
            raise RuntimeError(f"No image paths found in {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        # CheXpert images are grayscale; replicate to 3 channels for the RGB VAE.
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0