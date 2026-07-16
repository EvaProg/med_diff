# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset loader for the NIH ChestX-ray14 Kaggle mirror
(https://www.kaggle.com/datasets/nih-chest-xrays/data), used for unconditional
(no disease-label conditioning) image generation. A much smaller stand-in for
CheXpert to sanity-check the training pipeline end-to-end.
"""
import glob
import os

from PIL import Image
from torch.utils.data import Dataset


class NIHChestXrayDataset(Dataset):
    """
    The Kaggle mirror ships images split across nested images_NNN/images/*.png
    folders (Data_Entry_2017.csv only lists bare filenames, no subfolder), so
    this just globs for every .png under `root` instead of relying on the CSV.
    Returns (image, 0) pairs, ignoring disease labels, for unconditional training.
    """
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True))
        if not self.samples:
            raise RuntimeError(
                f"No .png images found under {root}. Point --data-path at the folder "
                f"you extracted the Kaggle NIH Chest X-ray dataset into."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        # NIH X-rays are grayscale (some are RGBA); flatten to 3-channel RGB for the VAE.
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0
