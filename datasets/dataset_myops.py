"""MyoPS-380 Dataset loader for SAMed.
Protocol: myops380_train304_test76_last_v1
Modalities: bSSFP (Ch 0), LGE (Ch 1), T2w (Ch 2)
Canonical labels: 0: Background, 1: Normal Myocardium, 2: Edema, 3: Scar
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def canonicalize_label(label: np.ndarray, label_order: str = "legacy") -> np.ndarray:
    """Map class IDs to Canonical (0: BG, 1: Normal Myo, 2: Edema, 3: Scar)."""
    label = np.asarray(label, dtype=np.uint8)
    if label_order == "legacy":
        # Raw cache order: 0: BG, 1: Myo, 2: Scar, 3: Edema -> map to 0: BG, 1: Myo, 2: Edema, 3: Scar
        return np.asarray([0, 1, 3, 2], dtype=np.uint8)[label]
    return label


def random_rot_flip(images: list[np.ndarray], label: np.ndarray):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)
    images = [np.flip(np.rot90(x, k), axis=axis).copy() for x in images]
    label = np.flip(np.rot90(label, k), axis=axis).copy()
    return images, label


def random_rotate(images: list[np.ndarray], label: np.ndarray):
    angle = np.random.uniform(-20, 20)
    images = [
        ndimage.rotate(x, angle, order=1, reshape=False, mode="constant", cval=0, prefilter=False)
        for x in images
    ]
    label = ndimage.rotate(label, angle, order=0, reshape=False, mode="constant", cval=0, prefilter=False)
    return images, label


class RandomGenerator:
    """Shared spatial warps and independent intensity variation on [0,1] images."""

    def __init__(self, output_size: list[int] | tuple[int, int], low_res: list[int] | tuple[int, int]):
        self.output_size = tuple(output_size)
        self.low_res = tuple(low_res)

    def __call__(self, sample: dict) -> dict:
        # images: [bSSFP, LGE, T2w], each (128, 128)
        images = [np.asarray(sample[k], dtype=np.float32).copy() for k in ("image", "image1", "image2")]
        label = np.asarray(sample["label"], dtype=np.uint8).copy()

        # Spatial transforms (synchronized across modalities and label)
        if random.random() < 0.5:
            images, label = random_rot_flip(images, label)
        elif random.random() < 0.5:
            images, label = random_rotate(images, label)

        if random.random() < 0.5:
            shift = (random.randint(-5, 5), random.randint(-5, 5))
            images = [
                ndimage.shift(x, shift, order=1, mode="constant", cval=0, prefilter=False)
                for x in images
            ]
            label = ndimage.shift(label, shift, order=0, mode="constant", cval=0, prefilter=False)

        # Intensity transforms (independent across modalities, label unchanged)
        for i, img in enumerate(images):
            if random.random() < 0.5:
                img = np.clip(img * random.uniform(0.8, 1.2), 0.0, 1.0)
            if random.random() < 0.5:
                img = np.clip(img, 0.0, 1.0) ** random.uniform(0.8, 1.2)
            if random.random() < 0.25:
                img = np.clip(img + np.random.normal(0.0, 0.02, img.shape), 0.0, 1.0)
            images[i] = img

        # Stack into 3-channel array (3, H, W)
        ch3_image = np.stack(images, axis=0).astype(np.float32)  # (3, 128, 128)
        image_tensor = torch.from_numpy(ch3_image)
        label_tensor = torch.from_numpy(label).long()

        # Resize image to output_size (e.g. 512x512 or 224x224) for SAM ViT encoder
        if image_tensor.shape[-2:] != self.output_size:
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0), size=self.output_size, mode="bilinear", align_corners=False
            ).squeeze(0)

        # Low-res label for SAMed loss calculation
        if label_tensor.shape[-2:] != self.low_res:
            low_res_label = F.interpolate(
                label_tensor.unsqueeze(0).unsqueeze(0).float(),
                size=self.low_res,
                mode="nearest"
            ).squeeze(0).squeeze(0).long()
        else:
            low_res_label = label_tensor.clone()

        return {
            "image": image_tensor,
            "label": label_tensor,
            "low_res_label": low_res_label,
            "case_name": sample.get("case_name", "")
        }


class MyopsDataset(Dataset):
    """MyoPS-380 multi-modality dataset for SAMed training and testing."""

    def __init__(
        self,
        base_dir: str | Path,
        list_dir: str | Path,
        split: str = "train",
        transform=None,
        label_order: str = "legacy"
    ):
        self.base_dir = Path(base_dir)
        self.list_dir = Path(list_dir)
        self.split = split
        self.transform = transform
        self.label_order = label_order

        # Modality subdirectories
        self.mod_dirs = {
            "bSSFP": self.base_dir / "bSSFP",
            "LGE": self.base_dir / "LGE",
            "T2w": self.base_dir / "T2w",
        }

        # Verify modality directories exist
        for mod, path in self.mod_dirs.items():
            if not path.is_dir():
                raise FileNotFoundError(f"Modality directory not found: {path}")

        # Read sample list
        list_file = self.list_dir / f"{split}.txt"
        if not list_file.is_file():
            raise FileNotFoundError(f"Split file not found: {list_file}")

        with open(list_file, "r", encoding="utf-8") as f:
            self.sample_list = [line.strip() for line in f if line.strip()]

        self.is_volume = split.endswith("vol") or "test" in split

    def __len__(self) -> int:
        return len(self.sample_list)

    def _load_slice(self, mod: str, name: str) -> tuple[np.ndarray, np.ndarray]:
        path = self.mod_dirs[mod] / "train_npz" / f"{name}.npz"
        with np.load(path) as data:
            image = np.asarray(data["image"], dtype=np.float32)
            label = np.asarray(data["label"], dtype=np.uint8) if "label" in data else None
        return image, label

    def _load_volume(self, mod: str, name: str) -> tuple[np.ndarray, np.ndarray]:
        path = self.mod_dirs[mod] / "test_vol_h5" / f"{name}.npy.h5"
        with h5py.File(path, "r") as data:
            image = np.asarray(data["image"], dtype=np.float32)
            label = np.asarray(data["label"], dtype=np.uint8) if "label" in data else None
        return image, label

    def __getitem__(self, idx: int) -> dict:
        name = self.sample_list[idx]

        if not self.is_volume:
            # 2D Slice for training
            img_bssfp, lbl_bssfp = self._load_slice("bSSFP", name)
            img_lge, _ = self._load_slice("LGE", name)
            img_t2w, _ = self._load_slice("T2w", name)

            label = canonicalize_label(lbl_bssfp, self.label_order)

            sample = {
                "image": img_bssfp,
                "image1": img_lge,
                "image2": img_t2w,
                "label": label,
                "case_name": name
            }

            if self.transform is not None:
                return self.transform(sample)

            # Default no-transform output
            ch3_image = np.stack([img_bssfp, img_lge, img_t2w], axis=0).astype(np.float32)
            return {
                "image": torch.from_numpy(ch3_image),
                "label": torch.from_numpy(label).long(),
                "low_res_label": torch.from_numpy(label).long(),
                "case_name": name
            }
        else:
            # 3D Volume for testing (H, W, D)
            img_bssfp, lbl_bssfp = self._load_volume("bSSFP", name)
            img_lge, _ = self._load_volume("LGE", name)
            img_t2w, _ = self._load_volume("T2w", name)

            label = canonicalize_label(lbl_bssfp, self.label_order)

            # Stack into (3, H, W, D)
            ch3_vol = np.stack([img_bssfp, img_lge, img_t2w], axis=0).astype(np.float32)

            return {
                "image": torch.from_numpy(ch3_vol),
                "label": torch.from_numpy(label).long(),
                "case_name": name
            }
