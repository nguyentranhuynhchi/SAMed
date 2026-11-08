"""Aligned CINE/LGE/T2w dataset for SAMed on MyoPS380 with Canonical Target Adapter.

Input: Early-fusion 3-channel CMR [bSSFP, LGE, T2w] shape (B, 3, 128, 128).
Target Adapter: In-memory LUT [0, 1, 3, 2] converting Legacy to Canonical:
  0: Background
  1: Normal Myocardium
  2: Edema (exclusive)
  3: Scar
Augmentation: 100% exact RandomGenerator from SCAR / I-MMSeg standard.
"""
from __future__ import annotations

import os
from pathlib import Path
import random
import re

import h5py
import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


CLASS_NAMES = ("background", "normal_myocardium", "edema", "scar")
CANONICAL_LABEL_ORDER = "background_normal_edema_scar"
LEGACY_LABEL_ORDER = "background_normal_scar_edema"
LABEL_ORDERS = {
    "canonical": CANONICAL_LABEL_ORDER,
    "legacy": LEGACY_LABEL_ORDER,
    CANONICAL_LABEL_ORDER: CANONICAL_LABEL_ORDER,
    LEGACY_LABEL_ORDER: LEGACY_LABEL_ORDER,
}


def resolve_label_order(value):
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return LABEL_ORDERS[str(value)]
    except KeyError as exc:
        raise ValueError(f"Unknown label order {value!r}; use canonical or legacy.") from exc


def canonicalize_label(label, label_order="legacy"):
    """Convert 0..3 class IDs in memory once, never mutate the source array.
    
    Legacy (0: bg, 1: myo, 2: scar, 3: edema) -> Canonical (0: bg, 1: myo, 2: edema, 3: scar).
    """
    label = np.asarray(label)
    if not np.isfinite(label).all() or not np.equal(label, np.round(label)).all():
        raise ValueError("Segmentation masks must contain finite integer class IDs.")
    if label.size == 0 or label.min() < 0 or label.max() > 3:
        raise ValueError(f"Expected class IDs 0..3; found {np.unique(label).tolist()}.")
    result = label.astype(np.uint8, copy=True)
    if resolve_label_order(label_order) == LEGACY_LABEL_ORDER:
        lut = np.array([0, 1, 3, 2], dtype=np.uint8)
        result = lut[result]
    return result


def patient_id(sample_name):
    """Extract patient ID from caseXXXX_sliceNNN or caseXXXX."""
    name = str(sample_name).strip()
    for extension in (".npy.h5", ".nii.gz", ".npz", ".h5", ".nii"):
        if name.endswith(extension):
            name = name[:-len(extension)]
            break
    if not name or name in {".", ".."} or any(c in name for c in ("/", "\\", ":")):
        raise ValueError(f"Invalid sample name {sample_name!r}; manifests must contain basenames.")
    return re.sub(r"_slice\d+$", "", name)


def read_split_names(list_dir, split):
    path = Path(list_dir) / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate samples in {path}; regenerate manifests without append mode.")
    return names


# ---------------------------------------------------------------------------
# Data Augmentation: 100% Exact I-MMSeg / SCAR Standard
# ---------------------------------------------------------------------------

def random_rot_flip(image, image1, image2, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    image1 = np.rot90(image1, k)
    image2 = np.rot90(image2, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    image1 = np.flip(image1, axis=axis).copy()
    image2 = np.flip(image2, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, image1, image2, label


def random_rotate(image, image1, image2, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    image1 = ndimage.rotate(image1, angle, order=0, reshape=False)
    image2 = ndimage.rotate(image2, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, image1, image2, label


def add_gaussian_noise(image, image1, image2, label, mean=0, std=0.2):
    noise = np.random.normal(mean, std, image.shape)
    image = image + noise
    image1 = image1 + noise
    image2 = image2 + noise
    image = np.clip(image, 0, 1)
    image1 = np.clip(image1, 0, 1)
    image2 = np.clip(image2, 0, 1)
    return image, image1, image2, label


class ResizeGenerator:
    """Deterministic, alignment-preserving image resize and tensor conversion."""

    def __init__(self, output_size):
        if len(output_size) != 2 or any(int(v) <= 0 for v in output_size):
            raise ValueError("output_size must contain two positive dimensions.")
        self.output_size = tuple(int(v) for v in output_size)

    def __call__(self, sample):
        result = dict(sample)
        shape = sample["label"].shape
        if len(shape) != 2:
            raise ValueError("Slice transforms require 2D samples; volume resize belongs in inference.")
        for key in ("image", "image1", "image2", "label"):
            value = np.asarray(sample[key])
            if value.shape != shape:
                raise ValueError("Modalities and mask must have identical spatial shapes.")
            dtype = np.int64 if key == "label" else np.float32
            tensor = torch.from_numpy(np.array(value, dtype=dtype, copy=True, order="C"))
            if shape != self.output_size:
                if key == "label":
                    tensor = F.interpolate(tensor[None, None].float(), size=self.output_size, mode="nearest")[0, 0].long()
                else:
                    tensor = F.interpolate(tensor[None, None], size=self.output_size, mode="bilinear", align_corners=False)[0, 0]
            result[key] = tensor if key == "label" else tensor.unsqueeze(0)
        return result


class RandomGenerator:
    """100% exact reproduction of SCAR / I-MMSeg augmentation including RNG call order."""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, image1, image2, label = sample['image'], sample['image1'], sample['image2'], sample['label']

        if random.random() > 0.5:
            image, image1, image2, label = random_rot_flip(image, image1, image2, label)
        elif random.random() > 0.5:
            image, image1, image2, label = random_rotate(image, image1, image2, label)
        elif random.random() > 0.5:
            image, image1, image2, label = add_gaussian_noise(image, image1, image2, label)

        if random.random() > 0.5:
            shift_x = random.randint(-5, 5)
            shift_y = random.randint(-5, 5)
            image = np.roll(image, (shift_x, shift_y), axis=(0, 1))
            image1 = np.roll(image1, (shift_x, shift_y), axis=(0, 1))
            image2 = np.roll(image2, (shift_x, shift_y), axis=(0, 1))
            label = np.roll(label, (shift_x, shift_y), axis=(0, 1))

        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255)
            image1 = np.clip(image1 * factor, 0, 255)
            image2 = np.clip(image2 * factor, 0, 255)

        if random.random() > 0.5:
            gamma = random.uniform(0.8, 1.2)
            image = np.power(image / 255.0, gamma) * 255.0
            image1 = np.power(image1 / 255.0, gamma) * 255.0
            image2 = np.power(image2 / 255.0, gamma) * 255.0

        if random.random() > 0.5:
            top = random.randint(0, image.shape[0] // 5)
            left = random.randint(0, image.shape[1] // 5)
            image = image[top:top + self.output_size[0], left:left + self.output_size[1]]
            image1 = image1[top:top + self.output_size[0], left:left + self.output_size[1]]
            image2 = image2[top:top + self.output_size[0], left:left + self.output_size[1]]
            label = label[top:top + self.output_size[0], left:left + self.output_size[1]]

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            image1 = zoom(image1, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            image2 = zoom(image2, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image1 = torch.from_numpy(image1.astype(np.float32)).unsqueeze(0)
        image2 = torch.from_numpy(image2.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        result = dict(sample)
        result.update(image=image, image1=image1, image2=image2, label=label.long())
        return result


# ---------------------------------------------------------------------------
# Dataset Definition
# ---------------------------------------------------------------------------

class MyopsDataset(Dataset):
    """MyoPS380 dataset providing 3-channel early fusion tensor and canonical targets."""

    def __init__(
        self,
        base_dir,
        base_dir1=None,
        base_dir2=None,
        list_dir=None,
        split="train",
        transform=None,
        label_order="legacy",
        low_res=(32, 32),
        validate_files=False,
    ):
        # Support multiple calling conventions:
        # 1. SCAR style: MyopsDataset(bssfp_dir, lge_dir, t2w_dir, list_dir, split)
        # 2. Single root positional: MyopsDataset(data_root, list_dir, split)
        # 3. Single root keyword: MyopsDataset(base_dir=data_root, list_dir=list_dir, split="train")
        if base_dir1 is not None and base_dir2 is not None and list_dir is not None:
            self.roots = (Path(base_dir), Path(base_dir1), Path(base_dir2))
            actual_list_dir = list_dir
            actual_split = split
        else:
            root = Path(base_dir)
            if not (root / "bSSFP").exists() and (root / "Processed_data" / "bSSFP").exists():
                root = root / "Processed_data"
            actual_list_dir = list_dir if list_dir is not None else base_dir1
            actual_split = split if base_dir2 is None else base_dir2
            is_vol = actual_split.endswith("vol") or actual_split in {"test", "volume"}
            subfolder = "test_vol_h5" if is_vol else "train_npz"
            self.roots = (
                root / "bSSFP" / subfolder,
                root / "LGE" / subfolder,
                root / "T2w" / subfolder,
            )

        self.split = actual_split
        self.list_dir = actual_list_dir
        self.transform = transform
        self.label_order = label_order
        self.low_res = tuple(low_res) if low_res is not None else (32, 32)
        self.sample_list = read_split_names(self.list_dir, self.split)
        self.is_volume = self.split.endswith("vol") or self.split in {"test", "volume"}
        self.extension = ".npy.h5" if self.is_volume else ".npz"

        if validate_files:
            required = [root / f"{name}{self.extension}" for name in self.sample_list for root in self.roots]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} aligned modality file(s); first: {missing[:5]}")

    def __len__(self):
        return len(self.sample_list)

    def _read(self, path):
        if path.name.endswith(".h5"):
            with h5py.File(path, "r") as data:
                image = data["image"][:]
                label = data["label"][:] if "label" in data else None
                metadata = dict(data.attrs)
                for key in ("spacing", "affine"):
                    if key in data:
                        metadata[key] = data[key][:]
        else:
            with np.load(path, allow_pickle=False) as data:
                image = data["image"].copy()
                label = data["label"].copy() if "label" in data else None
                metadata = {
                    key: data[key].copy()
                    for key in ("label_order", "spacing", "affine", "spacing_unit", "patient_id", "axis_order", "normalization")
                    if key in data
                }
        if label is not None:
            label = canonicalize_label(label, self.label_order)
        return np.asarray(image, dtype=np.float32), label, metadata

    def __getitem__(self, idx):
        name = self.sample_list[idx]
        modalities = [self._read(root / f"{name}{self.extension}") for root in self.roots]
        image, label, metadata = modalities[0]
        image1 = modalities[1][0]
        image2 = modalities[2][0]

        if self.is_volume:
            spacing = np.asarray(metadata.get("spacing", [np.nan] * 3), dtype=np.float64)
            affine = np.asarray(metadata.get("affine", np.full((4, 4), np.nan)), dtype=np.float64)
            raw_unit = metadata.get("spacing_unit", "unknown")
            if isinstance(raw_unit, np.ndarray):
                raw_unit = raw_unit.item()
            if isinstance(raw_unit, bytes):
                raw_unit = raw_unit.decode("utf-8")
            spacing_unit = str(raw_unit)

            valid_geom = bool(
                spacing.shape == (3,)
                and affine.shape == (4, 4)
                and np.isfinite(spacing).all()
                and (spacing > 0).all()
                and np.isfinite(affine).all()
                and abs(np.linalg.det(affine[:3, :3])) > 1e-12
            )
            has_geom = valid_geom and spacing_unit == "mm"

            return {
                "image": image,
                "image1": image1,
                "image2": image2,
                "label": label.astype(np.int64) if label is not None else None,
                "case_name": name,
                "patient_id": patient_id(name),
                "spacing": spacing,
                "affine": affine,
                "spacing_unit": spacing_unit,
                "has_geometry": has_geom,
                "has_affine": valid_geom,
            }

        sample = {
            "image": image,
            "image1": image1,
            "image2": image2,
            "label": label.astype(np.int64),
            "case_name": name,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        # Early-fusion: concatenate 3 modalities [bSSFP, LGE, T2w] -> shape (3, H, W)
        if isinstance(sample["image"], torch.Tensor):
            if sample["image"].ndim == 3 and sample["image"].shape[0] == 1:
                img_tensor = torch.cat([sample["image"], sample["image1"], sample["image2"]], dim=0)
            elif sample["image"].ndim == 2:
                img_tensor = torch.stack([sample["image"], sample["image1"], sample["image2"]], dim=0)
            else:
                img_tensor = sample["image"]
            label_tensor = sample["label"].long()
        else:
            img_tensor = torch.from_numpy(np.stack([sample["image"], sample["image1"], sample["image2"]], axis=0)).float()
            label_tensor = torch.from_numpy(sample["label"]).long()

        # Compute low-res label for SAMed mask decoder output resolution (H/4, W/4 = 32, 32)
        low_res_label = F.interpolate(
            label_tensor[None, None].float(),
            size=self.low_res,
            mode="nearest"
        )[0, 0].long()

        return {
            "image": img_tensor,
            "label": label_tensor,
            "low_res_label": low_res_label,
            "case_name": name,
        }
