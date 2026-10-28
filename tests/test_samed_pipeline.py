"""Pipeline integration test for SAMed on MyoPS-380.
Verifies dataset, augmentation, label mapping, split hashes, and evaluator.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F

SAMED_ROOT = Path(__file__).resolve().parents[1]
if str(SAMED_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMED_ROOT))

from datasets.dataset_myops import MyopsDataset, RandomGenerator, canonicalize_label
from metrics.surface_distance import protocol_for_dataset, dataset_rows, summarize_dataset_rows


class TestSAMedPipeline(unittest.TestCase):

    def test_01_split_hashes(self):
        """Verify train.txt and test_vol.txt hashes match QUY_TAC_BENCHMARK_NO_VAL.md."""
        list_dir = SAMED_ROOT / "lists" / "lists_MyoPS380"
        train_path = list_dir / "train.txt"
        test_path = list_dir / "test_vol.txt"

        self.assertTrue(train_path.is_file(), f"Missing {train_path}")
        self.assertTrue(test_path.is_file(), f"Missing {test_path}")

        with open(train_path, "rb") as f:
            train_hash = hashlib.sha256(f.read()).hexdigest()
        with open(test_path, "rb") as f:
            test_hash = hashlib.sha256(f.read()).hexdigest()

        self.assertEqual(train_hash, "16248ec3630bb0493fb421aab5ffb4dc9b1d0fd49c8614d69d3098a92305f919")
        self.assertEqual(test_hash, "3237b31411833cedcb3ecd86d36edae356bd3f880d3fcfd915066f1c850d9cbc")

    def test_02_dataset_loading(self):
        """Verify MyopsDataset loads correct shapes, channels, and types."""
        data_root = Path("D:/NCKH/ChanhThuc/MyoPS380/Processed_data")
        list_dir = SAMED_ROOT / "lists" / "lists_MyoPS380"

        if not data_root.is_dir():
            self.skipTest(f"Data root {data_root} not found on this machine.")

        ds_train = MyopsDataset(
            base_dir=data_root,
            list_dir=list_dir,
            split="train",
            transform=RandomGenerator(output_size=[512, 512], low_res=[128, 128])
        )
        self.assertEqual(len(ds_train), 1972)

        sample = ds_train[0]
        self.assertEqual(sample["image"].shape, (3, 512, 512))
        self.assertEqual(sample["label"].shape, (128, 128))
        self.assertEqual(sample["low_res_label"].shape, (128, 128))

        # Check classes in range 0..3
        unique_labels = torch.unique(sample["label"]).tolist()
        for c in unique_labels:
            self.assertIn(c, [0, 1, 2, 3])

        ds_test = MyopsDataset(
            base_dir=data_root,
            list_dir=list_dir,
            split="test_vol"
        )
        self.assertEqual(len(ds_test), 76)
        vol_sample = ds_test[0]
        self.assertEqual(vol_sample["image"].ndim, 4)
        self.assertEqual(vol_sample["image"].shape[0], 3)
        self.assertEqual(vol_sample["image"].shape[1:3], (128, 128))

    def test_03_canonical_label_mapping(self):
        """Verify canonicalize_label accurately maps [0, 1, 3, 2]."""
        raw_labels = np.array([0, 1, 2, 3], dtype=np.uint8)
        canonical = canonicalize_label(raw_labels, label_order="legacy")
        # 0: BG -> 0, 1: Normal -> 1, 2: Scar -> 3, 3: Edema -> 2
        self.assertEqual(canonical.tolist(), [0, 1, 3, 2])

    def test_04_model_architecture_and_forward(self):
        """Verify SAMed model builds and runs forward pass on synthetic input."""
        try:
            import torchvision
            from segment_anything import sam_model_registry
            from sam_lora_image_encoder import LoRA_Sam
            from trainer import calc_loss
            from torch.nn.modules.loss import CrossEntropyLoss
            from utils import DiceLoss
        except ImportError as e:
            self.skipTest(f"Skipping model test on CPU host without torchvision/dependencies: {e}")

        img_size = 224
        sam, img_embedding_size = sam_model_registry["vit_b"](
            image_size=img_size,
            num_classes=4,
            checkpoint=None,
            pixel_mean=[0, 0, 0],
            pixel_std=[1, 1, 1]
        )
        model = LoRA_Sam(sam, r=4)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        self.assertGreater(trainable, 0)
        self.assertLess(trainable, total)

        x = torch.rand(2, 3, img_size, img_size)
        outputs = model(x, multimask_output=True, image_size=img_size)

        self.assertIn("masks", outputs)
        self.assertIn("low_res_logits", outputs)
        self.assertEqual(outputs["masks"].shape, (2, 4, img_size, img_size))

        low_res = img_embedding_size * 4
        y_low_res = torch.randint(0, 4, (2, low_res, low_res))
        ce = CrossEntropyLoss()
        dice = DiceLoss(4 + 1)
        loss, loss_ce, loss_dice = calc_loss(outputs, y_low_res, ce, dice, dice_weight=0.8)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_05_evaluator_protocol(self):
        """Verify myops380_voxel_v3 protocol is defined and works."""
        proto = protocol_for_dataset("myops380")
        self.assertEqual(proto["id"], "myops380_voxel_v3")
        self.assertEqual(proto["distance_unit"], "voxel")

        # Test on dummy predictions
        gt = np.zeros((128, 128, 5), dtype=np.uint8)
        gt[30:60, 30:60, :] = 1
        gt[40:50, 40:50, :] = 2
        gt[50:55, 50:55, :] = 3

        pred = gt.copy()
        pred[50:55, 50:55, :] = 3

        rows = dataset_rows(pred, gt, "dummy_case")
        summary = summarize_dataset_rows(rows)

        self.assertAlmostEqual(summary["scar"]["mean_dice"], 1.0, places=4)
        self.assertAlmostEqual(summary["edema_inclusive"]["mean_dice"], 1.0, places=4)
        self.assertAlmostEqual(summary["myocardial_ring"]["mean_dice"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
