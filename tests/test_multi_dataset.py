"""tests/test_multi_dataset.py
===========================
Comprehensive Unit Test Suite for Multi-Dataset Support (MyoPS380 & MyoPS++),
Double-Swap Prevention, Spacing & Distance Metrics, Dynamic Updates, and CLI Parity in SAMed.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.dataset_myops import (
    CLASS_NAMES,
    MYOPS380_EXPECTED_TEST_VOLUMES,
    MYOPS380_EXPECTED_TRAIN_PATIENTS,
    MYOPS380_EXPECTED_TRAIN_SLICES,
    MYOPSPP_EXPECTED_TEST_VOLUMES,
    MYOPSPP_EXPECTED_TRAIN_PATIENTS,
    MYOPSPP_EXPECTED_TRAIN_SLICES,
    MyopsDataset,
    canonicalize_label,
    normalize_dataset_name,
    patient_id,
    read_split_names,
    resolve_default_list_dir,
    resolve_label_order,
)
from utils.metrics import (
    BENCHMARK_PROTOCOLS,
    BENCHMARK_PROTOCOL,
    binary_metrics,
    dataset_rows,
    summarize_dataset_rows,
)


class TestMultiDatasetSplits(unittest.TestCase):
    """Test data split manifests and line counts for both MyoPS380 and MyoPS++."""

    def test_normalize_dataset_name(self):
        self.assertEqual(normalize_dataset_name("myops380"), "myops380")
        self.assertEqual(normalize_dataset_name("myopspp"), "myopspp")
        self.assertEqual(normalize_dataset_name("myopspp_unet2d_roi128_1mm"), "myopspp")
        self.assertEqual(normalize_dataset_name("MYOPSPP"), "myopspp")
        self.assertEqual(normalize_dataset_name("MYOPS380"), "myops380")
        self.assertEqual(normalize_dataset_name(None), "myops380")
        with self.assertRaises(ValueError):
            normalize_dataset_name("unknown_dataset")

    def test_myops380_split_manifests(self):
        train_file = REPO_ROOT / "lists" / "myops380_no_val" / "train.txt"
        test_file = REPO_ROOT / "lists" / "myops380_no_val" / "test_vol.txt"

        self.assertTrue(train_file.is_file(), f"Missing {train_file}")
        self.assertTrue(test_file.is_file(), f"Missing {test_file}")

        train_lines = read_split_names(train_file.parent, "train")
        test_lines = read_split_names(test_file.parent, "test_vol")

        self.assertEqual(len(train_lines), MYOPS380_EXPECTED_TRAIN_SLICES, f"Expected {MYOPS380_EXPECTED_TRAIN_SLICES} slices")
        train_patients = {patient_id(line) for line in train_lines}
        self.assertEqual(len(train_patients), MYOPS380_EXPECTED_TRAIN_PATIENTS, f"Expected {MYOPS380_EXPECTED_TRAIN_PATIENTS} patients")

        self.assertEqual(len(test_lines), MYOPS380_EXPECTED_TEST_VOLUMES, f"Expected {MYOPS380_EXPECTED_TEST_VOLUMES} test volumes")
        test_patients = {patient_id(line) for line in test_lines}
        self.assertEqual(len(test_patients), MYOPS380_EXPECTED_TEST_VOLUMES)

        # Zero patient leakage
        self.assertEqual(len(train_patients & test_patients), 0, "Data leakage between train and test in MyoPS380!")

    def test_myopspp_split_manifests(self):
        train_file = REPO_ROOT / "lists" / "lists_MyoPSpp" / "train.txt"
        test_file = REPO_ROOT / "lists" / "lists_MyoPSpp" / "test_vol.txt"

        self.assertTrue(train_file.is_file(), f"Missing {train_file}")
        self.assertTrue(test_file.is_file(), f"Missing {test_file}")

        train_lines = read_split_names(train_file.parent, "train")
        test_lines = read_split_names(test_file.parent, "test_vol")

        self.assertEqual(len(train_lines), MYOPSPP_EXPECTED_TRAIN_SLICES, f"Expected {MYOPSPP_EXPECTED_TRAIN_SLICES} slices")
        train_patients = {patient_id(line) for line in train_lines}
        self.assertEqual(len(train_patients), MYOPSPP_EXPECTED_TRAIN_PATIENTS, f"Expected {MYOPSPP_EXPECTED_TRAIN_PATIENTS} patients")

        self.assertEqual(len(test_lines), MYOPSPP_EXPECTED_TEST_VOLUMES, f"Expected {MYOPSPP_EXPECTED_TEST_VOLUMES} test volumes")
        test_patients = {patient_id(line) for line in test_lines}
        self.assertEqual(len(test_patients), MYOPSPP_EXPECTED_TEST_VOLUMES)

        # Zero patient leakage
        self.assertEqual(len(train_patients & test_patients), 0, "Data leakage between train and test in MyoPS++!")

        # Verify case casing preserved (e.g., Case2011, Case3003)
        uppercase_cases = [p for p in test_patients if p.startswith("Case")]
        self.assertEqual(len(uppercase_cases), 16, "All 16 MyoPS++ test cases must preserve 'CaseXXXX' casing")


class TestDoubleSwapPreventionAndLabelOrder(unittest.TestCase):
    """Test label mapping adapter and double-swap prevention."""

    def test_canonicalize_label_legacy(self):
        # Legacy label order: 0: BG, 1: Normal Myo, 2: Scar, 3: Edema-exclusive
        # Should convert to Canonical: 0: BG, 1: Normal Myo, 2: Edema-exclusive, 3: Scar
        legacy = np.array([0, 1, 2, 3], dtype=np.uint8)
        canonical = canonicalize_label(legacy, label_order="legacy")
        np.testing.assert_array_equal(canonical, np.array([0, 1, 3, 2], dtype=np.uint8))

    def test_canonicalize_label_canonical_and_identity(self):
        # Canonical / Identity: MUST NOT swap 2 and 3!
        canonical = np.array([0, 1, 2, 3], dtype=np.uint8)
        out_canon = canonicalize_label(canonical, label_order="canonical")
        np.testing.assert_array_equal(out_canon, np.array([0, 1, 2, 3], dtype=np.uint8))

        out_ident = canonicalize_label(canonical, label_order="identity")
        np.testing.assert_array_equal(out_ident, np.array([0, 1, 2, 3], dtype=np.uint8))

    def test_double_swap_prevention_in_evaluation(self):
        # Synthetic volume matching canonical prediction:
        # 1: Myocardium, 2: Edema, 3: Scar
        pred_canonical = np.zeros((16, 16, 4), dtype=np.uint8)
        pred_canonical[2:6, 2:6, :] = 1
        pred_canonical[6:10, 6:10, :] = 2  # Edema
        pred_canonical[10:14, 10:14, :] = 3  # Scar

        # 1. On MyoPS++, GT is ALREADY Canonical on disk:
        gt_pp = pred_canonical.copy()
        rows_pp = dataset_rows(
            prediction=pred_canonical,
            target=gt_pp,
            case="Case2011",
            compute_distance=False,
            dataset_id="myopspp",
        )
        summary_pp = summarize_dataset_rows(rows_pp, dataset_id="myopspp")
        self.assertAlmostEqual(summary_pp["edema"]["mean_dice"], 1.0, places=5)
        self.assertAlmostEqual(summary_pp["scar"]["mean_dice"], 1.0, places=5)
        self.assertAlmostEqual(summary_pp["myocardial_ring"]["mean_dice"], 1.0, places=5)
        self.assertAlmostEqual(summary_pp["edema_inclusive"]["mean_dice"], 1.0, places=5)

        # 2. On MyoPS380, GT is LEGACY on disk: 2 is Scar, 3 is Edema.
        # Dataset adapter converts legacy GT to canonical via LUT [0, 1, 3, 2]
        gt_legacy = np.zeros((16, 16, 4), dtype=np.uint8)
        gt_legacy[2:6, 2:6, :] = 1
        gt_legacy[6:10, 6:10, :] = 3  # Edema is ID 3 in Legacy
        gt_legacy[10:14, 10:14, :] = 2  # Scar is ID 2 in Legacy

        gt_380_canonical = canonicalize_label(gt_legacy, label_order="legacy")
        rows_380 = dataset_rows(
            prediction=pred_canonical,
            target=gt_380_canonical,
            case="case0001",
            compute_distance=False,
            dataset_id="myops380",
        )
        summary_380 = summarize_dataset_rows(rows_380, dataset_id="myops380")
        self.assertAlmostEqual(summary_380["edema"]["mean_dice"], 1.0, places=5)
        self.assertAlmostEqual(summary_380["scar"]["mean_dice"], 1.0, places=5)
        self.assertAlmostEqual(summary_380["myocardial_ring"]["mean_dice"], 1.0, places=5)


class TestDistanceMetricsAndSpacing(unittest.TestCase):
    """Test physical spacing handling and mm vs voxel distance calculation."""

    def test_binary_metrics_mm_vs_voxel(self):
        pred = np.zeros((20, 20, 5), dtype=bool)
        truth = np.zeros((20, 20, 5), dtype=bool)

        pred[5:15, 5:15, 1:4] = True
        truth[6:16, 6:16, 1:4] = True

        # Voxel distance (spacing=None)
        m_voxel = binary_metrics(pred, truth, spacing=None, compute_distance=True)
        self.assertIsNotNone(m_voxel["hd95"])
        self.assertIsNotNone(m_voxel["asd"])

        # Physical spacing in mm (e.g. 2.0, 2.0, 5.0)
        spacing = [2.0, 2.0, 5.0]
        m_mm = binary_metrics(pred, truth, spacing=spacing, compute_distance=True)
        self.assertIsNotNone(m_mm["hd95"])
        self.assertIsNotNone(m_mm["asd"])

        # With 2x spacing, physical distance should be greater than voxel distance for in-plane shifts
        self.assertGreater(m_mm["hd95"], m_voxel["hd95"])

    def test_dataset_rows_and_summary_mm_vs_voxel(self):
        pred = np.zeros((20, 20, 5), dtype=np.uint8)
        truth = np.zeros((20, 20, 5), dtype=np.uint8)

        pred[5:15, 5:15, 1:4] = 2  # Edema
        truth[6:16, 6:16, 1:4] = 2

        # 1. MyoPS++ (mm distance)
        rows_pp = dataset_rows(
            prediction=pred,
            target=truth,
            case="Case2011",
            compute_distance=True,
            dataset_id="myopspp",
            spacing=[1.0, 1.0, 2.5],
        )
        self.assertEqual(rows_pp[0]["hd95_unit"], "mm")
        self.assertIn("hd95_mm", rows_pp[0])
        self.assertIn("asd_mm", rows_pp[0])

        summary_pp = summarize_dataset_rows(rows_pp, dataset_id="myopspp")
        self.assertIn("mean_hd95_mm", summary_pp["edema"])
        self.assertIn("avg_pathology_hd95_mm", summary_pp)

        # 2. MyoPS380 (voxel distance)
        rows_380 = dataset_rows(
            prediction=pred,
            target=truth,
            case="case0001",
            compute_distance=True,
            dataset_id="myops380",
            spacing=None,
        )
        self.assertEqual(rows_380[0]["hd95_unit"], "voxel")
        self.assertIn("hd95_voxel", rows_380[0])
        self.assertIn("asd_voxel", rows_380[0])

        summary_380 = summarize_dataset_rows(rows_380, dataset_id="myops380")
        self.assertIn("mean_hd95_voxel", summary_380["edema"])
        self.assertIn("avg_pathology_hd95_voxel", summary_380)

    def test_clinical_edema_inclusive_regions(self):
        # Test that edema_inclusive captures both ID 2 and ID 3 (LGE-abnormal)
        pred = np.zeros((10, 10, 2), dtype=np.uint8)
        truth = np.zeros((10, 10, 2), dtype=np.uint8)

        pred[1:4, 1:4, :] = 2  # Edema
        pred[5:8, 5:8, :] = 3  # Scar

        truth[1:4, 1:4, :] = 2
        truth[5:8, 5:8, :] = 3

        rows = dataset_rows(pred, truth, case="CaseTest", compute_distance=False, dataset_id="myopspp")
        summary = summarize_dataset_rows(rows, dataset_id="myopspp")

        self.assertAlmostEqual(summary["edema"]["mean_dice"], 1.0)
        self.assertAlmostEqual(summary["scar"]["mean_dice"], 1.0)
        self.assertAlmostEqual(summary["edema_inclusive"]["mean_dice"], 1.0)


class TestDynamicUpdatesAndSchedule(unittest.TestCase):
    """Test dynamic total update calculation for both datasets."""

    def test_total_updates_myops380(self):
        # 1,972 slices / batch_size 16 = 124 batches per epoch
        # 300 epochs * 124 batches = 37,200 updates
        batches_per_epoch = int(np.ceil(1972 / 16))
        self.assertEqual(batches_per_epoch, 124)
        updates = 300 * batches_per_epoch
        self.assertEqual(updates, 37200)

    def test_total_updates_myopspp(self):
        # 296 slices / batch_size 16 = 19 batches per epoch (18 batches of 16 + 1 batch of 8)
        # 300 epochs * 19 batches = 5,700 updates
        batches_per_epoch = int(np.ceil(296 / 16))
        self.assertEqual(batches_per_epoch, 19)
        updates = 300 * batches_per_epoch
        self.assertEqual(updates, 5700)


class TestCLIParity(unittest.TestCase):
    """Test CLI argument parsing and flags across all entrypoints."""

    def test_train_myops_parser_flags(self):
        import train_myops
        parser = train_myops.build_parser()

        # Test --dataset and --split_dir
        args = parser.parse_args([
            "--dataset", "myopspp",
            "--split_dir", "lists/lists_MyoPSpp",
            "--measure_efficiency",
        ])
        self.assertEqual(args.dataset, "myopspp")
        self.assertEqual(args.list_dir, "lists/lists_MyoPSpp")
        self.assertTrue(args.measure_efficiency)

    def test_evaluate_myops_parser_flags(self):
        import evaluate_myops
        parser = evaluate_myops.build_parser()

        args = parser.parse_args([
            "--checkpoint", "runs/samed_seed1234/checkpoints/samed_seed1234.pt",
            "--dataset", "myopspp",
            "--split_dir", "lists/lists_MyoPSpp",
        ])
        self.assertEqual(args.dataset, "myopspp")
        self.assertEqual(args.list_dir, "lists/lists_MyoPSpp")

    def test_predict_parser_flags(self):
        import predict
        parser = predict.build_parser()

        args = parser.parse_args([
            "--checkpoint", "runs/samed_seed1234/checkpoints/samed_seed1234.pt",
            "--dataset", "myopspp",
            "--split_dir", "lists/lists_MyoPSpp",
        ])
        self.assertEqual(args.dataset, "myopspp")
        self.assertEqual(args.list_dir, "lists/lists_MyoPSpp")

    def test_evaluate_predictions_parser_flags(self):
        import evaluate_predictions
        parser = evaluate_predictions.build_parser()

        args = parser.parse_args([
            "--pred_dir", "runs/samed_seed1234/test_predictions",
            "--dataset", "myopspp",
            "--split_dir", "lists/lists_MyoPSpp",
        ])
        self.assertEqual(args.dataset, "myopspp")
        self.assertEqual(args.list_dir, "lists/lists_MyoPSpp")


class TestEndToEndSmokeMultiDataset(unittest.TestCase):
    """Smoke test running train_myops and evaluate_myops on synthetic data for both datasets."""

    def test_smoke_myopspp_and_myops380(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_root = tmp_path / "Processed_data"
            for mod in ("bSSFP", "LGE", "T2w"):
                (data_root / mod / "train_npz").mkdir(parents=True)
                (data_root / mod / "test_vol_h5").mkdir(parents=True)

            # 1. MyoPS++ synthetic data (Canonical labels: 1=Myo, 2=Edema, 3=Scar)
            list_pp = tmp_path / "lists_MyoPSpp"
            list_pp.mkdir(parents=True)
            pp_slices = ["Case2001_slice000", "Case2001_slice001"]
            (list_pp / "train.txt").write_text("\n".join(pp_slices) + "\n", encoding="utf-8")
            (list_pp / "test_vol.txt").write_text("Case2001\n", encoding="utf-8")

            lbl_canonical = np.zeros((128, 128), dtype=np.uint8)
            lbl_canonical[30:70, 30:70] = 1
            lbl_canonical[40:55, 40:55] = 2
            lbl_canonical[55:65, 40:55] = 3

            for sl in pp_slices:
                for mod in ("bSSFP", "LGE", "T2w"):
                    img = np.random.rand(128, 128).astype(np.float32)
                    np.savez(data_root / mod / "train_npz" / f"{sl}.npz", image=img, label=lbl_canonical)

            vol_lbl_pp = np.zeros((128, 128, 2), dtype=np.uint8)
            vol_lbl_pp[30:70, 30:70, :] = 1
            vol_lbl_pp[40:55, 40:55, :] = 2
            vol_lbl_pp[55:65, 40:55, :] = 3

            for mod in ("bSSFP", "LGE", "T2w"):
                vol_img = np.random.rand(128, 128, 2).astype(np.float32)
                with h5py.File(data_root / mod / "test_vol_h5" / "Case2001.npy.h5", "w") as hf:
                    hf.create_dataset("image", data=vol_img)
                    hf.create_dataset("label", data=vol_lbl_pp)
                    hf.create_dataset("spacing", data=np.array([1.0, 1.0, 2.5], dtype=np.float32))

            # Train 1 epoch on MyoPS++ in CPU mode
            run_pp = tmp_path / "run_pp"
            cmd_train_pp = [
                sys.executable,
                str(REPO_ROOT / "train_myops.py"),
                "--dataset", "myopspp",
                "--data_root", str(data_root),
                "--list_dir", str(list_pp),
                "--output_dir", str(run_pp),
                "--batch_size", "2",
                "--max_epochs", "1",
                "--amp", "none",
                "--num_workers", "0",
                "--device", "cpu",
            ]
            res_train_pp = subprocess.run(cmd_train_pp, capture_output=True, text=True)
            self.assertEqual(res_train_pp.returncode, 0, msg=res_train_pp.stderr)

            ckpt_pp = run_pp / "checkpoints" / "samed_seed1234.pt"
            self.assertTrue(ckpt_pp.is_file())

            # Evaluate on MyoPS++
            eval_pp = run_pp / "evaluation"
            cmd_eval_pp = [
                sys.executable,
                str(REPO_ROOT / "evaluate_myops.py"),
                "--dataset", "myopspp",
                "--checkpoint", str(ckpt_pp),
                "--data_root", str(data_root),
                "--list_dir", str(list_pp),
                "--output_dir", str(eval_pp),
                "--batch_size", "2",
                "--amp", "none",
                "--device", "cpu",
            ]
            res_eval_pp = subprocess.run(cmd_eval_pp, capture_output=True, text=True)
            self.assertEqual(res_eval_pp.returncode, 0, msg=res_eval_pp.stderr)

            self.assertTrue((eval_pp / "metrics.json").is_file())
            self.assertTrue((eval_pp / "per_case.csv").is_file())
            self.assertTrue((eval_pp / "test_predictions" / "Case2001_pred.npz").is_file())

            pp_metrics = json.loads((eval_pp / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(pp_metrics["dataset_id"], "myopspp")
            self.assertEqual(pp_metrics["distance_unit"], "mm")
            self.assertIn("mean_hd95_mm", pp_metrics["edema"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
