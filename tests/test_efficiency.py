"""Unit and CLI integration tests for SAMed efficiency benchmarking on MyoPS380."""
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

from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
from utils.efficiency import (
    BYTES_PER_GB,
    BYTES_PER_MB,
    SAMedInferenceWrapper,
    build_efficiency_payload,
    check_rule_compliance,
    count_parameters,
    format_efficiency_report,
    measure_flops,
    measure_inference_efficiency,
    save_efficiency_report,
    summarize_training_efficiency,
)


class SAMedEfficiencyTests(unittest.TestCase):
    def test_samed_complexity_and_inference_cpu(self) -> None:
        sam, emb_size = sam_model_registry["vit_b"](
            image_size=128,
            num_classes=3,
            checkpoint=None,
            pixel_mean=[0, 0, 0],
            pixel_std=[1, 1, 1],
        )
        self.assertEqual(emb_size, 8)
        net = LoRA_Sam(sam, r=4).float()
        net.train()
        orig_requires_grad = [p.requires_grad for p in net.parameters()]

        params_info = count_parameters(net)
        self.assertEqual(params_info["total_params"], 90_729_008)
        self.assertEqual(params_info["trainable_params"], 4_212_016)
        self.assertEqual(params_info["trainable_params_excel"], "4.21M")
        self.assertEqual(params_info["total_params_excel"], "90.73M")

        wrapper = SAMedInferenceWrapper(net, image_size=128, multimask_output=True)
        with torch.no_grad():
            out = wrapper(torch.randn(1, 3, 128, 128, dtype=torch.float32))
        self.assertEqual(tuple(out.shape), (1, 4, 128, 128))

        flops_wrapped = measure_flops(wrapper, input_shape=(1, 3, 128, 128), device=torch.device("cpu"))
        flops_direct = measure_flops(net, input_shape=(1, 3, 128, 128), device=torch.device("cpu"))
        self.assertEqual(flops_wrapped["flops_total"], flops_direct["flops_total"])
        self.assertGreater(flops_wrapped["flops_total"], 17_000_000_000)
        self.assertLess(flops_wrapped["flops_total"], 18_000_000_000)
        self.assertAlmostEqual(flops_wrapped["flops_g_excel"], 17.23, places=2)

        # Ensure exact per-parameter requires_grad flags and training mode are restored after measure_flops
        self.assertTrue(net.training)
        self.assertEqual([p.requires_grad for p in net.parameters()], orig_requires_grad)
        restored_params = count_parameters(net)
        self.assertEqual(restored_params["trainable_params"], 4_212_016)

        inf_info = measure_inference_efficiency(
            wrapper, input_shape=(1, 3, 128, 128), device=torch.device("cpu"), warmup=2, repeats=3
        )
        self.assertTrue(net.training)
        self.assertGreater(inf_info["latency_ms_per_slice"], 0.0)
        self.assertEqual(inf_info["batch_size"], 1)
        self.assertEqual(inf_info["peak_vram_mb"], 0.0)
        self.assertEqual(inf_info["peak_vram_gb"], 0.0)

    def test_summarize_training_efficiency_skips_first_5_epochs(self) -> None:
        epoch_times = [60.0] * 5 + [12.0] * 295
        peak_bytes = int(3072 * BYTES_PER_MB)
        summary = summarize_training_efficiency(
            epoch_times_s=epoch_times,
            total_wall_clock_s=sum(epoch_times) + 90.0,
            peak_vram_bytes=peak_bytes,
            planned_epochs=300,
            warmup_epochs=5,
            peak_reserved_bytes=int(3.5 * BYTES_PER_GB),
        )
        self.assertEqual(summary["warmup_epochs_skipped"], 5)
        self.assertEqual(summary["measured_epoch_range"], "6..300")
        self.assertEqual(summary["time_per_epoch_s"], 12.0)
        self.assertEqual(summary["epoch_times_1_to_5_s"], [60.0] * 5)
        self.assertAlmostEqual(summary["total_time_hours"], sum(epoch_times) / 3600.0, places=3)
        self.assertEqual(summary["peak_vram_mb"], 3072.0)
        self.assertEqual(summary["peak_vram_gb"], 3.0)
        self.assertEqual(summary["peak_vram_reserved_mb"], 3584.0)
        self.assertEqual(summary["peak_vram_reserved_gb"], 3.5)

        short_summary = summarize_training_efficiency(
            epoch_times_s=[5.0, 7.0],
            total_wall_clock_s=12.5,
            peak_vram_bytes=0,
            planned_epochs=2,
            warmup_epochs=5,
        )
        self.assertEqual(short_summary["warmup_epochs_skipped"], 0)
        self.assertEqual(short_summary["measured_epoch_range"], "1..2")
        self.assertEqual(short_summary["time_per_epoch_s"], 6.0)

    def test_rule_compliance_flags(self) -> None:
        ok, warnings, hw = check_rule_compliance(
            device=torch.device("cpu"),
            amp_mode="none",
            num_workers=0,
            pin_memory=True,
            batch_size=16,
            planned_epochs=300,
            accum_steps=1,
        )
        self.assertFalse(ok)
        self.assertTrue(any("CPU" in w for w in warnings))
        self.assertEqual(hw["precision"], "fp32")

        ok2, warnings2, _ = check_rule_compliance(
            device=torch.device("cpu"),
            amp_mode="auto",
            num_workers=4,
            pin_memory=True,
            batch_size=8,
            planned_epochs=10,
            accum_steps=2,
        )
        self.assertFalse(ok2)
        self.assertGreaterEqual(len(warnings2), 6)

    def test_format_efficiency_report_contains_both_units(self) -> None:
        payload = build_efficiency_payload(
            model_name="SAMed",
            model_id="samed",
            seed=1234,
            params_info={
                "total_params": 90_729_008,
                "trainable_params": 4_212_016,
                "total_params_m": 90.729,
                "trainable_params_m": 4.212,
                "trainable_params_excel": "4.21M",
                "total_params_excel": "90.73M",
            },
            flops_info={"flops_total": 17_230_188_544, "flops_g": 17.2302},
            inference_info={
                "latency_ms_per_slice": 4.5,
                "warmup_runs": 30,
                "measure_runs": 100,
                "peak_vram_mb": 512.0,
                "peak_vram_gb": 0.5,
                "peak_vram_reserved_mb": 600.0,
            },
            training_info={
                "planned_epochs": 300,
                "completed_epochs": 300,
                "warmup_epochs_skipped": 5,
                "measured_epoch_range": "6..300",
                "time_per_epoch_s": 11.2,
                "total_time_seconds": 3360.0,
                "total_time_hours": 0.9333,
                "total_wall_clock_hours": 0.95,
                "peak_vram_mb": 2048.0,
                "peak_vram_gb": 2.0,
                "peak_vram_reserved_mb": 2400.0,
            },
            rule_compliant=True,
            compliance_warnings=[],
            hardware_info={
                "gpu_name": "NVIDIA A100-SXM4-40GB",
                "visible_gpu_count": 1,
                "pytorch_version": "2.4.0",
                "cuda_version": "12.1",
                "precision": "fp32",
                "amp_mode": "none",
                "num_workers": 0,
                "pin_memory": True,
                "batch_size": 16,
            },
        )
        txt = format_efficiency_report(payload)
        self.assertIn("2048.00 MB (2.0000 GB)", txt)
        self.assertIn("512.00 MB (0.5000 GB)", txt)
        self.assertIn("4.21M", txt)
        self.assertIn("17.23 G", txt)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_json = tmp_path / "eff.v1.json"
            out_txt = tmp_path / "summary.v1.txt"
            save_efficiency_report(payload, [out_json], [out_txt])
            self.assertTrue(out_json.is_file())
            self.assertTrue(out_txt.is_file())
            self.assertFalse((tmp_path / "eff.v1.json.tmp").exists())
            self.assertFalse((tmp_path / "summary.v1.txt.tmp").exists())
            self.assertEqual(json.loads(out_json.read_text(encoding="utf-8"))["seed"], 1234)

    def test_fast_fail_invalid_args_and_explicit_missing_ckpt(self) -> None:
        import train_myops

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with self.assertRaises(ValueError):
                train_myops.main([
                    "--output_dir", str(tmp_path / "bad_epochs"),
                    "--max_epochs", "0",
                    "--device", "cpu",
                    "--amp", "none",
                ])

            with self.assertRaises(FileNotFoundError):
                train_myops.main([
                    "--output_dir", str(tmp_path / "missing_ckpt"),
                    "--ckpt", str(tmp_path / "nonexistent_sam.pth"),
                    "--efficiency_only",
                    "--device", "cpu",
                    "--amp", "none",
                ])
            self.assertFalse((tmp_path / "missing_ckpt").exists())

    def test_cli_flag_aliases(self) -> None:
        import train_myops

        parser = train_myops.build_parser()
        a1 = parser.parse_args(["--no_measure_efficiency", "--no_pin_memory", "--num-workers", "0"])
        self.assertFalse(a1.measure_efficiency)
        self.assertFalse(a1.pin_memory)
        self.assertEqual(a1.num_workers, 0)

        a2 = parser.parse_args(["--no-measure-efficiency", "--pin-memory", "--efficiency-only"])
        self.assertFalse(a2.measure_efficiency)
        self.assertTrue(a2.pin_memory)
        self.assertTrue(a2.efficiency_only)

        a3 = parser.parse_args(["--resume"])
        self.assertEqual(a3.resume, "auto")

    def test_cli_train_and_eval_efficiency_modes_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_root = tmp_path / "Processed_data"
            list_dir = tmp_path / "lists"
            list_dir.mkdir(parents=True)

            for mod in ("bSSFP", "LGE", "T2w"):
                (data_root / mod / "train_npz").mkdir(parents=True)
                (data_root / mod / "test_vol_h5").mkdir(parents=True)

            train_lines: list[str] = []
            for idx in range(2):
                name = f"case9999_slice{idx:03d}"
                train_lines.append(name)
                label = np.zeros((128, 128), dtype=np.uint8)
                label[40:80, 40:80] = 1
                label[50:65, 50:65] = 2
                label[65:75, 50:65] = 3
                for mod in ("bSSFP", "LGE", "T2w"):
                    img = np.random.rand(128, 128).astype(np.float32)
                    np.savez(data_root / mod / "train_npz" / f"{name}.npz", image=img, label=label)

            (list_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

            # Synthetic 3D test volume for evaluate_myops smoke test
            vol_lbl = np.zeros((128, 128, 2), dtype=np.uint8)
            vol_lbl[40:80, 40:80, :] = 1
            vol_lbl[50:65, 50:65, :] = 2
            vol_lbl[65:75, 50:65, :] = 3
            for mod in ("bSSFP", "LGE", "T2w"):
                vol_img = np.random.rand(128, 128, 2).astype(np.float32)
                with h5py.File(data_root / mod / "test_vol_h5" / "case9001.npy.h5", "w") as hf:
                    hf.create_dataset("image", data=vol_img)
                    hf.create_dataset("label", data=vol_lbl)
            (list_dir / "test_vol.txt").write_text("case9001\n", encoding="utf-8")

            # 1) --efficiency_only mode
            eff_only_out = tmp_path / "eff_only" / "efficiency_metrics.json"
            eff_only_dir = tmp_path / "run_eff_only"
            cmd_only = [
                sys.executable,
                str(REPO_ROOT / "train_myops.py"),
                "--data_root", str(data_root),
                "--list_dir", str(list_dir),
                "--output_dir", str(eff_only_dir),
                "--batch_size", "2",
                "--max_epochs", "1",
                "--amp", "none",
                "--num_workers", "0",
                "--device", "cpu",
                "--efficiency_output", str(eff_only_out),
                "--efficiency_only",
            ]
            r_only = subprocess.run(cmd_only, capture_output=True, text=True)
            self.assertEqual(r_only.returncode, 0, msg=r_only.stderr)
            self.assertTrue(eff_only_out.is_file())
            self.assertTrue((eff_only_dir / "efficiency_metrics.json").is_file())
            self.assertTrue((eff_only_dir / "logs" / "efficiency_summary.txt").is_file())
            payload = json.loads(eff_only_out.read_text(encoding="utf-8"))
            self.assertEqual(payload["excel_summary"]["Trainable Params (M)"], 4.21)
            self.assertEqual(payload["excel_summary"]["FLOPs (G)"], 17.23)
            self.assertIsNone(payload["training"])

            # 2) --no_measure_efficiency mode skips efficiency JSON
            no_eff_out = tmp_path / "no_eff" / "efficiency_metrics.json"
            no_eff_dir = tmp_path / "run_no_eff"
            cmd_no_eff = [
                sys.executable,
                str(REPO_ROOT / "train_myops.py"),
                "--data_root", str(data_root),
                "--list_dir", str(list_dir),
                "--output_dir", str(no_eff_dir),
                "--batch_size", "2",
                "--max_epochs", "1",
                "--amp", "none",
                "--num_workers", "0",
                "--device", "cpu",
                "--no_measure_efficiency",
                "--efficiency_output", str(no_eff_out),
            ]
            r_no_eff = subprocess.run(cmd_no_eff, capture_output=True, text=True)
            self.assertEqual(r_no_eff.returncode, 0, msg=r_no_eff.stderr)
            self.assertFalse(no_eff_out.exists())
            self.assertTrue((no_eff_dir / "checkpoints" / "samed_seed1234.pt").is_file())

            # 3) Full 2-epoch training with efficiency measurement (non-default seed 1098)
            run_dir = tmp_path / "run_train"
            eff_train_out = tmp_path / "eff_train" / "efficiency_metrics.json"
            base_cmd = [
                sys.executable,
                str(REPO_ROOT / "train_myops.py"),
                "--data_root", str(data_root),
                "--list_dir", str(list_dir),
                "--output_dir", str(run_dir),
                "--seed", "1098",
                "--batch_size", "2",
                "--max_epochs", "2",
                "--amp", "none",
                "--num_workers", "0",
                "--device", "cpu",
                "--measure_efficiency",
                "--efficiency_output", str(eff_train_out),
            ]
            r1 = subprocess.run(base_cmd, capture_output=True, text=True)
            self.assertEqual(r1.returncode, 0, msg=r1.stderr)
            ckpt_file = run_dir / "checkpoints" / "samed_seed1098.pt"
            self.assertTrue(ckpt_file.is_file())
            self.assertTrue(eff_train_out.is_file())
            self.assertTrue((run_dir / "efficiency_metrics.json").is_file())
            self.assertTrue((run_dir / "logs" / "efficiency_summary.txt").is_file())
            train_payload = json.loads(eff_train_out.read_text(encoding="utf-8"))
            self.assertEqual(train_payload["seed"], 1098)
            self.assertEqual(train_payload["training"]["completed_epochs"], 2)
            self.assertEqual(len(train_payload["training"]["all_epoch_times_s"]), 2)

            # 4) Simulate interruption at epoch 0 and verify bare --resume <ckpt> inherits all saved args & run_dir
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["completed_epochs"], 2)
            self.assertTrue(ckpt["training_complete"])
            first_epoch_time = ckpt["epoch_times"][0]
            ckpt["epoch"] = 0
            ckpt["completed_epochs"] = 1
            ckpt["training_complete"] = False
            ckpt["global_step"] = 1
            ckpt["successful"] = 1
            ckpt["scheduler"]["last_epoch"] = 1
            ckpt["scheduler_state_dict"]["last_epoch"] = 1
            ckpt["epoch_times"] = [first_epoch_time]
            torch.save(ckpt, ckpt_file)
            eff_train_out.unlink()

            r2 = subprocess.run(
                [sys.executable, str(REPO_ROOT / "train_myops.py"), "--resume", str(ckpt_file), "--device", "cpu"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r2.returncode, 0, msg=r2.stderr)
            self.assertTrue(eff_train_out.is_file())
            resumed_payload = json.loads(eff_train_out.read_text(encoding="utf-8"))
            self.assertEqual(resumed_payload["seed"], 1098)
            self.assertEqual(resumed_payload["environment"]["amp_mode"], "none")
            self.assertEqual(resumed_payload["environment"]["num_workers"], 0)
            self.assertEqual(resumed_payload["training"]["completed_epochs"], 2)
            self.assertEqual(len(resumed_payload["training"]["all_epoch_times_s"]), 2)
            self.assertAlmostEqual(
                resumed_payload["training"]["all_epoch_times_s"][0],
                round(first_epoch_time, 4),
                places=4,
            )

            # 5) Resuming an already-complete checkpoint (even if training_complete=False at completed==max_epochs)
            # exits cleanly without overwriting JSON
            eff_train_out.unlink()
            r3 = subprocess.run(base_cmd + ["--resume", str(ckpt_file)], capture_output=True, text=True)
            self.assertEqual(r3.returncode, 0, msg=r3.stderr)
            self.assertFalse(eff_train_out.exists())

            ckpt_done = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            ckpt_done["training_complete"] = False
            torch.save(ckpt_done, ckpt_file)
            r3b = subprocess.run(
                [sys.executable, str(REPO_ROOT / "train_myops.py"), "--resume", str(ckpt_file), "--device", "cpu"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r3b.returncode, 0, msg=r3b.stderr)
            self.assertFalse(eff_train_out.exists())

            # 6) Verify evaluate_myops.py runs cleanly on the produced checkpoint
            eval_dir = run_dir / "evaluation"
            cmd_eval = [
                sys.executable,
                str(REPO_ROOT / "evaluate_myops.py"),
                "--checkpoint", str(ckpt_file),
                "--data_root", str(data_root),
                "--list_dir", str(list_dir),
                "--output_dir", str(eval_dir),
                "--batch_size", "2",
                "--amp", "none",
                "--device", "cpu",
            ]
            r_eval = subprocess.run(cmd_eval, capture_output=True, text=True)
            self.assertEqual(r_eval.returncode, 0, msg=r_eval.stderr)
            self.assertTrue((eval_dir / "metrics.json").is_file())
            self.assertTrue((eval_dir / "per_case.csv").is_file())
            self.assertTrue((eval_dir / "test_predictions" / "case9001_pred.npz").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
