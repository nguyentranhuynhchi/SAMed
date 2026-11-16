"""Evaluation script for SAMed on MyoPS380 (76 cases) and MyoPS++ (16 cases).

Protocols:
  - MyoPS380: myops380_voxel_v3 (voxel distance, empty mask policy, equal patient weight)
  - MyoPS++:  myopspp_unet2d_roi128_1mm_predicted_roi_mm_v1 (physical mm distance [1.0, 1.0, s_z])
Exports canonical predictions to <run_dir>/test_predictions/<CaseID>_pred.npz.
Computes and saves per_case.csv and metrics.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset_myops import (
    CLASS_NAMES,
    MyopsDataset,
    canonicalize_label,
    normalize_dataset_name,
    resolve_default_list_dir,
)
from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
from utils.metrics import (
    BENCHMARK_PROTOCOLS,
    dataset_rows,
    summarize_dataset_rows,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to samed_seed<seed>.pt checkpoint")
    parser.add_argument("--data_root", "--data-root",
                        default=os.environ.get("MYOPS380_DATA_ROOT", "/content/MyoPS380_dataset/Processed_data"),
                        help="Root folder containing bSSFP, LGE, T2w subdirectories")
    parser.add_argument("--dataset", "--dataset_id", "--dataset-id", default="myops380",
                        choices=["myops380", "myopspp", "myopspp_unet2d_roi128_1mm", "myops_380"],
                        help="Benchmark dataset to evaluate")
    parser.add_argument("--list_dir", "--list-dir", "--split_dir", "--split-dir", default=None,
                        help="Directory containing test_vol.txt (defaults dynamically by dataset)")
    parser.add_argument("--output_dir", "--output-dir", default=None,
                        help="Evaluation output directory; defaults to checkpoint run directory")
    parser.add_argument("--split", default="test_vol", help="Split name to evaluate (default: test_vol)")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8, help="Inference slices per batch")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto", help="AMP mode")
    parser.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=True,
                        help="Save <CaseID>_pred.npz files")
    parser.add_argument("--vit_name", default="vit_b", choices=["vit_b", "vit_l", "vit_h"], help="ViT variant")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--img_size", type=int, default=128, help="Model image size")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of canonical classes")
    parser.add_argument("--measure_efficiency", "--measure-efficiency", action=argparse.BooleanOptionalAction,
                        default=False, help="Measure Complexity and Inference efficiency metrics")
    parser.add_argument("--efficiency_output", "--efficiency-output", default=None,
                        help="Optional JSON file to save measured efficiency metrics")
    return parser


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this environment.")
    return device


def resolve_amp(mode: str, device: torch.device):
    if mode == "none" or (mode == "auto" and device.type == "cpu"):
        return None
    if device.type != "cuda":
        raise ValueError("Mixed precision requires CUDA; use --amp none on CPU.")
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bfloat16 is not supported on this CUDA device.")
        return torch.bfloat16
    if mode == "auto" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.inference_mode()
def predict_volume(model, images, img_size=128, batch_size=8, device=None, amp_dtype=None):
    """Predict on 3 aligned (128, 128, D) modalities and return canonical (128, 128, D) uint8 labels."""
    bSSFP, LGE, T2w = [np.asarray(m, dtype=np.float32) for m in images]
    height, width, depth = bSSFP.shape
    prediction = np.empty((height, width, depth), dtype=np.uint8)

    model.eval()
    for start in range(0, depth, batch_size):
        stop = min(start + batch_size, depth)

        # Slices to shape (B, 128, 128)
        s_bssfp = torch.from_numpy(np.ascontiguousarray(bSSFP[:, :, start:stop].transpose(2, 0, 1))).to(device)
        s_lge = torch.from_numpy(np.ascontiguousarray(LGE[:, :, start:stop].transpose(2, 0, 1))).to(device)
        s_t2w = torch.from_numpy(np.ascontiguousarray(T2w[:, :, start:stop].transpose(2, 0, 1))).to(device)

        # Early-fusion: stack modalities along channel dim -> (B, 3, 128, 128)
        batch_input = torch.stack([s_bssfp, s_lge, s_t2w], dim=1)

        if batch_input.shape[-2:] != (img_size, img_size):
            batch_input = F.interpolate(batch_input, (img_size, img_size), mode="bilinear", align_corners=False)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
            outputs = model(batch_input, multimask_output=True, image_size=img_size)
            logits = outputs["masks"]  # (B, 4, 128, 128)

        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite logits encountered during volume inference.")

        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(logits.float(), (height, width), mode="bilinear", align_corners=False)

        # Canonical argmax directly: 0: bg, 1: myo, 2: edema, 3: scar
        pred_slices = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        prediction[:, :, start:stop] = pred_slices.transpose(1, 2, 0)

    return prediction


def main(argv=None):
    args = build_parser().parse_args(argv)
    norm_dataset = normalize_dataset_name(args.dataset)
    list_dir = Path(args.list_dir) if args.list_dir else resolve_default_list_dir(norm_dataset)

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    # Determine run directory and output directory
    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "test_predictions"
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_file = output_dir / "test.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
        handlers=[
            logging.FileHandler(str(log_file), mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    device = resolve_device(args.device)
    amp_dtype = resolve_amp(args.amp, device)
    amp_str = "none" if amp_dtype is None else ("bf16" if amp_dtype == torch.bfloat16 else "fp16")

    logging.info(f"=== Starting SAMed Evaluation on {norm_dataset} ({args.split}) ===")
    logging.info(f"Checkpoint: {checkpoint_path}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"List directory: {list_dir}")
    logging.info(f"Device: {device} | AMP: {amp_str}")

    # Load checkpoint
    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_protocol = ckpt_data.get("training_protocol", {})
    completed_epochs = ckpt_data.get("completed_epochs", ckpt_data.get("epoch", -1) + 1)
    planned_epochs = ckpt_data.get("planned_epochs", 300)
    training_complete = ckpt_data.get("training_complete", (completed_epochs == planned_epochs))

    logging.info(f"Checkpoint Epoch: {completed_epochs}/{planned_epochs} | Training Complete: {training_complete}")
    if not training_complete and completed_epochs < planned_epochs:
        logging.warning(f"Evaluating an incomplete checkpoint ({completed_epochs}/{planned_epochs} epochs)!")

    # Build model architecture
    num_multimask = args.num_classes - 1 if args.num_classes > 1 else 1
    sam, _ = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=num_multimask,
        checkpoint=None,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    net = LoRA_Sam(sam, r=args.rank)

    # Load weights
    if "model_state_dict" in ckpt_data:
        net.load_state_dict(ckpt_data["model_state_dict"])
    elif "model" in ckpt_data:
        net.load_state_dict(ckpt_data["model"])
    else:
        net.load_lora_parameters(str(checkpoint_path))

    net.float().to(device).eval()

    if args.measure_efficiency:
        from utils.efficiency import (
            SAMedInferenceWrapper,
            count_parameters,
            measure_flops,
            measure_inference_efficiency,
        )
        params_info = count_parameters(net)
        wrapper = SAMedInferenceWrapper(net, image_size=args.img_size, multimask_output=True)
        flops_info = measure_flops(wrapper, input_shape=(1, 3, args.img_size, args.img_size), device=device)
        inference_info = measure_inference_efficiency(
            wrapper,
            input_shape=(1, 3, args.img_size, args.img_size),
            device=device,
            warmup=30 if device.type == "cuda" else 2,
            repeats=100 if device.type == "cuda" else 5,
        )
        logging.info(
            "Efficiency | Params: Trainable %.2fM / Total %.2fM | FLOPs: %.2f G | Latency: %.2f ms",
            params_info["trainable_params_m"],
            params_info["total_params_m"],
            flops_info["flops_g"],
            inference_info["latency_ms_per_slice"],
        )
        eff_payload = {
            "model_name": "SAMed",
            "model_id": "samed",
            "dataset_id": norm_dataset,
            "params_info": params_info,
            "flops_info": flops_info,
            "inference_info": inference_info,
        }
        eff_out = Path(args.efficiency_output) if args.efficiency_output else (output_dir / "efficiency_metrics.json")
        eff_out.parent.mkdir(parents=True, exist_ok=True)
        with open(eff_out, "w", encoding="utf-8") as f:
            json.dump(eff_payload, f, indent=2)
        logging.info(f"Saved efficiency metrics to: {eff_out}")

    # Load dataset
    # On MyoPS++, disk cache is already canonical -> default label_order='canonical' prevents double-swap!
    # On MyoPS380, disk cache is legacy -> default label_order='legacy' maps 2<->3 via LUT [0, 1, 3, 2].
    dataset = MyopsDataset(
        base_dir=args.data_root,
        list_dir=list_dir,
        split=args.split,
        dataset=norm_dataset,
    )
    exp_cases = 16 if norm_dataset == "myopspp" else 76
    logging.info(f"Evaluation dataset contains {len(dataset)} cases (expected: {exp_cases})")

    rows = []
    total_seconds = 0.0
    total_slices = 0

    benchmark_proto = BENCHMARK_PROTOCOLS.get(norm_dataset, BENCHMARK_PROTOCOLS["myops380"])
    dist_unit = benchmark_proto["distance_unit"]

    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    for sample in dataset:
        case = sample["case_name"]
        images = [sample["image"], sample["image1"], sample["image2"]]
        target = np.asarray(sample["label"]) if sample["label"] is not None else None

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()

        prediction = predict_volume(
            net,
            images,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=device,
            amp_dtype=amp_dtype,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        total_seconds += elapsed
        total_slices += prediction.shape[2]

        spacing = sample.get("spacing")
        if target is not None:
            case_rows = dataset_rows(
                prediction,
                target,
                case,
                compute_distance=True,
                dataset_id=norm_dataset,
                spacing=spacing,
            )
            for r in case_rows:
                r["inference_seconds"] = elapsed
            rows.extend(case_rows)
        else:
            case_rows = []

        if args.save_predictions:
            np.savez_compressed(
                pred_dir / f"{case}_pred.npz",
                prediction=prediction,
                class_names=np.asarray(CLASS_NAMES),
                spacing=np.asarray(spacing if spacing is not None else [np.nan, np.nan, np.nan]),
                affine=np.asarray(sample.get("affine", np.full((4, 4), np.nan))),
                spacing_unit=sample.get("spacing_unit", dist_unit),
                metric_distance_unit=dist_unit,
                exported_label_order="canonical",
                adapter_version="canonical_v1",
                dataset=norm_dataset,
            )
            if sample.get("has_affine", False):
                try:
                    import nibabel as nib
                    nifti = nib.Nifti1Image(prediction, np.asarray(sample["affine"]))
                    if sample.get("spacing_unit") == "mm":
                        nifti.header.set_xyzt_units("mm")
                    nib.save(nifti, pred_dir / f"{case}_pred.nii.gz")
                except Exception:
                    pass

        dist_col = f"hd95_{dist_unit}"
        logging.info(f"{case}: {prediction.shape[2]} slices, {elapsed:.3f}s")
        for r in case_rows:
            logging.info(
                f"{case} | {r['region']:<18} | Dice={r['dice']} IoU={r['iou']} "
                f"Precision={r['precision']} Recall={r['recall']} {dist_col.upper()}={r.get(dist_col)}"
            )

    if not rows and any(s["label"] is not None for s in dataset):
        raise ValueError("No evaluation results were generated.")

    # Save per_case.csv
    if rows:
        csv_path = output_dir / "per_case.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logging.info(f"Saved per-case metrics to: {csv_path}")

    # Summarize metrics
    summary = summarize_dataset_rows(rows, dataset_id=norm_dataset) if rows else {}
    summary.update({
        "model_name": "SAMed",
        "model_id": "samed",
        "dataset_id": norm_dataset,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": completed_epochs,
        "planned_epochs": planned_epochs,
        "training_complete": training_complete,
        "training_protocol": training_protocol,
        "metric_protocol": benchmark_proto["id"],
        "distance_unit": dist_unit,
        "exported_label_order": "canonical",
        "total_cases": len(dataset),
        "total_slices": total_slices,
        "total_inference_time_sec": total_seconds,
        "avg_time_per_case_sec": total_seconds / len(dataset) if len(dataset) else 0.0,
    })

    # Save metrics.json
    json_path = output_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Saved summary metrics to: {json_path}")

    # Display Master Benchmark Clinical Table
    hd95_key = f"mean_hd95_{dist_unit}"
    scar_dice = (summary.get("scar", {}).get("mean_dice") or 0.0) * 100
    scar_hd95 = summary.get("scar", {}).get(hd95_key) or 0.0
    edema_dice = (summary.get("edema_inclusive", {}).get("mean_dice") or 0.0) * 100
    edema_hd95 = summary.get("edema_inclusive", {}).get(hd95_key) or 0.0
    edema_excl_dice = (summary.get("edema", {}).get("mean_dice") or 0.0) * 100
    edema_excl_hd95 = summary.get("edema", {}).get(hd95_key) or 0.0
    myo_dice = (summary.get("myocardial_ring", {}).get("mean_dice") or 0.0) * 100
    myo_hd95 = summary.get("myocardial_ring", {}).get(hd95_key) or 0.0
    avg_path_dice = (summary.get("avg_pathology_inclusive_dice") or 0.0) * 100
    avg_path_hd95 = summary.get(f"avg_pathology_inclusive_{hd95_key}") or summary.get("avg_pathology_inclusive_hd95_mm") or summary.get("avg_pathology_inclusive_hd95_voxel") or 0.0
    avg_path_excl_dice = (summary.get("avg_pathology_dice") or 0.0) * 100
    avg_path_excl_hd95 = summary.get(f"avg_pathology_{hd95_key}") or summary.get("avg_pathology_hd95_mm") or summary.get("avg_pathology_hd95_voxel") or 0.0

    table = f"""
=============================================================================================================
MASTER BENCHMARK EVALUATION RESULTS (Protocol: {benchmark_proto['id']})
Model: SAMed | Dataset: {norm_dataset} | Checkpoint: {checkpoint_path.name} | Cases: {len(dataset)}
-------------------------------------------------------------------------------------------------------------
Region                   | Dice (%)       | HD95 ({dist_unit})
-------------------------------------------------------------------------------------------------------------
Scar                     | {scar_dice:6.2f}         | {scar_hd95:6.2f}
Edema (Inclusive / AAR)  | {edema_dice:6.2f}         | {edema_hd95:6.2f}
Edema (Exclusive)        | {edema_excl_dice:6.2f}         | {edema_excl_hd95:6.2f}
Myocardium (Whole Ring)  | {myo_dice:6.2f}         | {myo_hd95:6.2f}
-------------------------------------------------------------------------------------------------------------
Avg. Pathology (Incl.)   | {avg_path_dice:6.2f}         | {avg_path_hd95:6.2f}
Avg. Pathology (Excl.)   | {avg_path_excl_dice:6.2f}         | {avg_path_excl_hd95:6.2f}
=============================================================================================================
"""
    logging.info(table)


if __name__ == "__main__":
    main()
