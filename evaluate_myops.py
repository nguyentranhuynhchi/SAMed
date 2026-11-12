"""Evaluation script for SAMed on MyoPS380 held-out test volumes (76 cases).

Protocol: myops380_voxel_v3 (voxel distance, empty mask policy, equal patient weight).
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

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset_myops import (
    MyopsDataset,
    canonicalize_label,
    CLASS_NAMES,
)
from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
from utils.metrics import (
    BENCHMARK_PROTOCOL,
    dataset_rows,
    summarize_dataset_rows,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to samed_seed<seed>.pt checkpoint")
    parser.add_argument("--data_root", "--data-root",
                        default=os.environ.get("MYOPS380_DATA_ROOT", "/content/MyoPS380_dataset/Processed_data"),
                        help="Root folder containing bSSFP, LGE, T2w subdirectories")
    parser.add_argument("--list_dir", "--list-dir", default=str(PROJECT_ROOT / "lists" / "myops380_no_val"),
                        help="Directory containing test_vol.txt")
    parser.add_argument("--output_dir", "--output-dir", default=None,
                        help="Evaluation output directory; defaults to checkpoint run directory")
    parser.add_argument("--split", default="test_vol", help="Split name to evaluate (default: test_vol)")
    parser.add_argument("--dataset_id", "--dataset-id", default="myops380", help="Dataset identifier")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8, help="Inference slices per batch")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto", help="AMP mode")
    parser.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=True,
                        help="Save <CaseID>_pred.npz files")
    parser.add_argument("--vit_name", default="vit_b", choices=["vit_b", "vit_l", "vit_h"], help="ViT variant")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--img_size", type=int, default=128, help="Model image size")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of canonical classes")
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
        cur_batch = stop - start

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

    logging.info(f"=== Starting SAMed Evaluation on MyoPS380 ({args.split}) ===")
    logging.info(f"Checkpoint: {checkpoint_path}")
    logging.info(f"Output directory: {output_dir}")
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
        # Fallback for lora-only saved dict
        net.load_lora_parameters(str(checkpoint_path))

    net.float().to(device).eval()

    # Load dataset
    dataset = MyopsDataset(
        base_dir=args.data_root,
        list_dir=args.list_dir,
        split=args.split,
        label_order="legacy",
    )
    logging.info(f"Evaluation dataset contains {len(dataset)} cases (expected: 76)")

    rows = []
    total_seconds = 0.0
    total_slices = 0

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

        if target is not None:
            case_rows = dataset_rows(prediction, target, case, compute_distance=True, dataset_id=args.dataset_id)
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
                spacing=np.asarray(sample.get("spacing", [np.nan, np.nan, np.nan])),
                affine=np.asarray(sample.get("affine", np.full((4, 4), np.nan))),
                spacing_unit=sample.get("spacing_unit", "voxel"),
                metric_distance_unit="voxel",
                exported_label_order="canonical",
                adapter_version="canonical_v1",
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

        logging.info(f"{case}: {prediction.shape[2]} slices, {elapsed:.3f}s")
        for r in case_rows:
            logging.info(
                f"{case} | {r['region']:<18} | Dice={r['dice']} IoU={r['iou']} "
                f"Precision={r['precision']} Recall={r['recall']} HD95={r['hd95_voxel']} ASD={r['asd_voxel']}"
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
    summary = summarize_dataset_rows(rows, dataset_id=args.dataset_id) if rows else {}
    summary.update({
        "model_name": "SAMed",
        "model_id": "samed",
        "dataset_id": args.dataset_id,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": completed_epochs,
        "planned_epochs": planned_epochs,
        "training_complete": training_complete,
        "training_protocol": training_protocol,
        "metric_protocol": BENCHMARK_PROTOCOL["id"],
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
    scar_dice = (summary.get("scar", {}).get("mean_dice") or 0.0) * 100
    scar_hd95 = summary.get("scar", {}).get("mean_hd95_voxel") or 0.0
    edema_dice = (summary.get("edema_inclusive", {}).get("mean_dice") or 0.0) * 100
    edema_hd95 = summary.get("edema_inclusive", {}).get("mean_hd95_voxel") or 0.0
    myo_dice = (summary.get("myocardial_ring", {}).get("mean_dice") or 0.0) * 100
    myo_hd95 = summary.get("myocardial_ring", {}).get("mean_hd95_voxel") or 0.0
    avg_path_dice = (summary.get("avg_pathology_inclusive_dice") or 0.0) * 100
    avg_path_hd95 = summary.get("avg_pathology_inclusive_hd95_voxel") or 0.0

    table = f"""
=============================================================================================================
MASTER BENCHMARK EVALUATION RESULTS (Protocol: {BENCHMARK_PROTOCOL['id']})
Model: SAMed | Checkpoint: {checkpoint_path.name} | Cases: {len(dataset)}
-------------------------------------------------------------------------------------------------------------
Region                   | Dice (%)       | HD95 (voxel)
-------------------------------------------------------------------------------------------------------------
Scar                     | {scar_dice:6.2f}         | {scar_hd95:6.2f}
Edema (Inclusive / AAR)  | {edema_dice:6.2f}         | {edema_hd95:6.2f}
Myocardium (Whole Ring)  | {myo_dice:6.2f}         | {myo_hd95:6.2f}
-------------------------------------------------------------------------------------------------------------
Avg. Pathology           | {avg_path_dice:6.2f}         | {avg_path_hd95:6.2f}
=============================================================================================================
"""
    logging.info(table)


if __name__ == "__main__":
    main()
