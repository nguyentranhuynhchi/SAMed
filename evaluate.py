"""Independent evaluation script for SAMed on MyoPS-380 test set.
Protocol: myops380_voxel_v3
Inputs: 76 held-out test volumes (test_vol.txt)
Outputs: 3D predictions (.npz), metrics.json, per_case.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_myops import MyopsDataset
from metrics.surface_distance import protocol_for_dataset, dataset_rows, summarize_dataset_rows


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate SAMed on MyoPS-380 test volumes.")
    parser.add_argument("--root_path", type=str, default="/content/MyoPS380/Processed_data",
                        help="Root directory for MyoPS-380 Processed_data")
    parser.add_argument("--list_dir", type=str, default="./lists/lists_MyoPS380",
                        help="Directory containing test_vol.txt")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save test_results and test_predictions")
    parser.add_argument("--lora_ckpt", type=str, required=True,
                        help="Path to trained last.pth or last_lora.pth checkpoint")
    parser.add_argument("--ckpt", type=str, default="checkpoints/sam_vit_b_01ec64.pth",
                        help="Path to pretrained SAM ViT-B checkpoint")
    parser.add_argument("--vit_name", type=str, default="vit_b", help="ViT model variant")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--module", type=str, default="sam_lora_image_encoder")
    parser.add_argument("--img_size", type=int, default=512, help="Network input image size")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of segmentation classes")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size of slices")
    parser.add_argument("--cpu_threads", type=int, default=2, help="torch cpu threads")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    return parser


@torch.no_grad()
def predict_test_volume(
    model: torch.nn.Module,
    vol_image: torch.Tensor,
    img_size: int = 512,
    batch_size: int = 8,
    device: str = "cuda"
) -> np.ndarray:
    """Predict a 3D volume (3, H, W, D) slice-by-slice along depth D.

    vol_image: torch.Tensor of shape (3, H, W, D)
    Returns: numpy uint8 array of shape (H, W, D)
    """
    _, height, width, depth = vol_image.shape
    prediction_3d = np.zeros((height, width, depth), dtype=np.uint8)

    model.eval()

    # Iterate over depth in batches of slices
    for start_idx in range(0, depth, batch_size):
        end_idx = min(start_idx + batch_size, depth)
        curr_batch_len = end_idx - start_idx

        # Extract slices: (3, H, W, B) -> (B, 3, H, W)
        slices = vol_image[:, :, :, start_idx:end_idx].permute(3, 0, 1, 2).to(device)

        # Bilinear resize to img_size if needed
        if slices.shape[-2:] != (img_size, img_size):
            inputs = F.interpolate(slices, size=(img_size, img_size), mode="bilinear", align_corners=False)
        else:
            inputs = slices

        # Forward through SAMed
        outputs = model(inputs, multimask_output=True, image_size=img_size)
        logits = outputs["masks"]  # (B, num_classes, img_size, img_size)

        # Resize logits back to native grid (height, width) = (128, 128)
        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(logits.float(), size=(height, width), mode="bilinear", align_corners=False)

        # Argmax to obtain discrete prediction
        preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)  # (B, H, W)

        for b in range(curr_batch_len):
            prediction_3d[:, :, start_idx + b] = preds[b]

    return prediction_3d


def main():
    parser = build_parser()
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)

    output_dir = Path(args.output_dir)
    test_results_dir = output_dir / "test_results"
    pred_dir = output_dir / "test_predictions"
    test_results_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    log_file = test_results_dir / "test.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler(sys.stdout)]
    )

    logging.info(f"Evaluating SAMed on MyoPS-380 test set")
    logging.info(f"Model LoRA Checkpoint: {args.lora_ckpt}")
    logging.info(f"SAM ViT Backbone Checkpoint: {args.ckpt}")
    logging.info(f"Data Root: {args.root_path}")
    logging.info(f"List Dir: {args.list_dir}")

    # Build SAMed model: pass num_classes - 1 (3 foreground + 1 bg token = 4 channels)
    num_multimask = args.num_classes - 1
    sam, _ = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=num_multimask,
        checkpoint=args.ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1]
    )

    pkg = import_module(args.module)
    model = pkg.LoRA_Sam(sam, args.rank)

    # Load weights
    ckpt_path = Path(args.lora_ckpt)
    if not ckpt_path.is_file():
        alt_path = ckpt_path.parent / "last.pth"
        if alt_path.is_file():
            logging.warning(f"Checkpoint {ckpt_path} not found, falling back to {alt_path}")
            ckpt_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.lora_ckpt}")

    logging.info(f"Loading checkpoint weights from {ckpt_path}")
    ckpt_data = torch.load(str(ckpt_path), map_location="cpu")
    if "model" in ckpt_data and isinstance(ckpt_data["model"], dict):
        # Full state dict
        model.load_state_dict(ckpt_data["model"], strict=False)
    else:
        # LoRA state dict
        try:
            model.load_lora_parameters(str(ckpt_path))
        except Exception as e:
            logging.warning(f"load_lora_parameters failed ({e}), attempting direct load_state_dict...")
            model.load_state_dict(ckpt_data, strict=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_dataset = MyopsDataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="test_vol",
        label_order="legacy"
    )
    logging.info(f"Loaded test dataset: {len(test_dataset)} cases")

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    eval_items = []
    start_time = time.time()

    for idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
        vol_image = batch["image"].squeeze(0)  # (3, H, W, D)
        gt_label = batch["label"].squeeze(0).numpy().astype(np.uint8)  # (H, W, D)
        case_name = batch["case_name"][0]

        pred_3d = predict_test_volume(
            model=model,
            vol_image=vol_image,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=str(device)
        )

        # Save prediction array
        pred_path = pred_dir / f"{case_name}_pred.npz"
        np.savez_compressed(pred_path, prediction=pred_3d)

        eval_items.append({
            "case_id": case_name,
            "prediction": pred_3d,
            "ground_truth": gt_label,
            "spacing": [1.0, 1.0, 1.0],  # Unit voxel grid per myops380_voxel_v3 protocol
        })

    inference_duration = time.time() - start_time
    logging.info(f"Inference completed in {inference_duration:.2f}s for {len(test_dataset)} cases")

    # Evaluate metrics using standard protocol myops380_voxel_v3
    logging.info("Computing metrics with protocol: myops380_voxel_v3")
    protocol = protocol_for_dataset("myops380")

    all_rows = []
    for item in eval_items:
        case_rows = dataset_rows(
            prediction=item["prediction"],
            target=item["ground_truth"],
            case=item["case_id"]
        )
        all_rows.extend(case_rows)

    summary_metrics = summarize_dataset_rows(all_rows)

    # Add run metadata
    summary_metrics["checkpoint"] = str(args.lora_ckpt)
    summary_metrics["checkpoint_epoch"] = 300
    summary_metrics["checkpoint_metric_protocol_id"] = protocol["id"]
    summary_metrics["architecture"] = "samed"
    summary_metrics["split"] = "test_vol"
    summary_metrics["case_count"] = len(eval_items)
    summary_metrics["inference_seconds"] = inference_duration

    # Save metrics.json
    metrics_json_path = test_results_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    logging.info(f"Saved summary metrics to {metrics_json_path}")

    # Save per_case.csv
    per_case_csv_path = test_results_dir / "per_case.csv"
    if all_rows:
        with open(per_case_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        logging.info(f"Saved per-case metrics to {per_case_csv_path}")

    # Print summary table
    logging.info("=== SUMMARY RESULTS ===")
    logging.info(f"Scar Dice: {summary_metrics['scar']['mean_dice'] * 100:.2f}%, HD95: {summary_metrics['scar']['mean_hd95_voxel']:.2f}")
    logging.info(f"Edema (Inclusive) Dice: {summary_metrics['edema_inclusive']['mean_dice'] * 100:.2f}%, HD95: {summary_metrics['edema_inclusive']['mean_hd95_voxel']:.2f}")
    logging.info(f"Edema (Exclusive) Dice: {summary_metrics['edema']['mean_dice'] * 100:.2f}%, HD95: {summary_metrics['edema']['mean_hd95_voxel']:.2f}")
    logging.info(f"Myocardial Ring Dice: {summary_metrics['myocardial_ring']['mean_dice'] * 100:.2f}%, HD95: {summary_metrics['myocardial_ring']['mean_hd95_voxel']:.2f}")
    logging.info(f"Avg Pathology (Inclusive) Dice: {summary_metrics['avg_pathology_inclusive_dice'] * 100:.2f}%, HD95: {summary_metrics['avg_pathology_inclusive_hd95_voxel']:.2f}")


if __name__ == "__main__":
    main()
