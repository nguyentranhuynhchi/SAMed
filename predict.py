"""Single-pass deterministic volume inference script for SAMed.

Supports:
  - MyoPS380 (76 Held-Out Test Volumes)
  - MyoPS++ (16 Held-Out Test Volumes)

Key Requirements:
  - Checkpoint loading: <run_dir>/checkpoints/samed_seed<seed>.pt
  - Single-pass model.eval(), torch.inference_mode()
  - Process all cases from test_vol.txt (76 cases for myops380, 16 cases for myopspp)
  - Slice batching with bilinear spatial resizing (if H, W != img_size)
  - Strict NO test-time augmentation (TTA)
  - Strict NO post-processing / largest connected component (LCC)
  - Export: <run_dir>/test_predictions/<CaseID>_pred.npz with key 'prediction' (uint8, [0..3] canonical)
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset_myops import (
    CLASS_NAMES,
    MyopsDataset,
    normalize_dataset_name,
    resolve_default_list_dir,
)
from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
from utils.metrics import BENCHMARK_PROTOCOLS


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to samed_seed<seed>.pt checkpoint")
    parser.add_argument("--data_root", "--data-root",
                        default=os.environ.get("MYOPS380_DATA_ROOT", "/content/MyoPS380_dataset/Processed_data"),
                        help="Root folder containing bSSFP, LGE, T2w subdirectories")
    parser.add_argument("--dataset", "--dataset_id", "--dataset-id", default="myops380",
                        choices=["myops380", "myopspp", "myopspp_unet2d_roi128_1mm", "myops_380"],
                        help="Benchmark dataset to predict")
    parser.add_argument("--list_dir", "--list-dir", "--split_dir", "--split-dir", default=None,
                        help="Directory containing test_vol.txt (defaults dynamically by dataset)")
    parser.add_argument("--output_dir", "--output-dir", default=None,
                        help="Inference output directory; defaults to checkpoint run directory")
    parser.add_argument("--split", default="test_vol", help="Split name to predict (default: test_vol)")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8, help="Inference slices per batch")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto", help="AMP mode")
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

    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    pred_dir = output_dir / "test_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    device = resolve_device(args.device)
    amp_dtype = resolve_amp(args.amp, device)
    amp_str = "none" if amp_dtype is None else ("bf16" if amp_dtype == torch.bfloat16 else "fp16")

    logging.info(f"=== SAMed Volume Inference on {norm_dataset} ({args.split}) ===")
    logging.info(f"Checkpoint: {checkpoint_path}")
    logging.info(f"Output: {pred_dir}")
    logging.info(f"Device: {device} | AMP: {amp_str}")

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
    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt_data:
        net.load_state_dict(ckpt_data["model_state_dict"])
    elif "model" in ckpt_data:
        net.load_state_dict(ckpt_data["model"])
    else:
        net.load_lora_parameters(str(checkpoint_path))

    net.float().to(device).eval()

    if args.measure_efficiency:
        import json
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

    dataset = MyopsDataset(
        base_dir=args.data_root,
        list_dir=list_dir,
        split=args.split,
        dataset=norm_dataset,
    )
    exp_cases = 16 if norm_dataset == "myopspp" else 76
    logging.info(f"Inference dataset contains {len(dataset)} cases (expected: {exp_cases})")

    benchmark_proto = BENCHMARK_PROTOCOLS.get(norm_dataset, BENCHMARK_PROTOCOLS["myops380"])
    dist_unit = benchmark_proto["distance_unit"]

    for sample in dataset:
        case = sample["case_name"]
        images = [sample["image"], sample["image1"], sample["image2"]]

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

        spacing = sample.get("spacing")
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

        logging.info(f"Saved: {case}_pred.npz ({prediction.shape[2]} slices, {elapsed:.3f}s)")

    logging.info(f"=== Completed volume inference for {len(dataset)} cases -> {pred_dir} ===")


if __name__ == "__main__":
    main()
