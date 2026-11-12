"""Efficiency benchmarking utilities for SAMed on MyoPS380.

Measures and exports the 7 standardized computational efficiency metrics:
  Complexity:
    1) Trainable Params (M) and Total Params (M)
    2) FLOPs (G) on (1, 3, 128, 128) FP32 input via PyTorch FlopCounterMode (eval mode)
  Inference (Batch = 1, FP32, eval, no_grad):
    3) Latency (ms/slice): 30 warmup passes + 100 timed passes with cuda.synchronize()
    4) Peak VRAM (MB) + GB conversion
  Training (Batch = 16, FP32, num_workers = 0, pin_memory = True, 300 epochs):
    5) Time / Epoch (s): average over epochs 6..300 (skipping first 5 warmup epochs)
    6) Total Time (hours): full 300-epoch duration
    7) Peak VRAM (MB) + GB conversion

Memory unit convention: binary MiB/GiB (1 MB = 1024^2 bytes, 1 GB = 1024^3 bytes).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

BYTES_PER_MB = float(1024**2)
BYTES_PER_GB = float(1024**3)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _bytes_to_mb_gb(num_bytes: int) -> tuple[float, float]:
    return num_bytes / BYTES_PER_MB, num_bytes / BYTES_PER_GB


class SAMedInferenceWrapper(nn.Module):
    """Thin nn.Module adapter exposing standard (B, 3, H, W) -> mask logits forward for SAMed."""

    def __init__(self, model: nn.Module, image_size: int = 128, multimask_output: bool = True) -> None:
        super().__init__()
        self.model = model
        self.image_size = int(image_size)
        self.multimask_output = bool(multimask_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x, multimask_output=self.multimask_output, image_size=self.image_size)
        if isinstance(out, dict):
            return out["masks"]
        return out


def _as_single_input_module(model: nn.Module, image_size: int = 128) -> nn.Module:
    if isinstance(model, SAMedInferenceWrapper):
        return model
    if hasattr(model, "sam") or type(model).__name__ in {"LoRA_Sam", "Sam"}:
        return SAMedInferenceWrapper(model, image_size=image_size, multimask_output=True)
    return model


def count_parameters(model: nn.Module) -> dict[str, Any]:
    """Count total and trainable parameters."""
    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_m": round(total_params / 1e6, 4),
        "trainable_params_m": round(trainable_params / 1e6, 4),
        "trainable_params_excel": f"{trainable_params / 1e6:.2f}M",
        "total_params_excel": f"{total_params / 1e6:.2f}M",
    }


def measure_flops(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 128, 128),
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Measure theoretical forward FLOPs in FP32 eval mode using PyTorch FlopCounterMode."""
    if device is None:
        device = next(model.parameters()).device

    target_model = _as_single_input_module(model, image_size=input_shape[-1])
    was_training = target_model.training
    saved_requires_grad = [(p, p.requires_grad) for p in target_model.parameters()]
    target_model.eval()
    try:
        # PromptEncoder returns a view of a trainable parameter (`no_mask_embed.weight`),
        # which keeps requires_grad=True under torch.no_grad() and causes ModuleTracker inside
        # FlopCounterMode to attach grad hooks to a tensor without grad_fn. Temporarily disabling
        # requires_grad during FLOP counting avoids that hook bug and restores flags in finally.
        for p, _ in saved_requires_grad:
            if p.requires_grad:
                p.requires_grad_(False)

        dummy_input = torch.randn(*input_shape, dtype=torch.float32, device=device)
        flop_counter = FlopCounterMode(display=False)
        with torch.no_grad(), flop_counter:
            out = target_model(dummy_input)
            del out
        sync_if_cuda(device)
        total_flops = int(flop_counter.get_total_flops())
    finally:
        for p, orig_rg in saved_requires_grad:
            if p.requires_grad != orig_rg:
                p.requires_grad_(orig_rg)
        target_model.train(was_training)

    return {
        "input_shape": list(input_shape),
        "dtype": "float32",
        "counter": "torch.utils.flop_counter.FlopCounterMode",
        "flops_total": total_flops,
        "flops_g": round(total_flops / 1e9, 4),
        "flops_g_excel": round(total_flops / 1e9, 2),
    }


def measure_inference_efficiency(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 128, 128),
    device: torch.device | None = None,
    warmup: int = 30,
    repeats: int = 100,
) -> dict[str, Any]:
    """Measure Batch=1 FP32 inference latency and peak allocated VRAM in eval mode."""
    if device is None:
        device = next(model.parameters()).device

    target_model = _as_single_input_module(model, image_size=input_shape[-1])
    was_training = target_model.training
    target_model.eval()
    try:
        dummy_input = torch.randn(*input_shape, dtype=torch.float32, device=device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            for _ in range(warmup):
                out = target_model(dummy_input)
                del out
            sync_if_cuda(device)

            t_start = time.perf_counter()
            for _ in range(repeats):
                out = target_model(dummy_input)
                del out
            sync_if_cuda(device)
            t_end = time.perf_counter()

        elapsed_s = t_end - t_start
        latency_ms = (elapsed_s / max(1, repeats)) * 1000.0

        if device.type == "cuda" and torch.cuda.is_available():
            peak_alloc_bytes = int(torch.cuda.max_memory_allocated(device))
            peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_alloc_bytes = 0
            peak_reserved_bytes = 0
    finally:
        target_model.train(was_training)

    peak_mb, peak_gb = _bytes_to_mb_gb(peak_alloc_bytes)
    reserved_mb, reserved_gb = _bytes_to_mb_gb(peak_reserved_bytes)

    return {
        "batch_size": int(input_shape[0]),
        "input_shape": list(input_shape),
        "dtype": "float32",
        "warmup_runs": int(warmup),
        "measure_runs": int(repeats),
        "latency_ms_per_slice": round(latency_ms, 4),
        "peak_vram_bytes": peak_alloc_bytes,
        "peak_vram_mb": round(peak_mb, 2),
        "peak_vram_gb": round(peak_gb, 4),
        "peak_vram_reserved_mb": round(reserved_mb, 2),
        "peak_vram_reserved_gb": round(reserved_gb, 4),
    }


def summarize_training_efficiency(
    epoch_times_s: Sequence[float],
    total_wall_clock_s: float,
    peak_vram_bytes: int,
    planned_epochs: int = 300,
    warmup_epochs: int = 5,
    peak_reserved_bytes: int = 0,
) -> dict[str, Any]:
    """Summarize training timing (skipping first 5 epochs) and peak VRAM."""
    times = [float(t) for t in epoch_times_s]
    completed_epochs = len(times)

    if completed_epochs > warmup_epochs:
        measured_times = times[warmup_epochs:]
        skipped = warmup_epochs
        measured_range = f"{warmup_epochs + 1}..{completed_epochs}"
    elif completed_epochs > 0:
        measured_times = times
        skipped = 0
        measured_range = f"1..{completed_epochs}"
    else:
        measured_times = []
        skipped = 0
        measured_range = "none"

    time_per_epoch_s = (sum(measured_times) / len(measured_times)) if measured_times else 0.0
    total_epoch_time_s = sum(times)
    total_time_hours = total_epoch_time_s / 3600.0
    wall_clock_hours = float(total_wall_clock_s) / 3600.0

    peak_mb, peak_gb = _bytes_to_mb_gb(int(peak_vram_bytes))
    reserved_mb, reserved_gb = _bytes_to_mb_gb(int(peak_reserved_bytes))

    return {
        "planned_epochs": int(planned_epochs),
        "completed_epochs": completed_epochs,
        "warmup_epochs_skipped": skipped,
        "measured_epoch_range": measured_range,
        "time_per_epoch_s": round(time_per_epoch_s, 4),
        "total_time_seconds": round(total_epoch_time_s, 2),
        "total_time_hours": round(total_time_hours, 4),
        "total_wall_clock_seconds": round(float(total_wall_clock_s), 2),
        "total_wall_clock_hours": round(wall_clock_hours, 4),
        "peak_vram_bytes": int(peak_vram_bytes),
        "peak_vram_mb": round(peak_mb, 2),
        "peak_vram_gb": round(peak_gb, 4),
        "peak_vram_reserved_mb": round(reserved_mb, 2),
        "peak_vram_reserved_gb": round(reserved_gb, 4),
        "epoch_times_1_to_5_s": [round(t, 4) for t in times[:5]],
        "all_epoch_times_s": [round(t, 4) for t in times],
    }


def check_rule_compliance(
    device: torch.device,
    amp_mode: str,
    num_workers: int,
    pin_memory: bool,
    batch_size: int,
    planned_epochs: int,
    accum_steps: int = 1,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Verify compliance with Quy tac do hieu suat tinh toan."""
    warnings: list[str] = []
    gpu_name = "cpu"
    gpu_count = 0

    if device.type == "cuda" and torch.cuda.is_available():
        gpu_count = int(torch.cuda.device_count())
        gpu_name = torch.cuda.get_device_name(device)
        if "A100" not in gpu_name.upper():
            warnings.append(f"Hardware is '{gpu_name}', benchmark rule specifies NVIDIA A100.")
        if gpu_count != 1:
            warnings.append(f"Visible CUDA device count is {gpu_count}, rule requires strictly 1 GPU.")
    else:
        warnings.append("Running on CPU; benchmark rule requires 01 NVIDIA A100 GPU.")

    if str(amp_mode).lower() != "none":
        warnings.append(f"AMP mode is '{amp_mode}', rule requires pure FP32 (--amp none).")
    if num_workers != 0:
        warnings.append(f"num_workers={num_workers}, rule requires --num_workers 0.")
    if device.type == "cuda" and not pin_memory:
        warnings.append("pin_memory=False, rule requires pin_memory=True.")
    if batch_size != 16:
        warnings.append(f"batch_size={batch_size}, rule requires batch_size=16.")
    if accum_steps != 1:
        warnings.append(f"accum_steps={accum_steps}, rule requires accum_steps=1.")
    if planned_epochs != 300:
        warnings.append(f"max_epochs={planned_epochs}, rule requires 300 epochs.")

    hardware_info = {
        "device": str(device),
        "gpu_name": gpu_name,
        "visible_gpu_count": gpu_count,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "amp_mode": str(amp_mode),
        "precision": "fp32" if str(amp_mode).lower() == "none" else str(amp_mode),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "batch_size": int(batch_size),
        "accum_steps": int(accum_steps),
        "planned_epochs": int(planned_epochs),
    }
    return len(warnings) == 0, warnings, hardware_info


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def build_efficiency_payload(
    *,
    model_name: str,
    model_id: str,
    seed: int,
    params_info: dict[str, Any],
    flops_info: dict[str, Any],
    inference_info: dict[str, Any],
    training_info: dict[str, Any] | None,
    rule_compliant: bool,
    compliance_warnings: list[str],
    hardware_info: dict[str, Any],
) -> dict[str, Any]:
    """Assemble structured efficiency metrics JSON payload."""
    excel_summary: dict[str, Any] = {
        "Models": model_name,
        "Trainable Params (M)": round(params_info["trainable_params_m"], 2),
        "Total Params (M)": round(params_info["total_params_m"], 2),
        "FLOPs (G)": round(flops_info["flops_g"], 2),
        "Inference Latency (ms/slice)": round(inference_info["latency_ms_per_slice"], 2),
        "Inference Peak VRAM (MB)": round(inference_info["peak_vram_mb"], 2),
        "Inference Peak VRAM (GB)": round(inference_info["peak_vram_gb"], 4),
    }
    if training_info is not None:
        excel_summary.update(
            {
                "Training Time / Epoch (s)": round(training_info["time_per_epoch_s"], 2),
                "Training Total Time (hours)": round(training_info["total_time_hours"], 4),
                "Training Peak VRAM (MB)": round(training_info["peak_vram_mb"], 2),
                "Training Peak VRAM (GB)": round(training_info["peak_vram_gb"], 4),
            }
        )
    else:
        excel_summary.update(
            {
                "Training Time / Epoch (s)": None,
                "Training Total Time (hours)": None,
                "Training Peak VRAM (MB)": None,
                "Training Peak VRAM (GB)": None,
            }
        )

    return _json_safe(
        {
            "schema_version": "efficiency_metrics_v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "model_id": model_id,
            "seed": int(seed),
            "unit_conventions": {
                "params": "millions (10^6)",
                "flops": "GFLOPs (10^9 FLOPs, 2*MACs via PyTorch FlopCounterMode in eval mode)",
                "vram": "binary MiB/GiB (1 MB = 1024^2 bytes, 1 GB = 1024^3 bytes)",
            },
            "rule_compliant": bool(rule_compliant),
            "compliance_warnings": list(compliance_warnings),
            "environment": hardware_info,
            "excel_summary": excel_summary,
            "complexity": {
                **params_info,
                **flops_info,
            },
            "inference": inference_info,
            "training": training_info,
        }
    )


def format_efficiency_report(payload: dict[str, Any]) -> str:
    """Render clear, copy-paste friendly terminal summary of all 7 metrics."""
    ex = payload["excel_summary"]
    comp = payload["complexity"]
    inf = payload["inference"]
    trn = payload.get("training")
    env = payload["environment"]

    lines = [
        "================================================================================",
        f"COMPUTATIONAL EFFICIENCY BENCHMARK REPORT — {payload['model_name']} (Seed {payload['seed']})",
        "================================================================================",
        f"Hardware : {env['gpu_name']} (GPUs={env['visible_gpu_count']}) | PyTorch {env['pytorch_version']} | CUDA {env['cuda_version']}",
        f"Config   : Precision={env['precision']} | AMP={env['amp_mode']} | num_workers={env['num_workers']} | pin_memory={env['pin_memory']} | batch={env['batch_size']}",
        f"Rule OK  : {'YES (100% COMPLIANT)' if payload['rule_compliant'] else 'NO — ' + '; '.join(payload['compliance_warnings'])}",
        "--------------------------------------------------------------------------------",
        "[1. Complexity]",
        f"  - Trainable Params     : {comp['trainable_params_m']:.2f}M ({comp['trainable_params']:,})",
        f"  - Total Params         : {comp['total_params_m']:.2f}M ({comp['total_params']:,})",
        f"  - FLOPs (1x3x128x128)  : {comp['flops_g']:.2f} G ({comp['flops_total']:,} FLOPs)",
        "[2. Inference (Batch = 1, FP32, eval)]",
        f"  - Latency (ms/slice)   : {inf['latency_ms_per_slice']:.2f} ms (warmup={inf['warmup_runs']}, runs={inf['measure_runs']})",
        f"  - Peak VRAM            : {inf['peak_vram_mb']:.2f} MB ({inf['peak_vram_gb']:.4f} GB) [reserved: {inf['peak_vram_reserved_mb']:.2f} MB]",
    ]
    if trn is not None:
        lines.extend(
            [
                f"[3. Training (Batch = {env['batch_size']}, Epochs = {trn['completed_epochs']}/{trn['planned_epochs']})]",
                f"  - Time / Epoch (ep {trn['measured_epoch_range']}): {trn['time_per_epoch_s']:.2f} s (skipped first {trn['warmup_epochs_skipped']} epochs)",
                f"  - Total Time           : {trn['total_time_hours']:.4f} hours ({trn['total_time_seconds']:.1f} s | wall-clock: {trn['total_wall_clock_hours']:.4f} h)",
                f"  - Peak VRAM            : {trn['peak_vram_mb']:.2f} MB ({trn['peak_vram_gb']:.4f} GB) [reserved: {trn['peak_vram_reserved_mb']:.2f} MB]",
            ]
        )
    else:
        lines.append("[3. Training]            : Skipped (--efficiency-only mode)")

    lines.extend(
        [
            "--------------------------------------------------------------------------------",
            "EXCEL PASTE ROW (MyoPS380 Efficiency Table):",
            (
                f"  Trainable Params: {ex['Trainable Params (M)']:.2f}M | "
                f"FLOPs: {ex['FLOPs (G)']:.2f} G | "
                f"Inf Latency: {ex['Inference Latency (ms/slice)']:.2f} ms | "
                f"Inf VRAM: {ex['Inference Peak VRAM (MB)']:.2f} MB ({ex['Inference Peak VRAM (GB)']:.4f} GB) | "
                + (
                    f"Time/Epoch: {ex['Training Time / Epoch (s)']:.2f} s | "
                    f"Total Time: {ex['Training Total Time (hours)']:.4f} h | "
                    f"Train VRAM: {ex['Training Peak VRAM (MB)']:.2f} MB ({ex['Training Peak VRAM (GB)']:.4f} GB)"
                    if trn is not None
                    else "Training: N/A"
                )
            ),
            "================================================================================",
        ]
    )
    return "\n".join(lines)


def save_efficiency_report(
    payload: dict[str, Any],
    output_paths: Sequence[Path],
    summary_text_paths: Sequence[Path] = (),
    log: logging.Logger | None = None,
) -> None:
    """Atomically write efficiency_metrics.json (and optional text summary) to target paths."""
    report_str = format_efficiency_report(payload)
    for out_path in output_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(out_path)
        if log is not None:
            log.info("Efficiency metrics JSON saved to: %s", out_path)

    for txt_path in summary_text_paths:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_txt = txt_path.with_suffix(txt_path.suffix + ".tmp")
        tmp_txt.write_text(report_str + "\n", encoding="utf-8")
        tmp_txt.replace(txt_path)

    if log is not None:
        for line in report_str.splitlines():
            log.info("%s", line)
