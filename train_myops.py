"""Entry point for training SAMed on MyoPS380 without validation.

Protocol: myops380_train304_test76_last_v1 (304 train patients / 1,972 slices, 300 epochs, no validation).
Total updates: 300 epochs * 124 batches = 37,200 updates.
Optimizer: AdamW (base_lr=0.0003333, weight_decay=0.0001, betas=(0.9, 0.999), eps=1e-8).
Scheduler: Polynomial decay power 0.9 with zero-based offset (im_mseg_adamw_poly_v1).
Loss: SAMed original objective: 0.5 * CrossEntropyLoss + 0.5 * DiceLoss.
Checkpointing: Atomic write to <run_dir>/checkpoints/samed_seed<seed>.pt after each epoch.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure local imports work cleanly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset_myops import (
    MyopsDataset,
    RandomGenerator,
    CLASS_NAMES,
    CANONICAL_LABEL_ORDER,
    LEGACY_LABEL_ORDER,
)
from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
from utils.efficiency import (
    SAMedInferenceWrapper,
    build_efficiency_payload,
    check_rule_compliance,
    count_parameters,
    measure_flops,
    measure_inference_efficiency,
    save_efficiency_report,
    summarize_training_efficiency,
    sync_if_cuda,
)
from utils.loss import DiceLoss, calc_loss
from utils.metrics import BENCHMARK_PROTOCOL
from utils.optimization import (
    OPTIMIZATION_PROTOCOL,
    build_optimizer,
    build_scheduler,
    validate_optimization_checkpoint,
)


NO_VAL_PROTOCOL = {
    "id": "myops380_train304_test76_last_v1",
    "validation": False,
    "checkpoint_selection": "last_completed_epoch",
    "budget_unit": "epochs",
    "train_patients": 304,
    "test_patients": 76,
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", "--data-root",
                        default=os.environ.get("MYOPS380_DATA_ROOT", "/content/MyoPS380_dataset/Processed_data"),
                        help="Root folder containing bSSFP, LGE, T2w subdirectories")
    parser.add_argument("--list_dir", "--list-dir", default=str(PROJECT_ROOT / "lists" / "myops380_no_val"),
                        help="Directory containing train.txt and test_vol.txt")
    parser.add_argument("--output_dir", "--output-dir", "--run_dir", "--run-dir", default=None,
                        help="Run output directory; defaults to runs/samed_seed<seed>")
    parser.add_argument("--run_root", "--run-root", default=None,
                        help="Optional run root directory when --output_dir is not provided")
    parser.add_argument("--run_name", "--run-name", default=None,
                        help="Optional run folder name under --run_root (default: samed_seed<seed>)")
    parser.add_argument("--model_id", default="samed", help="Canonical lowercase model ID")
    parser.add_argument("--dataset_id", "--dataset-id", default="myops380", help="Dataset identifier")
    parser.add_argument("--vit_name", default="vit_b", choices=["vit_b", "vit_l", "vit_h"], help="ViT backbone variant")
    parser.add_argument("--ckpt", default="checkpoints/sam_vit_b_01ec64.pth", help="Pretrained SAM weights path")
    parser.add_argument("--lora_ckpt", default=None, help="Optional checkpoint to initialize LoRA weights")
    parser.add_argument("--rank", type=int, default=4, help="Rank for LoRA adaptation")
    parser.add_argument("--img_size", "--img-size", type=int, default=128, help="Model input spatial size (128x128)")
    parser.add_argument("--num_classes", "--num-classes", type=int, default=4, help="Number of canonical classes (4)")
    parser.add_argument("--max_epochs", "--epochs", type=int, default=300, help="Planned training epochs (300)")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16, help="Physical batch size on device (16)")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps (1)")
    parser.add_argument("--base_lr", "--lr", type=float, default=0.0003333, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for AdamW")
    parser.add_argument("--clip_grad", type=float, default=0.0, help="Gradient clipping norm (0.0: disabled)")
    parser.add_argument("--dice_param", "--dice_weight", type=float, default=0.5, help="Dice weight in 0.5 CE + 0.5 Dice")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=4,
                        help="DataLoader worker processes (use 0 for efficiency benchmark)")
    parser.add_argument("--pin_memory", "--pin-memory", dest="pin_memory",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Pin host memory for CUDA transfers (default: True on CUDA)")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto", help="Mixed precision mode")
    parser.add_argument("--no_val", "--no-val", action="store_true", default=True, help="Fixed no-val training protocol")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True, help="Deterministic algorithms")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Path to checkpoint to resume training from (or pass without path for auto-resume)")
    parser.add_argument("--log_every", "--log-every", dest="log_every", type=int, default=20, help="Logging step frequency")
    parser.add_argument("--measure_efficiency", "--measure-efficiency", dest="measure_efficiency",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Measure Complexity, Inference, and Training efficiency metrics and export JSON")
    parser.add_argument("--no_measure_efficiency", dest="measure_efficiency", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--efficiency_output", "--efficiency-output", dest="efficiency_output",
                        default="outputs/runs/samed/efficiency_metrics.json",
                        help="Primary JSON output path for efficiency metrics")
    parser.add_argument("--efficiency_only", "--efficiency-only", dest="efficiency_only",
                        action="store_true",
                        help="Measure Complexity + Inference efficiency only (skip training loop)")
    return parser


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def atomic_save(path: Path | str, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Resolve paths
    if args.output_dir:
        run_dir = Path(args.output_dir)
    elif args.run_root or args.run_name:
        base_root = Path(args.run_root) if args.run_root else (PROJECT_ROOT / "runs")
        run_dir = base_root / (args.run_name or f"{args.model_id}_seed{args.seed}")
    else:
        run_dir = PROJECT_ROOT / "runs" / f"{args.model_id}_seed{args.seed}"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    for d in (checkpoints_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    final_ckpt_path = checkpoints_dir / f"{args.model_id}_seed{args.seed}.pt"

    # Setup logger
    log_file = logs_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
        handlers=[
            logging.FileHandler(str(log_file), mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    seed_everything(args.seed, args.deterministic)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp(args.amp, device)
    amp_str = "none" if amp_dtype is None else ("bf16" if amp_dtype == torch.bfloat16 else "fp16")
    pin_memory = (device.type == "cuda") if args.pin_memory is None else bool(args.pin_memory)

    logging.info(f"=== Starting SAMed Training on MyoPS380 (Seed {args.seed}) ===")
    logging.info(f"Run directory: {run_dir}")
    logging.info(f"Data root: {args.data_root} | List dir: {args.list_dir}")

    rule_compliant, compliance_warnings, hardware_info = check_rule_compliance(
        device=device,
        amp_mode=args.amp,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        batch_size=args.batch_size,
        planned_epochs=args.max_epochs,
        accum_steps=args.accum_steps,
    )
    if args.measure_efficiency or args.efficiency_only:
        logging.info(
            "Efficiency Config | GPU: %s (count=%d) | Precision: %s | num_workers: %d | pin_memory: %s | batch: %d | epochs: %d",
            hardware_info["gpu_name"],
            hardware_info["visible_gpu_count"],
            hardware_info["precision"],
            args.num_workers,
            pin_memory,
            args.batch_size,
            args.max_epochs,
        )
        if not rule_compliant:
            for warn_msg in compliance_warnings:
                logging.warning("[EFFICIENCY RULE MISMATCH] %s", warn_msg)

    # Check resume checkpoint on CPU before allocating optimizer/CUDA memory
    resume_target: Path | None = None
    if args.resume is not None:
        if args.resume == "auto":
            if final_ckpt_path.is_file():
                resume_target = final_ckpt_path
            else:
                logging.info("--resume auto requested, but no checkpoint exists at %s; starting fresh.", final_ckpt_path)
        else:
            resume_target = Path(args.resume)
            if not resume_target.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_target}")

    resume_ckpt: dict[str, Any] | None = None
    if resume_target is not None:
        logging.info(f"Loading checkpoint metadata from: {resume_target}")
        resume_ckpt = torch.load(resume_target, map_location="cpu", weights_only=False)
        validate_optimization_checkpoint(resume_ckpt)
        if resume_ckpt.get("training_complete", False) and not args.efficiency_only:
            completed = resume_ckpt.get("completed_epochs", resume_ckpt.get("epoch", -1) + 1)
            logging.info(f"Training was already marked complete at epoch {completed}/{args.max_epochs}. Nothing to run.")
            return

    # Build SAM with LoRA on CPU first
    # In SAMed, num_multimask_outputs = args.num_classes - 1 (e.g. 4 - 1 = 3),
    # producing exactly num_mask_tokens = 3 + 1 = 4 output channels.
    num_multimask = args.num_classes - 1 if args.num_classes > 1 else 1
    has_full_resume_weights = (
        resume_ckpt is not None
        and (resume_ckpt.get("model_state_dict") is not None or resume_ckpt.get("model") is not None)
    )
    sam_checkpoint = args.ckpt if (not has_full_resume_weights and args.ckpt and os.path.exists(args.ckpt)) else None
    if sam_checkpoint:
        logging.info(f"Loading pretrained SAM backbone from: {sam_checkpoint}")
    elif has_full_resume_weights:
        logging.info("Resuming full SAMed weights from checkpoint; skipping base SAM checkpoint load.")
    else:
        logging.info("No pretrained SAM backbone checkpoint found on local disk. Initializing architecture.")

    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=num_multimask,
        checkpoint=sam_checkpoint,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    net = LoRA_Sam(sam, r=args.rank)
    if resume_ckpt is not None:
        model_weights = resume_ckpt.get("model_state_dict", resume_ckpt.get("model"))
        if model_weights is not None:
            net.load_state_dict(model_weights)
        elif args.lora_ckpt and os.path.exists(args.lora_ckpt):
            logging.info(f"Loading initial LoRA checkpoint from: {args.lora_ckpt}")
            net.load_lora_parameters(args.lora_ckpt)
    elif args.lora_ckpt and os.path.exists(args.lora_ckpt):
        logging.info(f"Loading initial LoRA checkpoint from: {args.lora_ckpt}")
        net.load_lora_parameters(args.lora_ckpt)

    # Ensure FP32 base weights and move model to target device
    net.float().to(device)

    params_info = count_parameters(net)
    total_params = params_info["total_params"]
    trainable_params = params_info["trainable_params"]

    # Master benchmark log line
    logging.info(
        f"INFO | Device {device.type} | AMP {amp_str} | "
        f"parameters {total_params:,} (trainable: {trainable_params:,}) | "
        f"train/val slices 1972/0 | effective batch <= {args.batch_size}"
    )

    # Save initial config early (including efficiency-only runs)
    config_dict = {
        "model_id": args.model_id,
        "model_name": "SAMed",
        "vit_name": args.vit_name,
        "rank": args.rank,
        "img_size": args.img_size,
        "num_classes": args.num_classes,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "base_lr": args.base_lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "amp": args.amp,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "measure_efficiency": bool(args.measure_efficiency),
        "no_val": True,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "training_protocol": NO_VAL_PROTOCOL,
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "benchmark_protocol": BENCHMARK_PROTOCOL,
        "label_order": "legacy",
        "exported_label_order": "canonical",
    }
    with open(logs_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    efficiency_json_targets = _dedupe_paths(
        [Path(args.efficiency_output), run_dir / "efficiency_metrics.json"]
    )
    efficiency_summary_targets = _dedupe_paths([logs_dir / "efficiency_summary.txt"])

    # Measure Complexity & Batch=1 Inference cleanly BEFORE optimizer/DataLoader GPU state allocation
    flops_info: dict[str, Any] | None = None
    inference_info: dict[str, Any] | None = None
    if args.measure_efficiency or args.efficiency_only:
        eval_wrapper = SAMedInferenceWrapper(net, image_size=args.img_size, multimask_output=True)
        logging.info(
            "Measuring Complexity (FP32 FLOPs on 1x3x%dx%d eval) & Inference (Batch=1, 30 warmup / 100 runs)...",
            args.img_size,
            args.img_size,
        )
        flops_info = measure_flops(
            eval_wrapper, input_shape=(1, 3, args.img_size, args.img_size), device=device
        )
        inference_info = measure_inference_efficiency(
            eval_wrapper, input_shape=(1, 3, args.img_size, args.img_size), device=device, warmup=30, repeats=100
        )
        logging.info(
            "Complexity | Trainable Params: %.2fM (Total: %.2fM) | FLOPs: %.2f G",
            params_info["trainable_params_m"],
            params_info["total_params_m"],
            flops_info["flops_g"],
        )
        logging.info(
            "Inference (B=1, FP32) | Latency: %.2f ms/slice | Peak VRAM: %.2f MB (%.4f GB)",
            inference_info["latency_ms_per_slice"],
            inference_info["peak_vram_mb"],
            inference_info["peak_vram_gb"],
        )
        if args.efficiency_only:
            payload = build_efficiency_payload(
                model_name="SAMed",
                model_id=args.model_id,
                seed=args.seed,
                params_info=params_info,
                flops_info=flops_info,
                inference_info=inference_info,
                training_info=None,
                rule_compliant=rule_compliant,
                compliance_warnings=compliance_warnings,
                hardware_info=hardware_info,
            )
            save_efficiency_report(
                payload, efficiency_json_targets, efficiency_summary_targets, log=logging.getLogger()
            )
            return

    # Dataset & Loader
    # Low-res resolution for SAMed mask decoder is (img_embedding_size * 4, img_embedding_size * 4) = (32, 32)
    low_res = (img_embedding_size * 4, img_embedding_size * 4)
    db_train = MyopsDataset(
        base_dir=args.data_root,
        list_dir=args.list_dir,
        split="train",
        transform=RandomGenerator(output_size=[args.img_size, args.img_size]),
        label_order="legacy",
        low_res=low_res,
    )
    logging.info(f"Training dataset size: {len(db_train)} slices (expected: 1,972)")

    g = torch.Generator()
    g.manual_seed(args.seed)
    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )

    batches_per_epoch = len(trainloader)
    updates_per_epoch = (batches_per_epoch + args.accum_steps - 1) // args.accum_steps
    total_planned_updates = args.max_epochs * updates_per_epoch
    logging.info(
        f"Batches per epoch: {batches_per_epoch} | Accumulation steps: {args.accum_steps} | "
        f"Planned epochs: {args.max_epochs} | Total updates: {total_planned_updates} (expected: 37,200)"
    )

    # Clear CUDA cache & reset peak memory stats right before allocating training state
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Optimizer & Scheduler
    optimizer = build_optimizer(net, base_lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, total_updates=total_planned_updates)

    # Scaler for FP16
    scaler = torch.amp.GradScaler("cuda") if (amp_dtype == torch.float16 and device.type == "cuda") else None

    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=args.num_classes)

    start_epoch = 0
    global_step = 0
    amp_skipped_updates = 0
    epoch_times: list[float] = []
    prev_wall_clock_s = 0.0
    max_train_peak_vram_bytes = 0
    max_train_reserved_bytes = 0

    # Restore optimizer, scheduler, RNG, and efficiency stats on resume
    if resume_ckpt is not None:
        logging.info(f"Resuming optimizer, scheduler, and RNG states from: {resume_target}")

        opt_state = resume_ckpt.get("optimizer_state_dict", resume_ckpt.get("optimizer"))
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
            # Ensure optimizer state tensors are on the correct device
            for state_item in optimizer.state.values():
                for k, v in state_item.items():
                    if torch.is_tensor(v):
                        state_item[k] = v.to(device)

        sched_state = resume_ckpt.get("scheduler_state_dict", resume_ckpt.get("scheduler"))
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)

        scaler_state = resume_ckpt.get("scaler_state_dict", resume_ckpt.get("scaler"))
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)

        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        global_step = int(resume_ckpt.get("global_step", resume_ckpt.get("successful", start_epoch * updates_per_epoch)))
        amp_skipped_updates = int(resume_ckpt.get("amp_skipped_updates", 0))
        epoch_times = [float(t) for t in resume_ckpt.get("epoch_times", [])][: max(0, start_epoch)]
        prev_wall_clock_s = float(resume_ckpt.get("total_wall_clock_seconds", 0.0))
        max_train_peak_vram_bytes = int(resume_ckpt.get("train_peak_vram_bytes", 0))
        max_train_reserved_bytes = int(resume_ckpt.get("train_peak_reserved_bytes", 0))

        # Restore RNG states safely onto CPU tensors
        rng = resume_ckpt.get("rng_states", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng and rng["torch"] is not None:
            torch.set_rng_state(rng["torch"].cpu())
        if "torch_cuda" in rng and torch.cuda.is_available() and rng["torch_cuda"] is not None:
            try:
                cuda_states = [s.cpu() for s in rng["torch_cuda"]] if isinstance(rng["torch_cuda"], list) else rng["torch_cuda"].cpu()
                if isinstance(cuda_states, list) and len(cuda_states) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(cuda_states)
                elif isinstance(cuda_states, list) and len(cuda_states) > 0:
                    torch.cuda.set_rng_state(cuda_states[0])
            except Exception as e:
                logging.warning(f"Could not restore CUDA RNG state: {e}")

        if "dataloader_generator_state" in resume_ckpt and resume_ckpt["dataloader_generator_state"] is not None:
            try:
                g.set_state(resume_ckpt["dataloader_generator_state"].cpu())
            except Exception as e:
                logging.warning(f"Could not restore DataLoader generator state: {e}")

        logging.info(f"Resumed successfully at epoch {start_epoch}, global_step {global_step}")
        del resume_ckpt

    # Training Loop (Strictly NO validation)
    net.train()
    session_start_time = time.perf_counter()
    for epoch in range(start_epoch, args.max_epochs):
        sync_if_cuda(device)
        epoch_start_time = time.perf_counter()
        running_loss = 0.0
        running_ce = 0.0
        running_dice = 0.0

        optimizer.zero_grad()

        for i_batch, sampled in enumerate(trainloader):
            images = sampled["image"].to(device, non_blocking=pin_memory)
            low_res_labels = sampled["low_res_label"].to(device, non_blocking=pin_memory)

            if device.type == "cuda" and amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    outputs = net(images, multimask_output=True, image_size=args.img_size)
                    loss, loss_ce, loss_dice = calc_loss(outputs, low_res_labels, ce_loss, dice_loss, args.dice_param)
                    loss_scaled = loss / args.accum_steps
            else:
                outputs = net(images, multimask_output=True, image_size=args.img_size)
                loss, loss_ce, loss_dice = calc_loss(outputs, low_res_labels, ce_loss, dice_loss, args.dice_param)
                loss_scaled = loss / args.accum_steps

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss ({loss.item()}) at epoch {epoch}, batch {i_batch}")

            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # Perform optimizer and scheduler update on accumulation boundary or epoch end
            if (i_batch + 1) % args.accum_steps == 0 or (i_batch + 1) == batches_per_epoch:
                if scaler is not None:
                    if args.clip_grad > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], args.clip_grad)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    new_scale = scaler.get_scale()
                    if new_scale < old_scale:
                        amp_skipped_updates += 1
                    else:
                        scheduler.step()
                        global_step += 1
                else:
                    if args.clip_grad > 0.0:
                        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], args.clip_grad)
                    optimizer.step()
                    scheduler.step()
                    global_step += 1

                optimizer.zero_grad()

            running_loss += loss.item()
            running_ce += loss_ce.item()
            running_dice += loss_dice.item()

            if (i_batch + 1) % args.log_every == 0 or (i_batch + 1) == batches_per_epoch:
                current_lr = optimizer.param_groups[0]["lr"]
                logging.info(
                    f"Epoch [{epoch+1:03d}/{args.max_epochs:03d}] "
                    f"Batch [{i_batch+1:03d}/{batches_per_epoch:03d}] "
                    f"Loss: {loss.item():.4f} (CE: {loss_ce.item():.4f}, Dice: {loss_dice.item():.4f}) | "
                    f"LR: {current_lr:.7f} | Step: {global_step}"
                )

        sync_if_cuda(device)
        epoch_duration = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_duration)

        if device.type == "cuda" and torch.cuda.is_available():
            max_train_peak_vram_bytes = max(max_train_peak_vram_bytes, int(torch.cuda.max_memory_allocated(device)))
            max_train_reserved_bytes = max(max_train_reserved_bytes, int(torch.cuda.max_memory_reserved(device)))

        total_wall_clock_s = prev_wall_clock_s + (time.perf_counter() - session_start_time)
        avg_loss = running_loss / batches_per_epoch
        avg_ce = running_ce / batches_per_epoch
        avg_dice = running_dice / batches_per_epoch
        logging.info(
            f"--> Epoch {epoch+1:03d} Completed in {epoch_duration:.1f}s | "
            f"Avg Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Dice: {avg_dice:.4f})"
        )

        # Save atomic checkpoint after each epoch
        is_complete = (epoch + 1 == args.max_epochs)
        parsed_args = vars(args).copy()
        parsed_args["no_val"] = True
        parsed_args["dataset_id"] = args.dataset_id
        parsed_args["pin_memory"] = pin_memory

        checkpoint_payload = {
            "epoch": epoch,
            "planned_epochs": args.max_epochs,
            "completed_epochs": epoch + 1,
            "training_complete": is_complete,
            "global_step": global_step,
            "successful": global_step,
            "total_updates": total_planned_updates,
            "amp_skipped_updates": amp_skipped_updates,
            "epoch_times": epoch_times,
            "total_wall_clock_seconds": total_wall_clock_s,
            "train_peak_vram_bytes": max_train_peak_vram_bytes,
            "train_peak_reserved_bytes": max_train_reserved_bytes,
            "model": net.state_dict(),
            "model_state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "rng_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "dataloader_generator_state": g.get_state(),
            "args": parsed_args,
            "model_name": "SAMed",
            "model_id": args.model_id,
            "dataset_id": "myops380",
            "training_protocol": NO_VAL_PROTOCOL,
            "optimization_protocol": OPTIMIZATION_PROTOCOL,
            "benchmark_protocol": BENCHMARK_PROTOCOL,
            "label_order": "legacy",
            "source_to_internal": "LUT [0, 1, 3, 2]",
            "native_to_export": "identity",
            "split_hashes": {
                "train": "16248ec3630bb0493fb421aab5ffb4dc9b1d0fd49c8614d69d3098a92305f919",
                "test_vol": "3237b31411833cedcb3ecd86d36edae356bd3f880d3fcfd915066f1c850d9cbc",
            },
            "benchmark_data": {
                "schema_version": 2,
                "split_protocol": "myops380_train304_test76_last_v1",
                "source_label_order": "background_normal_scar_edema",
                "canonical_class_names": list(CLASS_NAMES),
                "official_test_ids_sha256": "3237b31411833cedcb3ecd86d36edae356bd3f880d3fcfd915066f1c850d9cbc",
                "split_ids_sha256": {
                    "train": "16248ec3630bb0493fb421aab5ffb4dc9b1d0fd49c8614d69d3098a92305f919",
                    "test_vol": "3237b31411833cedcb3ecd86d36edae356bd3f880d3fcfd915066f1c850d9cbc",
                },
                "patient_counts": {"train": 304, "test_vol": 76},
            },
        }
        atomic_save(final_ckpt_path, checkpoint_payload)
        logging.info(f"Saved atomic checkpoint -> {final_ckpt_path}")

    logging.info(f"=== Training Complete for {args.max_epochs} Epochs (Final Checkpoint: {final_ckpt_path}) ===")

    if args.measure_efficiency and flops_info is not None and inference_info is not None:
        final_wall_clock_s = prev_wall_clock_s + (time.perf_counter() - session_start_time)
        training_info = summarize_training_efficiency(
            epoch_times_s=epoch_times,
            total_wall_clock_s=final_wall_clock_s,
            peak_vram_bytes=max_train_peak_vram_bytes,
            planned_epochs=args.max_epochs,
            warmup_epochs=5,
            peak_reserved_bytes=max_train_reserved_bytes,
        )
        payload = build_efficiency_payload(
            model_name="SAMed",
            model_id=args.model_id,
            seed=args.seed,
            params_info=params_info,
            flops_info=flops_info,
            inference_info=inference_info,
            training_info=training_info,
            rule_compliant=rule_compliant,
            compliance_warnings=compliance_warnings,
            hardware_info=hardware_info,
        )
        save_efficiency_report(
            payload, efficiency_json_targets, efficiency_summary_targets, log=logging.getLogger()
        )


if __name__ == "__main__":
    main()
