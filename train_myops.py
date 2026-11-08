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
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time

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
    parser.add_argument("--data_root", "--data-root", default="/content/MyoPS380_dataset/Processed_data",
                        help="Root folder containing bSSFP, LGE, T2w subdirectories")
    parser.add_argument("--list_dir", "--list-dir", default=str(PROJECT_ROOT / "lists" / "myops380_no_val"),
                        help="Directory containing train.txt and test_vol.txt")
    parser.add_argument("--output_dir", "--output-dir", "--run_dir", default=None,
                        help="Run output directory; defaults to runs/samed_seed<seed>")
    parser.add_argument("--model_id", default="samed", help="Canonical lowercase model ID")
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
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto", help="Mixed precision mode")
    parser.add_argument("--no_val", "--no-val", action="store_true", default=True, help="Fixed no-val training protocol")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True, help="Deterministic algorithms")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--log_every", type=int, default=20, help="Logging step frequency")
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
    if mode == "bf16" or (mode == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def atomic_save(path: Path | str, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Resolve paths
    run_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "runs" / f"{args.model_id}_seed{args.seed}"
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

    logging.info(f"=== Starting SAMed Training on MyoPS380 (Seed {args.seed}) ===")
    logging.info(f"Run directory: {run_dir}")
    logging.info(f"Data root: {args.data_root} | List dir: {args.list_dir}")

    # Build SAM with LoRA
    # In SAMed, num_multimask_outputs = args.num_classes - 1 (e.g. 4 - 1 = 3),
    # producing exactly num_mask_tokens = 3 + 1 = 4 output channels.
    num_multimask = args.num_classes - 1 if args.num_classes > 1 else 1
    sam_checkpoint = args.ckpt if (args.ckpt and os.path.exists(args.ckpt)) else None
    if sam_checkpoint:
        logging.info(f"Loading pretrained SAM backbone from: {sam_checkpoint}")
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
    if args.lora_ckpt and os.path.exists(args.lora_ckpt):
        logging.info(f"Loading initial LoRA checkpoint from: {args.lora_ckpt}")
        net.load_lora_parameters(args.lora_ckpt)

    net.to(device)

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # Master benchmark log line
    logging.info(
        f"INFO | Device {device.type} | AMP {amp_str} | "
        f"parameters {total_params:,} (trainable: {trainable_params:,}) | "
        f"train/val slices 1972/0 | effective batch <= {args.batch_size}"
    )

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
        pin_memory=(device.type == "cuda"),
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

    # Resume handling
    if args.resume:
        logging.info(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_optimization_checkpoint(checkpoint)

        model_weights = checkpoint.get("model_state_dict", checkpoint.get("model"))
        if model_weights is not None:
            net.load_state_dict(model_weights)

        opt_state = checkpoint.get("optimizer_state_dict", checkpoint.get("optimizer"))
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
            # Ensure optimizer state tensors are on the correct device
            for state_item in optimizer.state.values():
                for k, v in state_item.items():
                    if torch.is_tensor(v):
                        state_item[k] = v.to(device)

        sched_state = checkpoint.get("scheduler_state_dict", checkpoint.get("scheduler"))
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)

        scaler_state = checkpoint.get("scaler_state_dict", checkpoint.get("scaler"))
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)

        start_epoch = checkpoint.get("epoch", -1) + 1
        global_step = checkpoint.get("global_step", checkpoint.get("successful", start_epoch * updates_per_epoch))
        amp_skipped_updates = checkpoint.get("amp_skipped_updates", 0)

        # Restore RNG states
        rng = checkpoint.get("rng_states", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "torch_cuda" in rng and torch.cuda.is_available() and rng["torch_cuda"] is not None:
            try:
                if isinstance(rng["torch_cuda"], list) and len(rng["torch_cuda"]) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(rng["torch_cuda"])
                elif isinstance(rng["torch_cuda"], list) and len(rng["torch_cuda"]) > 0:
                    torch.cuda.set_rng_state(rng["torch_cuda"][0])
            except Exception as e:
                logging.warning(f"Could not restore CUDA RNG state: {e}")

        if "dataloader_generator_state" in checkpoint:
            try:
                g.set_state(checkpoint["dataloader_generator_state"])
            except Exception as e:
                logging.warning(f"Could not restore DataLoader generator state: {e}")

        logging.info(f"Resumed successfully at epoch {start_epoch}, global_step {global_step}")

    # Save initial config
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

    # Training Loop (Strictly NO validation)
    net.train()
    for epoch in range(start_epoch, args.max_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        running_ce = 0.0
        running_dice = 0.0

        optimizer.zero_grad()

        for i_batch, sampled in enumerate(trainloader):
            images = sampled["image"].to(device, non_blocking=True)
            low_res_labels = sampled["low_res_label"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
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

        epoch_duration = time.time() - epoch_start_time
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

        checkpoint_payload = {
            "epoch": epoch,
            "planned_epochs": args.max_epochs,
            "completed_epochs": epoch + 1,
            "training_complete": is_complete,
            "global_step": global_step,
            "successful": global_step,
            "total_updates": total_planned_updates,
            "amp_skipped_updates": amp_skipped_updates,
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

    logging.info(f"=== Training Complete for 300 Epochs (Final Checkpoint: {final_ckpt_path}) ===")


if __name__ == "__main__":
    main()
