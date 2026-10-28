import argparse
import hashlib
import logging
import math
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from torchvision import transforms
from icecream import ic


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight: float = 0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


def trainer_synapse(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                output_masks = outputs['masks']
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output_masks[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 20 # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_myops(args, model, snapshot_path, multimask_output, low_res):
    """Benchmark No-Val Trainer for MyoPS-380 protocol myops380_train304_test76_last_v1."""
    from datasets.dataset_myops import MyopsDataset, RandomGenerator

    log_file = os.path.join(snapshot_path, "train.log")
    logging.basicConfig(
        filename=log_file, level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Verify dataset split SHA-256
    train_split_path = os.path.join(args.list_dir, "train.txt")
    if os.path.isfile(train_split_path):
        with open(train_split_path, "rb") as fp:
            train_hash = hashlib.sha256(fp.read()).hexdigest()
        logging.info("train.txt SHA-256: " + train_hash + " (Expected: 16248ec3630bb0493fb421aab5ffb4dc9b1d0fd49c8614d69d3098a92305f919)")
        if train_hash != "16248ec3630bb0493fb421aab5ffb4dc9b1d0fd49c8614d69d3098a92305f919":
            raise RuntimeError(f"train.txt checksum mismatch: {train_hash}. Aborting to protect benchmark integrity.")

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    effective_batch_size = getattr(args, "effective_batch_size", 16) or 16
    accum_steps = max(1, effective_batch_size // batch_size)

    logging.info(f"Configuration: batch_size={batch_size}, accum_steps={accum_steps}, effective_batch_size={batch_size * accum_steps}")
    logging.info(f"Protocol: max_epochs={args.max_epochs}, num_classes={num_classes}, base_lr={base_lr}, dice_param={args.dice_param}")

    db_train = MyopsDataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res]),
        label_order="legacy"
    )
    logging.info(f"Loaded MyoPS-380 train dataset: {len(db_train)} slices")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    num_workers = getattr(args, "num_workers", 2)
    trainloader = DataLoader(
        db_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn, drop_last=False
    )

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: total={total_params:,} | trainable(LoRA+Head)={trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    weight_decay = getattr(args, "weight_decay", 0.0001)
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, momentum=0.9, weight_decay=weight_decay)

    writer = SummaryWriter(os.path.join(snapshot_path, "tensorboard"))
    max_epochs = args.max_epochs
    total_updates = max_epochs * math.ceil(len(trainloader) / accum_steps)
    logging.info(f"Total updates: {total_updates} over {max_epochs} epochs ({len(trainloader)} batches/epoch)")

    checkpoints_dir = os.path.join(snapshot_path, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    update_step = 0
    group_samples = 0

    for epoch_num in range(max_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_dice = 0.0
        batch_count_total = 0

        optimizer.zero_grad(set_to_none=True)
        group_samples = 0

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch["image"].cuda()
            low_res_label_batch = sampled_batch["low_res_label"].cuda()

            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)

            curr_bs = image_batch.shape[0]
            (loss * curr_bs).backward()
            group_samples += curr_bs

            epoch_loss += loss.item() * curr_bs
            epoch_loss_ce += loss_ce.item() * curr_bs
            epoch_loss_dice += loss_dice.item() * curr_bs
            batch_count_total += curr_bs

            is_boundary = ((i_batch + 1) % accum_steps == 0) or ((i_batch + 1) == len(trainloader))
            if is_boundary:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(group_samples)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                group_samples = 0
                update_step += 1

                # Learning rate schedule: Warmup + Polynomial Decay
                if args.warmup and update_step < args.warmup_period:
                    curr_lr = base_lr * (update_step / args.warmup_period)
                else:
                    shift = args.warmup_period if args.warmup else 0
                    curr_lr = base_lr * (max(0.0, 1.0 - (update_step - shift) / (total_updates - shift))) ** 0.9

                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr

        epoch_duration = time.time() - epoch_start_time
        avg_loss = epoch_loss / batch_count_total
        avg_ce = epoch_loss_ce / batch_count_total
        avg_dice = epoch_loss_dice / batch_count_total

        writer.add_scalar("train/loss", avg_loss, epoch_num + 1)
        writer.add_scalar("train/ce", avg_ce, epoch_num + 1)
        writer.add_scalar("train/dice_loss", avg_dice, epoch_num + 1)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch_num + 1)

        logging.info(f"Epoch {epoch_num+1}/{max_epochs} | loss: {avg_loss:.4f} | ce: {avg_ce:.4f} | dice_loss: {avg_dice:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f} | time: {epoch_duration:.1f}s")

        # Save last checkpoint (atomically overwrite every epoch)
        last_lora_path = os.path.join(checkpoints_dir, "last_lora.pth")
        try:
            model.save_lora_parameters(last_lora_path)
        except Exception:
            model.module.save_lora_parameters(last_lora_path)

        full_ckpt = {
            "epoch": epoch_num + 1,
            "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args)
        }
        torch.save(full_ckpt, os.path.join(checkpoints_dir, "last.pth"))
        torch.save(full_ckpt, os.path.join(checkpoints_dir, "last.pt"))

    writer.close()
    logging.info(f"Training completed {max_epochs} epochs! Final checkpoint saved to {checkpoints_dir}")
    return "Training Finished!"
