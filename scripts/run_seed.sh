#!/usr/bin/env bash
set -e

SEED=${1:-1234}
IMG_SIZE=${2:-512}
BATCH_SIZE=${3:-8}
ACCUM_STEPS=${4:-2}
DATA_ROOT=${5:-"/content/MyoPS380/Processed_data"}
CKPT="checkpoints/sam_vit_b_01ec64.pth"

RUN_DIR="./outputs/runs/SAMed_MyoPS380_Seed_${SEED}"

echo "=========================================================="
echo " Starting SAMed Benchmark: Seed ${SEED} (Resolution: ${IMG_SIZE})"
echo "=========================================================="

export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1

# 1. Download pretrained SAM ViT-B checkpoint if missing
if [ ! -f "$CKPT" ]; then
    echo "[SAMed] Downloading Meta SAM ViT-B pretrained weights..."
    mkdir -p checkpoints
    wget -q --show-progress -O "$CKPT" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi

# 2. Train for 300 epochs under protocol myops380_train304_test76_last_v1
echo "[SAMed] Starting training 300 epochs for Seed ${SEED}..."
python train.py \
    --dataset MyoPS380 \
    --root_path "$DATA_ROOT" \
    --list_dir ./lists/lists_MyoPS380 \
    --output ./outputs/runs \
    --ckpt "$CKPT" \
    --seed "$SEED" \
    --max_epochs 300 \
    --batch_size "$BATCH_SIZE" \
    --effective_batch_size 16 \
    --num_workers 2 \
    --cpu_threads 2 \
    --base_lr 0.005 \
    --warmup \
    --AdamW \
    --img_size "$IMG_SIZE" \
    --num_classes 4

# 3. Evaluate checkpoint last.pth / last_lora.pth on 76 test volumes
echo "[SAMed] Evaluating Seed ${SEED} on 76 test volumes..."
python evaluate.py \
    --root_path "$DATA_ROOT" \
    --list_dir ./lists/lists_MyoPS380 \
    --output_dir "$RUN_DIR" \
    --lora_ckpt "$RUN_DIR/checkpoints/last_lora.pth" \
    --ckpt "$CKPT" \
    --img_size "$IMG_SIZE" \
    --num_classes 4 \
    --cpu_threads 2 \
    --batch_size 8

# 4. Package results zip
ZIP_NAME="/content/SAMed_Seed_${SEED}_Results.zip"
echo "[SAMed] Packaging results to $ZIP_NAME..."
(cd "$RUN_DIR" && zip -r -q "$ZIP_NAME" .)

echo "=========================================================="
echo " Completed SAMed Seed ${SEED} successfully!"
echo " Results packaged at: $ZIP_NAME"
echo "=========================================================="
