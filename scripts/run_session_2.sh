#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "🚀 STARTING SAMED BENCHMARK SESSION 2: Seeds 2026, 6868"
echo "=========================================================="

# 1. Tải Dataset MyoPS-380 nếu chưa có
LOCAL_DATASET="/content/MyoPS380_dataset/Processed_data"
if [ ! -d "$LOCAL_DATASET" ]; then
    echo "📥 Đang tải Dataset MyoPS-380..."
    gdown --id "1UhD3-ncoARjcv_1IqUcMa_G4oaU9pY-b" -O /content/MyoPS380_dataset.zip
    mkdir -p /content/MyoPS380_dataset
    unzip -q -o /content/MyoPS380_dataset.zip -d /content/MyoPS380_dataset/
fi

# 2. Tải SAM ViT-B pretrained weights nếu chưa có
CKPT="checkpoints/sam_vit_b_01ec64.pth"
mkdir -p checkpoints
if [ ! -f "$CKPT" ]; then
    echo "📥 Đang tải Pretrained SAM ViT-B weights (375MB)..."
    wget -c -q --show-progress -O "$CKPT" "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
fi

export OMP_NUM_THREADS=2

# 3. Chạy song song 2 Seed trên GPU A100
echo "🔥 Bắt đầu Train SONG SONG 2 Seed: 2026, 6868..."
bash scripts/run_seed.sh 2026 512 8 2 "$LOCAL_DATASET" > train_2026.log 2>&1 &
bash scripts/run_seed.sh 6868 512 8 2 "$LOCAL_DATASET" > train_6868.log 2>&1 &
wait

echo "=========================================================="
echo "✅ HOÀN TẤT CẢ 2 SEEDS SESSION 2 (2026, 6868)!"
echo "📦 Các file zip kết quả đã sẵn sàng tại /content/"
echo "🕒 Kích hoạt lệnh treo phiên ngầm chống Colab tự ngắt..."
echo "=========================================================="
nohup bash -c 'while true; do sleep 60; done' >/dev/null 2>&1 &
echo "🎉 ĐÃ TREO PHIÊN THÀNH CÔNG! BẠN CÓ THỂ ĐÓNG MÁY HOẶC TẢI KẾT QUẢ BẤT CỨ LÚC NÀO!"
