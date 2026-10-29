#!/bin/bash
# ==============================================================================
# Script cài đặt môi trường và huấn luyện cho SAMed trên Google Colab CLI
# Repository: https://github.com/nguyentranhuynhchi/SAMed.git (Nhánh: main)
# Benchmark Protocol: myops380_train304_test76_last_v1 (No-Val, 300 Epochs)
# Phỏng theo quy chuẩn kiến trúc Environment của dự án (Enveronment.txt)
# ==============================================================================

set -e

echo "=== [1/7] Kiểm tra và Kích hoạt GPU NVIDIA ==="
cd /content

# Tự động liên kết thư viện driver NVIDIA nếu có trên Colab (/usr/lib64-nvidia)
if [ -d "/usr/lib64-nvidia" ]; then
    if ! grep -q "/usr/lib64-nvidia" /etc/ld.so.conf.d/nvidia.conf 2>/dev/null; then
        echo "/usr/lib64-nvidia" >> /etc/ld.so.conf.d/nvidia.conf
        ldconfig 2>/dev/null || true
    fi
    export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH}
fi
if ! grep -q "/usr/lib64-nvidia" /root/.bashrc 2>/dev/null; then
    echo 'export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH' >> /root/.bashrc
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ CẢNH BÁO: Chưa tìm thấy NVIDIA GPU. Hãy kiểm tra lại Runtime > Change runtime type > GPU trên Colab!"
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

echo "=== [2/7] Mount Google Drive (Lưu Checkpoint an toàn) ==="
python3 -c "
import os
if not os.path.exists('/content/drive/MyDrive'):
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print('✅ Mount Google Drive thành công!')
    except Exception as e:
        print('⚠️ Không thể mount Drive tự động (bỏ qua nếu không dùng):', e)
" || true

DRIVE_DIR="/content/drive/MyDrive/MyoPS_Results/SAMed"
mkdir -p "$DRIVE_DIR" || true

echo "=== [3/7] Clone Repository SAMed (Nhánh: main) ==="
if [ ! -d "/content/SAMed" ]; then
    git clone https://github.com/nguyentranhuynhchi/SAMed.git /content/SAMed
else
    echo "Thư mục /content/SAMed đã tồn tại. Tiến hành cập nhật nhánh main..."
    cd /content/SAMed
    git checkout main
    git pull origin main || true
fi
cd /content/SAMed

echo "=== [4/7] Cài đặt Dependencies cho SAMed ==="
python3 -m pip install --upgrade pip -q
python3 -m pip install -q icecream tensorboardX gdown medpy SimpleITK einops safetensors monai nibabel

echo "=== [5/7] Chuẩn bị Pretrained Checkpoint (sam_vit_b_01ec64.pth) ==="
mkdir -p /content/SAMed/checkpoints
CKPT_FILE="/content/SAMed/checkpoints/sam_vit_b_01ec64.pth"
DRIVE_CKPT="$DRIVE_DIR/checkpoints/sam_vit_b_01ec64.pth"

if [ -s "$CKPT_FILE" ]; then
    echo "✅ Checkpoint đã có sẵn tại: $CKPT_FILE"
elif [ -s "$DRIVE_CKPT" ]; then
    echo "Đang sao chép checkpoint từ Google Drive đã mount..."
    cp "$DRIVE_CKPT" "$CKPT_FILE"
    echo "✅ Sao chép checkpoint từ Google Drive thành công!"
else
    echo "Đang tải Meta SAM ViT-B pretrained weights (375MB)..."
    wget -c -q --show-progress -O "$CKPT_FILE" "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" || {
        echo "Tải qua wget lỗi, thử gdown..."
        gdown --id "1UhD3-ncoARjcv_1IqUcMa_G4oaU9pY-b" -O /content/sam_vit_b_01ec64.pth || true
    }
    # Tự động lưu 1 bản vào Drive để lần sau không phải tải lại
    if [ -d "/content/drive/MyDrive" ]; then
        mkdir -p "$DRIVE_DIR/checkpoints"
        cp "$CKPT_FILE" "$DRIVE_CKPT" 2>/dev/null || true
    fi
    echo "✅ Tải checkpoint SAM ViT-B hoàn tất!"
fi

echo "=== [6/7] Tải Dataset MyoPS-380 ==="
LOCAL_DATASET="/content/MyoPS380_dataset/Processed_data"
if [ ! -d "$LOCAL_DATASET" ]; then
    echo "Đang tải Dataset MyoPS-380 từ Google Drive..."
    gdown --id "1UhD3-ncoARjcv_1IqUcMa_G4oaU9pY-b" -O /content/MyoPS380_dataset.zip
    mkdir -p /content/MyoPS380_dataset
    command -v unzip &>/dev/null || (apt-get update -qq && apt-get install -y unzip)
    unzip -q -o /content/MyoPS380_dataset.zip -d /content/MyoPS380_dataset/
fi

# Tự động định vị thư mục Processed_data chuẩn
FOUND_BSSFP=$(find /content/MyoPS380_dataset -type d -name "bSSFP" 2>/dev/null | head -n 1)
if [ -n "$FOUND_BSSFP" ]; then
    LOCAL_DATASET=$(dirname "$FOUND_BSSFP")
    echo "✅ Đã xác định thư mục dataset chuẩn tại: $LOCAL_DATASET"
else
    echo "✅ Thư mục dataset mặc định: $LOCAL_DATASET"
fi

echo "=== [7/7] Cài đặt tmux và Kiểm tra PyTorch ==="
command -v tmux &> /dev/null || (apt-get update -qq && apt-get install -y tmux)
python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=============================================================================="
echo "🎉 HOÀN TẤT THIẾT LẬP MÔI TRƯỜNG CHO SAMED!"
echo "📁 Dữ liệu: $LOCAL_DATASET"
echo "💾 Checkpoint: $CKPT_FILE"
echo "☁️ Thư mục lưu Google Drive: $DRIVE_DIR"
echo "=============================================================================="
