#!/bin/bash
# ==============================================================================
# Script cài đặt môi trường tự động hóa cho SAMed trên Google Colab CLI
# Tập dữ liệu: MyoPS380 (304 Train / 76 Test, 300 Epochs, No-Val)
# Ghi chú: STRICTLY NO train command inside this script!
# ==============================================================================

set -e

echo "=== [1/7] Kiểm tra và Kích hoạt GPU NVIDIA ==="
cd /content

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
    echo "⚠️ CẢNH BÁO: Chưa tìm thấy NVIDIA GPU. Hãy kiểm tra lại Runtime Type trên Colab!"
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

echo "=== [3/7] Điều hướng thư mục SAMed ==="
if [ ! -d "/content/SAMed" ]; then
    echo "Thư mục /content/SAMed chưa tồn tại. Clone repo..."
    git clone https://github.com/nguyentranhuynhchi/SAMed.git /content/SAMed
else
    cd /content/SAMed
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
    echo "Thư mục /content/SAMed đã tồn tại (nhánh hiện tại: $CURRENT_BRANCH). Tiến hành cập nhật..."
    if [ "$CURRENT_BRANCH" != "HEAD" ]; then
        git pull origin "$CURRENT_BRANCH" || true
    fi
fi
cd /content/SAMed

echo "=== [4/7] Cài đặt Dependencies cho SAMed ==="
python3 -m pip install --upgrade pip -q
python3 -m pip install -r requirements.txt gdown -q

echo "=== [5/7] Chuẩn bị Pretrained SAM Backbone Checkpoint (sam_vit_b_01ec64.pth) ==="
mkdir -p /content/SAMed/checkpoints
CKPT_FILE="/content/SAMed/checkpoints/sam_vit_b_01ec64.pth"

if [ -s "$CKPT_FILE" ]; then
    echo "✅ Checkpoint đã có sẵn tại: $CKPT_FILE"
else
    # Tìm kiếm checkpoint nếu đã có sẵn trên Google Drive để copy trực tiếp
    for CANDIDATE in \
        "/content/drive/MyDrive/SAMed/checkpoints/sam_vit_b_01ec64.pth" \
        "/content/drive/MyDrive/sam_vit_b_01ec64.pth" \
        "/content/drive/MyDrive/NCKH/SAMed/checkpoints/sam_vit_b_01ec64.pth" \
        "/content/drive/MyDrive/NCKH_V2/SAMed/checkpoints/sam_vit_b_01ec64.pth" \
        "/content/drive/MyDrive/checkpoints/sam_vit_b_01ec64.pth"; do
        if [ -s "$CANDIDATE" ]; then
            echo "✅ Tìm thấy checkpoint trên Drive: $CANDIDATE. Đang sao chép..."
            cp "$CANDIDATE" "$CKPT_FILE"
            break
        fi
    done
fi

if [ ! -s "$CKPT_FILE" ]; then
    echo "Đang tải sam_vit_b_01ec64.pth từ server chính thức Meta..."
    wget -c -O "$CKPT_FILE" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth || {
        echo "Tải trực tiếp thất bại, dọn dẹp file dở dang và tải qua gdown từ Drive dự phòng..."
        rm -f "$CKPT_FILE"
        gdown --id "1j_n2fWbCjL6fW77oFmEw47J56P8nC_mP" -O "$CKPT_FILE" || true
    }
fi
if [ -s "$CKPT_FILE" ]; then
    echo "✅ Checkpoint sẵn sàng tại: $CKPT_FILE"
else
    echo "⚠️ CẢNH BÁO: Chưa tải được SAM backbone checkpoint. Kiểm tra lại kết nối mạng!"
fi

echo "=== [6/7] Tải Dataset MyoPS380 từ Google Drive ==="
DATASET_FOLDER_ID="13T6892xy-pUwC1BSjGl4sRxoypMu-SIS"
DOWNLOAD_DIR="/content/dataset_download"
TARGET_DATASET_DIR="/content/MyoPS380_dataset"

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$TARGET_DATASET_DIR"

# Kiểm tra nếu dataset đã có sẵn trên Google Drive để copy trực tiếp, tránh rate limit của gdown
for DRIVE_CANDIDATE in \
    "/content/drive/MyDrive/MyoPS380_dataset/Processed_data" \
    "/content/drive/MyDrive/Processed_data" \
    "/content/drive/MyDrive/NCKH_V2/MyoPS380_dataset/Processed_data" \
    "/content/drive/MyDrive/NCKH/MyoPS380_dataset/Processed_data"; do
    if [ -d "$DRIVE_CANDIDATE" ] && [ -z "$(ls -A "$TARGET_DATASET_DIR/Processed_data" 2>/dev/null)" ]; then
        echo "✅ Phát hiện dataset trên Google Drive: $DRIVE_CANDIDATE. Đang sao chép vào SSD cục bộ..."
        mkdir -p "$TARGET_DATASET_DIR/Processed_data"
        cp -r "$DRIVE_CANDIDATE/." "$TARGET_DATASET_DIR/Processed_data/"
        break
    fi
done

if [ -z "$(ls -A "$DOWNLOAD_DIR" 2>/dev/null)" ] && [ -z "$(ls -A "$TARGET_DATASET_DIR/Processed_data" 2>/dev/null)" ]; then
    echo "Đang tải toàn bộ thư mục dataset từ Google Drive (Folder ID: $DATASET_FOLDER_ID)..."
    gdown --folder --id "$DATASET_FOLDER_ID" -O "$DOWNLOAD_DIR" --remaining-ok || {
        echo "⚠️ Tải folder qua gdown gặp cảnh báo, tiếp tục kiểm tra các tệp đã tải..."
    }
fi

if [ ! -d "$TARGET_DATASET_DIR/Processed_data/bSSFP" ]; then
    for zipfile in $(find "$DOWNLOAD_DIR" -maxdepth 2 -name "*.zip" 2>/dev/null); do
        echo "Đang giải nén $zipfile vào $TARGET_DATASET_DIR ..."
        command -v unzip &>/dev/null || (apt-get update -qq && apt-get install -y unzip)
        unzip -q -o "$zipfile" -d "$TARGET_DATASET_DIR/"
    done

    if [ -d "$DOWNLOAD_DIR/Processed_data" ] && [ -z "$(ls -A "$TARGET_DATASET_DIR/Processed_data" 2>/dev/null)" ]; then
        mkdir -p "$TARGET_DATASET_DIR/Processed_data"
        cp -r "$DOWNLOAD_DIR/Processed_data/." "$TARGET_DATASET_DIR/Processed_data/"
    fi
fi

FOUND_BSSFP=$(find "$TARGET_DATASET_DIR" "$DOWNLOAD_DIR" -type d -name "bSSFP" 2>/dev/null | head -n 1)
if [ -n "$FOUND_BSSFP" ]; then
    LOCAL_DATASET=$(dirname "$FOUND_BSSFP")
    echo "✅ Đã xác định thư mục dataset chuẩn tại: $LOCAL_DATASET"
else
    LOCAL_DATASET="/content/MyoPS380_dataset/Processed_data"
    echo "⚠️ Chưa phát hiện thư mục bSSFP, đặt mặc định: $LOCAL_DATASET"
fi

export MYOPS380_DATA_ROOT="$LOCAL_DATASET"
if ! grep -q "MYOPS380_DATA_ROOT" /root/.bashrc 2>/dev/null; then
    echo "export MYOPS380_DATA_ROOT=\"$LOCAL_DATASET\"" >> /root/.bashrc
fi

echo "=== [7/7] Cài đặt tmux và Kiểm tra PyTorch ==="
command -v tmux &> /dev/null || (apt-get update -qq && apt-get install -y tmux)
python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo "=== HOÀN TẤT THIẾT LẬP MÔI TRƯỜNG CHO SAMED ==="
echo ""
echo "💡 HƯỚNG DẪN CHẠY ĐO HIỆU SUẤT TÍNH TOÁN (1 SEED ĐỘC LẬP 1234, KHÔNG GHI ĐÈ 5 SEED CŨ):"
echo "  1. Mở session tmux mới:"
echo "     tmux new -s samed_eff"
echo ""
echo "  2. Chạy huấn luyện và đánh giá nối tiếp (FP32, num_workers=0, pin_memory=True):"
echo "     cd /content/SAMed && export OMP_NUM_THREADS=2"
echo "     python3 train_myops.py --data_root $LOCAL_DATASET --ckpt checkpoints/sam_vit_b_01ec64.pth --seed 1234 --batch_size 16 --max_epochs 300 --amp none --num_workers 0 --pin_memory --measure_efficiency --output_dir /content/drive/MyDrive/MyoPS_Results/SAMed_efficiency/runs/samed_seed1234 && \\"
echo "     python3 evaluate_myops.py --checkpoint /content/drive/MyDrive/MyoPS_Results/SAMed_efficiency/runs/samed_seed1234/checkpoints/samed_seed1234.pt --data_root $LOCAL_DATASET --output_dir /content/drive/MyDrive/MyoPS_Results/SAMed_efficiency/runs/samed_seed1234/evaluation --amp none"
echo ""
echo "  3. Tách khỏi tmux (detach): Nhấn Ctrl+B rồi thả ra, sau đó nhấn D"
echo "  4. Xem lại tiến trình: tmux attach -t samed_eff"
echo ""
