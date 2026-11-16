# BỘ TIÊU CHÍ & CÂU HỎI KIỂM ĐỊNH ĐỘC LẬP CHO AI (AUDIT VERIFICATION QUESTIONNAIRE)
**Dự án**: SAMed Baseline Standardization & Multi-Dataset Parity (MyoPS380 & MyoPS++)  
**Mục tiêu**: Cung cấp bộ câu hỏi trắc nghiệm/tự luận kỹ thuật chi tiết để AI khác (hoặc giám khảo độc lập) đọc toàn bộ mã nguồn repo `SAMed` và chấm điểm mức độ tuân thủ quy chuẩn nghiên cứu.

---

## TỔNG QUAN THANG ĐIỂM (TỔNG CỘNG: 100 ĐIỂM)
- **Phần 1: Kiến trúc mô hình, Objective & Protocol MyoPS380 (Prompt 1)**: 30 điểm
- **Phần 2: Mở rộng Multi-Dataset Parity MyoPS380 & MyoPS++**: 35 điểm
- **Phần 3: Đo lường Hiệu suất Tính toán 7 chỉ số (Quy tắc đo hiệu suất tính toán)**: 20 điểm
- **Phần 4: Độ tin cậy Kỹ thuật, CLI Consistency, Unit Test & Edge Cases**: 15 điểm

---

## PHẦN 1: KIẾN TRÚC MÔ HÌNH, OBJECTIVE & PROTOCOL MYOPS380 (30 ĐIỂM)

### 1.1. Kiến trúc Cốt lõi & Cơ chế LoRA PEFT (10 điểm)
- **Q1.1.1**: Mô hình có giữ nguyên backbone Vision Transformer (ViT-B/16) với 100% tham số backbone bị đóng băng (`requires_grad = False`, ~86.52M tham số) không?
  - *Kiểm tra*: `sam_lora_image_encoder.py`, `utils/efficiency.py` (`count_parameters`).
  - *Tiêu chuẩn*: Trainable params = 4,212,016 (~4.21M), Total params = 90,729,008 (~90.73M).
- **Q1.1.2**: LoRA rank $r=4$ có được chèn chính xác vào ma trận Query ($W_q$) và Value ($W_v$) của tất cả 12 khối Multi-Head Attention không?
  - *Kiểm tra*: `sam_lora_image_encoder.py` (class `_LoRA_qkv`).
  - *Tiêu chuẩn*: Ma trận A khởi tạo Kaiming Uniform, ma trận B khởi tạo bằng 0 (`zeros_`).
- **Q1.1.3**: Dữ liệu đầu vào CMR có được gộp theo cơ chế Early-fusion [bSSFP, LGE, T2w] tạo tensor shape `(B, 3, 128, 128)` không?
  - *Kiểm tra*: `datasets/dataset_myops.py` (hàm `__getitem__`), `train_myops.py`, `evaluate_myops.py`.
  - *Tiêu chuẩn*: Tuyệt đối không dùng 3 encoder tách rời, không nhân bản kênh hay zero-fill.

### 1.2. Objective Loss & Optimization Recipe (10 điểm)
- **Q1.2.1**: Hàm loss có bảo toàn nguyên vẹn objective đặc thù của SAMed: $0.5 \times \text{CE} + 0.5 \times \text{Dice}$ không?
  - *Kiểm tra*: `train_myops.py` (hàm `calc_loss`), `utils/loss.py`.
  - *Tiêu chuẩn*: Không tự ý thay đổi sang hàm loss của SCAR (0.2 CE + 0.8 BatchDice).
- **Q1.2.2**: Optimizer có được cấu hình bằng AdamW với `base_lr = 0.0003333`, `weight_decay = 0.0001`, `betas = (0.9, 0.999)`, `eps = 1e-8` và tắt gradient clipping (`clip_grad = 0.0`) không?
  - *Kiểm tra*: `utils/optimization.py` (`build_optimizer`), `train_myops.py`.
- **Q1.2.3**: Learning rate scheduler có tuân thủ đúng công thức Polynomial decay lũy thừa $0.9$ có bù trừ zero-based offset:
  $$\text{lr}_u = \text{base\_lr} \times \max\left(0, 1 - \frac{\max(0, u - 1)}{U_{\text{total}}}\right)^{0.9}$$
  không?
  - *Kiểm tra*: `utils/optimization.py` (`PolynomialLRWithOffset`, `build_scheduler`).
  - *Tiêu chuẩn*: Hai update đầu giữ nguyên `base_lr`, update cuối cùng $u=U_{\text{total}}$ có $\text{lr} > 0$.

### 1.3. Ngân sách Huấn luyện & Checkpoint Atomic (10 điểm)
- **Q1.3.1**: Protocol MyoPS380 có khóa đúng ngân sách 300 epochs, batch size 16, 124 batches/epoch, tổng cộng đúng 37,200 updates không?
  - *Kiểm tra*: `train_myops.py` (biến `NO_VAL_PROTOCOLS['myops380']`, `total_planned_updates`).
- **Q1.3.2**: Quá trình huấn luyện có tuân thủ nguyên tắc STRICTLY NO VALIDATION (không val loader, không val giữa epoch, không early stopping) không?
  - *Kiểm tra*: `train_myops.py` (vòng lặp huấn luyện thuần túy qua các epoch).
- **Q1.3.3**: Checkpoint có được lưu theo cơ chế atomic write (ghi file tạm `.tmp` rồi rename đè an toàn) với tên chuẩn `samed_seed<seed>.pt` không?
  - *Kiểm tra*: `train_myops.py` (hàm `atomic_save`).
- **Q1.3.4**: Checkpoint có lưu trữ đầy đủ trạng thái model, optimizer, scheduler, scaler, RNG states (Python, NumPy, PyTorch CPU & CUDA) và hỗ trợ cờ `--resume` an toàn không?
  - *Kiểm tra*: `train_myops.py` (hàm `_run_training`, khối `resume_ckpt`).

---

## PHẦN 2: MỞ RỘNG MULTI-DATASET PARITY (MYOPS380 & MYOPS++) (35 ĐIỂM)

### 2.1. Quản lý Manifest & Phân chia Dữ liệu (10 điểm)
- **Q2.1.1**: Repo có chứa đầy đủ cả 2 bộ manifest chuẩn trong thư mục `lists/`:
  - `lists/myops380_no_val/` (hoặc `lists/lists_MyoPS380/`): `train.txt` (1,972 dòng / 304 bệnh nhân), `test_vol.txt` (76 dòng / 76 bệnh nhân).
  - `lists/lists_MyoPSpp/` (hoặc `lists/myopspp_no_val/`): `train.txt` (296 dòng / 64 bệnh nhân), `test_vol.txt` (16 dòng / 16 bệnh nhân).
  - *Kiểm tra*: Kiểm tra các tệp tin trong `lists/`.
- **Q2.1.2**: Các ca bệnh nhân trong tập train và test có đảm bảo không bị rò rỉ dữ liệu (Zero Data Leakage: $\text{Train} \cap \text{Test} = \emptyset$) trên cả 2 dataset không?
  - *Kiểm tra*: `tests/test_multi_dataset.py` (`test_myops380_split_manifests`, `test_myopspp_split_manifests`).
- **Q2.1.3**: Tên ca bệnh nhân dạng chữ hoa chữ thường (case casing) của MyoPS++ (`Case2011`, `Case3003`...) có được bảo toàn tuyệt đối, không bị ép về chữ thường không?
  - *Kiểm tra*: `datasets/dataset_myops.py` (`patient_id`).

### 2.2. Cơ chế Chống Double-Swap Nhãn (Double-Swap Prevention) (10 điểm)
- **Q2.2.1**: Dữ liệu MyoPS380 lưu trên đĩa ở chuẩn Legacy (2: Scar, 3: Edema), mô hình học theo Canonical (2: Edema, 3: Scar) -> Dataset Adapter có áp dụng LUT `[0, 1, 3, 2]` để chuyển đổi nhãn không?
  - *Kiểm tra*: `datasets/dataset_myops.py` (`canonicalize_label`, `label_order="legacy"`).
- **Q2.2.2**: Dữ liệu MyoPS++ lưu trên đĩa ĐÃ Ở CHUẨN CANONICAL (2: Edema, 3: Scar) -> Dataset Adapter có kích hoạt Identity Mapping (TUYỆT ĐỐI KHÔNG áp dụng LUT swap lần nữa) để tránh lỗi đảo ngược bệnh lý không?
  - *Kiểm tra*: `datasets/dataset_myops.py` (`label_order="canonical"` khi `dataset="myopspp"`).
- **Q2.2.3**: Khi đánh giá (`evaluate_myops.py`, `evaluate_predictions.py`), nhãn ground truth của MyoPS++ có được giữ nguyên chuẩn Canonical, đảm bảo mô hình dự đoán đúng thì đạt Dice = 1.0 không?
  - *Kiểm tra*: `tests/test_multi_dataset.py` (`test_double_swap_prevention_in_evaluation`).

### 2.3. Ngân sách Huấn luyện Động (Dynamic Optimizer Budget) (5 điểm)
- **Q2.3.1**: Ngân sách huấn luyện có tự động tính toán chính xác theo từng dataset không:
  - MyoPS380: 1,972 lát cắt / 16 = 124 batches/epoch $\times$ 300 epochs = đúng 37,200 updates.
  - MyoPS++: 296 lát cắt / 16 = 19 batches/epoch $\times$ 300 epochs = đúng 5,700 updates.
  - *Kiểm tra*: `train_myops.py` (`total_planned_updates = args.max_epochs * updates_per_epoch`).

### 2.4. Trích xuất Spacing & Khoảng cách Bề mặt Vật lý (10 điểm)
- **Q2.4.1**: Đối với MyoPS380, thước đo khoảng cách có tuân thủ đúng đơn vị voxel (`spacing=None`, protocol `myops380_voxel_v3`) không?
  - *Kiểm tra*: `utils/metrics.py` (`BENCHMARK_PROTOCOLS['myops380']`).
- **Q2.4.2**: Đối với MyoPS++, thước đo khoảng cách có tự động trích xuất thông tin khoảng cách vật lý động `[1.0, 1.0, s_z]` từ tệp H5 và tính khoảng cách theo đơn vị milimet (`mm`, protocol `myopspp_unet2d_roi128_1mm_predicted_roi_mm_v1`) không?
  - *Kiểm tra*: `datasets/dataset_myops.py` (`__getitem__`), `utils/metrics.py` (`dataset_rows`, `summarize_rows`).
- **Q2.4.3**: Báo cáo tổng kết (`metrics.json` và bảng hiển thị terminal) có xuất đúng `mean_hd95_mm` và `mean_asd_mm` cho MyoPS++, và `mean_hd95_voxel` cho MyoPS380 không?
  - *Kiểm tra*: `evaluate_myops.py`, `evaluate_predictions.py`.
- **Q2.4.4**: Báo cáo lâm sàng có trình bày chỉ số Phù nề (Edema) theo vùng bao hàm `edema_inclusive` (LGE-abnormal / AAR, nhãn {2, 3}) kết hợp cùng Edema-exclusive và Scar không?
  - *Kiểm tra*: `evaluate_myops.py` (Bảng Master Benchmark Clinical Table).

---

## PHẦN 3: ĐO LƯỜNG HIỆU SUẤT TÍNH TOÁN 7 CHỈ SỐ (20 ĐIỂM)

### 3.1. Tuân thủ Điều kiện Phần cứng & Môi trường (5 điểm)
- **Q3.1.1**: Repo có module `check_rule_compliance` kiểm tra nghiêm ngặt 7 điều kiện thực thi: 1 GPU NVIDIA A100, độ chính xác thuần FP32 (`--amp none`), `num_workers = 0`, `pin_memory = True`, `batch_size = 16`, `max_epochs = 300`, `accum_steps = 1` không?
  - *Kiểm tra*: `utils/efficiency.py` (`check_rule_compliance`).
- **Q3.1.2**: Các cờ lệnh `--measure_efficiency`, `--no_measure_efficiency`, `--efficiency_only` có hoạt động độc lập và không làm gián đoạn luồng huấn luyện thông thường không?
  - *Kiểm tra*: `train_myops.py` (`build_parser`).

### 3.2. Đo đạc Chính xác 7 Chỉ số (15 điểm)
- **Q3.2.1 (Complexity)**: 
  - Trainable params có được đếm chính xác bằng tổng `requires_grad=True` (~4.21M) và Total params (~90.73M) không?
  - FLOPs có được đo bằng `torch.utils.flop_counter.FlopCounterMode` trên dummy tensor `(1, 3, 128, 128)` FP32 ở chế độ `model.eval()` không?
  - *Kiểm tra*: `utils/efficiency.py` (`count_parameters`, `measure_flops`).
- **Q3.2.2 (Inference)**:
  - Latency có được đo ở Batch=1, FP32, eval mode, no_grad với đúng 30 lần warmup và 100 lần lặp benchmark kèm `torch.cuda.synchronize()` không?
  - Inference Peak VRAM có được đo bằng `torch.cuda.max_memory_allocated()` sau khi gọi `empty_cache()` và `reset_peak_memory_stats()` không?
  - *Kiểm tra*: `utils/efficiency.py` (`measure_inference_efficiency`).
- **Q3.2.3 (Training)**:
  - Time / Epoch (s) có được tính bằng trung bình từ **epoch 6 đến epoch 300** (bắt buộc bỏ qua 5 epoch đầu để loại trừ nhiễu khởi tạo) không?
  - Total Time (hours) có phản ánh đúng tổng thời gian thực tế hoàn thành 300 epochs không?
  - Training Peak VRAM có ghi nhận giá trị đỉnh cao nhất của `torch.cuda.max_memory_allocated()` trong suốt 300 epochs không?
  - *Kiểm tra*: `utils/efficiency.py` (`summarize_training_efficiency`).
- **Q3.2.4 (Báo cáo & Đơn vị)**:
  - Tất cả các trường Peak VRAM có được báo cáo song song cả 2 đơn vị **MB** (`/ 1024**2`) và **GB** (`/ 1024**3`) với định dạng `XXXX.XX MB (X.XX GB)` không?
  - File `efficiency_metrics.json` có được xuất tự động tại đường dẫn chỉ định không?
  - *Kiểm tra*: `utils/efficiency.py` (`format_efficiency_report`, `build_efficiency_payload`).

---

## PHẦN 4: ĐỘ TIN CẬY KỸ THUẬT, CLI CONSISTENCY & UNIT TEST (15 ĐIỂM)

### 4.1. Tính Nhất quán của Giao diện CLI (5 điểm)
- **Q4.1.1**: Các script chính gồm `train_myops.py`, `evaluate_myops.py`, `predict.py`, `evaluate_predictions.py` có cung cấp bộ cờ lệnh đồng nhất:
  - `--dataset {myops380, myopspp}`
  - `--list_dir` / `--split_dir`
  - `--data_root`
  - `--checkpoint`
  - `--output_dir`
  không?
  - *Kiểm tra*: Kiểm tra `build_parser()` trong từng file.
- **Q4.1.2**: Khi không truyền `--list_dir`, script có tự động nhận diện thư mục manifest mặc định phù hợp với từng dataset không?
  - *Kiểm tra*: `datasets/dataset_myops.py` (`resolve_default_list_dir`).

### 4.2. Bộ Kiểm thử Đơn vị & Tích hợp (Unit & Integration Tests) (10 điểm)
- **Q4.2.1**: Repo có file kiểm thử `tests/test_multi_dataset.py` bao phủ toàn diện các ca kiểm thử:
  - Kiểm tra manifest, số lượng lát cắt, số bệnh nhân và tính độc lập của tập test.
  - Kiểm tra hàm `canonicalize_label` với mode `legacy` (đổi nhãn) và mode `canonical`/`identity` (không đổi nhãn).
  - Kiểm tra tính toán khoảng cách voxel vs khoảng cách milimet có nhân với spacing.
  - Kiểm tra tính toán tổng updates 37,200 (MyoPS380) vs 5,700 (MyoPS++).
  - Kiểm tra giao diện dòng lệnh CLI flags.
  - Chạy smoke test end-to-end huấn luyện 1 epoch và đánh giá volume trên dữ liệu giả lập.
  - *Kiểm tra*: Chạy lệnh `python -m unittest tests/test_multi_dataset.py`.
- **Q4.2.2**: Repo có đảm bảo 100% các bài test trong cả `tests/test_efficiency.py` và `tests/test_multi_dataset.py` vượt qua (Exit code 0, 0 failures, 0 errors) không?
  - *Kiểm tra*: Chạy lệnh `python -m unittest discover tests`.

---

## BẢNG TỔNG KẾT ĐIỂM DÀNH CHO AI CHẤM ĐỘC LẬP

| Hạng mục kiểm định | Điểm tối đa | Điểm đạt được | Ghi chú & Dẫn chứng mã nguồn |
| :--- | :---: | :---: | :--- |
| **Phần 1: Kiến trúc mô hình & Protocol MyoPS380** | 30 | [ /30] | |
| 1.1. Kiến trúc cốt lõi & LoRA PEFT (4.21M / 90.73M) | 10 | [ /10] | |
| 1.2. Objective Loss (0.5 CE + 0.5 Dice) & Optimizer AdamW Poly | 10 | [ /10] | |
| 1.3. Ngân sách 300 epochs, no-val, atomic checkpoints, resume | 10 | [ /10] | |
| **Phần 2: Mở rộng Multi-Dataset Parity (MyoPS380 & MyoPS++)** | 35 | [ /35] | |
| 2.1. Manifests MyoPS380 & MyoPS++, Zero Leakage, Case Casing | 10 | [ /10] | |
| 2.2. Chống lỗi Double-Swap (Identity trên MyoPS++, LUT trên MyoPS380) | 10 | [ /10] | |
| 2.3. Ngân sách updates động (37,200 updates vs 5,700 updates) | 5 | [ /5] | |
| 2.4. Spacing vật lý `[1.0, 1.0, s_z]`, mm vs voxel, Edema inclusive | 10 | [ /10] | |
| **Phần 3: Đo lường Hiệu suất Tính toán 7 Chỉ số Chuẩn hóa** | 20 | [ /20] | |
| 3.1. Tuân thủ quy chuẩn phần cứng A100 FP32 num_workers=0 | 5 | [ /5] | |
| 3.2. Đo đạc chính xác 7 chỉ số, bỏ qua 5 epoch đầu, đơn vị MB & GB | 15 | [ /15] | |
| **Phần 4: Độ tin cậy Kỹ thuật, CLI Consistency & Tests** | 15 | [ /15] | |
| 4.1. Tính nhất quán CLI trên 4 scripts chính | 5 | [ /5] | |
| 4.2. Bộ test suite tự động (100% passing tests) | 10 | [ /10] | |
| **TỔNG ĐIỂM** | **100** | **[ /100]** | |
