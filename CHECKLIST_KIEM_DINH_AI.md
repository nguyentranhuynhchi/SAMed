# BỘ CÂU HỎI KIỂM ĐỊNH ĐỐI SÁNH CHUYÊN SÂU DÀNH CHO AI (AUDIT BENCHMARK QUESTIONNAIRE)
**Đề tài**: Kiểm toán tính bảo toàn và mức độ tuân thủ khi chuẩn hóa baseline **SAMed** sang giao thức **SCAR Benchmark** trên 2 tập dữ liệu (**MyoPS380** & **MyoPS++**).
**Tài liệu đối chiếu bắt buộc**:
1. `Promt_train_test_no_val.txt` (Quy định chuẩn hóa baseline đối chứng với SCAR).
2. `Quy tắc đo hiệu suất tính toán.txt` (Quy chuẩn 7 chỉ số hiệu năng trên NVIDIA A100).
3. `Promt_train_test_no_val_for_Samed.txt` (Đặc thù kỹ thuật kiến trúc SAMed).
4. Mã nguồn repo tham chiếu: `D:\NCKH\ChanhThucNhatCoThe\SCAR`.
5. Mã nguồn repo kiểm toán: `D:\NCKH\ChanhThucNhatCoThe\SAMed` (Branch: `main`).

---

### HƯỚNG DẪN DÀNH CHO AI ĐÁNH GIÁ (SYSTEM PROMPT FOR EVALUATOR AI):
> Bạn là chuyên gia kiểm toán độc lập về AI y tế. Hãy đọc toàn bộ mã nguồn repo `SAMed` (chú trọng `datasets/dataset_myops.py`, `utils/metrics.py`, `utils/efficiency.py`, `utils/loss.py`, `train_myops.py`, `evaluate_myops.py`, `predict.py`, `evaluate_predictions.py`, thư mục `lists/` và `tests/`), đối chiếu từng dòng code với 2 tài liệu prompt gốc và trả lời lần lượt từng câu hỏi bên dưới. 
> Mỗi câu hỏi yêu cầu:
> - **Phán quyết**: [ĐẠT / KHÔNG ĐẠT / ĐẠT MỘT PHẦN]
> - **Bằng chứng mã nguồn**: Chỉ rõ tên tệp và số dòng (file:line) chứng minh.
> - **Điểm số**: Ghi rõ điểm đạt được theo barem.

---

## PHẦN 1: NGUYÊN TẮC BẢO TOÀN KIẾN TRÚC & OBJECTIVE GỐC CỦA SAMed (25 ĐIỂM)
*(Mục tiêu: Đảm bảo việc chuẩn hóa không làm biến chất mô hình, không biến SAMed thành bản sao của SCAR)*

* **Câu 1.1 (8đ) - Bảo toàn Objective Loss gốc vs Không bị ép sang Loss của SCAR**:
  * *Tiêu chí đối chiếu*: SCAR sử dụng loss `0.2 CE + 0.8 BatchDice` (weights `[0.5, 1, 1, 1]`). `Promt_train_test_no_val.txt` (Mục 5) quy định: *"Giữ loss, class weights, objective gốc của baseline... Không gọi 0.5 CE + 0.5 Dice là loss SCAR; không đổi loss baseline sang SCAR"*.
  * *Câu hỏi kiểm toán*: Trong `utils/loss.py` và `train_myops.py`, hàm loss của SAMed có bảo toàn nguyên vẹn công thức gốc của baseline tim mạch là `0.5 * CrossEntropy + 0.5 * DiceLoss` (`dice_param = 0.5`) không? Có bằng chứng nào cho thấy SAMed bị ép sang công thức 0.2/0.8 hay weights của SCAR không?

* **Câu 1.2 (10đ) - Bảo toàn Kiến trúc Foundation Model & Cơ chế LoRA PEFT**:
  * *Tiêu chí đối chiếu*: SAMed là Segment Anything Model thích ứng qua LoRA. SCAR là kiến trúc Multi-stream Encoder đa nhánh với Spatial SS-PANet.
  * *Câu hỏi kiểm toán*: Mã nguồn có bảo toàn nguyên vẹn backbone Vision Transformer (`ViT-B/16`) với 100% tham số backbone bị đóng băng (`requires_grad = False`, ~86.52M tham số) không? Cơ chế LoRA (rank $r=4$) có được chèn chính xác vào các ma trận chiếu Query ($W_q$) và Value ($W_v$) tại 12 khối Attention không? Tổng tham số tối ưu có đúng `4,212,016` (~4.21M) và tổng tham số mô hình có đúng `90,729,008` (~90.73M) không?

* **Câu 1.3 (7đ) - Giao thức Promptless Tự động vs Bảo toàn Đầu vào Early-Fusion**:
  * *Tiêu chí đối chiếu*: SAM gốc cần điểm/box tương tác. `Promt_train_test_no_val_for_Samed.txt` quy định SAMed chạy hoàn toàn tự động (không tương tác người dùng, không lấy GT test làm prompt).
  * *Câu hỏi kiểm toán*: Lớp prompt encoder có sử dụng `no_mask_embed` mặc định của SAM để thực hiện inference tự động hoàn toàn không? Input đầu vào có giữ nguyên cơ chế Early-fusion ghép kênh `[bSSFP, LGE, T2w]` thành tensor shape `(B, 3, 128, 128)` đi qua 1 convolution chuyển kênh thích ứng `3 -> 3` trước khi vào ViT không?

---

## PHẦN 2: BẢO TOÀN DỮ LIỆU, CHUẨN NHÃN & AUGMENTATION (25 ĐIỂM)
*(Mục tiêu: Đảm bảo dữ liệu và không gian nhãn được xử lý nhất quán, phòng ngừa rủi ro sai lệch giải phẫu)*

* **Câu 2.1 (7đ) - Bảo toàn Tính toàn vẹn Dữ liệu Cache & Không chuẩn hóa thừa**:
  * *Tiêu chí đối chiếu*: `Promt_train_test_no_val.txt` (Mục 1) cấm chia lại 255 nếu cache đã ở `[0, 1]`, cấm z-score, percentile normalization hay histogram matching.
  * *Câu hỏi kiểm toán*: `datasets/dataset_myops.py` có nạp trực tiếp mảng ảnh từ cache đĩa mà không áp đặt thêm phép chia 255 hay z-score không? Dữ liệu đầu ra sau augment có giữ nguyên tính chất của pipeline SCAR không?

* **Câu 2.2 (10đ) - Cơ chế Adapter Nhãn & Phòng Chống Lỗi Double-Swap (MyoPS380 vs MyoPS++)**:
  * *Tiêu chí đối chiếu*: 
    * MyoPS380 trên đĩa là Legacy (2=Scar, 3=Edema) $\to$ Bắt buộc chuyển đổi Canonical trên RAM qua LUT `[0, 1, 3, 2]` đúng 1 lần duy nhất.
    * MyoPS++ trên đĩa **đã là Canonical** (2=Edema, 3=Scar) $\to$ Bắt buộc dùng Identity mapping (tuyệt đối cấm áp dụng LUT swap lần 2).
  * *Câu hỏi kiểm toán*: Kiểm tra hàm `canonicalize_label()` và lớp `MyopsDataset`:
    1. Khi `--dataset myops380`: Code có áp dụng đúng LUT `[0, 1, 3, 2]` để đưa về Canonical không?
    2. Khi `--dataset myopspp`: Code có tự động nhận diện và giữ nguyên nhãn gốc (Identity mapping, không hoán vị 2 và 3) không?
    3. Unit test `test_myopspp_label_identity_preservation` có chứng minh nhãn MyoPS++ không bị đảo ngược không?

* **Câu 2.3 (8đ) - Tái tạo Chuẩn xác 100% Thuật toán Augmentation từ SCAR (`RandomGenerator`)**:
  * *Tiêu chí đối chiếu*: `Promt_train_test_no_val.txt` (Mục 3) yêu cầu tái tạo chính xác chuỗi biến đổi hình học và cường độ của `training/dataset/myops_dataset.py` (SCAR / I-MMSeg).
  * *Câu hỏi kiểm toán*: Đối chiếu `RandomGenerator` trong `datasets/dataset_myops.py` với SCAR:
    1. Chuỗi `if/elif/elif`: Nhánh 1 xoay $k \times 90^\circ$ + lật (50%), Nhánh 2 xoay tự do $\pm20^\circ$ (25%), Nhánh 3 Gaussian noise $\sigma=0.2$ (12.5%), Nhánh không biến đổi (12.5%) có đúng thứ tự không?
    2. Xác suất 50% shift roll $\pm5$ pixel vòng qua biên?
    3. Xác suất 50% brightness factor $[0.8, 1.2]$?
    4. Xác suất 50% gamma $(x / 255.0)^\gamma \times 255.0$?
    5. Resize/zoom: Order 3 (cubic) cho ảnh, Order 0 (nearest) cho mask?

---

## PHẦN 3: CÔNG THỨC TRAIN, OPTIMIZER & CHÍNH SÁCH NO-VAL CHUẨN HÓA (20 ĐIỂM)
*(Mục tiêu: Đảm bảo ngân sách cập nhật và chính sách huấn luyện đồng nhất tuyệt đối với SCAR benchmark)*

* **Câu 3.1 (7đ) - Ngân sách Huấn luyện Động (Dynamic Updates Budget)**:
  * *Tiêu chí đối chiếu*: `Promt_train_test_no_val.txt` (Mục 4) quy định đúng 300 epochs, batch size 16, `accum_steps=1`, `drop_last=False`.
  * *Câu hỏi kiểm toán*: 
    1. Với MyoPS380 (1.972 lát cắt $\to$ 124 batches/epoch): Tổng updates có đạt đúng **37.200 updates** không?
    2. Với MyoPS++ (~304 lát cắt $\to$ 19 batches/epoch): Tổng updates có tự động điều chỉnh đạt đúng **5.700 updates** không?
    3. Code có bị hardcode 37.200 cho MyoPS++ không?

* **Câu 3.2 (7đ) - Optimizer AdamW & Polynomial LR Scheduler chuẩn SCAR**:
  * *Tiêu chí đối chiếu*: AdamW toàn bộ tham số trainable (`base_lr=0.0003333`, `weight_decay=0.0001`, `betas=(0.9, 0.999)`, `eps=1e-8`, `clip_grad=0.0`). Scheduler polynomial decay mũ 0.9 zero-based offset: $lr_u = base\_lr \times \max(0, 1 - \max(0, u - 1) / N)^{0.9}$.
  * *Câu hỏi kiểm toán*: Trong `utils/optimization.py` và `train_myops.py`:
    1. Cấu hình AdamW có đúng các siêu tham số trên không? Có loại trừ bias/norm khỏi decay không?
    2. Scheduler có được cập nhật **sau** mỗi optimizer step thành công và có offset zero-based $(u-1)$ không?
    3. Hai update đầu tiên có sử dụng đúng `base_lr` không?

* **Câu 3.3 (6đ) - Tuân thủ Nghiêm ngặt Chính sách No-Validation**:
  * *Tiêu chí đối chiếu*: Cấm tuyệt đối việc tạo val loader, cấm đánh giá validation mỗi epoch, cấm early stopping, cấm sinh file `best.pt`.
  * *Câu hỏi kiểm toán*: Trong vòng lặp huấn luyện của `train_myops.py`, có tồn tại bất kỳ validation step nào không? Checkpoint lưu sau mỗi epoch có phải là atomic save đè file duy nhất `samed_seed<seed>.pt` không? Checkpoint cuối tại epoch 300 có đánh dấu `training_complete=True` không?

---

## PHẦN 4: SHARED EVALUATOR & QUY CÁCH ĐÁNH GIÁ ĐỐI SÁNH (15 ĐIỂM)
*(Mục tiêu: Đảm bảo độ tin cậy của các chỉ số Dice, HD95, ASD theo đúng chuẩn SCAR)*

* **Câu 4.1 (5đ) - Bảo toàn Inference Tiêu chuẩn (Không Postprocessing gian lận)**:
  * *Tiêu chí đối chiếu*: `Promt_train_test_no_val.txt` (Mục 7) cấm TTA, ensemble, connected-component filtering (LCC), ngưỡng loại voxel nhỏ hay morphology postprocessing.
  * *Câu hỏi kiểm toán*: Trong `evaluate_myops.py`, `predict.py` và `evaluate_predictions.py`, quá trình inference có chạy thuần túy `model.eval()`, `torch.no_grad()`, single pass không? Có bước lọc thành phần liên thông hay can thiệp hậu xử lý nào không?

* **Câu 4.2 (5đ) - Thước đo Khoảng cách Mặt ngoài (Voxel vs Physical mm Spacing)**:
  * *Tiêu chí đối chiếu*: MyoPS380 dùng đơn vị `voxel` (protocol `myops380_voxel_v3`). MyoPS++ bắt buộc dùng khoảng cách milimet vật lý `mm` với spacing động $[1.0, 1.0, s_z]$ (protocol `myopspp_unet2d_roi128_1mm_predicted_roi_mm_v1`).
  * *Câu hỏi kiểm toán*: `utils/metrics.py` và `datasets/dataset_myops.py` có xử lý phân tách rạch ròi 2 giao thức này không? Khi đánh giá MyoPS++, code có tự động trích xuất $s_z$ từ file H5 và khóa in-plane spacing là 1.0 mm không?

* **Câu 4.3 (5đ) - Báo cáo Lâm sàng: Minh bạch Edema-Exclusive vs Edema-Inclusive**:
  * *Tiêu chí đối chiếu*: Nhãn ID 2 là Edema-exclusive, nhãn ID 3 là Scar, vùng $\{2, 3\}$ là Edema-inclusive (LGE-abnormal).
  * *Câu hỏi kiểm toán*: Bảng kết quả in ra màn hình và file `metrics.json` có hiển thị đồng thời cả:
    1. `Avg. Pathology (Inclusive)` = Mean(Scar, Edema-Inclusive)?
    2. `Avg. Pathology (Exclusive)` = Mean(Scar, Edema-Exclusive)?
    3. Empty mask policy có tuân thủ đúng quy tắc SCAR (cả hai rỗng $\to$ undefined; một rỗng $\to$ Dice=0, khoảng cách undefined) không?

---

## PHẦN 5: ĐO LƯỜNG HIỆU SUẤT TÍNH TOÁN 7 CHỈ SỐ (15 ĐIỂM)
*(Mục tiêu: Tuân thủ 100% tài liệu 'Quy tắc đo hiệu suất tính toán.txt')*

* **Câu 5.1 (5đ) - Module Kiểm soát Điều kiện Phần cứng & Quy định Benchmark**:
  * *Tiêu chí đối chiếu*: Kiểm tra đúng 01 GPU NVIDIA A100, FP32 (`--amp none`), `num_workers=0`, `pin_memory=True`, `batch_size=16`, `max_epochs=300`.
  * *Câu hỏi kiểm toán*: Hàm `check_rule_compliance()` trong `utils/efficiency.py` có bắt buộc kiểm tra đủ các điều kiện trên và đưa ra cảnh báo nếu vi phạm không? Cờ `--measure_efficiency` có hỗ trợ trên cả 4 script CLI (`train_myops.py`, `evaluate_myops.py`, `predict.py`, `evaluate_predictions.py`) không?

* **Câu 5.2 (10đ) - Phương pháp Đo 7 Chỉ số Chuẩn mực**:
  * *Tiêu chí đối chiếu*: 
    1. Params: Trainable & Total qua `p.requires_grad`.
    2. FLOPs: Đo trên input `(1, 3, 128, 128)` FP32 `eval()` bằng `FlopCounterMode`.
    3. Latency: Batch=1, 30 warmup + 100 repeats kèm `cuda.synchronize()`.
    4. Inference Peak VRAM: `max_memory_allocated()`, quy đổi nhị phân MiB/GiB.
    5. Training Time/Epoch: Bỏ qua 5 epoch đầu, trung bình epoch 6..300.
    6. Training Total Time: Tổng thời gian 300 epochs chia 3600.
    7. Training Peak VRAM: Max allocated VRAM trong 300 epochs.
  * *Câu hỏi kiểm toán*: Module `utils/efficiency.py` có hiện thực chuẩn xác từng phương pháp đo trên không? Có đúng 5 epoch đầu bị loại bỏ khi tính thời gian epoch không? Quy đổi đơn vị có dùng lũy thừa nhị phân ($1024^2$ cho MB, $1024^3$ cho GB) không?

---

## BẢNG TỔNG HỢP CHẤM ĐIỂM DÀNH CHO AI (SCORECARD):
| Phần | Tiêu chí Kiểm toán | Điểm tối đa | Điểm đạt được | Đánh giá tóm tắt |
| :---: | :--- | :---: | :---: | :--- |
| **Phần 1** | Bảo toàn Kiến trúc & Objective gốc SAMed | 25 | ... / 25 | |
| **Phần 2** | Bảo toàn Dữ liệu, Nhãn & Augmentation | 25 | ... / 25 | |
| **Phần 3** | Công thức Train, Optimizer & No-Val | 20 | ... / 20 | |
| **Phần 4** | Shared Evaluator & Báo cáo Lâm sàng | 15 | ... / 15 | |
| **Phần 5** | Đo lường Hiệu suất 7 Chỉ số A100 | 15 | ... / 15 | |
| **TỔNG** | **ĐÁNH GIÁ TOÀN DIỆN CHUẨN HÓA BASELINE** | **100** | **... / 100** | |
