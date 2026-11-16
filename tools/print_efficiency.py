"""Print efficiency table and seed 1234 metrics cleanly without any shell escape errors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def print_metrics(run_dir: str | Path | None = None) -> None:
    cand_dirs = [
        Path(run_dir) if run_dir else None,
        Path("/content/drive/MyDrive/MyoPS_Results/SAMed/runs/myopspp_seed1234"),
        Path("/content/SAMed/runs/myopspp_seed1234"),
        Path("runs/myopspp_seed1234"),
    ]
    valid_dir = None
    for c in cand_dirs:
        if c and (c / "efficiency_metrics.json").exists():
            valid_dir = c
            break

    if not valid_dir:
        # Check without efficiency metrics just in case
        for c in cand_dirs:
            if c and (c / "evaluation" / "metrics.json").exists():
                valid_dir = c
                break

    if not valid_dir:
        print("❌ Chưa tìm thấy thư mục chạy của Seed 1234. Kiểm tra lại đường dẫn.")
        return

    eff_file = valid_dir / "efficiency_metrics.json"
    metric_file = valid_dir / "evaluation" / "metrics.json"

    # 1. Print Bảng 1: Hiệu năng tính toán
    if eff_file.exists():
        with open(eff_file, "r", encoding="utf-8") as f:
            eff_data = json.load(f)
        ex = eff_data.get("excel_summary", {})
        print("\n" + "=" * 85)
        print("=== BẢNG 1: HIỆU NĂNG TÍNH TOÁN SAMed (NVIDIA A100 FP32) ===")
        print("=" * 85)
        for k, v in ex.items():
            print(f"  {k:<32}: {v}")

        params = ex.get("Trainable Params (M)", 4.21)
        flops = ex.get("FLOPs (G)", 17.23)
        lat = ex.get("Inference Latency (ms/slice)", 0.0)
        vram_inf = ex.get("Inference Peak VRAM (MB)", 0.0)
        time_ep = ex.get("Training Time / Epoch (s)", 0.0)
        tot_time = ex.get("Training Total Time (hours)", 0.0)
        vram_tr = ex.get("Training Peak VRAM (MB)", 0.0)

        print("\n📋 DÒNG COPY EXCEL (BẢNG 1 - ẢNH 1, CLICK Ô TRAINABLE PARAMS DÒNG SAMed):")
        print("-" * 85)
        line_dot = f"{params}M\t{flops}\t{lat}\t{vram_inf}\t{time_ep}\t{tot_time}\t{vram_tr}"
        print("Chuẩn dấu chấm (.):")
        print(line_dot)
        print("Chuẩn dấu phẩy (,):")
        print(line_dot.replace(".", ","))
        print("=" * 85 + "\n")
    else:
        print(f"⚠️ Không tìm thấy file {eff_file}")

    # 2. Print Seed 1234 Segmentation metrics
    if metric_file.exists():
        with open(metric_file, "r", encoding="utf-8") as f:
            m_data = json.load(f)

        dist_unit = m_data.get("distance_unit", "mm")
        hd_key = f"mean_hd95_{dist_unit}"
        s = m_data.get("scar", {})
        e = m_data.get("edema_inclusive", {})
        m = m_data.get("myocardial_ring", {})

        s_d = s.get("mean_dice", 0.0) * 100
        s_hd = s.get(hd_key, 0.0)
        e_d = e.get("mean_dice", 0.0) * 100
        e_hd = e.get(hd_key, 0.0)
        m_d = m.get("mean_dice", 0.0) * 100
        m_hd = m.get(hd_key, 0.0)

        avg_d = (s_d + e_d) / 2.0
        avg_hd = (s_hd + e_hd) / 2.0

        print("=" * 85)
        print("=== KẾT QUẢ PHÂN VÙNG SEED 1234 (MYOPS++) ===")
        print("=" * 85)
        print(f"Scar Dice: {s_d:.2f}% | HD: {s_hd:.2f} {dist_unit}")
        print(f"Edema (Inclusive) Dice: {e_d:.2f}% | HD: {e_hd:.2f} {dist_unit}")
        print(f"Myocardium Dice: {m_d:.2f}% | HD: {m_hd:.2f} {dist_unit}")
        print(f"Avg Pathology Dice: {avg_d:.2f}% | HD: {avg_hd:.2f} {dist_unit}")
        print("-" * 85)
        print("📋 DÒNG COPY EXCEL SEED 1234:")
        row_dot = f"{s_d:.2f}\t{s_hd:.2f}\t{e_d:.2f}\t{e_hd:.2f}\t{m_d:.2f}\t{m_hd:.2f}\t{avg_d:.2f}\t{avg_hd:.2f}"
        print("Chuẩn dấu chấm (.):")
        print(row_dot)
        print("Chuẩn dấu phẩy (,):")
        print(row_dot.replace(".", ","))
        print("=" * 85 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print efficiency and seed 1234 results")
    parser.add_argument("--run_dir", default=None, help="Path to seed 1234 run directory")
    args = parser.parse_args()
    print_metrics(args.run_dir)


if __name__ == "__main__":
    main()
