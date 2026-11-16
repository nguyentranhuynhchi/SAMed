"""Summarize all 5 seeds for SAMed benchmark on MyoPS++ and MyoPS380.
Outputs Excel-ready tab-separated values (both dot and comma decimals).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def summarize_seeds(
    base_dir: str | Path = "/content/drive/MyDrive/MyoPS_Results/SAMed/runs",
    dataset: str = "myopspp",
) -> None:
    base_dir = Path(base_dir).resolve()
    seeds = [1234, 1098, 1024, 2026, 6868]

    prefix = "myopspp_seed" if dataset in ("myopspp", "myopspp_unet2d_roi128_1mm") else "myops380_seed"
    dist_unit = "mm" if dataset in ("myopspp", "myopspp_unet2d_roi128_1mm") else "voxel"

    rows = []

    print("\n" + "=" * 90)
    print(f"=== TỔNG HỢP KẾT QUẢ 5 SEEDS SAMED ({dataset.upper()} - {dist_unit.upper()}) ===")
    print("=" * 90)

    for seed in seeds:
        # Check potential run folders
        cand_dirs = [
            base_dir / f"{prefix}{seed}",
            base_dir / f"samed_seed{seed}",
            base_dir / f"seed_{seed}",
            base_dir / f"seed{seed}",
        ]
        metric_file = None
        for cd in cand_dirs:
            mf = cd / "evaluation" / "metrics.json"
            if mf.exists():
                metric_file = mf
                break

        if metric_file is None:
            print(f"[!] Seed {seed}: Chưa tìm thấy metrics.json trong các thư mục {cand_dirs[0]}")
            continue

        with open(metric_file, "r", encoding="utf-8") as f:
            d = json.load(f)

        s = d.get("scar", {})
        e = d.get("edema_inclusive", {})
        m = d.get("myocardial_ring", {})

        hd_key = f"mean_hd95_{dist_unit}"
        s_d = s.get("mean_dice", 0.0) * 100
        s_hd = s.get(hd_key, 0.0)
        e_d = e.get("mean_dice", 0.0) * 100
        e_hd = e.get(hd_key, 0.0)
        m_d = m.get("mean_dice", 0.0) * 100
        m_hd = m.get(hd_key, 0.0)

        avg_d = (s_d + e_d) / 2.0
        avg_hd = (s_hd + e_hd) / 2.0

        rows.append({
            "seed": seed,
            "s_d": s_d, "s_hd": s_hd,
            "e_d": e_d, "e_hd": e_hd,
            "m_d": m_d, "m_hd": m_hd,
            "avg_d": avg_d, "avg_hd": avg_hd,
        })

    # Print Table
    print(f"{'Seed':<8} | {'Scar Dice':<10} {'HD':<8} | {'Edema Dice':<10} {'HD':<8} | {'Myo Dice':<10} {'HD':<8} | {'Avg Path':<10} {'HD':<8}")
    print("-" * 90)
    for r in rows:
        print(f"{r['seed']:<8} | {r['s_d']:<9.2f}% {r['s_hd']:<7.2f} | {r['e_d']:<9.2f}% {r['e_hd']:<7.2f} | {r['m_d']:<9.2f}% {r['m_hd']:<7.2f} | {r['avg_d']:<9.2f}% {r['avg_hd']:<7.2f}")

    if rows:
        print("-" * 90)
        means = {k: float(np.mean([r[k] for r in rows])) for k in ["s_d", "s_hd", "e_d", "e_hd", "m_d", "m_hd", "avg_d", "avg_hd"]}
        stds = {k: float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0 for k in ["s_d", "s_hd", "e_d", "e_hd", "m_d", "m_hd", "avg_d", "avg_hd"]}
        print(f"{'Avg':<8} | {means['s_d']:<9.2f}% {means['s_hd']:<7.2f} | {means['e_d']:<9.2f}% {means['e_hd']:<7.2f} | {means['m_d']:<9.2f}% {means['m_hd']:<7.2f} | {means['avg_d']:<9.2f}% {means['avg_hd']:<7.2f}")
        print(f"{'Std':<8} | {stds['s_d']:<9.2f}% {stds['s_hd']:<7.2f} | {stds['e_d']:<9.2f}% {stds['e_hd']:<7.2f} | {stds['m_d']:<9.2f}% {stds['m_hd']:<7.2f} | {stds['avg_d']:<9.2f}% {stds['avg_hd']:<7.2f}")

        print("\n" + "=" * 90)
        print("📋 KHỐI DỮ LIỆU COPY THẲNG VÀO EXCEL (CHUẨN DẤU CHẤM .):")
        print("=" * 90)
        for r in rows:
            print(f"{r['s_d']:.2f}\t{r['s_hd']:.2f}\t{r['e_d']:.2f}\t{r['e_hd']:.2f}\t{r['m_d']:.2f}\t{r['m_hd']:.2f}\t{r['avg_d']:.2f}\t{r['avg_hd']:.2f}")
        avg_line_dot = f"{means['s_d']:.2f} ± {stds['s_d']:.2f}\t{means['s_hd']:.2f} ± {stds['s_hd']:.2f}\t{means['e_d']:.2f} ± {stds['e_d']:.2f}\t{means['e_hd']:.2f} ± {stds['e_hd']:.2f}\t{means['m_d']:.2f} ± {stds['m_d']:.2f}\t{means['m_hd']:.2f} ± {stds['m_hd']:.2f}\t{means['avg_d']:.2f} ± {stds['avg_d']:.2f}\t{means['avg_hd']:.2f} ± {stds['avg_hd']:.2f}"
        print(avg_line_dot)

        print("\n" + "=" * 90)
        print("📋 KHỐI DỮ LIỆU COPY THẲNG VÀO EXCEL (CHUẨN DẤU PHẨY ,):")
        print("=" * 90)
        for r in rows:
            line_comma = f"{r['s_d']:.2f}\t{r['s_hd']:.2f}\t{r['e_d']:.2f}\t{r['e_hd']:.2f}\t{r['m_d']:.2f}\t{r['m_hd']:.2f}\t{r['avg_d']:.2f}\t{r['avg_hd']:.2f}"
            print(line_comma.replace(".", ","))
        avg_line_comma = avg_line_dot.replace(".", ",")
        print(avg_line_comma)
        print("=" * 90 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize 5 seeds for SAMed")
    parser.add_argument("--base_dir", default="/content/drive/MyDrive/MyoPS_Results/SAMed/runs", help="Base runs directory")
    parser.add_argument("--dataset", default="myopspp", choices=["myopspp", "myops380"], help="Dataset benchmark")
    args = parser.parse_args()
    summarize_seeds(args.base_dir, args.dataset)


if __name__ == "__main__":
    main()
