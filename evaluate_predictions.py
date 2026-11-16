"""Standalone Benchmark Prediction Evaluation script for SAMed.

Protocols:
  - myops380: myops380_voxel_v3 (Exact SCAR Parity, 76 Cases, Voxel Distance)
  - myopspp:  myopspp_unet2d_roi128_1mm_predicted_roi_mm_v1 (16 Cases, Physical mm Distance)

Evaluates patient volumes across 5 target regions:
  1. Myocardium (MYO / myocardial_ring): IDs [1, 2, 3]
  2. Edema (EDEMA / edema): ID [2] (exclusive)
  3. Scar (SCAR / scar): ID [3]
  4. LGE-abnormal (LGE-abn / edema_inclusive): IDs [2, 3]
  5. Normal Myocardium (NORM / normal_myocardium): ID [1]

Outputs:
  - <output_dir>/per_case.csv
  - <output_dir>/metrics.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset_myops import (
    CLASS_NAMES,
    MyopsDataset,
    normalize_dataset_name,
    resolve_default_list_dir,
)
from utils.metrics import (
    BENCHMARK_PROTOCOLS,
    dataset_rows,
    summarize_dataset_rows,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pred_dir", "--pred-dir", default=None,
                        help="Directory containing <CaseID>_pred.npz files")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional checkpoint path; if provided and pred_dir is not set, runs evaluate_myops")
    parser.add_argument("--data_root", "--data-root",
                        default=os.environ.get("MYOPS380_DATA_ROOT", "/content/MyoPS380_dataset/Processed_data"),
                        help="Root folder containing bSSFP, LGE, T2w subdirectories")
    parser.add_argument("--dataset", "--dataset_id", "--dataset-id", default="myops380",
                        choices=["myops380", "myopspp", "myopspp_unet2d_roi128_1mm", "myops_380"],
                        help="Benchmark dataset to evaluate")
    parser.add_argument("--list_dir", "--list-dir", "--split_dir", "--split-dir", default=None,
                        help="Directory containing test_vol.txt (defaults dynamically by dataset)")
    parser.add_argument("--output_dir", "--output-dir", default=None,
                        help="Evaluation output directory; defaults to pred_dir parent")
    parser.add_argument("--split", default="test_vol", help="Split name to evaluate (default: test_vol)")
    parser.add_argument("--compute_distance", action=argparse.BooleanOptionalAction, default=True,
                        help="Compute Hausdorff and surface distance metrics")
    parser.add_argument("--measure_efficiency", "--measure-efficiency", action=argparse.BooleanOptionalAction,
                        default=False, help="Measure Complexity and Inference efficiency metrics (requires --checkpoint)")
    parser.add_argument("--efficiency_output", "--efficiency-output", default=None,
                        help="Optional JSON file to save measured efficiency metrics")
    return parser


def load_prediction_file(pred_dir: Path, case_name: str) -> np.ndarray:
    candidates = [
        pred_dir / f"{case_name}_pred.npz",
        pred_dir / f"{case_name}.npz",
        pred_dir / "test_predictions" / f"{case_name}_pred.npz",
        pred_dir / "test_predictions" / f"{case_name}.npz",
    ]
    for c in candidates:
        if c.is_file():
            with np.load(c) as d:
                for k in ("prediction", "pred", "arr_0"):
                    if k in d:
                        return np.asarray(d[k], dtype=np.uint8)
    raise FileNotFoundError(f"Prediction file not found for case {case_name} in {pred_dir}")


def main(argv=None):
    args = build_parser().parse_args(argv)

    # If checkpoint is provided and no precomputed pred_dir, delegate to evaluate_myops
    if args.checkpoint and (not args.pred_dir or not Path(args.pred_dir).is_dir()):
        import evaluate_myops
        return evaluate_myops.main(argv)

    norm_dataset = normalize_dataset_name(args.dataset)
    list_dir = Path(args.list_dir) if args.list_dir else resolve_default_list_dir(norm_dataset)

    if not args.pred_dir:
        raise ValueError("Must provide either --pred_dir with existing predictions or --checkpoint.")

    pred_dir = Path(args.pred_dir).resolve()
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"pred_dir does not exist: {pred_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else (
        pred_dir.parent if pred_dir.name == "test_predictions" else pred_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "evaluation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
        handlers=[
            logging.FileHandler(str(log_file), mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(f"=== Standalone Evaluation on {norm_dataset} ({args.split}) ===")
    logging.info(f"Predictions: {pred_dir}")
    logging.info(f"Output: {output_dir}")

    # Load dataset for ground truth
    # MyoPS++: already canonical on disk -> label_order='canonical'
    # MyoPS380: legacy on disk -> label_order='legacy'
    dataset = MyopsDataset(
        base_dir=args.data_root,
        list_dir=list_dir,
        split=args.split,
        dataset=norm_dataset,
    )
    logging.info(f"Loaded ground truth for {len(dataset)} cases")

    benchmark_proto = BENCHMARK_PROTOCOLS.get(norm_dataset, BENCHMARK_PROTOCOLS["myops380"])
    dist_unit = benchmark_proto["distance_unit"]

    rows = []
    for sample in dataset:
        case = sample["case_name"]
        target = np.asarray(sample["label"]) if sample["label"] is not None else None
        if target is None:
            continue

        prediction = load_prediction_file(pred_dir, case)
        spacing = sample.get("spacing")

        case_rows = dataset_rows(
            prediction=prediction,
            target=target,
            case=case,
            compute_distance=args.compute_distance,
            dataset_id=norm_dataset,
            spacing=spacing,
        )
        rows.extend(case_rows)

    if not rows:
        raise ValueError("No evaluation results were generated.")

    csv_path = output_dir / "per_case.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Saved per-case metrics to: {csv_path}")

    summary = summarize_dataset_rows(rows, dataset_id=norm_dataset)
    summary.update({
        "dataset_id": norm_dataset,
        "metric_protocol": benchmark_proto["id"],
        "distance_unit": dist_unit,
        "exported_label_order": "canonical",
        "total_cases": len(dataset),
    })

    json_path = output_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Saved summary metrics to: {json_path}")

    # Display Master Benchmark Clinical Table
    hd95_key = f"mean_hd95_{dist_unit}"
    scar_dice = (summary.get("scar", {}).get("mean_dice") or 0.0) * 100
    scar_hd95 = summary.get("scar", {}).get(hd95_key) or 0.0
    edema_dice = (summary.get("edema_inclusive", {}).get("mean_dice") or 0.0) * 100
    edema_hd95 = summary.get("edema_inclusive", {}).get(hd95_key) or 0.0
    edema_excl_dice = (summary.get("edema", {}).get("mean_dice") or 0.0) * 100
    edema_excl_hd95 = summary.get("edema", {}).get(hd95_key) or 0.0
    myo_dice = (summary.get("myocardial_ring", {}).get("mean_dice") or 0.0) * 100
    myo_hd95 = summary.get("myocardial_ring", {}).get(hd95_key) or 0.0
    avg_path_dice = (summary.get("avg_pathology_inclusive_dice") or 0.0) * 100
    avg_path_hd95 = summary.get(f"avg_pathology_inclusive_{hd95_key}") or summary.get("avg_pathology_inclusive_hd95_mm") or summary.get("avg_pathology_inclusive_hd95_voxel") or 0.0
    avg_path_excl_dice = (summary.get("avg_pathology_dice") or 0.0) * 100
    avg_path_excl_hd95 = summary.get(f"avg_pathology_{hd95_key}") or summary.get("avg_pathology_hd95_mm") or summary.get("avg_pathology_hd95_voxel") or 0.0

    table = f"""
=============================================================================================================
MASTER BENCHMARK EVALUATION RESULTS (Protocol: {benchmark_proto['id']})
Dataset: {norm_dataset} | Protocol: {benchmark_proto['id']} | Cases: {len(dataset)}
-------------------------------------------------------------------------------------------------------------
Region                   | Dice (%)       | HD95 ({dist_unit})
-------------------------------------------------------------------------------------------------------------
Scar                     | {scar_dice:6.2f}         | {scar_hd95:6.2f}
Edema (Inclusive / AAR)  | {edema_dice:6.2f}         | {edema_hd95:6.2f}
Edema (Exclusive)        | {edema_excl_dice:6.2f}         | {edema_excl_hd95:6.2f}
Myocardium (Whole Ring)  | {myo_dice:6.2f}         | {myo_hd95:6.2f}
-------------------------------------------------------------------------------------------------------------
Avg. Pathology (Incl.)   | {avg_path_dice:6.2f}         | {avg_path_hd95:6.2f}
Avg. Pathology (Excl.)   | {avg_path_excl_dice:6.2f}         | {avg_path_excl_hd95:6.2f}
=============================================================================================================
"""
    logging.info(table)


if __name__ == "__main__":
    main()
