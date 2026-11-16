"""Shared Evaluator surface distance and clinical metrics for MyoPS benchmarks.

Protocols:
  - MyoPS380: myops380_voxel_v3 (Exact SCAR Parity, 76 Cases, Voxel Distance)
  - MyoPS++:  myopspp_unet2d_roi128_1mm_predicted_roi_mm_v1 (16 Cases, Physical mm Distance)
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


CLASS_NAMES = ("background", "normal_myocardium", "edema", "scar")

BENCHMARK_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "myops380": {
        "id": "myops380_voxel_v3",
        "distance_unit": "voxel",
        "voxelspacing": None,
        "axis_order": "HWD",
        "regions": {
            "normal_myocardium": [1],
            "edema": [2],
            "scar": [3],
            "edema_inclusive": [2, 3],
            "myocardial_ring": [1, 2, 3],
        },
        "region_aliases": {
            "NORM": "normal_myocardium",
            "EDEMA": "edema",
            "SCAR": "scar",
            "LGE-abn": "edema_inclusive",
            "MYO": "myocardial_ring",
        },
        "primary_empty_policy": "both_empty: excluded; one_empty: Dice/IoU=0, distances undefined",
        "aggregation": "equal patient weight within each region; exclude undefined values and report counts",
        "selection_metric": "avg_pathology_inclusive_dice",
        "pathology_regions": ["scar", "edema"],
        "pathology_aggregation": "arithmetic mean of scar and exclusive-edema patient means; both required",
    },
    "myopspp": {
        "id": "myopspp_unet2d_roi128_1mm_predicted_roi_mm_v1",
        "distance_unit": "mm",
        "voxelspacing": "preprocessed ROI NIfTI spacing in mm, orthogonal grid",
        "axis_order": "HWD",
        "regions": {
            "normal_myocardium": [1],
            "edema": [2],
            "scar": [3],
            "edema_inclusive": [2, 3],
            "myocardial_ring": [1, 2, 3],
        },
        "region_aliases": {
            "NORM": "normal_myocardium",
            "EDEMA": "edema",
            "SCAR": "scar",
            "LGE-abn": "edema_inclusive",
            "MYO": "myocardial_ring",
        },
        "primary_empty_policy": "both_empty: excluded; one_empty: Dice/IoU=0, distances undefined",
        "aggregation": "equal patient weight within each region; exclude undefined values and report counts",
        "selection_metric": "avg_pathology_inclusive_dice",
        "pathology_regions": ["scar", "edema"],
        "pathology_aggregation": "arithmetic mean of scar and exclusive-edema patient means; both required",
    },
}

# Backward compatibility alias
BENCHMARK_PROTOCOL = BENCHMARK_PROTOCOLS["myops380"]


def normalize_dataset_id(dataset_id: Optional[str]) -> str:
    """Normalizes dataset identifier string."""
    if dataset_id is None:
        return "myops380"
    ds = str(dataset_id).strip().lower()
    if not ds:
        return "myops380"
    if ds in ("myopspp", "myopspp_unet2d_roi128_1mm", "myops_plus_plus", "myops++"):
        return "myopspp"
    if ds in ("myops380", "myops_380", "default"):
        return "myops380"
    return "myops380"


def binary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: Optional[Sequence[float]] = None,
    compute_distance: bool = True,
    empty_mode: str = "undefined",
) -> Dict[str, Any]:
    """Symmetric HD95 and ASD on 1-connected surfaces; spacing follows array axis order.

    Missing surfaces have no finite distance. Report their status separately instead
    of assigning zero distance to a completely missed lesion.
    """
    pred, truth = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and target shapes differ.")
    if spacing is not None:
        spacing = np.asarray(spacing, dtype=float)
        if spacing.shape != (pred.ndim,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError("Spacing must contain one finite positive value per array axis.")
    n_pred, n_true = int(pred.sum()), int(truth.sum())
    if n_pred == 0 and n_true == 0:
        if empty_mode == "defined":
            return {"dice": 1.0, "iou": 1.0, "hd95": 0.0, "asd": 0.0, "status": "both_empty"}
        return {"dice": None, "iou": None, "hd95": None, "asd": None, "status": "both_empty"}
    overlap = np.logical_and(pred, truth).sum()
    result: Dict[str, Any] = {
        "dice": float(2 * overlap / (n_pred + n_true)),
        "iou": float(overlap / (n_pred + n_true - overlap)),
        "hd95": None,
        "asd": None,
        "status": "ok",
    }
    if not n_pred or not n_true:
        if empty_mode == "defined":
            result["hd95"] = float("inf")
            result["asd"] = float("inf")
            result["status"] = "one_empty"
        else:
            result["status"] = "prediction_empty" if not n_pred else "target_empty"
        return result
    if not compute_distance:
        return result
    structure = generate_binary_structure(pred.ndim, 1)
    pred_surface = pred ^ binary_erosion(pred, structure=structure, border_value=0)
    true_surface = truth ^ binary_erosion(truth, structure=structure, border_value=0)
    distances = np.concatenate((
        distance_transform_edt(~true_surface, sampling=spacing)[pred_surface],
        distance_transform_edt(~pred_surface, sampling=spacing)[true_surface],
    ))
    if distances.size > 0:
        result["hd95"] = float(np.percentile(distances, 95))
        result["asd"] = float(np.mean(distances))
    return result


def benchmark_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    case: str,
    compute_distance: bool = True,
    spacing: Optional[Sequence[float]] = None,
    dataset_id: str = "myops380",
) -> List[Dict[str, Any]]:
    """Primary metrics on canonical HWD predictions; never mutate masks."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.ndim != 3 or prediction.shape != target.shape or not prediction.size:
        raise ValueError("Benchmark requires matching nonempty HWD volumes")
    if not np.isin(prediction, [0, 1, 2, 3]).all() or not np.isin(target, [0, 1, 2, 3]).all():
        raise ValueError("Benchmark requires canonical integer labels 0..3")

    norm_ds = normalize_dataset_id(dataset_id)
    protocol = BENCHMARK_PROTOCOLS.get(norm_ds, BENCHMARK_PROTOCOLS["myops380"])
    distance_unit = protocol["distance_unit"]
    use_spacing = spacing if distance_unit == "mm" else None

    rows = []
    for region, ids in protocol["regions"].items():
        pred, truth = np.isin(prediction, ids), np.isin(target, ids)
        metrics = binary_metrics(pred, truth, spacing=use_spacing, compute_distance=compute_distance)
        n_pred, n_true = int(pred.sum()), int(truth.sum())
        overlap = int(np.logical_and(pred, truth).sum())
        precision = float(overlap / n_pred) if n_pred else None
        recall = float(overlap / n_true) if n_true else None

        row = {
            "case": case,
            "region": region,
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "hd95": metrics["hd95"],
            "asd": metrics["asd"],
            "status": metrics["status"],
            "precision": precision,
            "recall": recall,
            "prediction_voxels": n_pred,
            "target_voxels": n_true,
            "hd95_unit": distance_unit,
            f"hd95_{distance_unit}": metrics["hd95"],
            f"asd_{distance_unit}": metrics["asd"],
        }
        if distance_unit == "mm":
            row["hd95_mm"] = metrics["hd95"]
            row["asd_mm"] = metrics["asd"]
            # Backward compatibility aliases
            row["hd95_voxel"] = metrics["hd95"]
            row["asd_voxel"] = metrics["asd"]
        else:
            row["hd95_voxel"] = metrics["hd95"]
            row["asd_voxel"] = metrics["asd"]

        rows.append(row)
    return rows


def dataset_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    case: str,
    compute_distance: bool = True,
    dataset_id: str = "myops380",
    spacing: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Generate per-case metric rows for all canonical regions."""
    return benchmark_rows(
        prediction=prediction,
        target=target,
        case=case,
        compute_distance=compute_distance,
        spacing=spacing,
        dataset_id=dataset_id,
    )


def summarize_rows(rows: List[Dict[str, Any]], dataset_id: str = "myops380") -> Dict[str, Any]:
    """Patient means and defined/undefined counts for primary metrics."""
    summary: Dict[str, Any] = {}
    identities = [(r["case"], r["region"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate case/region rows would bias patient means")

    norm_ds = normalize_dataset_id(dataset_id)
    protocol = BENCHMARK_PROTOCOLS.get(norm_ds, BENCHMARK_PROTOCOLS["myops380"])
    distance_unit = protocol["distance_unit"]

    expected = set(protocol["regions"])
    for case in {r["case"] for r in rows}:
        if {r["region"] for r in rows if r["case"] == case} != expected:
            raise ValueError(f"Incomplete benchmark regions for {case}")

    for name in protocol["regions"]:
        subset = [r for r in rows if r["region"] == name]
        region: Dict[str, Any] = {
            "cases": len(subset),
            "status_counts": dict(Counter(r["status"] for r in subset)),
        }
        metric_keys = ["dice", "iou", "precision", "recall", f"hd95_{distance_unit}", f"asd_{distance_unit}"]
        for metric in metric_keys:
            values = [r[metric] for r in subset if r.get(metric) is not None]
            if not all(np.isfinite(v) for v in values):
                raise ValueError(f"Non-finite {metric}; undefined values must be None")
            region[f"mean_{metric}"] = float(np.mean(values)) if values else None
            region[f"std_{metric}"] = float(np.std(values)) if values else None
            region[f"median_{metric}"] = float(np.median(values)) if values else None
            region[f"{metric}_defined_cases"] = len(values)
            region[f"{metric}_undefined_cases"] = len(subset) - len(values)

        if distance_unit == "mm":
            region["mean_hd95_mm"] = region.get("mean_hd95_mm")
            region["std_hd95_mm"] = region.get("std_hd95_mm")
            region["mean_asd_mm"] = region.get("mean_asd_mm")
            region["std_asd_mm"] = region.get("std_asd_mm")
            # Fallback alias for table extraction parsers
            region["mean_hd95_voxel"] = region.get("mean_hd95_mm")
            region["mean_asd_voxel"] = region.get("mean_asd_mm")
        else:
            region["mean_hd95_voxel"] = region.get("mean_hd95_voxel")
            region["std_hd95_voxel"] = region.get("std_hd95_voxel")
            region["mean_asd_voxel"] = region.get("mean_asd_voxel")
            region["std_asd_voxel"] = region.get("std_asd_voxel")

        summary[name] = region

    # Pathology composite metrics across regions
    for metric in ("dice", "iou", f"hd95_{distance_unit}", f"asd_{distance_unit}"):
        values_excl = [summary[name].get(f"mean_{metric}") for name in ("scar", "edema")]
        summary[f"avg_pathology_{metric}"] = (
            float(np.mean(values_excl)) if all(v is not None for v in values_excl) else None
        )

        values_incl = [summary[name].get(f"mean_{metric}") for name in ("scar", "edema_inclusive")]
        summary[f"avg_pathology_inclusive_{metric}"] = (
            float(np.mean(values_incl)) if all(v is not None for v in values_incl) else None
        )

    if distance_unit == "mm":
        summary["avg_pathology_hd95_mm"] = summary.get(f"avg_pathology_hd95_{distance_unit}")
        summary["avg_pathology_inclusive_hd95_mm"] = summary.get(f"avg_pathology_inclusive_hd95_{distance_unit}")
        summary["avg_pathology_asd_mm"] = summary.get(f"avg_pathology_asd_{distance_unit}")
        summary["avg_pathology_inclusive_asd_mm"] = summary.get(f"avg_pathology_inclusive_asd_{distance_unit}")
        # Backward compatibility aliases
        if summary.get("avg_pathology_hd95_voxel") is None:
            summary["avg_pathology_hd95_voxel"] = summary["avg_pathology_hd95_mm"]
        if summary.get("avg_pathology_inclusive_hd95_voxel") is None:
            summary["avg_pathology_inclusive_hd95_voxel"] = summary["avg_pathology_inclusive_hd95_mm"]

    return summary


def summarize_dataset_rows(rows: List[Dict[str, Any]], dataset_id: str = "myops380") -> Dict[str, Any]:
    return summarize_rows(rows, dataset_id=dataset_id)


def calculate_metric_percase(pred: np.ndarray, gt: np.ndarray, spacing: Optional[Sequence[float]] = None) -> Tuple[float, float]:
    """Legacy tuple API, honest empty-mask behavior, no input mutation."""
    result = binary_metrics(pred, gt, spacing=spacing, empty_mode="undefined")
    distance = result["hd95"]
    if distance is None:
        distance = np.nan if result["status"] == "both_empty" else np.inf
    dice = result["dice"] if result["dice"] is not None else (1.0 if result["status"] == "both_empty" else 0.0)
    return float(dice), float(distance)
