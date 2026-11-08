"""Shared Evaluator surface distance and clinical metrics for MyoPS380 (myops380_voxel_v3)."""
from __future__ import annotations

from collections import Counter
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


CLASS_NAMES = ("background", "normal_myocardium", "edema", "scar")

BENCHMARK_PROTOCOL = {
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
    "primary_empty_policy": "both_empty: excluded; one_empty: Dice/IoU=0, distances undefined",
    "aggregation": "equal patient weight within each region; exclude undefined values and report counts",
    "selection_metric": "avg_pathology_inclusive_dice",
    "pathology_regions": ["scar", "edema"],
    "pathology_aggregation": "arithmetic mean of scar and exclusive-edema patient means; both required",
}


def binary_metrics(prediction, target, spacing=None, compute_distance=True, empty_mode="undefined"):
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
    result = {
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
    result["hd95"] = float(np.percentile(distances, 95))
    result["asd"] = float(np.mean(distances))
    return result


def benchmark_rows(prediction, target, case, compute_distance=True):
    """Primary metrics on canonical HWD predictions; never mutate masks."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.ndim != 3 or prediction.shape != target.shape or not prediction.size:
        raise ValueError("Benchmark requires matching nonempty HWD volumes")
    if not np.isin(prediction, [0, 1, 2, 3]).all() or not np.isin(target, [0, 1, 2, 3]).all():
        raise ValueError("Benchmark requires canonical integer labels 0..3")
    rows = []
    for region, ids in BENCHMARK_PROTOCOL["regions"].items():
        pred, truth = np.isin(prediction, ids), np.isin(target, ids)
        metrics = binary_metrics(pred, truth, compute_distance=compute_distance)
        n_pred, n_true = int(pred.sum()), int(truth.sum())
        overlap = int(np.logical_and(pred, truth).sum())
        metrics["precision"] = overlap / n_pred if n_pred else None
        metrics["recall"] = overlap / n_true if n_true else None
        rows.append(dict(
            case=case,
            region=region,
            **metrics,
            prediction_voxels=n_pred,
            target_voxels=n_true,
            hd95_unit="voxel",
            hd95_voxel=metrics["hd95"],
            asd_voxel=metrics["asd"],
        ))
    return rows


def dataset_rows(prediction, target, case, compute_distance=True, dataset_id="myops380", spacing=None):
    """Generate per-case metric rows for all canonical regions."""
    return benchmark_rows(prediction, target, case, compute_distance=compute_distance)


def summarize_rows(rows):
    """Patient means and defined/undefined counts for primary metrics."""
    summary = {}
    identities = [(r["case"], r["region"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate case/region rows would bias patient means")
    expected = set(BENCHMARK_PROTOCOL["regions"])
    for case in {r["case"] for r in rows}:
        if {r["region"] for r in rows if r["case"] == case} != expected:
            raise ValueError(f"Incomplete benchmark regions for {case}")
    for name in BENCHMARK_PROTOCOL["regions"]:
        subset = [r for r in rows if r["region"] == name]
        region = {"cases": len(subset), "status_counts": dict(Counter(r["status"] for r in subset))}
        for metric in ("dice", "iou", "precision", "recall", "hd95_voxel", "asd_voxel"):
            values = [r[metric] for r in subset if r[metric] is not None]
            if not all(np.isfinite(v) for v in values):
                raise ValueError(f"Non-finite {metric}; undefined values must be None")
            region[f"mean_{metric}"] = float(np.mean(values)) if values else None
            region[f"{metric}_defined_cases"] = len(values)
            region[f"{metric}_undefined_cases"] = len(subset) - len(values)
        summary[name] = region
    for metric in ("dice", "iou", "hd95_voxel"):
        values = [summary[name][f"mean_{metric}"] for name in ("scar", "edema")]
        summary[f"avg_pathology_{metric}"] = float(np.mean(values)) if all(v is not None for v in values) else None
        values = [summary[name][f"mean_{metric}"] for name in ("scar", "edema_inclusive")]
        summary[f"avg_pathology_inclusive_{metric}"] = float(np.mean(values)) if all(v is not None for v in values) else None
    return summary


def summarize_dataset_rows(rows, dataset_id="myops380"):
    return summarize_rows(rows)


def calculate_metric_percase(pred, gt, spacing=None):
    """Legacy tuple API, honest empty-mask behavior, no input mutation."""
    result = binary_metrics(pred, gt, spacing, empty_mode="undefined")
    distance = result["hd95"]
    if distance is None:
        distance = np.nan if result["status"] == "both_empty" else np.inf
    return (result["dice"] if result["dice"] is not None else np.nan), distance
