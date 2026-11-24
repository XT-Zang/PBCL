from __future__ import annotations

import math
from itertools import combinations
from typing import Iterable

import numpy as np

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - scipy is installed in the project env
    ndi = None

try:
    from skimage.morphology import skeletonize
except Exception:  # pragma: no cover - optional fallback
    skeletonize = None


STRUCTURAL_METRIC_KEYS = (
    "boundary_f1",
    "boundary_iou",
    "skeleton_f1",
    "merge_rate",
    "close_pair_recall",
    "pq",
)


def _as_bool(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {arr.shape}")
    return arr.astype(bool, copy=False)


def _safe_div(numerator: float, denominator: float, empty_value: float = math.nan) -> float:
    return float(numerator / denominator) if denominator else float(empty_value)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = _as_bool(mask)
    if radius <= 0:
        return mask
    if ndi is None:  # pragma: no cover
        padded = np.pad(mask, radius, mode="constant", constant_values=False)
        out = np.zeros_like(mask, dtype=bool)
        width = 2 * radius + 1
        for y in range(width):
            for x in range(width):
                out |= padded[y : y + mask.shape[0], x : x + mask.shape[1]]
        return out
    return ndi.binary_dilation(mask, iterations=radius)


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = _as_bool(mask)
    if radius <= 0:
        return mask
    if ndi is None:  # pragma: no cover
        return ~_dilate(~mask, radius)
    return ndi.binary_erosion(mask, iterations=radius, border_value=0)


def boundary_band(mask: np.ndarray, width: int = 1) -> np.ndarray:
    mask = _as_bool(mask)
    return _dilate(mask, width) & ~_erode(mask, width)


def tolerance_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 1) -> float:
    pred = _as_bool(pred)
    gt = _as_bool(gt)
    pred_count = int(pred.sum())
    gt_count = int(gt.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0
    pred_match = int((pred & _dilate(gt, tolerance)).sum())
    gt_match = int((gt & _dilate(pred, tolerance)).sum())
    precision = pred_match / pred_count
    recall = gt_match / gt_count
    return float(2.0 * precision * recall / max(1e-9, precision + recall))


def boundary_metrics(pred: np.ndarray, gt: np.ndarray, width: int = 1, tolerance: int = 1) -> dict[str, float]:
    pred_b = boundary_band(pred, width)
    gt_b = boundary_band(gt, width)
    union = int(np.logical_or(pred_b, gt_b).sum())
    inter = int(np.logical_and(pred_b, gt_b).sum())
    return {
        "boundary_f1": tolerance_f1(pred_b, gt_b, tolerance=tolerance),
        "boundary_iou": _safe_div(inter, union, empty_value=1.0),
    }


def skeleton_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 1) -> float:
    pred = _as_bool(pred)
    gt = _as_bool(gt)
    if skeletonize is None:  # pragma: no cover
        return tolerance_f1(pred, gt, tolerance=tolerance)
    return tolerance_f1(skeletonize(pred), skeletonize(gt), tolerance=tolerance)


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = _as_bool(mask)
    if ndi is None:  # pragma: no cover
        raise RuntimeError("scipy is required for connected-component structural metrics")
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, count = ndi.label(mask, structure=structure)
    return labels.astype(np.int32, copy=False), int(count)


def _component_gap(first: np.ndarray, second: np.ndarray) -> int:
    if np.any(first & second):
        return 0
    if not np.any(first) or not np.any(second):
        return math.inf
    if ndi is None:  # pragma: no cover
        first_yx = np.argwhere(first)
        second_yx = np.argwhere(second)
        distances = np.sqrt(((first_yx[:, None, :] - second_yx[None, :, :]) ** 2).sum(axis=2))
        return max(0, int(math.ceil(float(distances.min()))) - 1)
    distances = ndi.distance_transform_edt(~first)
    center_distance = float(np.min(distances[second]))
    return max(0, int(math.ceil(center_distance)) - 1)


def _dominant_pred_label(gt_component: np.ndarray, pred_labels: np.ndarray, min_overlap_pixels: int) -> int:
    labels, counts = np.unique(pred_labels[gt_component], return_counts=True)
    best_label = 0
    best_count = 0
    for label, count in zip(labels.tolist(), counts.tolist()):
        if label == 0:
            continue
        if count > best_count:
            best_label = int(label)
            best_count = int(count)
    return best_label if best_count >= min_overlap_pixels else 0


def close_pair_recall(
    gt_labels: np.ndarray,
    gt_count: int,
    pred_labels: np.ndarray,
    max_distance: int = 8,
    min_overlap_pixels: int = 8,
) -> tuple[float, int, int]:
    pairs = 0
    separated = 0
    components = {idx: gt_labels == idx for idx in range(1, gt_count + 1)}
    for first_idx, second_idx in combinations(range(1, gt_count + 1), 2):
        first = components[first_idx]
        second = components[second_idx]
        if _component_gap(first, second) > max_distance:
            continue
        pairs += 1
        first_pred = _dominant_pred_label(first, pred_labels, min_overlap_pixels=min_overlap_pixels)
        second_pred = _dominant_pred_label(second, pred_labels, min_overlap_pixels=min_overlap_pixels)
        if first_pred and second_pred and first_pred != second_pred:
            separated += 1
    return _safe_div(separated, pairs, empty_value=math.nan), separated, pairs


def merge_rate(
    gt_labels: np.ndarray,
    gt_count: int,
    pred_labels: np.ndarray,
    pred_count: int,
    min_overlap_pixels: int = 8,
) -> float:
    if gt_count == 0:
        return 0.0 if pred_count == 0 else 1.0
    merged_gt_labels: set[int] = set()
    for pred_idx in range(1, pred_count + 1):
        overlapping, counts = np.unique(gt_labels[pred_labels == pred_idx], return_counts=True)
        valid = [int(label) for label, count in zip(overlapping.tolist(), counts.tolist()) if label > 0 and count >= min_overlap_pixels]
        if len(valid) >= 2:
            merged_gt_labels.update(valid)
    return float(len(merged_gt_labels) / gt_count)


def panoptic_quality(pred_labels: np.ndarray, pred_count: int, gt_labels: np.ndarray, gt_count: int, iou_threshold: float = 0.5) -> float:
    if pred_count == 0 and gt_count == 0:
        return 1.0
    candidates: list[tuple[float, int, int]] = []
    for pred_idx in range(1, pred_count + 1):
        pred_component = pred_labels == pred_idx
        for gt_idx in range(1, gt_count + 1):
            gt_component = gt_labels == gt_idx
            inter = int(np.logical_and(pred_component, gt_component).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(pred_component, gt_component).sum())
            iou = inter / union
            if iou >= iou_threshold:
                candidates.append((iou, pred_idx, gt_idx))
    candidates.sort(reverse=True)
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    iou_sum = 0.0
    for iou, pred_idx, gt_idx in candidates:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        iou_sum += float(iou)
    tp = len(matched_gt)
    fp = pred_count - len(matched_pred)
    fn = gt_count - len(matched_gt)
    return _safe_div(iou_sum, tp + 0.5 * fp + 0.5 * fn, empty_value=0.0)


def compute_structural_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    boundary_width: int = 2,
    boundary_tolerance: int = 2,
    skeleton_tolerance: int = 2,
    close_pair_max_distance: int = 8,
    min_overlap_pixels: int = 8,
) -> dict[str, float]:
    pred = _as_bool(pred)
    gt = _as_bool(gt)
    gt_labels, gt_count = label_components(gt)
    pred_labels, pred_count = label_components(pred)
    pair_recall, _, _ = close_pair_recall(
        gt_labels,
        gt_count,
        pred_labels,
        max_distance=close_pair_max_distance,
        min_overlap_pixels=min_overlap_pixels,
    )
    result = boundary_metrics(pred, gt, width=boundary_width, tolerance=boundary_tolerance)
    result.update(
        {
            "skeleton_f1": skeleton_f1(pred, gt, tolerance=skeleton_tolerance),
            "merge_rate": merge_rate(gt_labels, gt_count, pred_labels, pred_count, min_overlap_pixels=min_overlap_pixels),
            "close_pair_recall": pair_recall,
            "pq": panoptic_quality(pred_labels, pred_count, gt_labels, gt_count),
        }
    )
    return result


def aggregate_structural_rows(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    summary: dict[str, float] = {}
    for key in STRUCTURAL_METRIC_KEYS:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        summary[key] = float(np.mean(values)) if values else math.nan
    return summary
