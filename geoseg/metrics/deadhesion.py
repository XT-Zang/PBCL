from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - scipy is present in the AquaV3 environment
    ndi = None


PAIR_BUCKETS = ("touching", "1_2", "3_5", "6_10")

DEADHESION_SAMPLE_FIELDS = (
    "split",
    "tile_id",
    "scene_id",
    "species_domain",
    "height",
    "width",
    "gt_foreground_pixels",
    "pred_foreground_pixels",
    "fg_ratio",
    "pred_fg_ratio",
    "tp_pixels",
    "fp_pixels",
    "fn_pixels",
    "tn_pixels",
    "fg_iou",
    "fg_dice",
    "fg_precision",
    "fg_recall",
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "boundary_iou",
    "boundary_pred_match",
    "boundary_pred_pixels",
    "boundary_gt_match",
    "boundary_gt_pixels",
    "boundary_intersection",
    "boundary_union",
    "gt_instances",
    "pred_components",
    "pred_component_max_area",
    "object_count_error",
    "object_count_abs_error",
    "object_count_denominator",
    "normalized_object_count_abs_error",
    "merge_components",
    "merge_extra_gt",
    "merge_denominator",
    "merge_rate",
    "split_components",
    "split_extra_pred",
    "split_denominator",
    "split_rate",
    "pq_iou_sum",
    "pq_tp",
    "pq_fp",
    "pq_fn",
    "pq",
    "close_pair_touching_total",
    "close_pair_touching_separated",
    "close_pair_touching_separation_recall",
    "close_pair_1_2_total",
    "close_pair_1_2_separated",
    "close_pair_1_2_separation_recall",
    "close_pair_3_5_total",
    "close_pair_3_5_separated",
    "close_pair_3_5_separation_recall",
    "close_pair_6_10_total",
    "close_pair_6_10_separated",
    "close_pair_6_10_separation_recall",
    "close_pair_1_5_total",
    "close_pair_1_5_separated",
    "close_pair_1_5_separation_recall",
    "close_pair_separation_recall",
)


@dataclass(frozen=True)
class CachedTile:
    tile_id: str
    split: str
    scene_id: str
    species_domain: str
    gt_mask: np.ndarray
    label_map: np.ndarray
    gt_areas: np.ndarray
    pair_buckets: tuple[tuple[int, int, str], ...]


@torch.no_grad()
def binary_logits_to_masks(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4 or int(logits.shape[1]) != 1:
        raise ValueError(f"deadhesion validation expects single-channel binary logits, got shape {tuple(logits.shape)}")
    return torch.sigmoid(logits) > 0.5


def read_manifest(dataset_root: Path, split: str) -> list[dict[str, Any]]:
    splits = ["train", "val", "test"] if split == "all" else [split]
    rows: list[dict[str, Any]] = []
    for name in splits:
        path = dataset_root / "splits" / f"{name}.jsonl"
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["_split"] = name
                    rows.append(row)
    return rows


def load_coco_index(path: Path) -> dict[str, list[dict[str, Any]]]:
    coco = json.loads(path.read_text(encoding="utf-8"))
    tile_by_image_id: dict[int, str] = {}
    index: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in coco.get("images", []):
        tile_id = str(image.get("tile_id") or Path(str(image.get("file_name", ""))).stem)
        if not tile_id:
            continue
        tile_by_image_id[int(image["id"])] = tile_id
        index[tile_id]
    for annotation in coco.get("annotations", []):
        tile_id = tile_by_image_id.get(int(annotation.get("image_id", -1)))
        if tile_id is not None:
            index[tile_id].append(annotation)
    return dict(index)


def rasterize_coco_instances(annotations: list[dict[str, Any]], shape: tuple[int, int]) -> list[np.ndarray]:
    height, width = shape
    masks: list[np.ndarray] = []
    for annotation in annotations:
        segmentation = annotation.get("segmentation", [])
        if isinstance(segmentation, dict):
            raise ValueError("Compressed COCO RLE is not supported by this evaluator; convert annotations to polygons.")
        polygons = segmentation
        if polygons and isinstance(polygons[0], (int, float)):
            polygons = [polygons]
        canvas = Image.new("1", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for polygon in polygons:
            if len(polygon) < 6 or len(polygon) % 2:
                continue
            points = [(float(polygon[idx]), float(polygon[idx + 1])) for idx in range(0, len(polygon), 2)]
            draw.polygon(points, fill=1)
        mask = np.asarray(canvas, dtype=bool)
        if np.any(mask):
            masks.append(mask)
    return masks


def read_mask(path: Path, threshold: float = 0.0, foreground: bool = False) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if foreground:
        if arr.dtype.kind in {"f"}:
            return arr > threshold
        max_value = float(np.nanmax(arr)) if arr.size else 1.0
        cutoff = threshold * 255.0 if max_value > 1.0 else threshold
        return arr > cutoff
    return arr > 0


def max_filter(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    if ndi is not None:
        return ndi.binary_dilation(mask, iterations=radius)
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    size = 2 * radius + 1
    for y in range(size):
        for x in range(size):
            out |= padded[y : y + mask.shape[0], x : x + mask.shape[1]]
    return out


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    if ndi is not None:
        return ndi.binary_erosion(mask, iterations=radius, border_value=0)
    return ~max_filter(~mask, radius)


def boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return max_filter(mask, width) & ~erode(mask, width)


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _safe_match_rate(matches: int | float, predicted: int | float, reference: int | float) -> float:
    if predicted:
        return float(matches / predicted)
    return 1.0 if not reference else 0.0


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union else 1.0


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return {
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "tn_pixels": tn,
        "fg_iou": float(tp / (tp + fp + fn)) if (tp + fp + fn) else 1.0,
        "fg_dice": float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0,
        "fg_precision": float(tp / (tp + fp)) if (tp + fp) else (1.0 if not np.any(gt) else 0.0),
        "fg_recall": float(tp / (tp + fn)) if (tp + fn) else 1.0,
    }


def boundary_metrics(pred: np.ndarray, gt: np.ndarray, width: int, tolerance: int) -> dict[str, float | int]:
    gt_b = boundary_band(gt, width)
    return _boundary_metrics(pred, width, tolerance, gt_b, max_filter(gt_b, tolerance))


def _boundary_metrics(
    pred: np.ndarray, width: int, tolerance: int, gt_b: np.ndarray, gt_dilated: np.ndarray
) -> dict[str, float | int]:
    pred_b = boundary_band(pred, width)
    pred_match = pred_b & gt_dilated
    gt_match = gt_b & max_filter(pred_b, tolerance)
    pred_match_count = int(pred_match.sum())
    gt_match_count = int(gt_match.sum())
    pred_boundary_count = int(pred_b.sum())
    gt_boundary_count = int(gt_b.sum())
    boundary_intersection = int(np.logical_and(pred_b, gt_b).sum())
    boundary_union = int(np.logical_or(pred_b, gt_b).sum())
    precision = pred_match_count / max(1, pred_boundary_count)
    recall = gt_match_count / max(1, gt_boundary_count)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    return {
        "boundary_precision": float(precision),
        "boundary_recall": float(recall),
        "boundary_f1": float(f1),
        "boundary_iou": _safe_rate(boundary_intersection, boundary_union),
        "boundary_pred_match": pred_match_count,
        "boundary_pred_pixels": pred_boundary_count,
        "boundary_gt_match": gt_match_count,
        "boundary_gt_pixels": gt_boundary_count,
        "boundary_intersection": boundary_intersection,
        "boundary_union": boundary_union,
    }


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    if ndi is not None:
        labels, count = ndi.label(mask)
        return labels.astype(np.int32), int(count)
    labels = np.zeros(mask.shape, dtype=np.int32)
    count = 0
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            count += 1
            labels[y, x] = count
            queue: deque[tuple[int, int]] = deque([(y, x)])
            while queue:
                cy, cx = queue.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = count
                        queue.append((ny, nx))
    return labels, count


def _required_overlap(area: int, min_overlap_pixels: int, min_overlap_ratio: float) -> int:
    return min(area, max(min_overlap_pixels, int(math.ceil(area * min_overlap_ratio))))


def _instance_gap_pixels(first: np.ndarray, second: np.ndarray, distance_from_first: np.ndarray | None = None) -> int:
    if np.any(first & second):
        return 0
    if ndi is not None:
        distances = distance_from_first if distance_from_first is not None else ndi.distance_transform_edt(~first)
        center_distance = float(np.min(distances[second]))
    else:  # pragma: no cover
        first_yx = np.argwhere(first)
        second_yx = np.argwhere(second)
        if not first_yx.size or not second_yx.size:
            return 0
        deltas = first_yx[:, None, :] - second_yx[None, :, :]
        center_distance = float(np.sqrt(np.square(deltas).sum(axis=2)).min())
    return max(0, int(math.floor(center_distance + 1e-9)) - 1)


def _pair_bucket(gap: int) -> str | None:
    if gap == 0:
        return "touching"
    if gap <= 2:
        return "1_2"
    if gap <= 5:
        return "3_5"
    if gap <= 10:
        return "6_10"
    return None


def _instance_label_map(gt_instances: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int, str], ...]]:
    masks = [np.asarray(mask, dtype=bool) for mask in gt_instances if np.any(mask)]
    if not masks:
        return np.zeros((0, 0), dtype=np.int32), np.zeros(0, dtype=np.int64), ()
    label_map = np.zeros(masks[0].shape, dtype=np.int32)
    areas: list[int] = []
    for idx, mask in enumerate(masks, start=1):
        label_map[mask] = idx
        areas.append(int(mask.sum()))
    pairs: list[tuple[int, int, str]] = []
    for first_idx in range(len(masks)):
        distance_from_first = ndi.distance_transform_edt(~masks[first_idx]) if ndi is not None else None
        for second_idx in range(first_idx + 1, len(masks)):
            bucket = _pair_bucket(
                _instance_gap_pixels(masks[first_idx], masks[second_idx], distance_from_first=distance_from_first)
            )
            if bucket is not None:
                pairs.append((first_idx, second_idx, bucket))
    return label_map, np.asarray(areas, dtype=np.int64), tuple(pairs)


def compute_instance_metrics_from_labels(
    pred: np.ndarray,
    label_map: np.ndarray,
    gt_areas: np.ndarray,
    pair_buckets: tuple[tuple[int, int, str], ...] = (),
    min_overlap_pixels: int = 8,
    min_overlap_ratio: float = 0.05,
) -> dict[str, float | int]:
    pred_labels, pred_count = label_components(pred)
    pred_areas = np.bincount(pred_labels.ravel(), minlength=pred_count + 1)[1:]
    gt_count = int(len(gt_areas))
    intersections = np.zeros((gt_count, pred_count), dtype=np.int64)

    if gt_count and pred_count:
        valid = (label_map > 0) & (pred_labels > 0)
        if np.any(valid):
            encoded = (label_map[valid].astype(np.int64) - 1) * pred_count + (pred_labels[valid].astype(np.int64) - 1)
            counts = np.bincount(encoded, minlength=gt_count * pred_count)
            intersections = counts.reshape(gt_count, pred_count)

    return _instance_metrics_from_intersections(
        pred_areas=pred_areas,
        intersections=intersections,
        gt_areas=gt_areas,
        pair_buckets=pair_buckets,
        min_overlap_pixels=min_overlap_pixels,
        min_overlap_ratio=min_overlap_ratio,
    )


def compute_instance_metrics_from_masks(
    pred_masks: np.ndarray,
    label_map: np.ndarray,
    gt_areas: np.ndarray,
    pair_buckets: tuple[tuple[int, int, str], ...] = (),
    min_overlap_pixels: int = 8,
    min_overlap_ratio: float = 0.05,
) -> dict[str, float | int]:
    """Evaluate query masks directly, without connected-component relabeling."""

    masks = np.asarray(pred_masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"pred_masks must have shape [N,H,W], got {masks.shape}")
    if tuple(masks.shape[-2:]) != tuple(label_map.shape):
        raise ValueError(
            f"pred_masks shape {masks.shape[-2:]} does not match ground truth {label_map.shape}"
        )
    pred_count = int(masks.shape[0])
    gt_count = int(len(gt_areas))
    pred_areas = (
        masks.reshape(pred_count, -1).sum(axis=1, dtype=np.int64)
        if pred_count
        else np.zeros(0, dtype=np.int64)
    )
    intersections = np.zeros((gt_count, pred_count), dtype=np.int64)
    for pred_idx, mask in enumerate(masks):
        labels = label_map[mask]
        labels = labels[labels > 0]
        if labels.size:
            intersections[:, pred_idx] = np.bincount(
                labels.astype(np.int64) - 1,
                minlength=gt_count,
            )[:gt_count]
    return _instance_metrics_from_intersections(
        pred_areas=pred_areas,
        intersections=intersections,
        gt_areas=gt_areas,
        pair_buckets=pair_buckets,
        min_overlap_pixels=min_overlap_pixels,
        min_overlap_ratio=min_overlap_ratio,
    )


def _instance_metrics_from_intersections(
    *,
    pred_areas: np.ndarray,
    intersections: np.ndarray,
    gt_areas: np.ndarray,
    pair_buckets: tuple[tuple[int, int, str], ...],
    min_overlap_pixels: int,
    min_overlap_ratio: float,
) -> dict[str, float | int]:
    pred_count = int(len(pred_areas))
    gt_count = int(len(gt_areas))
    if intersections.shape != (gt_count, pred_count):
        raise ValueError(
            f"intersection matrix must have shape {(gt_count, pred_count)}, got {intersections.shape}"
        )
    linked = np.zeros((gt_count, pred_count), dtype=bool)
    dominant_pred = np.zeros(gt_count, dtype=np.int32)
    if gt_count and pred_count:
        for gt_idx, area in enumerate(gt_areas):
            required = _required_overlap(int(area), min_overlap_pixels, min_overlap_ratio)
            linked[gt_idx] = intersections[gt_idx] >= required
            best_idx = int(np.argmax(intersections[gt_idx]))
            if intersections[gt_idx, best_idx] >= required:
                dominant_pred[gt_idx] = best_idx + 1

    gt_per_pred = linked.sum(axis=0) if pred_count else np.zeros(0, dtype=int)
    pred_per_gt = linked.sum(axis=1) if gt_count else np.zeros(0, dtype=int)
    merge_extra = int(np.maximum(0, gt_per_pred - 1).sum())
    split_extra = int(np.maximum(0, pred_per_gt - 1).sum())

    pair_counts: dict[str, dict[str, int]] = {name: {"total": 0, "separated": 0} for name in PAIR_BUCKETS}
    for first_idx, second_idx, bucket in pair_buckets:
        pair_counts[bucket]["total"] += 1
        first_pred = int(dominant_pred[first_idx])
        second_pred = int(dominant_pred[second_idx])
        if first_pred > 0 and second_pred > 0 and first_pred != second_pred:
            pair_counts[bucket]["separated"] += 1

    pq_candidates: list[tuple[float, int, int]] = []
    for gt_idx, gt_area in enumerate(gt_areas):
        for pred_idx in range(pred_count):
            inter = int(intersections[gt_idx, pred_idx])
            if not inter:
                continue
            pred_area = int(pred_areas[pred_idx])
            iou = inter / max(1, int(gt_area) + pred_area - inter)
            if iou > 0.5:
                pq_candidates.append((iou, gt_idx, pred_idx))
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    pq_iou_sum = 0.0
    for iou, gt_idx, pred_idx in sorted(pq_candidates, reverse=True):
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        pq_iou_sum += iou
    pq_tp = len(matched_gt)
    pq_fp = pred_count - pq_tp
    pq_fn = gt_count - pq_tp
    pq_denominator = pq_tp + 0.5 * pq_fp + 0.5 * pq_fn

    result: dict[str, float | int] = {
        "gt_instances": gt_count,
        "pred_components": pred_count,
        "pred_component_max_area": int(pred_areas.max()) if pred_areas.size else 0,
        "object_count_error": pred_count - gt_count,
        "object_count_abs_error": abs(pred_count - gt_count),
        "object_count_denominator": gt_count,
        "normalized_object_count_abs_error": _safe_rate(abs(pred_count - gt_count), gt_count),
        "merge_components": int(np.count_nonzero(gt_per_pred > 1)),
        "merge_extra_gt": merge_extra,
        "merge_denominator": gt_count,
        "merge_rate": _safe_rate(merge_extra, gt_count),
        "split_components": int(np.count_nonzero(pred_per_gt > 1)),
        "split_extra_pred": split_extra,
        "split_denominator": gt_count,
        "split_rate": _safe_rate(split_extra, gt_count),
        "pq_iou_sum": float(pq_iou_sum),
        "pq_tp": pq_tp,
        "pq_fp": pq_fp,
        "pq_fn": pq_fn,
        "pq": float(pq_iou_sum / pq_denominator) if pq_denominator else 1.0,
    }
    for bucket, values in pair_counts.items():
        result[f"close_pair_{bucket}_total"] = values["total"]
        result[f"close_pair_{bucket}_separated"] = values["separated"]
        result[f"close_pair_{bucket}_separation_recall"] = _safe_rate(values["separated"], values["total"])
    main_total = pair_counts["1_2"]["total"] + pair_counts["3_5"]["total"]
    main_separated = pair_counts["1_2"]["separated"] + pair_counts["3_5"]["separated"]
    result["close_pair_1_5_total"] = main_total
    result["close_pair_1_5_separated"] = main_separated
    result["close_pair_1_5_separation_recall"] = _safe_rate(main_separated, main_total)
    result["close_pair_separation_recall"] = _safe_rate(main_separated, main_total)
    return result


def compute_instance_metrics(
    pred: np.ndarray,
    gt_instances: list[np.ndarray],
    min_overlap_pixels: int = 8,
    min_overlap_ratio: float = 0.05,
) -> dict[str, float | int]:
    label_map, gt_areas, pair_buckets = _instance_label_map(gt_instances)
    if label_map.size == 0:
        label_map = np.zeros(np.asarray(pred).shape, dtype=np.int32)
    return compute_instance_metrics_from_labels(
        pred,
        label_map,
        gt_areas,
        pair_buckets,
        min_overlap_pixels=min_overlap_pixels,
        min_overlap_ratio=min_overlap_ratio,
    )


def _sum_rows(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(row.get(key, 0)) for row in rows))


def aggregate_deadhesion_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = _sum_rows(rows, "tp_pixels")
    fp = _sum_rows(rows, "fp_pixels")
    fn = _sum_rows(rows, "fn_pixels")
    tn = _sum_rows(rows, "tn_pixels")
    boundary_pred_match = _sum_rows(rows, "boundary_pred_match")
    boundary_pred_pixels = _sum_rows(rows, "boundary_pred_pixels")
    boundary_gt_match = _sum_rows(rows, "boundary_gt_match")
    boundary_gt_pixels = _sum_rows(rows, "boundary_gt_pixels")
    boundary_precision = _safe_match_rate(boundary_pred_match, boundary_pred_pixels, boundary_gt_pixels)
    boundary_recall = _safe_match_rate(boundary_gt_match, boundary_gt_pixels, boundary_pred_pixels)
    boundary_f1 = (
        2.0 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
        if boundary_precision + boundary_recall
        else 0.0
    )
    boundary_intersection = _sum_rows(rows, "boundary_intersection")
    boundary_union = _sum_rows(rows, "boundary_union")
    merge_extra = _sum_rows(rows, "merge_extra_gt")
    merge_denominator = _sum_rows(rows, "merge_denominator")
    split_extra = _sum_rows(rows, "split_extra_pred")
    split_denominator = _sum_rows(rows, "split_denominator")
    count_error = _sum_rows(rows, "object_count_abs_error")
    count_denominator = _sum_rows(rows, "object_count_denominator")
    pq_iou_sum = _sum_rows(rows, "pq_iou_sum")
    pq_tp = _sum_rows(rows, "pq_tp")
    pq_fp = _sum_rows(rows, "pq_fp")
    pq_fn = _sum_rows(rows, "pq_fn")

    fg_iou = _safe_rate(tp, tp + fp + fn)
    bg_iou = _safe_rate(tn, tn + fp + fn)
    summary: dict[str, float | int] = {
        "sample_count": len(rows),
        "tp_pixels": int(tp),
        "fp_pixels": int(fp),
        "fn_pixels": int(fn),
        "tn_pixels": int(tn),
        "fg_iou": fg_iou,
        "bg_iou": bg_iou,
        "miou": 0.5 * (fg_iou + bg_iou) if math.isfinite(fg_iou) and math.isfinite(bg_iou) else math.nan,
        "fg_dice": _safe_rate(2.0 * tp, 2.0 * tp + fp + fn),
        "fg_precision": _safe_rate(tp, tp + fp),
        "fg_recall": _safe_rate(tp, tp + fn),
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": boundary_f1,
        "boundary_iou": _safe_rate(boundary_intersection, boundary_union) if boundary_union else 1.0,
        "merge_extra_gt": int(merge_extra),
        "merge_denominator": int(merge_denominator),
        "merge_rate": _safe_rate(merge_extra, merge_denominator),
        "split_extra_pred": int(split_extra),
        "split_denominator": int(split_denominator),
        "split_rate": _safe_rate(split_extra, split_denominator),
        "object_count_abs_error": int(count_error),
        "object_count_denominator": int(count_denominator),
        "normalized_object_count_abs_error": _safe_rate(count_error, count_denominator),
        "pq_iou_sum": pq_iou_sum,
        "pq_tp": int(pq_tp),
        "pq_fp": int(pq_fp),
        "pq_fn": int(pq_fn),
        "pq": _safe_rate(pq_iou_sum, pq_tp + 0.5 * pq_fp + 0.5 * pq_fn),
    }
    for bucket in PAIR_BUCKETS:
        separated = _sum_rows(rows, f"close_pair_{bucket}_separated")
        total = _sum_rows(rows, f"close_pair_{bucket}_total")
        summary[f"close_pair_{bucket}_separated"] = int(separated)
        summary[f"close_pair_{bucket}_total"] = int(total)
        summary[f"close_pair_{bucket}_separation_recall"] = _safe_rate(separated, total)
    main_separated = _sum_rows(rows, "close_pair_1_5_separated")
    main_total = _sum_rows(rows, "close_pair_1_5_total")
    summary["close_pair_1_5_separated"] = int(main_separated)
    summary["close_pair_1_5_total"] = int(main_total)
    summary["close_pair_1_5_separation_recall"] = _safe_rate(main_separated, main_total)
    summary["close_pair_separation_recall"] = _safe_rate(main_separated, main_total)
    return summary


def build_grouped_summary(rows: list[dict[str, Any]], missing_predictions: int = 0) -> dict[str, Any]:
    by_scene: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_species: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row.get("scene_id", "unknown"))].append(row)
        by_species[str(row.get("species_domain", "unknown"))].append(row)
    return {
        "sample_count": len(rows),
        "missing_predictions": missing_predictions,
        "overall": aggregate_deadhesion_rows(rows),
        "by_scene": {key: aggregate_deadhesion_rows(group) for key, group in sorted(by_scene.items())},
        "by_species_domain": {key: aggregate_deadhesion_rows(group) for key, group in sorted(by_species.items())},
    }


def _as_batch_bool_masks(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (B,H,W) or (B,1,H,W), got {arr.shape}")
    if arr.dtype != np.bool_:
        finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
        unique = np.unique(finite)
        if np.any((unique != 0) & (unique != 1)):
            raise ValueError(f"{name} must contain binary masks, got values outside {{0,1}}")
    return arr.astype(bool, copy=False)


class CachedDeadhesionEvaluator:
    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str = "val",
        instances_coco: str | Path | None = None,
        boundary_width: int = 2,
        boundary_tolerance: int = 2,
        min_overlap_pixels: int = 8,
        min_overlap_ratio: float = 0.05,
        tile_ids: set[str] | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.boundary_width = int(boundary_width)
        self.boundary_tolerance = int(boundary_tolerance)
        self.min_overlap_pixels = int(min_overlap_pixels)
        self.min_overlap_ratio = float(min_overlap_ratio)
        coco_path = Path(instances_coco) if instances_coco else self.dataset_root / "instances" / "annotations.coco.json"
        target_root = self.dataset_root / "instances" / "targets"
        if not coco_path.exists() and not target_root.is_dir():
            raise FileNotFoundError(
                f"Instance annotations not found: expected {coco_path} or {target_root}"
            )
        self.instances_coco = coco_path if coco_path.exists() else None
        self.instance_target_root = target_root if target_root.is_dir() else None
        self._tile_filter = tile_ids
        coco_index = load_coco_index(coco_path) if coco_path.exists() else {}
        self._tiles = self._build_cache(coco_index)
        self._gt_boundaries: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.reset()

    @property
    def cached_tile_count(self) -> int:
        return len(self._tiles)

    def reset(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.prediction_kind = "unset"

    def _tile_boundary_metrics(self, pred: np.ndarray, tile: CachedTile) -> dict[str, float | int]:
        bands = self._gt_boundaries.get(tile.tile_id)
        if bands is None:
            if tile.gt_mask.any():
                boundary = boundary_band(tile.gt_mask, self.boundary_width)
                bands = (boundary, max_filter(boundary, self.boundary_tolerance))
            else:
                bands = (tile.gt_mask, tile.gt_mask)
            self._gt_boundaries[tile.tile_id] = bands
        return _boundary_metrics(pred, self.boundary_width, self.boundary_tolerance, *bands)

    def _build_cache(self, coco_index: dict[str, list[dict[str, Any]]]) -> dict[str, CachedTile]:
        tiles: dict[str, CachedTile] = {}
        for item in read_manifest(self.dataset_root, self.split):
            tile_id = str(item.get("tile_id") or Path(str(item["semantic"])).stem)
            if self._tile_filter is not None and tile_id not in self._tile_filter:
                continue
            target_path = (
                self.instance_target_root / f"{tile_id}.npz"
                if self.instance_target_root is not None
                else None
            )
            if target_path is not None and target_path.is_file():
                with np.load(target_path, allow_pickle=False) as payload:
                    label_map = np.asarray(payload["instance_map"], dtype=np.int32)
                    all_pairs = np.asarray(
                        payload.get("all_instance_pairs", payload["instance_pairs"]),
                        dtype=np.int32,
                    )
                valid_height = min(
                    int(label_map.shape[0]),
                    int(item.get("valid_height", label_map.shape[0])),
                )
                valid_width = min(
                    int(label_map.shape[1]),
                    int(item.get("valid_width", label_map.shape[1])),
                )
                label_map = label_map[:valid_height, :valid_width]
                gt_mask = label_map > 0
                gt_count = int(label_map.max())
                gt_areas = np.bincount(
                    label_map.ravel(), minlength=gt_count + 1
                )[1:].astype(np.int64, copy=False)
                names = ("touching", "1_2", "3_5", "6_10")
                pair_buckets = tuple(
                    (int(first) - 1, int(second) - 1, names[int(bucket)])
                    for first, second, bucket in all_pairs
                )
                tiles[tile_id] = CachedTile(
                    tile_id=tile_id,
                    split=str(item.get("_split", self.split)),
                    scene_id=str(item.get("scene_id", "")),
                    species_domain=str(item.get("species_domain", "")),
                    gt_mask=gt_mask.astype(bool, copy=False),
                    label_map=label_map.astype(np.int32, copy=False),
                    gt_areas=gt_areas,
                    pair_buckets=pair_buckets,
                )
                continue
            semantic_path = item.get("semantic")
            if semantic_path:
                gt_full = read_mask(self.dataset_root / str(semantic_path), threshold=0.0, foreground=False)
            else:
                if not bool(item.get("is_background", False)):
                    raise ValueError(f"Positive tile has no semantic mask: {tile_id}")
                height = int(item.get("valid_height") or item.get("tile_size") or 0)
                width = int(item.get("valid_width") or item.get("tile_size") or 0)
                if height <= 0 or width <= 0:
                    raise ValueError(f"Background tile has no valid shape: {tile_id}")
                gt_full = np.zeros((height, width), dtype=bool)
            valid_height = min(gt_full.shape[0], int(item.get("valid_height", gt_full.shape[0])))
            valid_width = min(gt_full.shape[1], int(item.get("valid_width", gt_full.shape[1])))
            gt_mask = gt_full[:valid_height, :valid_width]
            gt_instances = [
                mask[:valid_height, :valid_width]
                for mask in rasterize_coco_instances(coco_index.get(tile_id, []), shape=gt_full.shape)
            ]
            gt_instances = [mask for mask in gt_instances if np.any(mask)]
            if np.any(gt_mask) and not gt_instances:
                raise ValueError(f"Foreground tile has no COCO instances: {tile_id}")
            if gt_instances:
                label_map, gt_areas, pair_buckets = _instance_label_map(gt_instances)
            else:
                label_map = np.zeros(gt_mask.shape, dtype=np.int32)
                gt_areas = np.zeros(0, dtype=np.int64)
                pair_buckets = ()
            tiles[tile_id] = CachedTile(
                tile_id=tile_id,
                split=str(item.get("_split", self.split)),
                scene_id=str(item.get("scene_id", "")),
                species_domain=str(item.get("species_domain", "")),
                gt_mask=gt_mask.astype(bool, copy=False),
                label_map=label_map.astype(np.int32, copy=False),
                gt_areas=gt_areas,
                pair_buckets=pair_buckets,
            )
        return tiles

    def update_batch(self, pred_masks: Any, gt_masks: Any, tile_ids: list[str] | tuple[str, ...]) -> None:
        if self.prediction_kind not in {"unset", "connected_components"}:
            raise RuntimeError("Cannot mix connected-component and query-instance predictions")
        self.prediction_kind = "connected_components"
        preds = _as_batch_bool_masks(pred_masks, name="pred_masks")
        _ = _as_batch_bool_masks(gt_masks, name="gt_masks")
        if len(tile_ids) != preds.shape[0]:
            raise ValueError(f"tile_ids length {len(tile_ids)} does not match batch size {preds.shape[0]}")
        for idx, tile_id in enumerate(tile_ids):
            tile = self._tiles.get(str(tile_id))
            if tile is None:
                raise KeyError(f"Tile {tile_id!r} is not present in cached split {self.split!r}")
            pred = preds[idx]
            valid_height, valid_width = tile.gt_mask.shape
            if pred.shape[0] < valid_height or pred.shape[1] < valid_width:
                raise ValueError(f"Prediction for {tile_id} has shape {pred.shape}, smaller than valid shape {tile.gt_mask.shape}")
            pred = pred[:valid_height, :valid_width]
            row: dict[str, Any] = {
                "split": tile.split,
                "tile_id": tile.tile_id,
                "scene_id": tile.scene_id,
                "species_domain": tile.species_domain,
                "height": valid_height,
                "width": valid_width,
                "gt_foreground_pixels": int(tile.gt_mask.sum()),
                "pred_foreground_pixels": int(pred.sum()),
                "fg_ratio": float(tile.gt_mask.mean()),
                "pred_fg_ratio": float(pred.mean()),
            }
            row.update(binary_metrics(pred, tile.gt_mask))
            row.update(self._tile_boundary_metrics(pred, tile))
            row.update(
                compute_instance_metrics_from_labels(
                    pred,
                    tile.label_map,
                    tile.gt_areas,
                    tile.pair_buckets,
                    min_overlap_pixels=self.min_overlap_pixels,
                    min_overlap_ratio=self.min_overlap_ratio,
                )
            )
            self.rows.append(row)

    def update_instance_batch(
        self,
        predictions: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
        gt_masks: Any,
        tile_ids: list[str] | tuple[str, ...],
    ) -> None:
        """Update with independent query masks rather than a connected-component union."""

        if self.prediction_kind not in {"unset", "query_instances"}:
            raise RuntimeError("Cannot mix connected-component and query-instance predictions")
        self.prediction_kind = "query_instances"
        targets = _as_batch_bool_masks(gt_masks, name="gt_masks")
        if len(predictions) != len(tile_ids) or len(tile_ids) != targets.shape[0]:
            raise ValueError("Instance predictions, targets, and tile_ids must have equal batch length")
        for prediction, tile_id in zip(predictions, tile_ids, strict=True):
            tile = self._tiles.get(str(tile_id))
            if tile is None:
                raise KeyError(f"Tile {tile_id!r} is not present in cached split {self.split!r}")
            masks_value = prediction.get("masks")
            if isinstance(masks_value, torch.Tensor):
                pred_instances = masks_value.detach().cpu().numpy().astype(bool, copy=False)
            else:
                pred_instances = np.asarray(masks_value, dtype=bool)
            if pred_instances.ndim != 3:
                raise ValueError(
                    f"Prediction masks for {tile_id} must have shape [N,H,W], got {pred_instances.shape}"
                )
            valid_height, valid_width = tile.gt_mask.shape
            if pred_instances.shape[1] < valid_height or pred_instances.shape[2] < valid_width:
                raise ValueError(
                    f"Prediction for {tile_id} has shape {pred_instances.shape}, "
                    f"smaller than valid shape {tile.gt_mask.shape}"
                )
            pred_instances = pred_instances[:, :valid_height, :valid_width]
            pred_union = (
                pred_instances.any(axis=0)
                if int(pred_instances.shape[0])
                else np.zeros(tile.gt_mask.shape, dtype=bool)
            )
            row: dict[str, Any] = {
                "split": tile.split,
                "tile_id": tile.tile_id,
                "scene_id": tile.scene_id,
                "species_domain": tile.species_domain,
                "height": valid_height,
                "width": valid_width,
                "gt_foreground_pixels": int(tile.gt_mask.sum()),
                "pred_foreground_pixels": int(pred_union.sum()),
                "fg_ratio": float(tile.gt_mask.mean()),
                "pred_fg_ratio": float(pred_union.mean()),
            }
            row.update(binary_metrics(pred_union, tile.gt_mask))
            row.update(self._tile_boundary_metrics(pred_union, tile))
            row.update(
                compute_instance_metrics_from_masks(
                    pred_instances,
                    tile.label_map,
                    tile.gt_areas,
                    tile.pair_buckets,
                    min_overlap_pixels=self.min_overlap_pixels,
                    min_overlap_ratio=self.min_overlap_ratio,
                )
            )
            row["zero_overlap_predictions"] = int(
                np.count_nonzero(~np.any(pred_instances & tile.gt_mask, axis=(1, 2)))
            )
            self.rows.append(row)

    def summary(self) -> dict[str, Any]:
        summary = build_grouped_summary(self.rows, missing_predictions=0)
        summary["cached_tile_count"] = self.cached_tile_count
        summary["settings"] = {
            "split": self.split,
            "threshold": 0.5,
            "boundary_width": self.boundary_width,
            "boundary_tolerance": self.boundary_tolerance,
            "min_overlap_pixels": self.min_overlap_pixels,
            "min_overlap_ratio": self.min_overlap_ratio,
            "prediction_kind": self.prediction_kind,
            "close_pair_bins_pixels": {"touching": [0, 0], "1_2": [1, 2], "3_5": [3, 5], "6_10": [6, 10]},
            "main_close_pair_range_pixels": [1, 5],
            "instances_source": (
                str(self.instances_coco)
                if self.instances_coco is not None
                else str(self.instance_target_root)
            ),
        }
        return summary

    @staticmethod
    def flat_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
        overall = summary["overall"]
        return {
            "boundary_f1": float(overall["boundary_f1"]),
            "boundary_iou": float(overall["boundary_iou"]),
            "merge_rate": float(overall["merge_rate"]),
            "split_rate": float(overall["split_rate"]),
            "close_pair_recall": float(overall["close_pair_separation_recall"]),
            "count_nmae": float(overall["normalized_object_count_abs_error"]),
            "pq": float(overall["pq"]),
        }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
