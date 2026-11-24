#!/usr/bin/env python3
"""Zero-overlap Test-corridor sliding-window mosaicking evaluation (Table V).

Implements the exact large-region mosaicking protocol reported in Section III-C
of the paper:
  * Edge-anchored 512x512 sliding windows at stride 384
  * Separable 2D triangular blending weights (1D floor 0.15, 2D min weight 0.0225)
  * Binary threshold p >= 0.50
  * Optional validation-frozen 64-px interior component filter (Post) that removes
    isolated open-water specks strictly smaller than 64 pixels while retaining
    all border-touching components
  * Out-of-corridor / NoData pixels zeroed prior to inference so no external
    context enters the receptive field
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import label as nd_label
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoseg.losses.pairwise_connectivity import PAIR_BUCKET_NAME  # noqa: E402
from geoseg.metrics.deadhesion import (  # noqa: E402
    binary_metrics,
    boundary_metrics,
    compute_instance_metrics_from_labels,
)


def evaluate_corridor_binary_prediction(
    pred_mask: np.ndarray,
    instance_map: np.ndarray,
    instance_pairs: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float | int]:
    """Compute Fg IoU, B-F1, PQ, Merge, Split, DCP Recall, and Count NMAE on a corridor."""
    pred_valid = np.asarray(pred_mask, dtype=bool) & np.asarray(valid_mask, dtype=bool)
    gt_labels = np.where(valid_mask, np.asarray(instance_map, dtype=np.int64), 0)
    gt_binary = gt_labels > 0

    unique_ids = [int(v) for v in np.unique(gt_labels) if int(v) > 0]
    remap = {old_id: idx + 1 for idx, old_id in enumerate(unique_ids)}
    dense_labels = np.zeros_like(gt_labels, dtype=np.int64)
    for old_id, new_id in remap.items():
        dense_labels[gt_labels == old_id] = new_id
    gt_areas = np.bincount(dense_labels.ravel(), minlength=len(unique_ids) + 1)[1:]

    pair_buckets: list[tuple[int, int, str]] = []
    for u, v, code in np.asarray(instance_pairs, dtype=np.int64).tolist():
        if int(u) in remap and int(v) in remap and int(code) in PAIR_BUCKET_NAME:
            pair_buckets.append((remap[int(u)], remap[int(v)], PAIR_BUCKET_NAME[int(code)]))

    pix = binary_metrics(pred_valid, gt_binary)
    bnd = boundary_metrics(pred_valid, gt_binary, width=2, tolerance=2)
    inst = compute_instance_metrics_from_labels(
        pred_valid,
        dense_labels,
        gt_areas,
        pair_buckets=tuple(pair_buckets),
    )
    return {**pix, **bnd, **inst}


def make_triangular_weight(window_size: int = 512, floor: float = 0.15) -> np.ndarray:
    """Return the separable 2D triangular blending weight matrix."""
    half = (window_size - 1) / 2.0
    coords = np.arange(window_size, dtype=np.float32)
    tri_1d = 1.0 - np.abs(coords - half) / max(half, 1.0)
    tri_1d = floor + (1.0 - floor) * tri_1d
    return np.outer(tri_1d, tri_1d).astype(np.float32)


def edge_anchored_starts(length: int, window: int, stride: int) -> list[int]:
    """Generate edge-anchored sliding-window start indices covering [0, length)."""
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def filter_interior_small_components(
    binary_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    min_area_px: int = 64,
) -> np.ndarray:
    """Validation-frozen Post filter: drop interior 4-connected components < min_area_px.

    Components touching the image boundary or the invalid-support boundary are
    always retained unchanged.
    """
    mask = np.asarray(binary_mask, dtype=bool)
    if min_area_px <= 1 or not np.any(mask):
        return mask.copy()

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    comp, num = nd_label(mask, structure=structure)
    if num == 0:
        return mask.copy()

    # Determine border pixels (image boundary or adjacent to invalid support).
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if valid_mask is not None:
        invalid = ~np.asarray(valid_mask, dtype=bool)
        if np.any(invalid):
            padded = np.pad(invalid, 1, mode="constant", constant_values=False)
            adj_invalid = (
                padded[:-2, 1:-1]
                | padded[2:, 1:-1]
                | padded[1:-1, :-2]
                | padded[1:-1, 2:]
            )
            border |= adj_invalid

    areas = np.bincount(comp.ravel())
    border_labels = set(np.unique(comp[border & (comp > 0)]).tolist())

    keep = np.zeros(num + 1, dtype=bool)
    for label_id in range(1, num + 1):
        if areas[label_id] >= min_area_px or label_id in border_labels:
            keep[label_id] = True
    return keep[comp]


@torch.inference_mode()
def mosaic_scene_probabilities(
    model: torch.nn.Module,
    image_chw: np.ndarray,
    valid_mask: np.ndarray,
    *,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    window_size: int = 512,
    stride: int = 384,
    tri_floor: float = 0.15,
    batch_size: int = 8,
    device: str = "cuda",
) -> np.ndarray:
    """Run zero-context sliding-window inference over a corridor scene."""
    from geoseg.models.aquav3_outputs import get_main_logits

    channels, height, width = image_chw.shape
    masked_image = np.where(valid_mask[None, :, :], image_chw.astype(np.float32), 0.0)
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(channels, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(channels, 1, 1)

    weight_2d = make_triangular_weight(window_size, floor=tri_floor)
    prob_sum = np.zeros((height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)

    ys = edge_anchored_starts(height, window_size, stride)
    xs = edge_anchored_starts(width, window_size, stride)
    windows = [
        (y, x)
        for y in ys
        for x in xs
        if np.any(valid_mask[y : y + window_size, x : x + window_size])
    ]

    model = model.to(device).eval()
    for start in range(0, len(windows), batch_size):
        batch_coords = windows[start : start + batch_size]
        patches = []
        for y, x in batch_coords:
            patch = masked_image[:, y : y + window_size, x : x + window_size]
            v = valid_mask[y : y + window_size, x : x + window_size]
            normed = np.where(v[None, :, :], (patch - mean_arr) / std_arr, 0.0)
            patches.append(normed)
        tensor = torch.from_numpy(np.stack(patches, axis=0)).to(device=device, dtype=torch.float32)
        logits = get_main_logits(model(tensor)).float()
        probs = torch.sigmoid(logits)[:, 0].cpu().numpy()
        for (y, x), prob in zip(batch_coords, probs, strict=True):
            prob_sum[y : y + window_size, x : x + window_size] += prob * weight_2d
            weight_sum[y : y + window_size, x : x + window_size] += weight_2d

    prob_map = np.zeros((height, width), dtype=np.float32)
    active = (weight_sum > 0) & valid_mask
    prob_map[active] = (prob_sum[active] / weight_sum[active]).astype(np.float32)
    return prob_map


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Zero-overlap Test-corridor sliding-window mosaicking evaluator"
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment config path")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint (.ckpt)")
    parser.add_argument("--image-tif", type=Path, required=True, help="4-band corridor GeoTIFF")
    parser.add_argument("--gt-instance-npz", type=Path, required=True, help="NPZ with instance_map, valid_mask, instance_pairs")
    parser.add_argument("--output-json", type=Path, required=True, help="Path to write Raw + Post metrics JSON")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--post-min-area", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import rasterio
    from geoseg.runtime import load_experiment
    from scripts.evaluate_aquav3_checkpoint import load_checkpoint_into_net

    experiment = load_experiment(args.config, build=True)
    net = experiment.components.model
    load_checkpoint_into_net(net, args.checkpoint)

    with rasterio.open(args.image_tif) as src:
        image_chw = src.read().astype(np.float32)
    scale = float(experiment.spec.data.get("raw_scale", 10000.0))
    image_chw = image_chw / scale

    gt_bundle = np.load(args.gt_instance_npz)
    instance_map = gt_bundle["instance_map"]
    valid_mask = gt_bundle["valid_mask"].astype(bool)
    instance_pairs = gt_bundle["instance_pairs"]

    prob_map = mosaic_scene_probabilities(
        net,
        image_chw,
        valid_mask,
        mean=tuple(experiment.spec.data["normalize_mean"]),
        std=tuple(experiment.spec.data["normalize_std"]),
        window_size=int(experiment.spec.data.get("image_size", 512)),
        stride=args.stride,
        batch_size=args.batch_size,
        device=args.device,
    )

    raw_pred = (prob_map >= args.threshold) & valid_mask
    post_pred = filter_interior_small_components(
        raw_pred, valid_mask, min_area_px=args.post_min_area
    )

    raw_metrics = evaluate_corridor_binary_prediction(
        raw_pred, instance_map, instance_pairs, valid_mask
    )
    post_metrics = evaluate_corridor_binary_prediction(
        post_pred, instance_map, instance_pairs, valid_mask
    )

    payload = {
        "protocol": {
            "window_size": int(experiment.spec.data.get("image_size", 512)),
            "stride": args.stride,
            "tri_floor": 0.15,
            "threshold": args.threshold,
            "post_min_interior_area_px": args.post_min_area,
        },
        "raw": raw_metrics,
        "post": post_metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
