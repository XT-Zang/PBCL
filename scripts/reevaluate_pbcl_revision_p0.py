#!/usr/bin/env python3
"""Replay frozen PBCL revision predictions for the two P0 comparisons."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean, stdev
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine

from geoseg.datasets.geographic_holdout import sha256_file
from geoseg.experiments.pbcl_revision import DATA_ROOT, OUTPUT_ROOT
from geoseg.metrics.deadhesion import (
    CachedDeadhesionEvaluator, _instance_label_map, aggregate_deadhesion_rows,
    compute_instance_metrics_from_labels, label_components, load_coco_index,
    rasterize_coco_instances,
)
from scripts.analyze_pbcl_revision import metric_values, write_csv
from scripts.run_pbcl_revision import write_json

METHODS = ("boundary", "full")
SEEDS = (42, 43, 44)
GRID = tuple(i / 100 for i in range(30, 71))
COUNTS = {"val": 879, "test": 563}


def frozen(method, seed, split, tiles):
    directory = OUTPUT_ROOT / "evaluation" / method / f"seed{seed}" / split
    stamp = json.loads((directory / "evaluation_complete.json").read_text())
    official = json.loads((directory / f"{split}_summary.json").read_text())
    if official["threshold"] != 0.5 or official["sample_count"] != len(tiles) or len(tiles) != COUNTS[split]:
        raise ValueError(f"Unexpected frozen evaluation protocol: {directory}")
    if official["checkpoint_sha256"] != stamp["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash disagrees with completion record: {directory}")
    probs = directory / split / "probs"
    if {p.name for p in probs.glob("*_prob.npy")} != {f"{tile_id}_prob.npy" for tile_id in tiles}:
        raise ValueError(f"Unexpected frozen probability coverage: {probs}")
    return probs, official


def prediction_source(method, seed, split, probs, official):
    return {"method": method, "seed": seed, "split": split,
            "checkpoint_sha256": official["checkpoint_sha256"], "probabilities": str(probs)}


def probability(probs, tile):
    path = probs / f"{tile.tile_id}_prob.npy"
    array = np.load(path, allow_pickle=False)
    height, width = tile.gt_mask.shape
    if (array.dtype != np.float32 or array.ndim != 2 or array.shape[0] < height
            or array.shape[1] < width or not np.isfinite(array).all()):
        raise ValueError(f"Invalid frozen probability map: {path}")
    return array[:height, :width]


def split_extra(pred, tile, required):
    """Replay the official GT-to-prediction link rule for Split."""
    labels, count = label_components(pred)
    valid = (tile.label_map > 0) & (labels > 0)
    if not valid.any():
        return 0
    gt_count = len(tile.gt_areas)
    keys = (tile.label_map[valid].astype(np.int64) - 1) * count + labels[valid] - 1
    overlaps = np.bincount(keys, minlength=gt_count * count).reshape(gt_count, count)
    return int(np.maximum((overlaps >= required[:, None]).sum(axis=1) - 1, 0).sum())


def sweep_val(tiles, method, seed):
    probs, official = frozen(method, seed, "val", tiles)
    thresholds = (0.5,) if method == "boundary" else GRID
    totals = np.zeros((len(thresholds), 3), dtype=np.int64)
    gt_count = sum(len(tile.gt_areas) for tile in tiles.values())
    for tile in tiles.values():
        scores = probability(probs, tile)
        required = np.minimum(tile.gt_areas, np.maximum(8, np.ceil(0.05 * tile.gt_areas)))
        for index, threshold in enumerate(thresholds):
            pred = scores > threshold
            totals[index, 0] += np.count_nonzero(pred & tile.gt_mask)
            totals[index, 1] += np.count_nonzero(pred | tile.gt_mask)
            if required.size and pred.any():
                totals[index, 2] += split_extra(pred, tile, required)
    rows = [
        {"method": method, "seed": seed, "threshold": threshold, "fg_iou": int(tp) / int(union),
         "split_rate": int(extra) / gt_count, "split_extra_pred": int(extra), "split_denominator": gt_count}
        for threshold, (tp, union, extra) in zip(thresholds, totals, strict=True)
    ]
    fixed = next(row for row in rows if row["threshold"] == 0.5)
    for key in ("fg_iou", "split_rate"):
        if not np.isclose(fixed[key], official["deadhesion"]["overall"][key], atol=1e-10, rtol=1e-10):
            raise ValueError(f"Val 0.5 replay mismatch for {method} seed {seed} {key}")
    print(f"Val sweep complete {method} seed {seed}", flush=True)
    return rows, prediction_source(method, seed, "val", probs, official)


def paired(rows, group_keys, metric_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in (*group_keys, "seed"))
        grouped.setdefault(key, {})[row["method"]] = row
    deltas = []
    for key, methods in grouped.items():
        if set(methods) != set(METHODS):
            continue
        deltas.append({
            **dict(zip((*group_keys, "seed"), key, strict=True)),
            **{metric: 100 * (methods["full"][metric] - methods["boundary"][metric]) for metric in metric_keys},
        })
    summaries = []
    for key in dict.fromkeys(tuple(row[k] for k in group_keys) for row in deltas):
        subset = [row for row in deltas if tuple(row[k] for k in group_keys) == key]
        if {row["seed"] for row in subset} != set(SEEDS):
            raise ValueError(f"Incomplete paired seeds for {key}")
        for metric in metric_keys:
            values = [row[metric] for row in subset]
            summaries.append({**dict(zip(group_keys, key, strict=True)), "metric": metric,
                              "mean_delta_pp": mean(values), "sample_sd_pp": stdev(values)})
    return deltas, summaries


def match(output, workers):
    evaluator = CachedDeadhesionEvaluator(DATA_ROOT, split="val")
    jobs = [(method, seed) for method in METHODS for seed in SEEDS]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        replays = list(pool.map(lambda job: sweep_val(evaluator._tiles, *job), jobs))
    sweeps = [row for rows, _ in replays for row in rows]
    provenance = [source for _, source in replays]
    write_csv(output / "val_grid.csv", sweeps)
    selection = []
    for seed in SEEDS:
        baseline = next(row for row in sweeps if row["seed"] == seed and row["method"] == "boundary")
        full = [row for row in sweeps if row["seed"] == seed and row["method"] == "full"]
        if any(row["split_denominator"] != baseline["split_denominator"] for row in full):
            raise ValueError(f"Split denominator mismatch for seed {seed}")
        target = baseline["split_extra_pred"]
        chosen = min(full, key=lambda row: (
            abs(row["split_extra_pred"] - target), abs(round(row["threshold"] * 100) - 50), row["threshold"],
        ))
        lower, upper = min(row["split_rate"] for row in full), max(row["split_rate"] for row in full)
        selection.append({"seed": seed, "boundary_threshold": 0.5, "full_threshold": chosen["threshold"],
                          "boundary_val_split": baseline["split_rate"], "full_val_split": chosen["split_rate"],
                          "full_val_split_min": lower, "full_val_split_max": upper,
                          "boundary_in_full_val_split_range": lower <= baseline["split_rate"] <= upper,
                          "boundary_val_fg_iou": baseline["fg_iou"], "full_val_fg_iou": chosen["fg_iou"]})
    write_csv(output / "val_selection.csv", selection)
    write_json(output / "frozen_operating_points.json", {
        "selection_data": "Val only", "grid": GRID,
        "rule": "Per seed, minimize the absolute difference in Split against Boundary@0.5; ties favor proximity to 0.5 then lower threshold.",
        "seeds": selection,
    })
    print("Val thresholds frozen; evaluating Test", flush=True)
    test = CachedDeadhesionEvaluator(DATA_ROOT, split="test")
    result = []
    for record in selection:
        seed = record["seed"]
        for method in METHODS:
            probs, official = frozen(method, seed, "test", test._tiles)
            provenance.append(prediction_source(method, seed, "test", probs, official))
            selected = record[f"{method}_threshold"]
            for threshold in dict.fromkeys((0.5, selected)):
                if threshold == 0.5:
                    metrics = metric_values(official["deadhesion"]["overall"])
                else:
                    test.reset()
                    for tile in test._tiles.values():
                        pred = probability(probs, tile) > threshold
                        test.update_batch(pred[None], tile.gt_mask[None], [tile.tile_id])
                    metrics = metric_values(aggregate_deadhesion_rows(test.rows))
                for label in ("fixed_0.5", "val_matched_split"):
                    if threshold == (0.5 if label == "fixed_0.5" else selected):
                        result.append({"condition": label, "seed": seed, "method": method,
                                       "threshold": threshold, **metrics})
    write_csv(output / "test_per_seed.csv", result)
    deltas, summary = paired(result, ("condition",), tuple(metrics))
    write_csv(output / "test_paired_deltas_pp.csv", deltas)
    write_csv(output / "test_summary_pp.csv", summary)
    write_json(output / "provenance.json", {
        "val_manifest_sha256": sha256_file(DATA_ROOT / "splits/val.jsonl"),
        "test_manifest_sha256": sha256_file(DATA_ROOT / "splits/test.jsonl"),
        "evaluations": provenance,
    })


def alternate_masks(annotations, shape):
    height, width = shape
    masks = []
    source_ids = []
    for ann in annotations:
        rings = [list(zip(poly[::2], poly[1::2])) for poly in ann["segmentation"] if len(poly) >= 6]
        geometries = [({"type": "Polygon", "coordinates": [ring]}, 1) for ring in rings]
        if not geometries:
            continue
        mask = rasterize(geometries, out_shape=(512, 512), transform=Affine.identity(),
                         all_touched=False).astype(bool)[:height, :width]
        if mask.any():
            masks.append(mask)
            source_ids.append(str(ann["source_instance_id"]))
    return masks, source_ids


def common_key(first, second, ids):
    return tuple(sorted((ids[first], ids[second])))


def rasterization(output):
    coco_path = DATA_ROOT / "instances/annotations.coco.json"
    index = load_coco_index(coco_path)
    splits = ("test",)
    results, pair_rows, provenance = [], [], []
    for split in splits:
        official_evaluator = CachedDeadhesionEvaluator(DATA_ROOT, split=split)
        alt = {}
        common_pairs = {}
        for tile_id, tile in official_evaluator._tiles.items():
            original_ids = []
            original_masks = []
            height, width = tile.gt_mask.shape
            for ann in index.get(tile_id, ()):
                masks = rasterize_coco_instances([ann], (512, 512))
                if masks and masks[0][:height, :width].any():
                    original_ids.append(str(ann["source_instance_id"]))
                    original_masks.append(masks[0][:height, :width])
            masks, ids = alternate_masks(index.get(tile_id, ()), tile.gt_mask.shape)
            if (len(original_ids) != len(tile.gt_areas) or len(set(original_ids)) != len(original_ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError(f"Cannot uniquely align reference instances for {tile_id}")
            labels_check, areas_check, pairs_check = _instance_label_map(original_masks)
            if original_ids and (not np.array_equal(labels_check, tile.label_map)
                                 or not np.array_equal(areas_check, tile.gt_areas) or pairs_check != tile.pair_buckets):
                raise ValueError(f"Archived PIL IDs do not replay for {tile_id}")
            labels, areas, pairs = _instance_label_map(masks)
            if not len(areas):
                labels = np.zeros(tile.gt_mask.shape, dtype=np.int32)
            alt[tile_id] = replace(tile, label_map=labels, gt_areas=areas, pair_buckets=pairs)
            buckets = {"1_2", "3_5"}
            pil = {common_key(a, b, original_ids): k for a, b, k in tile.pair_buckets if k in buckets}
            raster = {common_key(a, b, ids): k for a, b, k in pairs if k in buckets}
            mutual = pil.keys() & raster.keys()
            common_pairs[tile_id] = {
                name: tuple((a, b, k) for a, b, k in reference.pair_buckets
                            if k in buckets and common_key(a, b, source_ids) in mutual)
                for name, reference, source_ids in (
                    ("pil", tile, original_ids), ("pixel_center", alt[tile_id], ids),
                )
            }
            if any(len(pairs) != len(mutual) for pairs in common_pairs[tile_id].values()):
                raise ValueError(f"Common set denominator mismatch for {tile_id}")
            for pair in sorted(pil.keys() | raster.keys()):
                pair_rows.append({"split": split, "tile_id": tile_id, "first_source_id": pair[0],
                                  "second_source_id": pair[1], "pil_bucket": pil.get(pair, ""),
                                  "pixel_center_bucket": raster.get(pair, ""), "common_1_5": pair in mutual})
        alternate_evaluator = CachedDeadhesionEvaluator(DATA_ROOT, split=split, tile_ids=set())
        alternate_evaluator._tiles = alt
        # Keep the official semantic mask in both replays. Only the instance reference changes.
        for method in METHODS:
            for seed in SEEDS:
                probs, official = frozen(method, seed, split, official_evaluator._tiles)
                provenance.append(prediction_source(method, seed, split, probs, official))
                official_evaluator.reset()
                alternate_evaluator.reset()
                common_sums = {"pil": 0, "pixel_center": 0}
                common_total = 0
                for tile_id, tile in official_evaluator._tiles.items():
                    pred = probability(probs, tile) > 0.5
                    official_evaluator.update_batch(pred[None], tile.gt_mask[None], [tile_id])
                    alternate_evaluator.update_batch(pred[None], tile.gt_mask[None], [tile_id])
                    shared = common_pairs[tile_id]
                    if not shared["pil"]:
                        continue
                    common_total += len(shared["pil"])
                    for name, reference in (("pil", tile), ("pixel_center", alt[tile_id])):
                        counts = compute_instance_metrics_from_labels(
                            pred, reference.label_map, reference.gt_areas, shared[name],
                        )
                        if counts["close_pair_1_5_total"] != len(shared[name]):
                            raise ValueError(f"Common set denominator mismatch for {tile_id}")
                        common_sums[name] += counts["close_pair_1_5_separated"]
                for name, evaluator in (("pil", official_evaluator), ("pixel_center", alternate_evaluator)):
                    overall = aggregate_deadhesion_rows(evaluator.rows)
                    metrics = metric_values(overall)
                    if name == "pil":
                        frozen_metrics = metric_values(official["deadhesion"]["overall"])
                        for key in metrics:
                            if not np.isclose(metrics[key], frozen_metrics[key], atol=1e-10, rtol=1e-10):
                                raise ValueError(f"Official PIL replay mismatch for {method}/{seed}/{key}")
                    results.append({"split": split, "rasterizer": name, "method": method, "seed": seed,
                                    "pair_total": overall["close_pair_1_5_total"],
                                    "gt_instances": overall["object_count_denominator"],
                                    "common_pair_total": common_total,
                                    "common_separated": common_sums[name],
                                    "common_dcp_recall": common_sums[name] / common_total, **metrics})
                print(f"Dual rasterization complete {method} seed {seed}", flush=True)
    write_csv(output / "test_pair_cohort.csv", pair_rows)
    write_csv(output / "test_per_seed.csv", results)
    deltas, summary = paired(results, ("rasterizer",), (*metrics, "common_dcp_recall"))
    write_csv(output / "test_paired_deltas_pp.csv", deltas)
    write_csv(output / "test_summary_pp.csv", summary)
    write_json(output / "provenance.json", {"coco_sha256": sha256_file(coco_path),
               "test_manifest_sha256": sha256_file(DATA_ROOT / "splits/test.jsonl"),
               "semantic_reference": "Identical archived semantic mask in both evaluations",
               "common_cohort": "Same tile and unordered source IDs classified as 1–5 px under both rasterizers",
               "evaluations": provenance})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("match", "rasterization"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "match":
        match(args.output, args.workers)
    else:
        rasterization(args.output)
