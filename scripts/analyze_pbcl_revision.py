#!/usr/bin/env python3
"""Summarize frozen revision predictions and validation-only selection sensitivity."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from geoseg.experiments.pbcl_revision import DATA_ROOT, OUTPUT_ROOT
from geoseg.metrics.aquadataset import AQUADATASET_OVERALL_SCORE_WEIGHTS
from geoseg.metrics.deadhesion import CachedDeadhesionEvaluator, aggregate_deadhesion_rows, json_ready
from scripts.run_pbcl_revision import runs, write_json


METRICS = tuple(AQUADATASET_OVERALL_SCORE_WEIGHTS)
ERRORS = {"merge_rate", "split_rate", "count_nmae"}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_ready(rows))


def metric_values(overall: dict) -> dict:
    values = {"fg_iou": overall["fg_iou"], "boundary_f1": overall["boundary_f1"], "pq": overall["pq"],
            "close_pair_recall": overall["close_pair_separation_recall"], "merge_rate": overall["merge_rate"],
            "split_rate": overall["split_rate"], "count_nmae": overall["normalized_object_count_abs_error"]}
    return {key: float(value) if value is not None else np.nan for key, value in values.items()}


def weight_sensitivity() -> list[dict]:
    weights = {"original": dict(AQUADATASET_OVERALL_SCORE_WEIGHTS), "equal": {key: 1 / 7 for key in METRICS},
               "fg_iou_only": {"fg_iou": 1.0}, "pq_only": {"pq": 1.0}}
    for removed in METRICS:
        remaining = {key: value for key, value in AQUADATASET_OVERALL_SCORE_WEIGHTS.items() if key != removed}
        total = sum(remaining.values())
        weights[f"without_{removed}"] = {key: value / total for key, value in remaining.items()}
    selected = []
    histories = []
    for method, seed in runs():
        # The formal logger supplies the same seven values used for selection.
        metrics_path = ROOT / "lightning_logs/pbcl_revision" / method / f"seed{seed}/version_0/metrics.csv"
        by_epoch = {}
        with metrics_path.open() as handle:
            for row in csv.DictReader(handle):
                if row.get("val_overall_score"):
                    epoch = int(float(row["epoch"])) + 1
                    by_epoch[epoch] = {key: float(row[f"val_{key}"]) for key in METRICS}
                    by_epoch[epoch]["original_score"] = float(row["val_overall_score"])
        histories.extend({"method": method, "seed": seed, "epoch": epoch, **values}
                         for epoch, values in sorted(by_epoch.items()))
        epochs = [{"epoch": epoch, **values} for epoch, values in sorted(by_epoch.items()) if epoch >= 10]
        if len(epochs) != 31:
            raise RuntimeError(f"Expected Val epochs 10–40 for {method} seed {seed}, found {len(epochs)}")
        for label, scheme in weights.items():
            if label == "original":
                scores = [row["original_score"] for row in epochs]
            else:
                scores = [sum(weight * (1 - min(max(row[key], 0), 1) if key in ERRORS else row[key])
                              for key, weight in scheme.items()) for row in epochs]
            best = epochs[max(range(len(scores)), key=scores.__getitem__)]
            if label == "original":
                evaluation = OUTPUT_ROOT / "evaluation" / method / f"seed{seed}/val/val_summary.json"
                saved_epoch = int(json.loads(evaluation.read_text())["checkpoint_metadata"]["epoch"]) + 1
                if saved_epoch != best["epoch"]:
                    raise RuntimeError(f"Recorded validation maximum differs from selected checkpoint for {method} seed {seed}")
            selected.append({"method": method, "seed": seed, "scheme": label, "epoch": best["epoch"],
                             "validation_score": max(scores), **{key: best[key] for key in METRICS}})
    write_csv(OUTPUT_ROOT / "analysis/selection_sensitivity.csv", selected)
    write_csv(OUTPUT_ROOT / "analysis/validation_history.csv", histories)
    return selected


def threshold_scan(method: str, seed: int, split: str, evaluator: CachedDeadhesionEvaluator) -> list[dict]:
    base = OUTPUT_ROOT / "evaluation" / method / f"seed{seed}" / split
    if not (base / "evaluation_complete.json").exists():
        raise RuntimeError(f"Missing frozen evaluation: {base}")
    probabilities = base / split / "probs"
    result = []
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        evaluator.reset()
        for tile_id, tile in evaluator._tiles.items():
            path = probabilities / f"{tile_id}_prob.npy"
            probability = np.load(path, allow_pickle=False)
            if probability.dtype != np.float32:
                raise ValueError(f"Threshold analysis requires FP32 probabilities: {path}")
            height, width = tile.gt_mask.shape
            prediction = probability[:height, :width] > threshold
            evaluator.update_batch(prediction[None], tile.gt_mask[None], [tile_id])
        summary = evaluator.summary()
        result.append({"method": method, "seed": seed, "split": split, "threshold": threshold,
                       "group": "overall", **metric_values(summary["overall"])})
        for sensor in ("GF1", "GF6"):
            rows = [row for row in evaluator.rows if row["scene_id"].startswith(sensor)]
            overall = aggregate_deadhesion_rows(rows)
            result.append({"method": method, "seed": seed, "split": split, "threshold": threshold,
                           "group": sensor, **metric_values(overall), "samples": len(rows)})
        if threshold == 0.5:
            frozen = json.loads((base / f"{split}_summary.json").read_text())
            metrics = metric_values(frozen["deadhesion"]["overall"])
            for key, value in metrics.items():
                if not np.isclose(value, result[-3][key], atol=1e-10, rtol=1e-10, equal_nan=True):
                    raise RuntimeError(f"Cached threshold-0.5 replay differs for {method}/{seed}/{split}/{key}")
    return result


def factorial_interactions(records: list[dict]) -> tuple[list[dict], list[dict]]:
    indexed = {(row["method"], row["seed"]): row for row in records}
    per_seed, summary = [], []
    for metric in METRICS:
        values = []
        for seed in (42, 43, 44):
            base = indexed["base", seed][metric]
            boundary = indexed["boundary", seed][metric]
            pbcl_only = indexed["pbcl_only", seed][metric]
            full = indexed["full", seed][metric]
            interaction = 100 * ((full - boundary) - (pbcl_only - base))
            values.append(interaction)
            per_seed.append({"metric": metric, "seed": seed, "interaction_pp": interaction})
        summary.append({"metric": metric, "seeds": len(values), "mean_interaction_pp": statistics.mean(values),
                        "sample_sd_pp": statistics.stdev(values)})
    return per_seed, summary


def summary_tables() -> list[dict]:
    records, groups = [], []
    for method, seed in runs():
        path = OUTPUT_ROOT / "evaluation" / method / f"seed{seed}/test/test_summary.json"
        content = json.loads(path.read_text())
        if content["sample_count"] != 563 or content["threshold"] != 0.5 or content["deadhesion"]["overall"]["close_pair_1_5_total"] != 170:
            raise RuntimeError(f"Formal Test protocol or DCP denominator mismatch: {path}")
        records.append({"method": method, "seed": seed, **metric_values(content["deadhesion"]["overall"])})
        sample_path = path.parent / "test/deadhesion_sample_metrics.csv"
        rows = []
        for row in csv.DictReader(sample_path.open()):
            rows.append({key: float(value) if value and key not in {"split", "tile_id", "scene_id", "species_domain"} else value
                         for key, value in row.items()})
        for sensor in ("GF1", "GF6"):
            subset = [row for row in rows if row["scene_id"].startswith(sensor)]
            overall = aggregate_deadhesion_rows(subset)
            groups.append({"method": method, "seed": seed, "sensor": sensor, **metric_values(overall),
                           "gap_1_2_total": overall["close_pair_1_2_total"],
                           "gap_1_2_dcp_recall": overall["close_pair_1_2_separation_recall"],
                           "gap_3_5_total": overall["close_pair_3_5_total"],
                           "gap_3_5_dcp_recall": overall["close_pair_3_5_separation_recall"]})
    summary = []
    for method in dict.fromkeys(row["method"] for row in records):
        selected = [row for row in records if row["method"] == method]
        for key in METRICS:
            values = [row[key] for row in selected]
            summary.append({"method": method, "metric": key, "seeds": len(values), "mean": statistics.mean(values),
                            "sample_sd": statistics.stdev(values) if len(values) > 1 else None})
    paired, paired_seeds = [], []
    for treatment, reference in (("full", "boundary"), ("full", "pbcl_only"), ("pbcl_only", "base"), ("boundary", "base"),
                                 ("no_four_contact", "full"), ("noncontact", "full"),
                                 ("noncontact", "no_four_contact"),
                                 ("full", "wprf_gt"), ("full", "wprf_valid")):
        a = {row["seed"]: row for row in records if row["method"] == treatment}
        b = {row["seed"]: row for row in records if row["method"] == reference}
        common = sorted(a.keys() & b.keys())
        if not common:
            continue
        for key in METRICS:
            values = [a[seed][key] - b[seed][key] for seed in common]
            paired_seeds.extend({"treatment": treatment, "reference": reference, "metric": key,
                                 "seed": seed, "delta_pp": 100 * value} for seed, value in zip(common, values))
            paired.append({"treatment": treatment, "reference": reference, "metric": key, "seeds": len(common),
                           "mean_delta_pp": 100 * statistics.mean(values),
                           "sample_sd_pp": 100 * statistics.stdev(values) if len(values) > 1 else None})
    for name, rows in (("test_runs", records), ("test_mean_sd", summary), ("test_paired_deltas", paired),
                       ("test_paired_seeds", paired_seeds), ("test_sensor_gap", groups)):
        write_csv(OUTPUT_ROOT / "analysis" / f"{name}.csv", rows)
    factorial_seeds, factorial_summary = factorial_interactions(records)
    write_csv(OUTPUT_ROOT / "analysis/test_factorial_seeds.csv", factorial_seeds)
    write_csv(OUTPUT_ROOT / "analysis/test_factorial_interactions.csv", factorial_summary)
    return records


def make_figures(thresholds: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ("fg_iou", "merge_rate", "split_rate", "close_pair_recall")
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.3), constrained_layout=True)
    for axis, key in zip(axes, metrics):
        for method in ("boundary", "full"):
            x, y, spread = [], [], []
            for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
                values = [row[key] * 100 for row in thresholds if row["method"] == method and row["split"] == "test"
                          and row["group"] == "overall" and row["threshold"] == threshold]
                x.append(threshold)
                y.append(statistics.mean(values))
                spread.append(statistics.stdev(values))
            axis.errorbar(x, y, yerr=spread, marker="o", capsize=2, label=method.title())
        axis.set_xlabel("Prediction threshold")
        axis.set_ylabel(key + " (%)")
        axis.grid(alpha=0.2)
    axes[-1].legend()
    figure.savefig(OUTPUT_ROOT / "analysis/threshold_sensitivity.pdf")
    figure.savefig(OUTPUT_ROOT / "analysis/threshold_sensitivity.png", dpi=180)
    plt.close(figure)


def analyze() -> None:
    from scripts.profile_pbcl_revision import profile
    from scripts.run_pbcl_revision import snapshot

    snapshot("analysis_source_snapshot")
    write_json(OUTPUT_ROOT / "progress.json", {"stage": "analysis", "detail": "summaries and frozen-probability replay"})
    records = summary_tables()
    selection = weight_sensitivity()
    threshold_rows = []
    for split in ("val", "test"):
        evaluator = CachedDeadhesionEvaluator(DATA_ROOT, split=split)
        for method in ("boundary", "full"):
            for seed in (42, 43, 44):
                threshold_rows.extend(threshold_scan(method, seed, split, evaluator))
                print(f"Threshold replay complete {method} seed {seed} {split}", flush=True)
    write_csv(OUTPUT_ROOT / "analysis/threshold_sensitivity.csv", threshold_rows)
    make_figures(threshold_rows)
    if not (OUTPUT_ROOT / "analysis/profile/complete.json").exists():
        write_json(OUTPUT_ROOT / "progress.json", {"stage": "profile", "detail": "matched Boundary/Full training-step timings"})
        profile()
    write_json(OUTPUT_ROOT / "analysis_complete.json", {"formal_test_runs": len(records), "selection_comparisons": len(selection),
               "threshold_rows": len(threshold_rows), "primary_threshold": 0.5,
               "selection_source": "Val epochs 10–40 only", "test_role": "frozen retrospective sensitivity, no retuning"})
    write_json(OUTPUT_ROOT / "progress.json", {"stage": "complete", "formal_test_runs": len(records)})


if __name__ == "__main__":
    analyze()
