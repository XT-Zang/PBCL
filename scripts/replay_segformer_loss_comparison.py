"""Select recall-matched thresholds on Val and replay frozen Test probabilities."""
from collections import defaultdict
import json
from statistics import mean, stdev

import numpy as np

from geoseg.experiments.segformer_loss_comparison import DATASETS, OUTPUT, SEEDS
from geoseg.metrics.deadhesion import CachedDeadhesionEvaluator
from geoseg.runtime import load_experiment

FIXED = (0.3, 0.4, 0.5, 0.6, 0.7)
GRID = tuple(i / 100 for i in range(5, 96))


def predictions(root, evaluator):
    for tile_id, tile in evaluator._tiles.items():
        image = np.load(root / f"{tile_id}_prob.npy", allow_pickle=False)
        h, w = tile.gt_mask.shape
        if image.dtype != np.float32 or image.ndim != 2 or image.shape[0] < h or image.shape[1] < w:
            raise ValueError(f"Invalid FP32 prediction for {tile_id}")
        values = image[:h, :w]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite probabilities for {tile_id}")
        yield tile, values


def recall_grid(root, evaluator):
    thresholds = np.asarray(GRID, dtype=np.float32)
    histogram = np.zeros(len(GRID) + 1, dtype=np.int64)
    for tile, scores in predictions(root, evaluator):
        ranks = np.searchsorted(thresholds, scores[tile.gt_mask], side="left")
        histogram += np.bincount(ranks, minlength=len(histogram))
    return np.cumsum(histogram[:0:-1])[::-1] / histogram.sum()


def score_predictions(root, evaluator, thresholds):
    evaluators = {}
    from copy import copy
    for value in thresholds:
        cloned = copy(evaluator)
        cloned.reset()
        evaluators[value] = cloned
    for tile, scores in predictions(root, evaluator):
        for threshold, runner in evaluators.items():
            runner.update_batch((scores > threshold)[None], tile.gt_mask[None], [tile.tile_id])
    return {threshold: runner.summary()["overall"] for threshold, runner in evaluators.items()}


def replay_comparison(refs):
    from scripts.run_segformer_loss_comparison import (
        THRESHOLD_METHODS, config, ensure_evaluation, evaluation_checkpoint, progress,
    )
    from scripts.analyze_pbcl_revision import metric_values, write_csv
    from scripts.run_pbcl_revision import write_json

    output = OUTPUT / "analysis"
    selected, replay = [], []
    for dataset in DATASETS:
        spec = load_experiment(config(dataset, "boundary"), build=False).spec
        root = spec.data["root"]
        val = CachedDeadhesionEvaluator(root, split="val")
        for seed in SEEDS:
            cutoff = None
            for method in THRESHOLD_METHODS:
                checkpoint = evaluation_checkpoint(dataset, method, seed, refs)
                summary = ensure_evaluation(dataset, method, seed, checkpoint, "val", True)
                recall = recall_grid(summary.parent / "val/probs", val)
                if method == "boundary":
                    cutoff = float(recall[GRID.index(0.5)])
                index = min(range(len(GRID)), key=lambda i: (abs(float(recall[i]) - cutoff),
                                                             abs(i - GRID.index(0.5)), i))
                selected.append({"dataset": dataset, "method": method, "seed": seed, "threshold": GRID[index],
                                 "val_target_recall": cutoff, "val_selected_recall": float(recall[index]),
                                 "val_recall_gap": float(recall[index]) - cutoff,
                                 "checkpoint_sha256": json.loads(summary.read_text())["checkpoint_sha256"]})
    write_json(output / "frozen_recall_thresholds.json", selected)

    for dataset in DATASETS:
        spec = load_experiment(config(dataset, "boundary"), build=False).spec
        test = CachedDeadhesionEvaluator(spec.data["root"], split="test")
        for record in (r for r in selected if r["dataset"] == dataset):
            method, seed = record["method"], record["seed"]
            checkpoint = evaluation_checkpoint(dataset, method, seed, refs)
            summary = ensure_evaluation(dataset, method, seed, checkpoint, "test", True)
            progress("threshold_replay", dataset=dataset, method=method, seed=seed)
            scores = score_predictions(summary.parent / "test/probs", test, sorted(set((*FIXED, record["threshold"]))))
            official = json.loads(summary.read_text())["deadhesion"]["overall"]
            for key, value in metric_values(official).items():
                np.testing.assert_allclose(metric_values(scores[0.5])[key], value, atol=1e-10)
            for threshold, overall in scores.items():
                replay.append({"dataset": dataset, "method": method, "seed": seed, "threshold": threshold,
                               "fixed": threshold in FIXED, "recall_matched": threshold == record["threshold"],
                               **metric_values(overall), "fg_recall": overall["fg_recall"]})
    write_csv(output / "threshold_replay.csv", replay)
    fixed_groups = defaultdict(list)
    for row in replay:
        if row["fixed"]:
            for metric in (*metric_values(official), "fg_recall"):
                fixed_groups[row["dataset"], row["method"], row["threshold"], metric].append(100 * row[metric])
    write_csv(output / "fixed_threshold_test_mean_sd.csv", [
        {"dataset": dataset, "method": method, "threshold": threshold, "metric": metric,
         "mean_pct": mean(values), "sd_pp": stdev(values)}
        for (dataset, method, threshold, metric), values in fixed_groups.items()
    ])
    matched = {(r["dataset"], r["seed"]): r for r in replay if r["recall_matched"] and r["method"] == "boundary"}
    pairs = defaultdict(list)
    metrics = tuple(metric_values(official)) + ("fg_recall",)
    for row in replay:
        if row["recall_matched"] and row["method"] != "boundary":
            base = matched[row["dataset"], row["seed"]]
            for metric in metrics:
                pairs[row["dataset"], row["method"], metric].append(100 * (row[metric] - base[metric]))
    write_csv(output / "recall_matched_test_mean_sd.csv", [
        {"dataset": dataset, "method": method, "metric": metric, "mean_delta_pp": mean(values),
         "sd_pp": stdev(values)} for (dataset, method, metric), values in pairs.items()
    ])
    audits = []
    for dataset in DATASETS:
        for method in ("wprf_gt", "wprf_valid"):
            for seed in SEEDS:
                directory = OUTPUT / "train" / dataset / method / f"seed{seed}/wprf_gradient_audit"
                for path in sorted(directory.glob("epoch_*.json")):
                    record = json.loads(path.read_text())
                    for scale, values in record["scales"].items():
                        audits.append({"dataset": dataset, "method": method, "seed": seed,
                                       "epoch": record["epoch"], "scale": int(scale), **values})
    write_csv(output / "wprf_negative_audit.csv", audits)
    (output / "reviewer_response_experiment_update.md").write_text(
        "# Structural-loss experiments in the revision\n\n"
        "## Associate Editor and Reviewer 1 on SPLIT\n\n"
        "We obtained the published SPLIT chapter and implemented the boundary penalty in Eq. (1). "
        "We compared Boundary, Boundary plus PBCL, and Boundary plus the SPLIT penalty at weight 0.1 "
        "with SegFormer-B2 on the Aqua spatial holdout and Vegas buildings using seeds 42–44. "
        "The common Boundary objective provides the edge head. The original SPLIT chapter reports "
        "a BCE-only combination in Eq. (15) and a separate contour-head combination in Eq. (16). "
        "The [fixed-threshold comparison](test_mean_sd.csv) reports foreground overlap and recall, "
        "separation, fragmentation, PQ and count error. The [threshold replay](threshold_replay.csv) "
        "reports the same outcomes across five fixed cutoffs. The [recall-matched comparison]"
        "(recall_matched_test_mean_sd.csv) freezes each operating point on Val against Boundary at 0.5 "
        "and reports the remaining recall gap on Test.\n\n"
        "## Reviewer 3 on WPRF\n\n"
        "We audited the source sampler, edge loss and max-min reachability against the author code "
        "at commit 418e362b440df08e72ae6da54abeec6559a8f442. The comparison follows Table I for the "
        "grid stride, offsets and scales and uses the shared Boundary segmentation objective. "
        "The GT and valid adaptations differ in the propagation domain of negative queries. "
        "For each dataset and seed, the [negative-term audit](wprf_negative_audit.csv) reports "
        "the number and proportion of budget-reachable queries and the isolated gradient at epochs "
        "5, 20 and 40. The [source parity check](../preflight/upstream_parity.json) records the "
        "reachability and gradient comparisons across all five scales.\n"
    )
