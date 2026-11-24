#!/usr/bin/env python3
"""Qualify, train, evaluate and summarize the matched SegFormer-B2 loss matrix."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
from time import monotonic

import rasterio  # Load GDAL before the ML runtime.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoseg.datasets.aquadataset_ewc import EWC_MANIFEST_NAME, build_ewc_targets
from geoseg.datasets.geographic_holdout import sha256_file
from geoseg.datasets.segformer_wprf_targets import UPSTREAM
from geoseg.experiments.pbcl_revision import COMPARISON_ROOT
from geoseg.experiments.segformer_loss_comparison import DATASETS, METHODS, OUTPUT, SEEDS
from geoseg.runtime import load_experiment
from scripts.analyze_pbcl_revision import metric_values, write_csv
from scripts.run_pbcl_revision import write_json
from scripts.verify_pbcl_revision_wprf import verify_upstream

VEGAS_ARCHIVE = Path(os.environ.get("PBCL_VEGAS_ARCHIVE", str(ROOT / "outputs/vegas")))
REFERENCES = ("boundary", "full")
THRESHOLD_METHODS = (*REFERENCES, "split")


def config(dataset, method):
    if dataset == "aqua":
        name = f"segformer_b2_{method}.py"
        if method == "split":
            name = "segformer_b2_boundary_split.py"
        return ROOT / "configs/aquadataset" / name
    name = f"segformer_b2__{method}.py"
    if method == "split":
        name = "segformer_b2__boundary_split.py"
    return ROOT / "configs/vegas" / name


def progress(stage, **details):
    state = {"stage": stage, "time": datetime.now().astimezone().isoformat(), **details}
    write_json(OUTPUT / "progress.json", state)
    print(state, flush=True)


def command(options, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        subprocess.run([sys.executable, *options], cwd=ROOT, check=True, stdout=handle,
                       stderr=subprocess.STDOUT)


def read_json(path):
    return json.loads(path.read_text())


def archived_reference(dataset, method, seed):
    role = f"seed{seed}"
    if dataset == "aqua":
        root = COMPARISON_ROOT / "external"
        variant = f"segformer_b2_{method}"
        train = root / "train" / variant / role
        summary = root / "evaluation" / variant / role / "test/test_summary.json"
    else:
        variant = f"segformer_b2__{method}"
        train = VEGAS_ARCHIVE / "train_out/VegasDenseClosePBCL_v1/external_factorial" / variant / role
        summary = VEGAS_ARCHIVE / "eval_out/VegasDenseClosePBCL_v1/external_factorial" / variant / role / "best_overall/test_summary.json"

    checkpoint, = train.glob("*-best_overall.ckpt")
    archived = read_json(train / "resolved_config.json")
    expected = load_experiment(config(dataset, method), build=False, seed=seed).spec.to_dict()
    for group, fields in {
        "model": ("key", "pretrained_sha256", "external_boundary_head"),
        "data": ("split_hashes", "split_counts", "image_size", "normalize_mean", "normalize_std", "resize_mode"),
        "runtime": ("max_epochs", "seed", "precision"),
        "optimization": ("loss", "optimizer", "scheduler", "accumulate_grad_batches"),
        "loader": ("train_batch_size", "balanced_sampler", "drop_last", "sampler_generator_stream",
                   "loader_generator_streams"),
        "checkpoint": ("start_epoch", "monitor", "overall_score", "selection"),
    }.items():
        for field in fields:
            if archived[group].get(field) != expected[group].get(field):
                raise RuntimeError(f"{train} archived {group}.{field} differs from current protocol")
    payload = read_json(summary)
    digest = sha256_file(checkpoint)
    if (payload["checkpoint_sha256"] != digest or payload["threshold"] != 0.5
            or payload["sample_count"] != expected["data"]["split_counts"]["test"]):
        raise RuntimeError(f"Archived evaluation and reference checkpoint disagree: {summary}")
    return {"checkpoint": str(checkpoint), "sha256": digest, "summary": str(summary),
            "summary_sha256": sha256_file(summary)}


def prepare():
    refs = {}
    for dataset in DATASETS:
        for method in REFERENCES:
            for seed in SEEDS:
                refs[f"{dataset}/{method}/{seed}"] = archived_reference(dataset, method, seed)
        spec = load_experiment(config(dataset, "ewc"), build=False).spec
        root = Path(spec.data["ewc_weight_root"])
        if not (root / EWC_MANIFEST_NAME).is_file():
            build_ewc_targets(
                spec.data["root"], instance_target_root=spec.data["instance_target_root"],
                target_root=root, splits=spec.data["ewc_weight_splits"],
                use_manifest_valid_mask=spec.data.get("use_manifest_valid_mask", False),
                progress_every=500,
            )
    write_json(OUTPUT / "references.json", refs)
    return refs


def ensure_evaluation(dataset, method, seed, checkpoint, split, probabilities):
    base = OUTPUT / "evaluation" / dataset / method / f"seed{seed}" / split
    summary = base / f"{split}_summary.json"
    digest = sha256_file(checkpoint)
    count = load_experiment(config(dataset, method), build=False, seed=seed).spec.data["split_counts"][split]
    valid = False
    if summary.is_file():
        record = read_json(summary)
        valid = (record["checkpoint_sha256"] == digest and record["threshold"] == 0.5
                 and record["sample_count"] == count
                 and (not probabilities or len(list((base / split / "probs").glob("*_prob.npy"))) == count))
    if not valid:
        progress("evaluate", dataset=dataset, method=method, seed=seed, split=split)
        extra = ["--save-probabilities", "--probability-dtype", "float32"] if probabilities else []
        command(
            ["-m", "geoseg", "evaluate", "--config", str(config(dataset, method)), "--seed", str(seed),
             "--checkpoint", str(checkpoint), "--split", split, "--threshold", "0.5", "--out-dir",
             str(base), "--save-predictions", "--strict", *extra],
            OUTPUT / "logs" / dataset / method / f"seed{seed}_{split}.log",
        )
        record = read_json(summary)
        if (record["checkpoint_sha256"] != digest or record["sample_count"] != count
                or record["threshold"] != 0.5):
            raise RuntimeError(f"Evaluation metadata mismatch: {summary}")
        if probabilities and len(list((base / split / "probs").glob("*_prob.npy"))) != count:
            raise RuntimeError(f"Incomplete FP32 probabilities: {base}")
    return summary


def evaluation_checkpoint(dataset, method, seed, refs):
    if method in REFERENCES:
        return Path(refs[f"{dataset}/{method}/{seed}"]["checkpoint"])
    return OUTPUT / "train" / dataset / method / f"seed{seed}/best_overall.ckpt"


def train(dataset, method, seed):
    directory = OUTPUT / "train" / dataset / method / f"seed{seed}"
    final = directory / "training_complete.json"
    best = directory / "best_overall.ckpt"
    last = directory / "last.ckpt"
    if final.exists():
        record = read_json(final)
        if record["sha256"] != sha256_file(best) or record["last_sha256"] != sha256_file(last):
            raise RuntimeError(f"Completed training weights changed: {directory}")
        return record
    progress("train", dataset=dataset, method=method, seed=seed)
    start = monotonic()
    resume = ["--resume-checkpoint", str(last)] if last.exists() else []
    command(["-m", "geoseg", "train", "--config", str(config(dataset, method)), "--seed", str(seed), *resume],
            OUTPUT / "logs" / dataset / method / f"seed{seed}_train.log")
    if len(list((directory / "deadhesion_val").glob("epoch_*.json"))) != 40:
        raise RuntimeError(f"Expected 40 validation epochs: {directory}")
    record = {"sha256": sha256_file(best), "last_sha256": sha256_file(last),
              "wall_seconds": monotonic() - start, "resumed": bool(resume)}
    if method.startswith("wprf"):
        for epoch in (5, 20, 40):
            if not (directory / "wprf_gradient_audit" / f"epoch_{epoch:03}.json").exists():
                raise RuntimeError(f"Missing WPRF audit for {directory} epoch {epoch}")
    write_json(final, record)
    return record


def compare_report(refs, completed):
    rows = []
    for dataset in DATASETS:
        for method in (*REFERENCES, *METHODS):
            for seed in SEEDS:
                if method in REFERENCES:
                    path = Path(refs[f"{dataset}/{method}/{seed}"]["summary"])
                elif (dataset, method, seed) in completed:
                    path = OUTPUT / "evaluation" / dataset / method / f"seed{seed}/test/test_summary.json"
                else:
                    continue
                score = read_json(path)["deadhesion"]["overall"]
                rows.append({"dataset": dataset, "method": method, "seed": seed, **metric_values(score),
                             "fg_recall": score["fg_recall"]})
    directory = OUTPUT / "analysis"
    write_csv(directory / "test_per_seed.csv", rows)
    paired = []
    lookup = {(row["dataset"], row["method"], row["seed"]): row for row in rows}
    metrics = (*metric_values(read_json(Path(next(iter(refs.values()))["summary"]))["deadhesion"]["overall"]), "fg_recall")
    for row in rows:
        for ref in REFERENCES:
            if row["method"] == ref:
                continue
            base = lookup[row["dataset"], ref, row["seed"]]
            for metric in metrics:
                paired.append({key: row[key] for key in ("dataset", "method", "seed")} |
                              {"reference": ref, "metric": metric, "delta_pp": 100 * (row[metric] - base[metric])})
    write_csv(directory / "test_paired_deltas.csv", paired)
    paired_groups = {}
    for row in paired:
        key = tuple(row[field] for field in ("dataset", "method", "reference", "metric"))
        paired_groups.setdefault(key, []).append(row["delta_pp"])
    write_csv(directory / "test_paired_mean_sd.csv", [
        {"dataset": d, "method": m, "reference": ref, "metric": k,
         "mean_delta_pp": mean(values), "sd_pp": stdev(values)}
        for (d, m, ref, k), values in paired_groups.items() if len(values) == len(SEEDS)
    ])
    summaries = []
    for dataset in DATASETS:
        for method in (*REFERENCES, *METHODS):
            subset = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            if len(subset) == len(SEEDS):
                for metric in metrics:
                    values = [100 * row[metric] for row in subset]
                    summaries.append({"dataset": dataset, "method": method, "metric": metric,
                                      "mean_pct": mean(values), "sd_pp": stdev(values)})
    write_csv(directory / "test_mean_sd.csv", summaries)


def qualify():
    signature = source_signature()
    preflight = OUTPUT / "preflight"
    preflight.mkdir(parents=True, exist_ok=True)
    write_json(preflight / "upstream_parity.json", verify_upstream())
    from scripts.verify_segformer_loss_comparison import verify

    write_json(preflight / "comparison_parity.json", verify())
    for method in (*METHODS, "boundary"):
        for dataset in DATASETS:
            key = f"{dataset}__{method}"
            path = preflight / f"{key}.json"
            if path.exists() and read_json(path).get("signature") == signature:
                continue
            progress("profile", dataset=dataset, method=method)
            command(["scripts/profile_segformer_loss_comparison.py", str(config(dataset, method)), str(path)],
                    preflight / f"{key}.log")
            record = read_json(path)
            record["signature"] = signature
            write_json(path, record)
    from scripts.verify_segformer_loss_comparison import efficiency_review

    write_json(preflight / "efficiency_review.json", efficiency_review(preflight))
    return signature


def source_signature():
    paths = [Path(__file__), *sorted((ROOT / "configs/aquadataset").glob("segformer_b2_*.py")),
             *sorted((ROOT / "configs/vegas").glob("segformer_b2__*.py")),
             *sorted((ROOT / "geoseg").rglob("*.py")),
             ROOT / "scripts/profile_segformer_loss_comparison.py",
             ROOT / "scripts/verify_segformer_loss_comparison.py",
             ROOT / "scripts/replay_segformer_loss_comparison.py",
             ROOT / "scripts/evaluate_aquav3_checkpoint.py", ROOT / "train_supervision.py"]
    entries = {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}
    upstream = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain", "--", "src"], text=True).strip():
        raise RuntimeError("Pinned WPRF source has local modifications")
    entries["upstream"] = upstream
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GEOSEG_LIGHTNING_LOG_ROOT", str(OUTPUT / "lightning_logs"))
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    with (OUTPUT / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress("verify_references")
        refs = prepare()
        signature = qualify()
        if args.audit_only:
            progress("qualified")
            return
        protocol = OUTPUT / "protocol.json"
        if protocol.exists() and read_json(protocol)["signature"] != signature:
            raise RuntimeError("Campaign source differs from the locked formal protocol")
        queue = [(dataset, method, seed) for group in (METHODS[:3], METHODS[3:])
                 for seed in SEEDS for method in group for dataset in DATASETS]
        write_json(protocol, {"signature": signature, "queue": queue, "reference_runs": len(refs),
                              "threshold": "FP32 p > 0.5"})
        completed = set()
        for dataset, method, seed in queue:
            if source_signature() != signature:
                raise RuntimeError("Campaign source changed during execution")
            train(dataset, method, seed)
            checkpoint = evaluation_checkpoint(dataset, method, seed, refs)
            ensure_evaluation(dataset, method, seed, checkpoint, "test", method == "split")
            if method == "split":
                ensure_evaluation(dataset, method, seed, checkpoint, "val", True)
            completed.add((dataset, method, seed))
            compare_report(refs, completed)
            if len(completed) in (6, 18, 24, 36):
                records = {(d, m): read_json(OUTPUT / "train" / d / m / "seed42/training_complete.json")["wall_seconds"]
                           for d, m, s in completed if s == 42}
                remaining = queue[len(completed):]
                known_seconds = sum(records.get((d, m), 0) for d, m, _ in remaining)
                write_json(OUTPUT / "eta.json", {"completed": len(completed), "remaining": len(remaining),
                           "measured_remaining_hours": known_seconds / 3600,
                           "remaining_without_seed42_timing": sum((d, m) not in records for d, m, _ in remaining)})
            if len(completed) == 18:
                from scripts.replay_segformer_loss_comparison import replay_comparison

                replay_comparison(refs)
        progress("complete", new_runs=len(completed), reused=len(refs))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        progress("failed", error=str(error))
        raise
