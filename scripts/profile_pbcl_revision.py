#!/usr/bin/env python3
"""Profile matched revision treatments on real batches spanning pair densities."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from geoseg.datasets.aquav3_datamodule import AquaV3DataModule
from geoseg.datasets.geographic_holdout import sha256_file
from geoseg.experiments.pbcl_revision import OUTPUT_ROOT
from geoseg.experiments.reproducibility import seed_experiment
from geoseg.losses.pairwise_connectivity import _pair_crop_slices, pair_bottleneck_connectivity_loss
from geoseg.runtime import load_experiment
from scripts.analyze_pbcl_revision import write_csv
from scripts.run_pbcl_revision import write_json
from scripts.profile_aquadataset_training_overhead import tensor_sha


def prepare_batches() -> list[dict]:
    spec = load_experiment(ROOT / "configs/aquadataset/full.py", build=False, seed=42).spec
    spec = replace(spec, loader={**spec.loader, "train_workers": 0})
    seed_experiment(42)
    datamodule = AquaV3DataModule(spec)
    loader = datamodule.train_dataloader()
    dataset = datamodule.train_dataset
    pair_counts = []
    for row in dataset.entries:
        count = 0
        if not row["is_background"]:
            with np.load(dataset.instance_target_root / f"{row['tile_id']}.npz") as payload:
                count = len(payload["instance_pairs"])
        pair_counts.append(count)
    sampler_state = loader.sampler.generator.get_state()
    index_batches = list(loader.batch_sampler)
    loader.sampler.generator.set_state(sampler_state)
    counts = [sum(pair_counts[index] for index in batch) for batch in index_batches]
    nonzero = sorted(value for value in counts if value > 0)
    targets = {"zero": 0, "two": 2, "median_nonempty": float(np.median(nonzero)), "p90_nonempty": float(np.quantile(nonzero, 0.9))}
    selected = {label: min(range(len(counts)), key=lambda index: abs(counts[index] - value)) for label, value in targets.items()}
    directory = OUTPUT_ROOT / "analysis/profile"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for batch_index, batch in enumerate(loader):
        labels = [label for label, index in selected.items() if index == batch_index]
        if not labels:
            continue
        pairs = sum(len(item) for item in batch["instance_pairs"])
        if pairs != counts[batch_index]:
            raise RuntimeError("Real batch pair counts differ from the deterministic sampler")
        areas = []
        for image_index, image_pairs in enumerate(batch["instance_pairs"]):
            instance_map = batch["instance_map"][image_index:image_index + 1]
            for first, second, _ in image_pairs.tolist():
                crop = _pair_crop_slices(instance_map == first, instance_map == second, padding=3)
                if crop is not None:
                    y, x = crop
                    areas.append((y.stop - y.start) * (x.stop - x.start))
        for label in labels:
            path = directory / f"batch_{label}.pt"
            stored = {key: batch[key] for key in ("image", "mask", "instance_map", "instance_pairs", "tile_id", "valid_mask")}
            stored["valid_pair_count"] = pairs
            digest = tensor_sha(stored)
            stored["batch_sha256"] = digest
            torch.save(stored, path)
            records.append({"density": label, "target_pair_count": targets[label], "batch_index": batch_index,
                            "valid_pairs": pairs, "crop_pixels_sum": sum(areas), "crop_pixels_max": max(areas, default=0),
                            "path": str(path), "file_sha256": sha256_file(path), "tensor_batch_id": digest,
                            "tile_ids": batch["tile_id"]})
        if batch_index >= max(selected.values()):
            break
    write_json(directory / "batches.json", {"batch_pair_counts": counts, "sampler_seed": 42,
               "augmentation_workers": 0, "records": records})
    return records


def profile_pbcl_only(record: dict, iterations: int, repeats: int) -> dict:
    batch = torch.load(record["path"], map_location="cpu", weights_only=False)
    labels = batch["instance_map"].cuda()
    valid = batch["valid_mask"].cuda()
    pairs = [item.cuda() for item in batch["instance_pairs"]]
    logits = torch.zeros_like(batch["mask"], device="cuda", requires_grad=True)
    times = []
    for _ in range(repeats):
        for _ in range(10):
            logits.grad = None
            loss, _, count = pair_bottleneck_connectivity_loss(logits, labels, pairs, valid_masks=valid)
            loss.backward()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            logits.grad = None
            loss, _, count = pair_bottleneck_connectivity_loss(logits, labels, pairs, valid_masks=valid)
            loss.backward()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / iterations)
    if count != record["valid_pairs"]:
        raise RuntimeError("Profiled PBCL effective pair count differs from the batch")
    return {"density": record["density"], "valid_pairs": count, "scope": "PBCL forward and backward on neutral logits",
            "milliseconds_mean": statistics.mean(times), "milliseconds_sample_sd": statistics.stdev(times)}


def profile(warmup: int = 50, iterations: int = 200, repeats: int = 3) -> None:
    directory = OUTPUT_ROOT / "analysis/profile"
    records = prepare_batches()
    summaries, micro = [], []
    for record in records:
        output = directory / record["density"]
        for title in ("Boundary", "Full"):
            method = title.lower()
            checkpoint = OUTPUT_ROOT / "train" / method / f"seed42/{method}_seed42-best_overall.ckpt"
            subprocess.run([
                sys.executable, str(ROOT / "scripts/profile_aquadataset_training_overhead.py"),
                "--treatment", title, "--config", str(ROOT / "configs/aquadataset" / f"{method}.py"),
                "--checkpoint", str(checkpoint), "--batch-file", record["path"], "--output-dir", str(output),
                "--warmup", str(warmup), "--iterations", str(iterations), "--repeats", str(repeats),
            ], cwd=ROOT, check=True)
        boundary = json.loads((output / "boundary.json").read_text())
        full = json.loads((output / "full.json").read_text())
        summaries.append({"density": record["density"], "valid_pairs": record["valid_pairs"],
                          "crop_pixels_sum": record["crop_pixels_sum"], "crop_pixels_max": record["crop_pixels_max"],
                          "boundary_step_ms": boundary["step_ms_mean"], "full_step_ms": full["step_ms_mean"],
                          "pbcl_increment_pct": 100 * (full["step_ms_mean"] / boundary["step_ms_mean"] - 1),
                          "boundary_peak_gib": boundary["peak_allocated_memory_gib_max"],
                          "full_peak_gib": full["peak_allocated_memory_gib_max"]})
        micro.append(profile_pbcl_only(record, iterations, repeats))
    write_csv(directory / "matched_training_overhead.csv", summaries)
    write_csv(directory / "pbcl_only.csv", micro)
    write_json(directory / "complete.json", {"density_batches": len(records), "warmup": warmup, "iterations": iterations,
               "repeats": repeats, "checkpoint_role": "frozen best_overall", "batch_size": 12})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.prepare_only:
        prepare_batches()
    else:
        profile(args.warmup, args.iterations, args.repeats)
