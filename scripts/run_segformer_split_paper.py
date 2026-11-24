#!/usr/bin/env python3
"""Run and summarize the 3-seed SegFormer-B2 SPLIT-paper (Eq. 15) comparison on Aqua and Vegas."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
from time import monotonic, sleep

import rasterio  # Load GDAL before the ML runtime.
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoseg.datasets.geographic_holdout import sha256_file
from geoseg.datasets.vegas_validation_ram import with_vegas_validation_ram
from geoseg.experiments.reproducibility import seed_experiment
from geoseg.experiments.segformer_loss_comparison import DATASETS, OUTPUT, SEEDS
from geoseg.losses.binary_seg import loss_batch_kwargs
from geoseg.losses.split import SplitSegmentationLoss
from geoseg.metrics.deadhesion import CachedDeadhesionEvaluator
from geoseg.runtime import apply_runtime_optimizations, load_experiment, preflight_experiment
from geoseg.runtime.aquav3_builder import build_aquav3_components
from scripts.analyze_pbcl_revision import metric_values, write_csv
from scripts.replay_segformer_loss_comparison import FIXED, GRID, recall_grid, score_predictions
from scripts.run_pbcl_revision import write_json

METHOD = "split_paper"
WEIGHT = 0.1
KERNEL = 3
REPLAY_METHODS = ("boundary", "full", "split", METHOD)
ALL_METHODS = ("boundary", "full", "split", METHOD, "wprf_gt", "wprf_valid", "ewc", "cldice", "cmalis")


def now() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def progress(stage: str, **details: object) -> None:
    payload = {"stage": stage, "time": now(), **details}
    write_json(OUTPUT / "split_paper_progress.json", payload)
    print(payload, flush=True)


def base_config(dataset: str) -> Path:
    if dataset == "aqua":
        return ROOT / "configs/aquadataset/segformer_b2.py"
    if dataset == "vegas":
        return ROOT / "configs/vegas/segformer_b2__base.py"
    raise ValueError(f"Unsupported dataset: {dataset}")


def run_dir(dataset: str, seed: int) -> Path:
    return OUTPUT / "train" / dataset / METHOD / f"seed{seed}"


def eval_dir(dataset: str, method: str, seed: int, split: str) -> Path:
    return OUTPUT / "evaluation" / dataset / method / f"seed{seed}" / split


def spec_for(dataset: str, seed: int, *, resume: bool = False):
    from train_supervision import apply_training_overrides

    directory = run_dir(dataset, seed)
    latest = directory / "last.ckpt"
    experiment = apply_training_overrides(
        load_experiment(base_config(dataset), build=False, seed=seed),
        weights_path=directory,
        log_name=f"{dataset}__{METHOD}/seed{seed}",
        resume_checkpoint=latest if resume and latest.exists() else None,
    )
    spec = experiment.spec
    if spec.model.get("key") != "segformer_b2":
        raise ValueError(f"Expected segformer_b2 model, got {spec.model.get('key')}")
    if spec.model.get("external_boundary_head"):
        raise ValueError("SPLIT-paper requires Base SegFormer-B2 without an external boundary head")
    if spec.loader["train_batch_size"] != 12 or spec.optimization["accumulate_grad_batches"] != 1:
        raise ValueError("SegFormer-B2 comparison requires the original batch-12 protocol")
    if spec.optimization["loss"]["name"] != "BinarySegmentationLoss":
        raise ValueError(f"Expected BinarySegmentationLoss base, got {spec.optimization['loss']}")

    loss_spec = {
        "name": "SplitSegmentationLoss",
        "mask_bce_weight": 0.9,
        "split_weight": WEIGHT,
        "boundary_kernel": KERNEL,
        "paper_equation": 15,
    }
    spec = replace(
        spec,
        optimization={**spec.optimization, "loss": loss_spec},
        checkpoint={
            **spec.checkpoint,
            "monitors": {"overall": "val_overall_score"},
            "save_last": True,
            "declared_save_last": True,
            "allow_epoch_state_checkpoint": True,
            "epoch_state_every_n_epochs": 1,
            "save_final_state": True,
            "plain_checkpoint_names": True,
        },
        artifacts={
            **spec.artifacts,
            "weights_path": str(directory),
            "run_name": f"{dataset}__{METHOD}_seed{seed}",
            "log_name": f"{dataset}__{METHOD}/seed{seed}",
            "val_prediction_dir": str(directory / "val_predictions"),
            "guard_finite_training_loss": True,
        },
        provenance={
            **spec.provenance,
            "experiment_id": f"{dataset}__{METHOD}",
            "family": "segformer_loss_comparison",
            "comparison_method": METHOD,
            "base_experiment_id": "segformer_b2" if dataset == "aqua" else "segformer_b2__base",
        },
    )
    return replace(experiment, spec=spec)


def build(dataset: str, seed: int, *, resume: bool = False):
    experiment = spec_for(dataset, seed, resume=resume)
    spec = with_vegas_validation_ram(experiment.spec)
    spec, components = build_aquav3_components(spec)
    loss = components.loss
    expected_size = int(spec.data["image_size"])
    if (
        type(loss) is not SplitSegmentationLoss
        or loss.split_weight != WEIGHT
        or loss.boundary_kernel != KERNEL
        or loss.image_size != (expected_size, expected_size)
    ):
        raise RuntimeError(f"Unexpected SPLIT-paper loss configuration for {dataset}: {loss}")
    if hasattr(components.config, "training_protocol"):
        components.config.training_protocol["structural_addition"] = {
            "method": "BCE + SPLIT",
            "formula": "0.9 * mask BCE + 0.1 * SPLIT",
            "paper_equation": 15,
            "band_kernel": KERNEL,
        }
    return replace(experiment, spec=spec, components=components)


def check_paper_recipe() -> dict:
    import math

    # General Eq. (15) value & gradient identity with invalid mask exclusion.
    size = 8
    logits = torch.zeros((1, 1, size, size), requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 3:5, 3:5] = 1.0
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[0, 0, -1] = False
    objective = SplitSegmentationLoss(image_size=size, split_weight=WEIGHT, boundary_kernel=KERNEL)
    loss = objective({"out": logits}, target, valid_masks=valid)
    torch.testing.assert_close(loss, loss.new_tensor(math.log(2)))
    gradient, = torch.autograd.grad(loss, (logits,))
    assert (gradient[~valid] == 0).all()
    assert gradient[0, 0, 0, 0] > 0 and gradient[0, 0, 3, 3] < 0
    plain = SplitSegmentationLoss(image_size=size, split_weight=0.0, boundary_kernel=KERNEL)
    plain_gradient = torch.autograd.grad(plain({"out": logits}, target, valid_masks=valid), logits)[0]
    band = objective._boundary_band(target, valid)
    expected = 0.9 * plain_gradient + 0.05 * band / (band.sum() + 1e-6)
    torch.testing.assert_close(gradient, expected)

    # Check both Aqua (512) and Vegas (672 pad with 22px invalid right/bottom border).
    vegas_objective = SplitSegmentationLoss(image_size=672, split_weight=WEIGHT, boundary_kernel=KERNEL)
    v_logits = torch.randn((1, 1, 672, 672), requires_grad=True)
    v_target = torch.zeros((1, 1, 672, 672))
    v_target[:, :, :650, :650] = 1.0  # All-foreground valid region touching padding border at 650.
    v_valid = torch.zeros((1, 1, 672, 672), dtype=torch.bool)
    v_valid[:, :, :650, :650] = True
    v_loss = vegas_objective({"out": v_logits}, v_target, valid_masks=v_valid)
    v_plain = SplitSegmentationLoss(image_size=672, split_weight=0.0, boundary_kernel=KERNEL)
    torch.testing.assert_close(v_loss, 0.9 * v_plain({"out": v_logits}, v_target, valid_masks=v_valid))
    v_grad = torch.autograd.grad(v_loss, v_logits)[0]
    assert torch.count_nonzero(v_grad[~v_valid]) == 0
    return {"eq15_value_and_gradient": "passed", "vegas_padding_boundary_exclusion": "passed"}


def verify_protocol_and_caches() -> dict:
    refs_path = OUTPUT / "references.json"
    if not refs_path.is_file():
        raise FileNotFoundError(f"Missing {refs_path}")
    refs = read_json(refs_path)
    verified_refs = {}
    for dataset in DATASETS:
        split_ref_cfg = (
            ROOT / "configs/aquadataset/segformer_b2_boundary_split.py"
            if dataset == "aqua"
            else ROOT / "configs/vegas/segformer_b2__boundary_split.py"
        )
        for seed in SEEDS:
            sp_spec = spec_for(dataset, seed).spec.to_dict()
            cmp_spec = load_experiment(split_ref_cfg, build=False, seed=seed).spec.to_dict()
            for group, fields in {
                "model": ("key", "pretrained_sha256"),
                "data": ("split_hashes", "split_counts", "image_size", "normalize_mean", "normalize_std", "resize_mode"),
                "runtime": ("max_epochs", "seed", "precision"),
                "optimization": ("optimizer", "scheduler", "accumulate_grad_batches"),
                "loader": ("train_batch_size", "balanced_sampler", "drop_last",
                           "sampler_generator_stream", "loader_generator_streams"),
                "checkpoint": ("start_epoch", "monitor", "overall_score", "selection"),
            }.items():
                for field in fields:
                    if sp_spec[group].get(field) != cmp_spec[group].get(field):
                        raise RuntimeError(
                            f"{dataset} seed{seed}: split_paper {group}.{field}={sp_spec[group].get(field)!r} "
                            f"differs from split {cmp_spec[group].get(field)!r}"
                        )
            if sp_spec["model"].get("external_boundary_head") is not None:
                raise RuntimeError(f"{dataset} seed{seed}: expected external_boundary_head=None")

        for method in ("boundary", "full", "split"):
            for seed in SEEDS:
                expected_count = spec_for(dataset, seed).spec.data["split_counts"]
                for split in ("val", "test"):
                    ev_dir = eval_dir(dataset, method, seed, split)
                    summary_path = ev_dir / f"{split}_summary.json"
                    summary = read_json(summary_path)
                    probs = list((ev_dir / split / "probs").glob("*_prob.npy"))
                    if len(probs) != expected_count[split] or summary["sample_count"] != expected_count[split]:
                        raise RuntimeError(f"Incomplete FP32 probability cache in {ev_dir}")
                    if summary["threshold"] != 0.5:
                        raise RuntimeError(f"Unexpected threshold in {summary_path}")
                    if method in ("boundary", "full"):
                        ref_key = f"{dataset}/{method}/{seed}"
                        if summary["checkpoint_sha256"] != refs[ref_key]["sha256"]:
                            raise RuntimeError(f"Checkpoint sha256 mismatch for {ref_key} ({split})")
                        verified_refs[f"{ref_key}/{split}"] = summary["checkpoint_sha256"]
    return {"verified_reference_caches": len(verified_refs), "protocol_parity": "passed"}


def _filtered_loss_kwargs(loss, batch: dict, device: torch.device) -> dict:
    kwargs = loss_batch_kwargs(batch, device)
    if not bool(getattr(loss, "requires_instance_supervision", False)):
        kwargs.pop("instance_maps", None)
        kwargs.pop("instance_pairs", None)
    if not bool(getattr(loss, "requires_ewc_supervision", False)):
        kwargs.pop("ewc_weights", None)
    if not bool(getattr(loss, "requires_wprf_supervision", False)):
        kwargs.pop("wprf_support", None)
        kwargs.pop("wprf_dense_components", None)
        kwargs.pop("wprf_pairs", None)
    return kwargs


def cpu_smoke_build() -> dict:
    details = {}
    for dataset in DATASETS:
        experiment = build(dataset, 42)
        comp = experiment.components
        comp.datamodule.setup("fit")
        batch = next(iter(comp.datamodule.train_dataloader()))
        image = batch["img"][:1]
        mask = batch["mask"][:1]
        kwargs = {
            k: v[:1] if torch.is_tensor(v) else v
            for k, v in _filtered_loss_kwargs(comp.config.loss, batch, torch.device("cpu")).items()
        }
        outputs = comp.config.net(image)
        loss = comp.config.loss(outputs, mask, **kwargs)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite CPU smoke loss for {dataset}")
        loss.backward()
        comp.config.optimizer.zero_grad(set_to_none=True)
        details[dataset] = {
            "image_shape": list(image.shape),
            "loss_value": float(loss.detach()),
            "train_batches": len(comp.datamodule.train_dataloader()),
            "val_batches": len(comp.datamodule.val_dataloader()),
        }
    return details


def check(*, skip_cpu_smoke: bool = False) -> dict:
    recipe = check_paper_recipe()
    protocol = verify_protocol_and_caches()
    smoke = {} if skip_cpu_smoke else cpu_smoke_build()
    report = {
        "checked_at": now(),
        "recipe": recipe,
        "protocol": protocol,
        "cpu_smoke": smoke,
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(OUTPUT / "split_paper_preflight.json", report)
    return report


def gpu_free_mib() -> tuple[int, list[str]]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()[0]
    free_str, _ = [x.strip() for x in out.split(",")]
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    apps = [line.strip() for line in apps if line.strip()]
    return int(free_str), apps


def wait_for_gpu(wait_pid: int | None, min_free_mib: int = 20000, poll_seconds: int = 30) -> None:
    while True:
        pid_alive = wait_pid is not None and Path(f"/proc/{wait_pid}").exists()
        free_mib, apps = gpu_free_mib()
        if not pid_alive and free_mib >= min_free_mib and not apps:
            return
        if not pid_alive and free_mib >= min_free_mib:
            # No specified PID and ample VRAM is free.
            return
        progress(
            "waiting_for_gpu",
            waiting_for_pid=wait_pid,
            pid_alive=pid_alive,
            gpu_free_mib=free_mib,
            min_free_mib=min_free_mib,
            compute_apps=apps,
        )
        sleep(poll_seconds)


def profile_dataset(dataset: str) -> Path:
    path = OUTPUT / "preflight" / f"{dataset}__{METHOD}.json"
    sig = sha256_file(Path(__file__))
    if path.is_file() and read_json(path).get("script_sha256") == sig:
        return path
    progress("profile", dataset=dataset, method=METHOD)
    command(["profile", "--dataset", dataset, "--seed", "42"], OUTPUT / "preflight" / f"{dataset}__{METHOD}.log")
    record = read_json(path)
    record["script_sha256"] = sig
    write_json(path, record)
    return path


def run_profile_worker(dataset: str) -> None:
    from time import perf_counter
    from statistics import median

    experiment = build(dataset, 42)
    components = experiment.components
    config = components.config
    device = torch.device("cuda")
    net = config.net.to(device)
    loss = config.loss.to(device)
    optimizer = config.optimizer
    loader = iter(components.datamodule.train_dataloader())
    wait, train_t, forward_t, obj_t, values = [], [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(4):
        t0 = perf_counter()
        batch = next(loader)
        wait.append(perf_counter() - t0)
        optimizer.zero_grad(set_to_none=True)
        t0 = perf_counter()
        image = batch["img"].to(device)
        mask = batch["mask"].to(device)
        kwargs = _filtered_loss_kwargs(loss, batch, device)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        ev[0].record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = net(image)
            ev[1].record()
            value = loss(outputs, mask, **kwargs)
        ev[2].record()
        if not value.isfinite():
            raise RuntimeError("Nonfinite training loss in GPU profile")
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.gradient_clip_val, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        train_t.append(perf_counter() - t0)
        forward_t.append(ev[0].elapsed_time(ev[1]) / 1000)
        obj_t.append(ev[1].elapsed_time(ev[2]) / 1000)
        values.append(float(value.detach()))
    net.eval()
    loss.eval()
    t0 = perf_counter()
    batch = next(iter(components.datamodule.val_dataloader()))
    kwargs = _filtered_loss_kwargs(loss, batch, device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        val_loss = loss(net(batch["img"].to(device)), batch["mask"].to(device), **kwargs)
    if not val_loss.isfinite():
        raise RuntimeError("Nonfinite validation loss in GPU profile")
    torch.cuda.synchronize()
    out_path = OUTPUT / "preflight" / f"{dataset}__{METHOD}.json"
    write_json(out_path, {
        "dataset": dataset,
        "method": METHOD,
        "train_step_seconds": train_t,
        "step_median": median(train_t[1:]),
        "loader_wait_seconds": wait,
        "forward_seconds": forward_t,
        "objective_seconds": obj_t,
        "val_first_batch_seconds": perf_counter() - t0,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "training_losses": values,
        "validation_loss": float(val_loss),
    })


def fit_worker(dataset: str, seed: int) -> None:
    import pytorch_lightning as pl
    from train_supervision import Supervision_Train, build_trainer_kwargs, write_training_protocol_manifest

    preflight_experiment(spec_for(dataset, seed, resume=True))
    experiment = apply_runtime_optimizations(build(dataset, seed, resume=True))
    config = experiment.config
    seed_experiment(seed)
    experiment.write_resolved_config()
    write_training_protocol_manifest(config, experiment.config_path)
    trainer = pl.Trainer(**build_trainer_kwargs(config, experiment=experiment))
    trainer.fit(Supervision_Train(config), ckpt_path=config.resume_ckpt_path)


def evaluate_worker(dataset: str, seed: int) -> None:
    from geoseg.evaluation import EvaluationRequest, evaluate_checkpoint

    experiment = build(dataset, seed)
    checkpoint = run_dir(dataset, seed) / "best_overall.ckpt"
    digest = sha256_file(checkpoint)
    for split in ("val", "test"):
        directory = eval_dir(dataset, METHOD, seed, split)
        summary_path = directory / f"{split}_summary.json"
        expected_count = experiment.spec.data["split_counts"][split]
        if summary_path.is_file():
            report = read_json(summary_path)
            if (
                report["checkpoint_sha256"] == digest
                and report["threshold"] == 0.5
                and report["sample_count"] == expected_count
                and len(list((directory / split / "probs").glob("*_prob.npy"))) == expected_count
            ):
                continue
        evaluate_checkpoint(
            experiment,
            EvaluationRequest(
                checkpoint=checkpoint,
                split=split,
                threshold=0.5,
                out_dir=directory,
                save_predictions=True,
                strict=True,
                save_probabilities=True,
                probability_dtype="float32",
            ),
        )
        report = read_json(summary_path)
        if report["checkpoint_sha256"] != digest or report["sample_count"] != expected_count or report["threshold"] != 0.5:
            raise RuntimeError(f"Evaluation metadata mismatch: {summary_path}")
        if len(list((directory / split / "probs").glob("*_prob.npy"))) != expected_count:
            raise RuntimeError(f"Incomplete FP32 probabilities: {directory}")


def command(args_list: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), *args_list],
            cwd=ROOT,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def train_and_evaluate_one(dataset: str, seed: int) -> None:
    directory = run_dir(dataset, seed)
    final = directory / "training_complete.json"
    best = directory / "best_overall.ckpt"
    last = directory / "last.ckpt"
    if final.is_file():
        record = read_json(final)
        if record["sha256"] != sha256_file(best) or record["last_sha256"] != sha256_file(last):
            raise RuntimeError(f"Completed training weights changed: {directory}")
    else:
        progress("train", dataset=dataset, method=METHOD, seed=seed)
        t0 = monotonic()
        resumed = last.is_file()
        command(
            ["fit", "--dataset", dataset, "--seed", str(seed)],
            OUTPUT / "logs" / dataset / METHOD / f"seed{seed}_train.log",
        )
        if len(list((directory / "deadhesion_val").glob("epoch_*.json"))) != 40:
            raise RuntimeError(f"Expected 40 validation epochs: {directory}")
        if not best.is_file() or not last.is_file():
            raise RuntimeError(f"Missing best_overall.ckpt or last.ckpt in {directory}")
        write_json(
            final,
            {
                "sha256": sha256_file(best),
                "last_sha256": sha256_file(last),
                "wall_seconds": monotonic() - t0,
                "resumed": resumed,
            },
        )
    progress("evaluate", dataset=dataset, method=METHOD, seed=seed)
    command(
        ["evaluate", "--dataset", dataset, "--seed", str(seed)],
        OUTPUT / "logs" / dataset / METHOD / f"seed{seed}_eval.log",
    )


def summarize() -> None:
    progress("summarize")
    analysis = OUTPUT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    refs = read_json(OUTPUT / "references.json")

    selected = []
    replay = []
    for dataset in DATASETS:
        spec = spec_for(dataset, 42).spec
        val_eval = CachedDeadhesionEvaluator(spec.data["root"], split="val")
        for seed in SEEDS:
            cutoff = None
            for method in REPLAY_METHODS:
                ev_dir = eval_dir(dataset, method, seed, "val")
                summary = read_json(ev_dir / "val_summary.json")
                recall = recall_grid(ev_dir / "val/probs", val_eval)
                midpoint = GRID.index(0.5)
                if method == "boundary":
                    cutoff = float(recall[midpoint])
                idx = min(range(len(GRID)), key=lambda i: (abs(float(recall[i]) - cutoff), abs(i - midpoint), i))
                selected.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "threshold": GRID[idx],
                    "val_target_recall": cutoff,
                    "val_selected_recall": float(recall[idx]),
                    "val_recall_gap": float(recall[idx]) - cutoff,
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                })
    write_json(analysis / "split_paper_frozen_recall_thresholds.json", selected)

    for dataset in DATASETS:
        spec = spec_for(dataset, 42).spec
        test_eval = CachedDeadhesionEvaluator(spec.data["root"], split="test")
        for record in (r for r in selected if r["dataset"] == dataset):
            method, seed = record["method"], record["seed"]
            ev_dir = eval_dir(dataset, method, seed, "test")
            summary = read_json(ev_dir / "test_summary.json")
            if summary["checkpoint_sha256"] != record["checkpoint_sha256"]:
                raise RuntimeError(f"Val and Test checkpoints disagree for {dataset}/{method}/seed{seed}")
            scores = score_predictions(ev_dir / "test/probs", test_eval, sorted(set((*FIXED, record["threshold"]))))
            official = summary["deadhesion"]["overall"]
            for key, value in metric_values(official).items():
                np.testing.assert_allclose(metric_values(scores[0.5])[key], value, atol=1e-10)
            for threshold, overall in scores.items():
                replay.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "threshold": threshold,
                    "fixed": threshold in FIXED,
                    "recall_matched": threshold == record["threshold"],
                    **metric_values(overall),
                    "fg_recall": overall["fg_recall"],
                })
    write_csv(analysis / "split_paper_threshold_replay.csv", replay)

    sample_metrics = (*metric_values(official), "fg_recall")
    fixed_groups: dict[tuple[str, str, float, str], list[float]] = {}
    for row in replay:
        if row["fixed"]:
            for metric in sample_metrics:
                fixed_groups.setdefault((row["dataset"], row["method"], row["threshold"], metric), []).append(
                    100 * row[metric]
                )
    write_csv(analysis / "split_paper_fixed_threshold_test_mean_sd.csv", [
        {"dataset": d, "method": m, "threshold": t, "metric": k, "mean_pct": mean(v), "sd_pp": stdev(v)}
        for (d, m, t, k), v in fixed_groups.items()
    ])

    matched = {(r["dataset"], r["seed"]): r for r in replay if r["recall_matched"] and r["method"] == "boundary"}
    rm_pairs: dict[tuple[str, str, str], list[float]] = {}
    for row in replay:
        if row["recall_matched"] and row["method"] != "boundary":
            base = matched[row["dataset"], row["seed"]]
            for metric in sample_metrics:
                rm_pairs.setdefault((row["dataset"], row["method"], metric), []).append(
                    100 * (row[metric] - base[metric])
                )
    write_csv(analysis / "split_paper_recall_matched_test_mean_sd.csv", [
        {"dataset": d, "method": m, "metric": k, "mean_delta_pp": mean(v), "sd_pp": stdev(v)}
        for (d, m, k), v in rm_pairs.items()
    ])

    # Fixed 0.5 per-seed, mean±SD, and paired deltas across all SegFormer comparison methods.
    per_seed = []
    for dataset in DATASETS:
        for method in ALL_METHODS:
            for seed in SEEDS:
                if method in ("boundary", "full"):
                    path = Path(refs[f"{dataset}/{method}/{seed}"]["summary"])
                else:
                    path = eval_dir(dataset, method, seed, "test") / "test_summary.json"
                score = read_json(path)["deadhesion"]["overall"]
                per_seed.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    **metric_values(score),
                    "fg_recall": float(score["fg_recall"]),
                })
    write_csv(analysis / "split_paper_test_per_seed.csv", per_seed)

    summaries = []
    for dataset in DATASETS:
        for method in ALL_METHODS:
            subset = [r for r in per_seed if r["dataset"] == dataset and r["method"] == method]
            for metric in sample_metrics:
                vals = [100 * r[metric] for r in subset]
                summaries.append({
                    "dataset": dataset,
                    "method": method,
                    "metric": metric,
                    "mean_pct": mean(vals),
                    "sd_pp": stdev(vals),
                })
    write_csv(analysis / "split_paper_test_mean_sd.csv", summaries)

    lookup = {(r["dataset"], r["method"], r["seed"]): r for r in per_seed}
    comparisons = (
        ("full", "boundary"),
        ("split", "boundary"),
        ("split", "full"),
        ("split_paper", "boundary"),
        ("split_paper", "full"),
        ("split_paper", "split"),
    )
    paired = []
    for dataset in DATASETS:
        for treatment, reference in comparisons:
            for seed in SEEDS:
                a, b = lookup[dataset, treatment, seed], lookup[dataset, reference, seed]
                for metric in sample_metrics:
                    paired.append({
                        "dataset": dataset,
                        "method": treatment,
                        "reference": reference,
                        "seed": seed,
                        "metric": metric,
                        "delta_pp": 100 * (a[metric] - b[metric]),
                    })
    write_csv(analysis / "split_paper_test_paired_deltas.csv", paired)
    paired_groups: dict[tuple[str, str, str, str], list[float]] = {}
    for row in paired:
        paired_groups.setdefault((row["dataset"], row["method"], row["reference"], row["metric"]), []).append(
            row["delta_pp"]
        )
    write_csv(analysis / "split_paper_test_paired_mean_sd.csv", [
        {"dataset": d, "method": m, "reference": ref, "metric": k, "mean_delta_pp": mean(v), "sd_pp": stdev(v)}
        for (d, m, ref, k), v in paired_groups.items()
    ])

    # Render concise markdown summary for split_paper alongside Boundary, Full, and SPLIT.
    stat_lookup = {(r["dataset"], r["method"], r["metric"]): (r["mean_pct"], r["sd_pp"]) for r in summaries}
    rm_lookup = {(r["dataset"], r["method"], r["metric"]): (r["mean_delta_pp"], r["sd_pp"])
                 for r in [
                     {"dataset": d, "method": m, "metric": k, "mean_delta_pp": mean(v), "sd_pp": stdev(v)}
                     for (d, m, k), v in rm_pairs.items()
                 ]}
    lines = [
        "# SegFormer-B2 SPLIT-paper (Eq. 15) 3-Seed Comparison",
        "",
        "Protocol: Base SegFormer-B2 (`segformer_b2` on Aqua 512, `segformer_b2__base` on Vegas 672 pad; "
        "AdamW, 40 epochs, batch 12, no external boundary head) trained with `SplitSegmentationLoss` "
        "(`0.9 * BCE + 0.1 * SPLIT`, kernel 3) over seeds 42–44.",
        "",
        "## Fixed Threshold (p > 0.5) Mean ± SD (%)",
        "",
        "| Dataset | Method | fg_iou ↑ | boundary_f1 ↑ | pq ↑ | close_pair_recall ↑ | merge_rate ↓ | split_rate ↓ | count_nmae ↓ | fg_recall ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cols = ("fg_iou", "boundary_f1", "pq", "close_pair_recall", "merge_rate", "split_rate", "count_nmae", "fg_recall")
    for dataset in DATASETS:
        for method in REPLAY_METHODS:
            cells = [f"{stat_lookup[dataset, method, c][0]:.2f} ± {stat_lookup[dataset, method, c][1]:.2f}" for c in cols]
            lines.append(f"| {dataset} | {method} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Recall-Matched vs Boundary@0.5 (Val-selected threshold, Test Δ pp)",
        "",
        "| Dataset | Method | Δ fg_iou | Δ boundary_f1 | Δ pq | Δ close_pair_recall | Δ merge_rate | Δ split_rate | Δ count_nmae | Δ fg_recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for method in ("full", "split", "split_paper"):
            cells = [f"{rm_lookup[dataset, method, c][0]:+.2f} ± {rm_lookup[dataset, method, c][1]:.2f}" for c in cols]
            lines.append(f"| {dataset} | {method} | " + " | ".join(cells) + " |")
    (analysis / "split_paper_summary.md").write_text("\n".join(lines) + "\n")


def resume_systemd_unit(unit: str | None) -> dict | None:
    if not unit:
        return None
    subprocess.run(["systemctl", "--user", "start", unit], check=True)
    status_out = subprocess.check_output(
        ["systemctl", "--user", "show", unit, "--property=ActiveState,SubState,MainPID"],
        text=True,
    ).strip()
    return {"unit": unit, "status": status_out, "started_at": now()}


def execute(wait_pid: int | None, min_free_mib: int, poll_seconds: int, resume_unit: str | None = None) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "split_paper_campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress("check")
        command(["check"], OUTPUT / "preflight" / "split_paper_check.log")
        wait_for_gpu(wait_pid, min_free_mib=min_free_mib, poll_seconds=poll_seconds)
        for dataset in DATASETS:
            profile_dataset(dataset)
        queue = [(dataset, seed) for seed in SEEDS for dataset in DATASETS]
        write_json(OUTPUT / "split_paper_protocol.json", {
            "method": METHOD,
            "formula": "0.9 * mask BCE + 0.1 * SPLIT",
            "paper_equation": 15,
            "boundary_kernel": KERNEL,
            "split_weight": WEIGHT,
            "queue": queue,
            "resume_systemd_unit": resume_unit,
            "script_sha256": sha256_file(Path(__file__)),
        })
        for dataset, seed in queue:
            train_and_evaluate_one(dataset, seed)
        summarize()
        resumed = resume_systemd_unit(resume_unit)
        progress("complete", runs=len(queue), resumed_unit=resumed)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GEOSEG_LIGHTNING_LOG_ROOT", str(OUTPUT / "lightning_logs"))
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="run",
                        choices=("run", "check", "profile", "fit", "evaluate", "summarize"))
    parser.add_argument("--dataset", choices=DATASETS, default="aqua")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--wait-pid", type=int, default=None)
    parser.add_argument("--min-free-mib", type=int, default=20000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--resume-systemd-unit", type=str, default=None)
    parser.add_argument("--skip-cpu-smoke", action="store_true")
    args = parser.parse_args()

    if args.stage == "check":
        report = check(skip_cpu_smoke=args.skip_cpu_smoke)
        print(json.dumps(report, indent=2))
    elif args.stage == "profile":
        run_profile_worker(args.dataset)
    elif args.stage == "fit":
        fit_worker(args.dataset, args.seed)
    elif args.stage == "evaluate":
        evaluate_worker(args.dataset, args.seed)
    elif args.stage == "summarize":
        summarize()
    else:
        try:
            execute(args.wait_pid, args.min_free_mib, args.poll_seconds, args.resume_systemd_unit)
        except Exception as error:
            progress("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
