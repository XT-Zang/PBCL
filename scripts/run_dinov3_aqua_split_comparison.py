#!/usr/bin/env python3
"""Run the Aqua DINOv3 SPLIT and verified WPRF comparisons before SegFormer."""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import json
import os
import select
import signal
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

import rasterio  # Load GDAL before the ML runtime.
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoseg.datasets.geographic_holdout import sha256_file
from geoseg.experiments.pbcl_revision import DATA_ROOT, OUTPUT_ROOT as ARCHIVE
from geoseg.experiments.segformer_loss_comparison import OUTPUT as PRIOR
from geoseg.losses.split import SplitSegmentationLoss
from geoseg.losses.ssepnet import SSepNetLoss, SSepNetWPRFLoss
from geoseg.models.aquav3_external import HuggingFaceSegmentationWrapper
from geoseg.metrics.deadhesion import CachedDeadhesionEvaluator
from geoseg.runtime import load_experiment
from scripts.analyze_pbcl_revision import metric_values, write_csv
from scripts.reevaluate_pbcl_revision_p0 import frozen
from scripts.replay_segformer_loss_comparison import FIXED, GRID, recall_grid, score_predictions
from scripts.run_pbcl_revision import now, write_json

OUTPUT = PRIOR.with_name("dinov3_aqua_split_comparison")
CONFIG_ROOT = ROOT / "configs/aquadataset"
SEEDS = (42, 43, 44)
REPLAY_METHODS = ("boundary", "full", "split", "split_paper")
WPRF_METHODS = ("wprf_gt", "wprf_valid")
TREATMENTS = ("split", *WPRF_METHODS, "split_paper")
BASE_CONFIGS = {"split": "boundary", "split_paper": "base"}
WEIGHT = 0.1
KERNEL = 3


class SSepNetSplitLoss(SSepNetLoss):
    """Keep the original SSepNet objective and add the SPLIT mask penalty."""

    def __init__(self, base: SSepNetLoss):
        super().__init__(boundary_weight=base.boundary_weight, boundary_kernel=base.boundary_kernel,
                         pbcl_weight=base.pbcl_weight)
        self.split = SplitSegmentationLoss(image_size=512, split_weight=WEIGHT, boundary_kernel=KERNEL)

    def forward(self, outputs, target, *, valid_masks=None, **kwargs):
        total = super().forward(outputs, target, valid_masks=valid_masks, **kwargs)
        logits = outputs["out"].float()
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, target.shape[-2:], mode="bilinear", align_corners=False)
        target = target.float()
        valid = self._valid_mask(valid_masks, target).bool() if valid_masks is not None else None
        return total + WEIGHT * self.split.penalty(logits, target, valid)


class AuthorAffinityHead(torch.nn.Conv2d):
    """Reuse the verified striped implementation of upsampling and stride-4 convolution."""

    def __init__(self, previous: torch.nn.Conv2d):
        super().__init__(previous.in_channels, previous.out_channels, 3, stride=4, padding=1)
        # Keep optimizer references, parameter names and initial values.
        self.weight = previous.weight
        self.bias = previous.bias

    def affinity_head(self, features):
        return super().forward(features)

    def forward(self, features):
        height, width = features.shape[-2:]
        return HuggingFaceSegmentationWrapper._wprf_affinity(self, features, (4 * height, 4 * width))


def run_dir(method, seed):
    # Preserve the established paths for SPLIT controls and replay.
    root = OUTPUT / "train"
    return (root if method == "split" else root / method) / f"seed{seed}"


def eval_dir(method, seed, split):
    root = OUTPUT / "evaluation"
    return (root if method == "split" else root / method) / f"seed{seed}" / split


def status(stage, **details):
    payload = {"stage": stage, "time": now(), **details}
    write_json(OUTPUT / "progress.json", payload)
    print(payload, flush=True)


def read_json(path):
    return json.loads(path.read_text())


def spec_for(seed, method="split", *, resume=False):
    from train_supervision import apply_training_overrides

    directory = run_dir(method, seed)
    latest = directory / "last.ckpt"
    base_method = BASE_CONFIGS.get(method, method)
    experiment = apply_training_overrides(
        load_experiment(CONFIG_ROOT / f"{base_method}.py", build=False, seed=seed),
        weights_path=directory, log_name=f"dinov3_aqua_comparison/{method}/seed{seed}",
        resume_checkpoint=latest if resume and latest.exists() else None,
    )
    spec = experiment.spec
    spec = replace(
        spec,
        checkpoint={**spec.checkpoint, "monitors": {"overall": "val_overall_score"},
                    "save_last": True, "declared_save_last": True, "save_final_state": True,
                    "plain_checkpoint_names": True, "allow_epoch_state_checkpoint": True,
                    "epoch_state_every_n_epochs": 5},
        artifacts={**spec.artifacts, "run_name": f"dinov3_aqua_{method}_seed{seed}",
                   "guard_finite_training_loss": True},
        provenance={**spec.provenance, "revision_treatment": f"dinov3_aqua_{method}",
                    "base_revision_treatment": base_method},
    )
    if method.startswith("wprf"):
        spec = replace(
            spec,
            model={**spec.model, "affinity_projection": "fused_stride4_bilinear_then_conv_stride4"},
            data={**spec.data, "wprf_target_root": None,
                  "wprf_semantic_cache": str(OUTPUT / "cache" / "wprf_semantic_v1")},
        )
    return replace(experiment, spec=spec)


def check():
    generator = torch.Generator().manual_seed(42)
    outputs = {key: torch.randn(2, 1, 16, 16, generator=generator, requires_grad=True)
               for key in ("out", "aux", "edge")}
    mask = (torch.rand(2, 1, 16, 16, generator=generator) > 0.25).float()
    valid = torch.ones_like(mask, dtype=torch.bool)
    valid[:, :, -3:, :] = False
    base = SSepNetLoss(boundary_weight=0.6, pbcl_weight=0.0)
    loss = SSepNetSplitLoss(base)
    expected = base(outputs, mask, valid_masks=valid) + WEIGHT * loss.split.penalty(outputs["out"], mask, valid)
    actual = loss(outputs, mask, valid_masks=valid)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual, tuple(outputs.values())),
        torch.autograd.grad(expected, tuple(outputs.values())), strict=True,
    ):
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    if not torch.isfinite(actual):
        raise ValueError("Nonfinite SPLIT loss")
    refs = {}
    for seed in SEEDS:
        spec = spec_for(seed).spec.to_dict()
        for method in REPLAY_METHODS[:2]:
            directory = ARCHIVE / "train" / method / f"seed{seed}"
            archived = read_json(directory / "resolved_config.json")
            for group, fields in {
                "data": ("root", "split_hashes", "normalize_mean", "normalize_std", "image_size"),
                "runtime": ("max_epochs", "precision", "seed"),
                "loader": ("train_batch_size", "balanced_sampler", "drop_last"),
                "optimization": ("optimizer", "scheduler", "accumulate_grad_batches"),
                "checkpoint": ("start_epoch", "monitor", "overall_score"),
            }.items():
                for field in fields:
                    if spec[group].get(field) != archived[group].get(field):
                        raise ValueError(f"{method} seed{seed}: archived {group}.{field} differs")
            checkpoint, = directory.glob("*-best_overall.ckpt")
            # Each weights file is hashed by the archived evaluation. The two
            # split summaries must identify the same frozen checkpoint.
            summaries = []
            for split in ("val", "test"):
                evaluation = ARCHIVE / "evaluation" / method / f"seed{seed}" / split
                summary = read_json(evaluation / f"{split}_summary.json")
                receipt = read_json(evaluation / "evaluation_complete.json")
                if (summary["checkpoint_sha256"] != receipt["checkpoint_sha256"]
                        or summary["sample_count"] != spec["data"]["split_counts"][split]
                        or summary["threshold"] != 0.5):
                    raise ValueError(f"Invalid archived {method} seed{seed} {split} evaluation")
                summaries.append(summary["checkpoint_sha256"])
            if summaries[0] != summaries[1] or sha256_file(checkpoint) != summaries[0]:
                raise ValueError(f"Checkpoint hash mismatch for {checkpoint}")
            refs[f"{method}/seed{seed}"] = {"path": str(checkpoint), "sha256": summaries[0]}
    from geoseg.datasets.segformer_wprf_targets import UPSTREAM, WPRF_SOURCE_COMMIT
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain", "--", "src"], text=True).strip()
    if commit != WPRF_SOURCE_COMMIT or dirty:
        raise ValueError("WPRF source must match the pinned author commit")
    old_conv = torch.nn.Conv2d(4, 8, 3, padding=1)
    head = AuthorAffinityHead(old_conv)
    features = torch.randn(2, 4, 27, 31, generator=generator, requires_grad=True)
    direct = F.conv2d(F.interpolate(features, scale_factor=4, mode="bilinear", align_corners=False),
                      head.weight, head.bias, stride=4, padding=1)
    striped = head(features)
    torch.testing.assert_close(direct, striped, atol=1e-6, rtol=1e-6)
    signal = torch.randn(direct.shape, generator=generator) / direct.numel()
    parameters = (features, head.weight, head.bias)
    for reference, adapted in zip(torch.autograd.grad(direct, parameters, signal),
                                  torch.autograd.grad(striped, parameters, signal), strict=True):
        torch.testing.assert_close(reference, adapted, atol=1e-7, rtol=1e-5)
    for treatment in WPRF_METHODS:
        protocol = spec_for(42, treatment).spec
        if protocol.optimization["loss"]["wprf_negative_support"] != treatment.removeprefix("wprf_"):
            raise ValueError("Incorrect WPRF negative propagation domain")
    check_paper_recipe()
    write_json(OUTPUT / "preflight.json", {"loss_and_gradient": "passed", "paper_recipe_value_and_gradient": "passed",
               "references": refs,
               "script_sha256": sha256_file(Path(__file__)), "splits": spec["data"]["split_counts"],
               "seeds": SEEDS, "split_weight": WEIGHT, "split_kernel": KERNEL,
               "wprf_head_value_and_gradient": "passed", "wprf_source_commit": commit})


def check_paper_recipe():
    import math

    size = 8
    logits = torch.zeros((1, 1, size, size), requires_grad=True)
    ignored = torch.zeros_like(logits, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 3:5, 3:5] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[0, 0, -1] = False
    objective = SplitSegmentationLoss(image_size=size, split_weight=WEIGHT, boundary_kernel=KERNEL)
    loss = objective({"out": logits, "aux": ignored, "edge": ignored}, target, valid_masks=valid)
    torch.testing.assert_close(loss, loss.new_tensor(math.log(2)))
    gradient, unused = torch.autograd.grad(loss, (logits, ignored), allow_unused=True)
    assert unused is None and (gradient[~valid] == 0).all()
    assert gradient[0, 0, 0, 0] > 0 and gradient[0, 0, 3, 3] < 0
    plain = SplitSegmentationLoss(image_size=size, split_weight=0, boundary_kernel=KERNEL)
    plain_gradient = torch.autograd.grad(plain({"out": logits}, target, valid_masks=valid), logits)[0]
    band = objective._boundary_band(target, valid)
    expected = 0.9 * plain_gradient + 0.05 * band / (band.sum() + 1e-6)
    torch.testing.assert_close(gradient, expected)
    spec = spec_for(42, "split_paper").spec
    if spec.optimization["loss"]["boundary_weight"] != 0 or spec.optimization["loss"]["pbcl_weight"] != 0:
        raise RuntimeError("The SPLIT paper row requires the Base architecture")


def wprf_recompute_report(method):
    return OUTPUT / f"wprf_recompute_{method}.json"


def wprf_recompute_setting(seed, method, default):
    directory = run_dir(method, seed)
    manifest = directory / "resolved_config.json"
    if (directory / "last.ckpt").is_file() or (directory / "best_overall.ckpt").is_file():
        saved = read_json(manifest)["optimization"]["loss"]
        return bool(saved.get("wprf_activation_checkpoint", default))
    report = wprf_recompute_report(method)
    if report.is_file() and read_json(report).get("admitted") is True:
        return False
    return default


def qualify_wprf_recompute(seed, method):
    report = wprf_recompute_report(method)
    directory = run_dir(method, seed)
    if report.exists() or any((directory / name).exists() for name in ("last.ckpt", "best_overall.ckpt")):
        return
    command = [sys.executable, str(ROOT / "scripts/profile_dinov3_wprf_recompute.py"), method]
    print(f"Checking WPRF recomputation on fixed real batches before {method} seed{seed}", flush=True)
    try:
        subprocess.run(command, check=True, cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        write_json(report, {"method": method, "admitted": False, "reason": "benchmark failed",
                            "returncode": exc.returncode})

def build(seed, method, *, resume=False):
    from geoseg.runtime.aquav3_builder import build_aquav3_components

    experiment = spec_for(seed, method, resume=resume)
    spec, components = build_aquav3_components(experiment.spec)
    loss_spec = dict(spec.optimization["loss"])
    if method == "split":
        if type(components.loss) is not SSepNetLoss or components.loss.pbcl_weight != 0:
            raise ValueError("SPLIT requires the matched DINOv3 Boundary objective")
        loss = SSepNetSplitLoss(components.loss)
        loss_spec.update(name="SSepNetSplitLoss", split_weight=WEIGHT, split_band_kernel=KERNEL)
        description = {"method": "SPLIT-style", "weight": WEIGHT, "band_kernel": KERNEL}
    elif method == "split_paper":
        loss = SplitSegmentationLoss(
            image_size=int(spec.data["image_size"]), split_weight=WEIGHT, boundary_kernel=KERNEL,
        )
        loss_spec = {"name": "SplitSegmentationLoss", "mask_bce_weight": 0.9,
                     "split_weight": WEIGHT, "boundary_kernel": KERNEL,
                     "paper_equation": 15}
        description = {"method": "BCE + SPLIT", "formula": "0.9 * mask BCE + 0.1 * SPLIT",
                       "paper_equation": 15, "band_kernel": KERNEL}
    else:
        loss = components.loss
        if type(loss) is not SSepNetWPRFLoss or not loss.wprf_upstream:
            raise ValueError("The author-verified WPRF objective is required")
        original = components.model.head.affinity_head
        with torch.random.fork_rng(devices=[]):
            components.model.head.affinity_head = AuthorAffinityHead(original)
        loss.wprf_local_windows = True
        recompute = wprf_recompute_setting(seed, method, loss.wprf_activation_checkpoint)
        loss.wprf_activation_checkpoint = recompute
        loss_spec.update(wprf_local_windows=True, wprf_target_rule="semantic_author_sampler_row_major_vna",
                         wprf_activation_checkpoint=recompute)
        description = {"method": method, "head": spec.model["affinity_projection"],
                       "targets": loss_spec["wprf_target_rule"], "recompute": recompute}
    components.config.loss = loss
    if hasattr(components.config, "training_protocol"):
        components.config.training_protocol["structural_addition"] = description
    spec = replace(spec, optimization={**spec.optimization, "loss": loss_spec})
    return replace(experiment, spec=spec, components=replace(components, loss=loss))


def fit(seed, method):
    import pytorch_lightning as pl
    from geoseg.runtime import apply_runtime_optimizations, preflight_experiment
    from train_supervision import Supervision_Train, build_trainer_kwargs, write_training_protocol_manifest

    if method in WPRF_METHODS:
        qualify_wprf_recompute(seed, method)
    preflight_experiment(spec_for(seed, method, resume=True))
    experiment = apply_runtime_optimizations(build(seed, method, resume=True))
    config = experiment.config
    pl.seed_everything(seed)
    experiment.write_resolved_config()
    write_training_protocol_manifest(config, experiment.config_path)
    trainer = pl.Trainer(**build_trainer_kwargs(config, experiment=experiment))
    trainer.fit(Supervision_Train(config), ckpt_path=config.resume_ckpt_path)


def evaluate(seed, method):
    from geoseg.evaluation import EvaluationRequest, evaluate_checkpoint

    experiment = build(seed, method)
    checkpoint = run_dir(method, seed) / "best_overall.ckpt"
    digest = sha256_file(checkpoint)
    save_probabilities = method in {"split", "split_paper"}
    for split in ("val", "test"):
        directory = eval_dir(method, seed, split)
        summary = directory / f"{split}_summary.json"
        if summary.exists():
            report = read_json(summary)
            if (report["checkpoint_sha256"] == digest
                    and report["sample_count"] == experiment.spec.data["split_counts"][split]
                    and (not save_probabilities or len(list((directory / split / "probs").glob("*_prob.npy")))
                         == report["sample_count"])):
                continue
        evaluate_checkpoint(experiment, EvaluationRequest(
            checkpoint=checkpoint, split=split, threshold=0.5, out_dir=directory,
            save_predictions=True, strict=True, save_probabilities=save_probabilities, probability_dtype="float32",
        ))


def probabilities(method, seed, split, evaluator):
    if method in REPLAY_METHODS[:2]:
        return frozen(method, seed, split, evaluator._tiles)
    directory = eval_dir(method, seed, split)
    summary = read_json(directory / f"{split}_summary.json")
    root = directory / split / "probs"
    if (summary["threshold"] != 0.5 or summary["sample_count"] != evaluator.cached_tile_count
            or {p.name for p in root.glob("*_prob.npy")} != {f"{key}_prob.npy" for key in evaluator._tiles}):
        raise ValueError(f"Incomplete frozen probabilities in {directory}")
    return root, summary


def summarize():
    val = CachedDeadhesionEvaluator(DATA_ROOT, split="val")
    settings = []
    for seed in SEEDS:
        baseline = None
        for method in REPLAY_METHODS:
            root, summary = probabilities(method, seed, "val", val)
            recalls = recall_grid(root, val)
            midpoint = GRID.index(0.5)
            if method == "boundary":
                baseline = float(recalls[midpoint])
            index = min(range(len(GRID)), key=lambda i: (
                abs(float(recalls[i]) - baseline), abs(i - midpoint), i,
            ))
            settings.append({"method": method, "seed": seed, "threshold": GRID[index],
                             "val_target_recall": baseline, "val_recall": float(recalls[index]),
                             "checkpoint_sha256": summary["checkpoint_sha256"]})
    write_json(OUTPUT / "frozen_recall_thresholds.json", settings)

    test = CachedDeadhesionEvaluator(DATA_ROOT, split="test")
    rows = []
    for item in settings:
        method, seed, threshold = (item[name] for name in ("method", "seed", "threshold"))
        root, summary = probabilities(method, seed, "test", test)
        if summary["checkpoint_sha256"] != item["checkpoint_sha256"]:
            raise ValueError("Val and Test checkpoints disagree")
        metrics = score_predictions(root, test, sorted(set((*FIXED, threshold))))
        expected = metric_values(summary["deadhesion"]["overall"])
        np.testing.assert_allclose(list(metric_values(metrics[0.5]).values()), list(expected.values()), atol=1e-10)
        for cutoff, score in metrics.items():
            rows.append({"method": method, "seed": seed, "threshold": cutoff, "fixed": cutoff in FIXED,
                         "recall_matched": cutoff == threshold, **metric_values(score), "fg_recall": score["fg_recall"]})
    write_csv(OUTPUT / "threshold_replay.csv", rows)
    keys = (*expected, "fg_recall")
    delta_rows = []
    comparisons = (("full", "boundary"), ("split", "full"), ("split_paper", "split"))
    for treatment, reference in comparisons:
        for mode in ("fixed_0.5", "recall_matched"):
            sample = {(r["method"], r["seed"]): r for r in rows if
                      (r["threshold"] == 0.5 if mode == "fixed_0.5" else r["recall_matched"])}
            for seed in SEEDS:
                first, second = sample[treatment, seed], sample[reference, seed]
                delta_rows.append({"method": treatment, "reference": reference, "mode": mode, "seed": seed,
                                   **{key: 100 * (first[key] - second[key]) for key in keys}})
    write_csv(OUTPUT / "test_paired_deltas_pp.csv", delta_rows)
    from statistics import mean, stdev
    aggregates = []
    for treatment, reference in comparisons:
        for mode in ("fixed_0.5", "recall_matched"):
            matching = [r for r in delta_rows if r["method"] == treatment and r["mode"] == mode]
            for key in keys:
                values = [r[key] for r in matching]
                aggregates.append({"method": treatment, "reference": reference, "mode": mode,
                                   "metric": key, "mean_delta_pp": mean(values), "sd_pp": stdev(values)})
    write_csv(OUTPUT / "test_paired_mean_sd_pp.csv", aggregates)


def profile(method):
    from geoseg.losses.binary_seg import loss_batch_kwargs

    experiment = build(42, method)
    config = experiment.config
    device = torch.device("cuda")
    net, objective = config.net.to(device), config.loss.to(device)
    objective.set_epoch(4)
    loader = iter(experiment.components.datamodule.train_dataloader())
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(3):
        batch = next(loader)
        start = monotonic()
        config.optimizer.zero_grad(set_to_none=True)
        args = loss_batch_kwargs(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = net(batch["img"].to(device))
            loss = objective(outputs, batch["mask"].to(device), **args)
        if not loss.isfinite() or outputs["affinity"].shape[1:] != (8, 128, 128):
            raise RuntimeError("Invalid WPRF outputs in the real batch preflight")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.gradient_clip_val, error_if_nonfinite=True)
        if not net.head.affinity_head.weight.grad.abs().max().item():
            raise RuntimeError("The WPRF affinity head received no gradient")
        config.optimizer.step()
        torch.cuda.synchronize()
        times.append(monotonic() - start)
    write_json(OUTPUT / f"preflight_{method}.json", {
        "active_epoch_simulated": 5, "train_step_seconds": times,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "affinity_gradient": "nonzero", "negative_gradient_audit": objective.negative_gradient_audit,
    })


def summarize_wprf():
    rows, audits = [], []
    for method in TREATMENTS:
        for seed in SEEDS:
            report = read_json(eval_dir(method, seed, "test") / "test_summary.json")
            score = report["deadhesion"]["overall"]
            rows.append({"method": method, "seed": seed, **metric_values(score), "fg_recall": score["fg_recall"]})
            if not method.startswith("wprf"):
                continue
            for epoch in (5, 20, 40):
                report = read_json(run_dir(method, seed) / "wprf_gradient_audit" / f"epoch_{epoch:03d}.json")
                for scale, values in report["scales"].items():
                    audits.append({"method": method, "seed": seed, "epoch": epoch, "scale": int(scale), **values})
    for method in REPLAY_METHODS[:2]:
        for seed in SEEDS:
            report = read_json(ARCHIVE / "evaluation" / method / f"seed{seed}" / "test/test_summary.json")
            score = report["deadhesion"]["overall"]
            rows.append({"method": method, "seed": seed, **metric_values(score), "fg_recall": score["fg_recall"]})
    write_csv(OUTPUT / "test_per_seed.csv", rows)
    write_csv(OUTPUT / "wprf_negative_audit.csv", audits)
    from statistics import mean, stdev

    stats = []
    for method in (*REPLAY_METHODS[:2], *TREATMENTS):
        matched = [row for row in rows if row["method"] == method]
        for metric in (*metric_values(score), "fg_recall"):
            values = [100 * row[metric] for row in matched]
            stats.append({"method": method, "metric": metric, "mean_pct": mean(values), "sd_pp": stdev(values)})
    write_csv(OUTPUT / "test_mean_sd.csv", stats)


def wait_for_segformer_exit():
    while True:
        with (PRIOR / "campaign.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                if read_json(PRIOR / "progress.json")["stage"] != "queued_after_dinov3":
                    raise RuntimeError("SegFormer has no confirmed checkpoint handoff")
                return
        sleep(10)


def write_queue():
    write_json(OUTPUT / "run_queue.json", {
        "dataset": "aquav3-geographic-holdout-v1",
        "runs": [{"method": method, "seed": seed} for method in TREATMENTS for seed in SEEDS],
        "after": "resume_segformer_b2",
    })


def handoff():
    """Adopt the queue at the end of the current worker without stopping it."""
    parent = int((OUTPUT / "campaign.pid").read_text())
    proc = Path(f"/proc/{parent}")
    if (b"run_dinov3_aqua_split_comparison.py" not in (proc / "cmdline").read_bytes()
            or os.getpgid(parent) != parent):
        raise RuntimeError("The active DINOv3 queue PID failed verification")
    parent_fd = os.pidfd_open(parent)
    stopped = False
    try:
        # Pause only the controller. The active fit or evaluation owns the GPU
        # and keeps running until it has written its normal deliverables.
        signal.pidfd_send_signal(parent_fd, signal.SIGSTOP)
        stopped = True
        while not any(line.startswith(("State:\tT", "State:\tt"))
                      for line in (proc / "status").read_text().splitlines()):
            sleep(0.02)
        children = (proc / "task" / str(parent) / "children").read_text().split()
        if len(children) != 1:
            raise RuntimeError("Queue handoff requires exactly one active training or evaluation worker")
        worker = int(children[0])
        worker_fd = os.pidfd_open(worker)
        try:
            worker_args = Path(f"/proc/{worker}/cmdline").read_bytes().split(b"\0")
            if b"fit" not in worker_args and b"evaluate" not in worker_args:
                raise RuntimeError("The active worker is outside the training/evaluation handoff")
            write_queue()
            write_json(OUTPUT / "queue_update.json", {
                "state": "waiting_for_active_worker", "time": now(), "old_controller": parent,
                "active_worker": worker, "appended": "split_paper", "seeds": SEEDS,
            })
            print(f"Leaving worker {worker} running while the queue update waits", flush=True)
            poll = select.poll()
            poll.register(worker_fd, select.POLLIN)
            poll.poll()
        finally:
            os.close(worker_fd)
        signal.pidfd_send_signal(parent_fd, signal.SIGKILL)
        stopped = False
        poll = select.poll()
        poll.register(parent_fd, select.POLLIN)
        poll.poll()
        (OUTPUT / "campaign.pid").write_text(f"{os.getpid()}\n")
        write_json(OUTPUT / "queue_update.json", {
            "state": "active", "time": now(), "controller": os.getpid(),
            "appended": "split_paper", "seeds": SEEDS,
        })
    finally:
        if stopped:
            signal.pidfd_send_signal(parent_fd, signal.SIGCONT)
        os.close(parent_fd)
    execute()


def execute():
    os.environ.setdefault("GEOSEG_LIGHTNING_LOG_ROOT", str(OUTPUT / "lightning_logs"))
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    with (OUTPUT / "campaign.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status("audit", seeds=SEEDS)
        preflight = OUTPUT / "preflight.json"
        if not preflight.exists() or read_json(preflight)["script_sha256"] != sha256_file(Path(__file__)):
            check()
        wait_for_segformer_exit()
        write_queue()
        for method in WPRF_METHODS:
            if not (OUTPUT / f"preflight_{method}.json").exists():
                status("profile", method=method)
                command("profile", method, 42)
        for method in TREATMENTS:
            for seed in SEEDS:
                directory = run_dir(method, seed)
                if len(list((directory / "deadhesion_val").glob("epoch_*.json"))) < 40 or not (
                    directory / "best_overall.ckpt"
                ).exists():
                    status("train", method=method, seed=seed)
                    command("fit", method, seed)
                status("evaluate", method=method, seed=seed)
                command("evaluate", method, seed)
        status("summarize")
        summarize()
        summarize_wprf()
        status("complete", new_runs=len(SEEDS) * len(TREATMENTS))
        restart_segformer()


def command(stage, method, seed):
    name = f"seed{seed}_{stage}.log" if method == "split" else f"{method}_seed{seed}_{stage}.log"
    with (OUTPUT / name).open("a") as log:
        subprocess.run([sys.executable, __file__, stage, "--method", method, "--seed", str(seed)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def restart_segformer():
    # Use the SegFormer log root and inherit its existing protocol/recovery checks.
    env = {**os.environ, "GEOSEG_LIGHTNING_LOG_ROOT": str(PRIOR / "lightning_logs")}
    with (PRIOR / "campaign.log").open("a") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "scripts/run_segformer_loss_comparison.py")],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (PRIOR / "campaign.pid").write_text(f"{process.pid}\n")
    write_json(OUTPUT / "segformer_restart.json", {"pid": process.pid, "submitted": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "check", "check-paper", "fit", "evaluate", "profile", "handoff"))
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--method", choices=TREATMENTS, default="split")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage in {"run", "handoff"}:
        try:
            if args.stage == "run":
                execute()
            else:
                handoff()
        except Exception as error:
            status("failed", error=str(error))
            raise
    elif args.stage == "check":
        check()
    elif args.stage == "check-paper":
        check_paper_recipe()
        print("SPLIT paper objective, gradients, valid mask and Base protocol verified")
    elif args.stage == "fit":
        fit(args.seed, args.method)
    elif args.stage == "profile":
        profile(args.method)
    else:
        evaluate(args.seed, args.method)


if __name__ == "__main__":
    main()
