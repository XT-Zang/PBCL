#!/usr/bin/env python3
"""Measure current SSepNet supervision-treatment training-step overhead.

One real deterministic Full-model training batch is cached and reused for Base,
Boundary, PBCL, and Full.  Data loading and checkpoint I/O are outside the timed
region.  Each treatment runs in a fresh child process.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoseg.losses.binary_seg import loss_batch_kwargs  # noqa: E402
from geoseg.runtime import load_experiment  # noqa: E402
from scripts.evaluate_aquav3_checkpoint import load_checkpoint_into_net  # noqa: E402


TRAIN_ROOT = Path(
    os.environ.get(
        "AQUADATASET_TRAIN_ROOT",
        str(Path(__file__).resolve().parents[1] / "outputs" / "aquadataset" / "train"),
    )
)


TREATMENTS: dict[str, dict[str, str]] = {
    "Base": {
        "config": "config/aquadataset_stage_adapter_fpn_binary/ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_base.py",
        "checkpoint": "ssepnet_stage_adapter_fpn_ocr/ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_base/seed42/ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_base_seed42-best_overall.ckpt",
    },
    "Boundary": {
        "config": "config/aquadataset_stage_adapter_fpn_binary/ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_boundary.py",
        "checkpoint": "ssepnet_stage_adapter_fpn_ocr/ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_boundary/seed42/ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_boundary_seed42-best_overall.ckpt",
    },
    "PBCL": {
        "config": "config/aquadataset_stage_adapter_fpn_binary/ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_pbcl.py",
        "checkpoint": "ssepnet_stage_adapter_fpn_ocr/ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_pbcl/seed42/ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_pbcl_seed42-best_overall.ckpt",
    },
    "Full": {
        "config": "config/aquadataset_stage_adapter_fpn_ocr/ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_head.py",
        "checkpoint": "ssepnet_stage_adapter_fpn_ocr/ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_head/seed42/ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_head_seed42-best_overall.ckpt",
    },
}


def tensor_sha(batch: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    keys = ["tile_id", "image", "mask", "instance_map", "instance_pairs"]
    if "valid_mask" in batch:
        keys.append("valid_mask")
    for key in keys:
        value = batch[key]
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, torch.Tensor):
                    digest.update(item.detach().cpu().contiguous().numpy().tobytes())
                else:
                    digest.update(str(item).encode())
        else:
            digest.update(str(value).encode())
    return digest.hexdigest()


def prepare_batch(path: Path) -> dict[str, Any]:
    full = TREATMENTS["Full"]
    experiment = load_experiment(ROOT / full["config"], build=True, seed=42)
    if experiment.config is None:
        raise RuntimeError("Full config unavailable")
    selected = None
    for batch in experiment.config.train_loader:
        pair_count = sum(int(pairs.shape[0]) for pairs in batch.get("instance_pairs", []))
        if pair_count > 0:
            selected = {
                "image": batch["image"].cpu(),
                "mask": batch["mask"].cpu(),
                "instance_map": batch["instance_map"].cpu(),
                "instance_pairs": [pairs.cpu() for pairs in batch["instance_pairs"]],
                "tile_id": list(batch["tile_id"]),
                "valid_pair_count": pair_count,
            }
            break
    if selected is None:
        raise RuntimeError("no training batch with valid PBCL pairs")
    selected["batch_sha256"] = tensor_sha(selected)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(selected, path)
    return selected


def autocast_context():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def move_batch(batch: dict[str, Any]) -> dict[str, Any]:
    moved = {
        "image": batch["image"].cuda(non_blocking=False),
        "img": batch["image"].cuda(non_blocking=False),
        "mask": batch["mask"].cuda(non_blocking=False),
        "gt_semantic_seg": batch["mask"].cuda(non_blocking=False),
        "instance_map": batch["instance_map"].cuda(non_blocking=False),
        "instance_pairs": [pairs.cuda(non_blocking=False) for pairs in batch["instance_pairs"]],
        "tile_id": batch["tile_id"],
    }
    if "valid_mask" in batch:
        moved["valid_mask"] = batch["valid_mask"].cuda(non_blocking=False)
    return moved


def training_step(model, loss_fn, optimizer, batch: dict[str, Any]) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    kwargs = loss_batch_kwargs(batch, torch.device("cuda"))
    if not bool(getattr(loss_fn, "requires_instance_supervision", False)):
        kwargs.pop("instance_maps", None)
        kwargs.pop("instance_pairs", None)
    if not bool(getattr(loss_fn, "requires_ewc_supervision", False)):
        kwargs.pop("ewc_weights", None)
    model_kwargs: dict[str, object] = {}
    prepare = getattr(loss_fn, "model_forward_kwargs", None)
    if callable(prepare):
        model_kwargs = dict(prepare(**kwargs))
    with autocast_context():
        outputs = model(batch["image"], **model_kwargs)
        loss = loss_fn(outputs, batch["mask"], **kwargs)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.detach()


def profile_worker(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    spec = TREATMENTS[args.treatment]
    config_path = getattr(args, "config", None) or ROOT / spec["config"]
    checkpoint_path = getattr(args, "checkpoint", None) or args.train_root / spec["checkpoint"]
    experiment = load_experiment(config_path, build=True, seed=42)
    if experiment.config is None:
        raise RuntimeError(f"failed to build {args.treatment}")
    config = experiment.config
    model = config.net.cuda().train()
    loss_fn = config.loss.cuda().train()
    optimizer = config.optimizer
    load_checkpoint_into_net(model, checkpoint_path, strict=True)
    cpu_batch = torch.load(args.batch_file, map_location="cpu", weights_only=False)
    batch = move_batch(cpu_batch)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    measurements: list[float] = []
    peaks: list[float] = []
    losses: list[float] = []
    for repeat in range(args.repeats):
        load_checkpoint_into_net(model, checkpoint_path, strict=True)
        optimizer.state.clear()
        for _ in range(args.warmup):
            loss = training_step(model, loss_fn, optimizer, batch)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            loss = training_step(model, loss_fn, optimizer, batch)
        end.record()
        torch.cuda.synchronize()
        measurements.append(float(start.elapsed_time(end)) / args.iterations)
        peaks.append(float(torch.cuda.max_memory_allocated()) / (1024**3))
        losses.append(float(loss.cpu()))
    step_ms = statistics.fmean(measurements)
    result = {
        "treatment": args.treatment,
        "seed": 42,
        "checkpoint_role": "best_overall",
        "batch_size": int(batch["image"].shape[0]),
        "valid_pairs_per_batch": int(cpu_batch["valid_pair_count"]),
        "batch_sha256": str(cpu_batch["batch_sha256"]),
        "trainable_parameters": total_parameters,
        "warmup_steps": args.warmup,
        "timed_steps": args.iterations,
        "repeats": args.repeats,
        "step_ms_mean": step_ms,
        "step_ms_sd": statistics.stdev(measurements),
        "samples_per_second": int(batch["image"].shape[0]) * 1000.0 / step_ms,
        "peak_allocated_memory_gib_mean": statistics.fmean(peaks),
        "peak_allocated_memory_gib_max": max(peaks),
        "final_loss_by_repeat": losses,
        "precision": "bfloat16 autocast",
        "scope": "forward + active losses + backward + gradient clipping + optimizer step; excludes I/O, validation, and checkpoint writes",
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "python": sys.version.split()[0],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.treatment.lower()}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def combine(output_dir: Path) -> dict[str, Any]:
    rows = [
        json.loads((output_dir / f"{name.lower()}.json").read_text(encoding="utf-8"))
        for name in TREATMENTS
    ]
    csv_rows = [
        {
            "treatment": row["treatment"],
            "trainable_params_m": row["trainable_parameters"] / 1e6,
            "step_ms_mean": row["step_ms_mean"],
            "step_ms_sd": row["step_ms_sd"],
            "samples_per_second": row["samples_per_second"],
            "peak_allocated_memory_gib": row["peak_allocated_memory_gib_max"],
            "valid_pairs_per_batch": row["valid_pairs_per_batch"],
            "batch_sha256": row["batch_sha256"],
        }
        for row in rows
    ]
    with (output_dir / "submission_training_profile.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    payload = {
        "schema_version": 1,
        "scope": "current_ssepnet_supervision_training_step_profile",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    (output_dir / "submission_training_profile.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return payload


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    if not args.batch_file.is_file():
        prepare_batch(args.batch_file)
    for treatment in TREATMENTS:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--treatment",
            treatment,
            "--batch-file",
            str(args.batch_file),
            "--train-root",
            str(args.train_root),
            "--output-dir",
            str(args.output_dir),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--repeats",
            str(args.repeats),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    return combine(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-batch", action="store_true")
    group.add_argument("--treatment", choices=tuple(TREATMENTS))
    group.add_argument("--all", action="store_true")
    parser.add_argument("--batch-file", type=Path, default=Path("/tmp/aquadataset_training_profile_batch.pt"))
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "aquadataset_submission_training_profile")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prepare_batch:
        payload = prepare_batch(args.batch_file)
        print(json.dumps({"batch": str(args.batch_file), "valid_pairs": payload["valid_pair_count"]}))
    elif args.all:
        payload = run_all(args)
        print(json.dumps({"treatments": len(payload["rows"]), "output": str(args.output_dir)}))
    else:
        payload = profile_worker(args)
        print(json.dumps({"treatment": payload["treatment"], "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
