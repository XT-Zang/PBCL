#!/usr/bin/env python3
"""Run and resume the geographically isolated PBCL major-revision campaign."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoseg.datasets.geographic_holdout import prepare_instance_targets, sha256_file
from geoseg.experiments.pbcl_revision import DATA_ROOT, OUTPUT_ROOT
from geoseg.runtime import load_experiment


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runs() -> list[tuple[str, int]]:
    result = [(method, seed) for seed in (42, 43, 44) for method in ("boundary", "full")]
    result += [("pbcl_only", seed) for seed in (42, 43, 44)]
    result += [("base", seed) for seed in (42, 43, 44)]
    result += [(method, seed) for method in ("no_four_contact", "noncontact") for seed in (42, 43, 44)]
    result += [("wprf_gt", seed) for seed in (42, 43, 44)]
    result += [("four_contact", 42)]
    gate = OUTPUT_ROOT / "analysis" / "wprf_verification.json"
    if gate.exists() and json.loads(gate.read_text()).get("valid_support_gate_passed"):
        result += [("wprf_valid", seed) for seed in (42, 43, 44)]
    return result


def smoke() -> None:
    import rasterio  # Load GDAL before model dependencies.
    from dataclasses import replace
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, Subset
    from geoseg.datasets.aquav3_dataset import aquav3_collate
    from geoseg.engine.task import SegmentationTask
    from geoseg.runtime.aquav3_builder import build_aquav3_components

    for method in ("boundary", "full"):
        spec = load_experiment(ROOT / "configs/aquadataset" / f"{method}.py", build=False).spec
        directory = OUTPUT_ROOT / "preflight" / "smoke" / method
        spec = replace(spec, artifacts={**spec.artifacts, "weights_path": str(directory)})
        spec, components = build_aquav3_components(spec)
        components.datamodule.setup("fit")
        val_dataset = components.datamodule.val_dataset
        positives = [index for index, row in enumerate(val_dataset.entries) if not row["is_background"]][:8]
        negatives = [index for index, row in enumerate(val_dataset.entries) if row["is_background"]][:8]
        val_loader = DataLoader(Subset(val_dataset, positives + negatives), batch_size=8, collate_fn=aquav3_collate)
        trainer = pl.Trainer(accelerator="gpu", devices=1, precision="bf16-mixed", fast_dev_run=2,
                             logger=False, enable_checkpointing=False, enable_progress_bar=False,
                             gradient_clip_val=1.0)
        trainer.fit(SegmentationTask(components.config),
                    train_dataloaders=components.datamodule.train_dataloader(), val_dataloaders=val_loader)
        write_json(directory / "complete.json", {"success": True, "train_batches": 2, "val_batches": 2})
        del trainer, components
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    write_json(OUTPUT_ROOT / "preflight" / "smoke_complete.json", {"success": True, "methods": ["boundary", "full"], "finished": now()})


def train() -> None:
    if not (OUTPUT_ROOT / "preflight/smoke_complete.json").exists():
        raise RuntimeError("Run the Boundary/Full smoke first")
    plan = runs()
    write_json(OUTPUT_ROOT / "run_queue.json", [{"method": method, "seed": seed} for method, seed in plan])
    for method, seed in plan:
        config = ROOT / "configs/aquadataset" / f"{method}.py"
        spec = load_experiment(config, build=False, seed=seed).spec
        directory = Path(spec.artifacts["weights_path"])
        complete_path = directory / "training_complete.json"
        if complete_path.exists():
            record = json.loads(complete_path.read_text())
            if record["spec"] != spec.to_dict():
                raise RuntimeError(f"Completed run configuration differs: {directory}")
            if sha256_file(Path(record["checkpoint"])) != record["checkpoint_sha256"]:
                raise RuntimeError(f"Completed run checkpoint changed: {directory}")
            continue
        write_json(OUTPUT_ROOT / "progress.json", {"stage": "train", "method": method, "seed": seed, "updated": now()})
        resume = directory / "_state/resume_last.ckpt"
        for attempt in range(2):
            command = [sys.executable, "-m", "geoseg", "train", "--config", str(config), "--seed", str(seed)]
            if resume.exists():
                command += ["--resume-checkpoint", str(resume)]
            log = OUTPUT_ROOT / "logs" / f"{method}_seed{seed}_attempt{attempt}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                               GEOSEG_EPOCH_STATE_CHECKPOINT_EVERY="5")
            print(f"{now()} training {method} seed {seed}, log {log}", flush=True)
            with log.open("a") as handle:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                break
            if attempt == 1 or not resume.exists():
                raise RuntimeError(f"Training failed for {method} seed {seed}, see {log}")
        checkpoint = directory / f"{method}_seed{seed}-best_overall.ckpt"
        if not checkpoint.is_file() or not (directory / "deadhesion_val/epoch_040.json").is_file():
            raise RuntimeError(f"Training completion evidence missing: {directory}")
        write_json(complete_path, {"finished": now(), "spec": spec.to_dict(), "checkpoint": str(checkpoint),
                                  "checkpoint_sha256": sha256_file(checkpoint)})
    write_json(OUTPUT_ROOT / "training_complete.json", {"finished": now(), "runs": len(plan), "test_evaluated": False})


def evaluate() -> None:
    plan = runs()
    completed = OUTPUT_ROOT / "training_complete.json"
    if not completed.exists() or json.loads(completed.read_text())["runs"] != len(plan):
        raise RuntimeError("Finish all validation-selected training runs before formal Test evaluation")
    for method, seed in plan:
        config = ROOT / "configs/aquadataset" / f"{method}.py"
        spec = load_experiment(config, build=False, seed=seed).spec
        directory = Path(spec.artifacts["weights_path"])
        record = json.loads((directory / "training_complete.json").read_text())
        for split in ("val", "test"):
            output = OUTPUT_ROOT / "evaluation" / method / f"seed{seed}" / split
            if (output / "evaluation_complete.json").exists():
                continue
            write_json(OUTPUT_ROOT / "progress.json", {"stage": "evaluate", "method": method, "seed": seed, "split": split, "updated": now()})
            command = [sys.executable, "-m", "geoseg", "evaluate", "--config", str(config), "--seed", str(seed),
                       "--checkpoint", record["checkpoint"], "--split", split, "--out-dir", str(output),
                       "--save-predictions", "--save-probabilities", "--probability-dtype", "float32", "--strict"]
            log = OUTPUT_ROOT / "logs" / f"evaluate_{method}_seed{seed}_{split}.log"
            with log.open("a") as handle:
                subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            write_json(output / "evaluation_complete.json", {"finished": now(), "checkpoint_sha256": record["checkpoint_sha256"]})


def snapshot(name: str = "source_snapshot") -> None:
    destination = OUTPUT_ROOT / name
    if destination.exists():
        return
    destination.mkdir(parents=True)
    for directory in ("geoseg", "scripts", "config", "tools", "tests"):
        shutil.copytree(ROOT / directory, destination / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "train_supervision.py", destination)
    (destination / "git_head.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True))
    with (destination / "working_changes.patch").open("w") as handle:
        subprocess.run(["git", "diff", "--", "geoseg", "scripts", "config", "tests", "train_supervision.py"], cwd=ROOT, stdout=handle, check=True)
    files = {str(path.relative_to(destination)): sha256_file(path) for path in sorted(destination.rglob("*")) if path.is_file()}
    write_json(destination / "files_sha256.json", files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "train", "evaluate", "all"))
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "prepare":
            prepare_instance_targets(DATA_ROOT)
        if args.stage == "smoke":
            smoke()
        if args.stage in {"train", "all"}:
            snapshot()
            train()
        if args.stage in {"evaluate", "all"}:
            evaluate()
            from scripts.analyze_pbcl_revision import analyze
            analyze()


if __name__ == "__main__":
    main()
