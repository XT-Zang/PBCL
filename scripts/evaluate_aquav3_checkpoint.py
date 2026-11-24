#!/usr/bin/env python3
from __future__ import annotations

import rasterio  # Load GDAL/libtiff before model libraries bind an incompatible libjpeg.
import argparse
import csv
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Sampler

from geoseg.experiments.aquav3_protocol import RESULT_SCHEMA, coerce_result_row, write_result_rows
from geoseg.metrics.binary import add_binary_counts, binary_confusion_tensor, binary_global_stats, binary_valid_mask, empty_binary_counts
from geoseg.metrics.deadhesion import (
    DEADHESION_SAMPLE_FIELDS,
    CachedDeadhesionEvaluator,
    json_ready as deadhesion_json_ready,
)
from geoseg.models.aquav3_outputs import get_main_logits
from geoseg.losses.binary_seg import loss_batch_kwargs
from geoseg.evaluation import EvaluationRequest, evaluate_checkpoint
from geoseg.runtime import load_experiment


@contextmanager
def checkpoint_evaluation_environment(config_path: str | Path):
    """Skip initializer downloads only while rebuilding a model for checkpoint loading."""

    path = Path(config_path)
    variables = []
    if path.stem == "swinunet":
        variables.append("AQUAV3_SWINUNET_SKIP_PRETRAIN")
    if path.parent.name in {"aquav3_compare", "vegasdenseclosepbcl_v1"} and path.stem in {
        "unetformer",
        "abcnet",
    }:
        variables.append("AQUAV3_SWSL_RESNET18_SKIP_PRETRAIN")
    previous = {variable: os.environ.get(variable) for variable in variables}
    for variable in variables:
        os.environ[variable] = "1"
    try:
        yield
    finally:
        for variable in variables:
            if previous[variable] is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous[variable]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an AquaV3 binary segmentation checkpoint on val or test split.")
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("--checkpoint", default="", help="Checkpoint path; defaults to newest best checkpoint under config.weights_path.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", default="", help="Output directory; defaults to config.weights_path/eval_SPLIT.")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Save per-tile foreground probabilities as NumPy arrays.",
    )
    parser.add_argument(
        "--probability-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Storage dtype for saved probabilities (default: float16).",
    )
    parser.add_argument(
        "--separate-background-batches",
        action="store_true",
        help=(
            "Evaluate positive and background samples in separate sequential batches. "
            "This preserves the historical positive-only batch boundaries after backgrounds are added."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override val/test DataLoader workers; use 0 in socket-restricted evaluation environments.",
    )
    parser.add_argument("--measure-efficiency", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Require checkpoint keys to match exactly.")
    return parser.parse_args()


def validate_threshold_policy(split: str, threshold: float) -> None:
    if split == "test" and not math.isclose(float(threshold), 0.5, abs_tol=1e-12):
        raise ValueError("The test threshold is fixed at 0.5; tune thresholds on validation data only.")


def evaluation_loss_kwargs(loss, batch: Mapping, device: torch.device) -> dict[str, object]:
    requires_instances = bool(
        getattr(loss, "requires_instance_supervision", False)
    )
    requires_ewc = bool(getattr(loss, "requires_ewc_supervision", False))
    kwargs = loss_batch_kwargs(batch, device)
    if not requires_instances:
        kwargs.pop("instance_maps", None)
        kwargs.pop("instance_pairs", None)
    if not requires_ewc:
        kwargs.pop("ewc_weights", None)
    return kwargs


def can_compute_evaluation_loss(loss, kwargs: Mapping) -> bool:
    """Score the training objective only when its supervision is available."""
    return (
        loss is not None
        and not bool(getattr(loss, "training_only_affinity_head", False))
        and (
            not bool(getattr(loss, "requires_ewc_supervision", False))
            or "ewc_weights" in kwargs
        )
    )


def predict_with_optional_model_targets(
    net: torch.nn.Module,
    loss,
    image: torch.Tensor,
    batch: Mapping,
) -> tuple[object, dict[str, object]]:
    if (
        not net.training
        and bool(getattr(net, "training_only_affinity_head", False))
        and hasattr(net, "forward_inference")
    ):
        return {"out": net.forward_inference(image)}, {}
    loss_kwargs = evaluation_loss_kwargs(loss, batch, image.device)
    model_kwargs: dict[str, object] = {}
    prepare = getattr(loss, "model_forward_kwargs", None)
    if callable(prepare):
        model_kwargs = dict(prepare(**loss_kwargs))
    return net(image, **model_kwargs), loss_kwargs


def override_loader_workers(config, num_workers: int | None) -> None:
    if num_workers is None:
        return
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    for name in ("val_loader", "test_loader"):
        loader = getattr(config, name, None)
        if loader is None:
            continue
        if isinstance(loader, DataLoader):
            common = {
                "dataset": loader.dataset,
                "num_workers": num_workers,
                "collate_fn": loader.collate_fn,
                "pin_memory": bool(loader.pin_memory),
                "timeout": float(loader.timeout),
                "worker_init_fn": loader.worker_init_fn,
                "generator": loader.generator,
            }
            if loader.batch_size is None:
                common["batch_sampler"] = loader.batch_sampler
            else:
                common.update(
                    {
                        "batch_size": int(loader.batch_size),
                        "sampler": loader.sampler,
                        "drop_last": bool(loader.drop_last),
                    }
                )
            if num_workers > 0:
                common["persistent_workers"] = bool(loader.persistent_workers)
                if loader.prefetch_factor is not None:
                    common["prefetch_factor"] = int(loader.prefetch_factor)
                if loader.multiprocessing_context is not None:
                    common["multiprocessing_context"] = loader.multiprocessing_context
            rebuilt = DataLoader(**common)
            setattr(config, name, rebuilt)
            continue
        loader.num_workers = num_workers
        if num_workers == 0 and hasattr(loader, "persistent_workers"):
            loader.persistent_workers = False


class SequentialGroupBatchSampler(Sampler[list[int]]):
    """Yield stable positive batches followed by stable background batches."""

    def __init__(self, is_negative: list[bool], *, batch_size: int, drop_last: bool = False) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.groups = (
            [index for index, value in enumerate(is_negative) if not value],
            [index for index, value in enumerate(is_negative) if value],
        )

    def __iter__(self):
        for group in self.groups:
            for start in range(0, len(group), self.batch_size):
                batch = group[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return sum(len(group) // self.batch_size for group in self.groups)
        return sum(math.ceil(len(group) / self.batch_size) for group in self.groups)


def separate_background_batches(loader: DataLoader) -> tuple[DataLoader, dict[str, int]]:
    dataset = loader.dataset
    flags = list(getattr(dataset, "is_negative", []))
    if len(flags) != len(dataset):
        raise ValueError(
            "--separate-background-batches requires loader.dataset.is_negative "
            "with one Boolean value per sample"
        )
    if loader.batch_size is None:
        raise ValueError("--separate-background-batches requires a DataLoader with batch_size")
    sampler = SequentialGroupBatchSampler(
        flags,
        batch_size=int(loader.batch_size),
        drop_last=bool(loader.drop_last),
    )
    kwargs = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": int(loader.num_workers),
        "collate_fn": loader.collate_fn,
        "pin_memory": bool(loader.pin_memory),
        "timeout": float(loader.timeout),
        "worker_init_fn": loader.worker_init_fn,
        "generator": loader.generator,
    }
    if loader.num_workers > 0:
        kwargs["persistent_workers"] = bool(loader.persistent_workers)
        if loader.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(loader.prefetch_factor)
    grouped_loader = DataLoader(**kwargs)
    return grouped_loader, {
        "positive": sum(not value for value in flags),
        "negative": sum(flags),
        "batches": len(sampler),
    }


def resolve_checkpoint_path(config, checkpoint: str | Path | None = None) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    weights_path = Path(config.weights_path)
    weights_name = str(config.weights_name)
    if hasattr(config, "checkpoint_monitors"):
        primary_role = str(getattr(config, "checkpoint_primary_role", "overall"))
        prefix = "" if getattr(config, "plain_checkpoint_names", False) else f"{weights_name}-"
        primary = weights_path / f"{prefix}best_{primary_role}.ckpt"
        if primary.is_file():
            return primary
        raise FileNotFoundError(
            f"AquaDateset primary {primary_role} checkpoint is missing: {primary}. "
            "Pass --checkpoint explicitly to evaluate another validation-selected role."
        )
    candidates = []
    candidates.extend(weights_path.glob(f"{weights_name}.ckpt"))
    candidates.extend(weights_path.glob(f"{weights_name}-v*.ckpt"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError(
        f"No best checkpoint found under {weights_path} for weights_name={weights_name!r}. "
        "Pass --checkpoint explicitly to evaluate a periodic checkpoint."
    )


def load_checkpoint_into_net(net: torch.nn.Module, checkpoint_path: Path, strict: bool = False) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata: dict[str, object] = {}
    if isinstance(checkpoint, Mapping):
        for key in ("epoch", "global_step"):
            if key in checkpoint:
                metadata[key] = checkpoint[key]
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("net.") for key in state_dict):
        state_dict = {key.removeprefix("net."): value for key, value in state_dict.items() if key.startswith("net.")}
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
            if key.startswith("_orig_mod.")
        }
    incompatible = net.load_state_dict(state_dict, strict=strict)
    if strict:
        return metadata
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "checkpoint loaded with non-strict key differences:",
            {"missing": incompatible.missing_keys[:8], "unexpected": incompatible.unexpected_keys[:8]},
        )
    return metadata


def _finite(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_result_row(
    metadata: Mapping[str, object],
    seed: int,
    split: str,
    metrics: Mapping[str, float],
    efficiency: Mapping[str, float] | None = None,
) -> dict[str, object]:
    efficiency = efficiency or {}
    close_pair_recall = metrics.get("close_pair_separation_recall", metrics.get("close_pair_recall", ""))
    row = {
        "model": metadata.get("model", ""),
        "backbone": metadata.get("backbone", ""),
        "pretrain": metadata.get("pretrain", ""),
        "seed": seed,
        "split": split,
        "fg_iou": metrics.get("fg_iou", ""),
        "dice": metrics.get("fg_dice", metrics.get("dice", "")),
        "precision": metrics.get("fg_precision", metrics.get("precision", "")),
        "recall": metrics.get("fg_recall", metrics.get("recall", "")),
        "miou": metrics.get("miou", ""),
        "boundary_precision": metrics.get("boundary_precision", ""),
        "boundary_recall": metrics.get("boundary_recall", ""),
        "boundary_f1": metrics.get("boundary_f1", ""),
        "boundary_iou": metrics.get("boundary_iou", ""),
        "skeleton_f1": metrics.get("skeleton_f1", ""),
        "merge_rate": metrics.get("merge_rate", ""),
        "split_rate": metrics.get("split_rate", ""),
        "close_pair_recall": close_pair_recall,
        "close_pair_separation_recall": close_pair_recall,
        "object_count_abs_error": metrics.get("object_count_abs_error", ""),
        "normalized_object_count_abs_error": metrics.get("normalized_object_count_abs_error", ""),
        "count_nmae": metrics.get("normalized_object_count_abs_error", metrics.get("count_nmae", "")),
        "pq": metrics.get("pq", ""),
        "pq_tp": metrics.get("pq_tp", ""),
        "pq_fp": metrics.get("pq_fp", ""),
        "pq_fn": metrics.get("pq_fn", ""),
        "close_pair_touching_total": metrics.get("close_pair_touching_total", ""),
        "close_pair_touching_separated": metrics.get("close_pair_touching_separated", ""),
        "close_pair_touching_separation_recall": metrics.get("close_pair_touching_separation_recall", ""),
        "close_pair_1_2_total": metrics.get("close_pair_1_2_total", ""),
        "close_pair_1_2_separated": metrics.get("close_pair_1_2_separated", ""),
        "close_pair_1_2_separation_recall": metrics.get("close_pair_1_2_separation_recall", ""),
        "close_pair_3_5_total": metrics.get("close_pair_3_5_total", ""),
        "close_pair_3_5_separated": metrics.get("close_pair_3_5_separated", ""),
        "close_pair_3_5_separation_recall": metrics.get("close_pair_3_5_separation_recall", ""),
        "close_pair_6_10_total": metrics.get("close_pair_6_10_total", ""),
        "close_pair_6_10_separated": metrics.get("close_pair_6_10_separated", ""),
        "close_pair_6_10_separation_recall": metrics.get("close_pair_6_10_separation_recall", ""),
        "close_pair_1_5_total": metrics.get("close_pair_1_5_total", ""),
        "close_pair_1_5_separated": metrics.get("close_pair_1_5_separated", ""),
        "close_pair_1_5_separation_recall": metrics.get("close_pair_1_5_separation_recall", close_pair_recall),
        "params_m": efficiency.get("params_m", ""),
        "flops_g": efficiency.get("flops_g", ""),
        "ms_per_img": efficiency.get("ms_per_img", ""),
        "gpu_mem_mb": efficiency.get("gpu_mem_mb", ""),
    }
    return {key: _finite(row.get(key, "")) for key in RESULT_SCHEMA}


def _infer_dataset_root(config, loader) -> Path:
    dataset = getattr(loader, "dataset", None)
    root = getattr(dataset, "root", None)
    if root is None:
        root = getattr(config, "data_root", getattr(config, "DATA_ROOT", ""))
    if not root:
        raise ValueError("Could not infer AquaV3 dataset root from loader.dataset.root or config.data_root")
    return Path(root)


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_json_ready(dict(row)) for row in rows)


def _write_csv_with_fields(path: Path, rows: list[Mapping[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames and key not in extra_fields:
                extra_fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames) + extra_fields)
        writer.writeheader()
        writer.writerows(_json_ready(dict(row)) for row in rows)


def _legacy_sample_rows(deadhesion_rows: list[Mapping[str, object]]) -> list[dict[str, object]]:
    keys = (
        "tile_id",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "boundary_iou",
        "merge_rate",
        "split_rate",
        "close_pair_separation_recall",
        "object_count_abs_error",
        "normalized_object_count_abs_error",
        "pq",
    )
    return [{key: row.get(key, "") for key in keys} for row in deadhesion_rows]


def _write_deadhesion_outputs(out_dir: Path, split: str, rows: list[Mapping[str, object]], summary: Mapping[str, object]) -> None:
    split_dir = out_dir / split
    output_summary = deadhesion_json_ready(dict(summary))
    _write_csv_with_fields(split_dir / "deadhesion_sample_metrics.csv", rows, DEADHESION_SAMPLE_FIELDS)
    (split_dir / "deadhesion_summary.json").write_text(
        json.dumps(output_summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(split_dir / "deadhesion_summary.csv", [deadhesion_json_ready(dict(summary.get("overall", {})))])


def _prediction_path(out_dir: Path, split: str, tile_id: str) -> Path:
    return out_dir / split / "pred" / f"{tile_id}_pred.png"


def _probability_path(out_dir: Path, split: str, tile_id: str) -> Path:
    return out_dir / split / "probs" / f"{tile_id}_prob.npy"


def save_prediction_mask(path: Path, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred.astype(np.uint8) * 255).save(path)


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure_efficiency(net: torch.nn.Module, sample: torch.Tensor, device: torch.device) -> dict[str, float]:
    params_m = sum(param.numel() for param in net.parameters()) / 1e6
    result = {"params_m": float(params_m), "flops_g": math.nan, "ms_per_img": math.nan, "gpu_mem_mb": math.nan}
    net.eval()
    sample = sample[:1].to(device)
    inference = (
        net.forward_inference
        if bool(getattr(net, "training_only_affinity_head", False))
        and hasattr(net, "forward_inference")
        else net
    )
    try:
        from thop import profile

        flops, _ = profile(net, inputs=(sample,), verbose=False)
        result["flops_g"] = float(flops / 1e9)
    except Exception:
        pass
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(5):
            _ = inference(sample)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(20):
            _ = inference(sample)
        torch.cuda.synchronize(device)
        result["ms_per_img"] = float((time.perf_counter() - start) * 1000.0 / 20)
        result["gpu_mem_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    return result


@torch.no_grad()
def prepare_binary_batch_outputs(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float,
    probabilities_override: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> tuple[dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    """Prepare reusable evaluation arrays with one sigmoid and one bulk CPU transfer."""

    probabilities = (
        probabilities_override
        if probabilities_override is not None
        else torch.sigmoid(logits)
    )
    if probabilities.shape != logits.shape:
        raise ValueError(
            f"probability override shape {tuple(probabilities.shape)} does not match logits {tuple(logits.shape)}"
        )
    count_tensor = binary_confusion_tensor(
        logits,
        mask,
        threshold=threshold,
        probabilities=probabilities,
        valid_mask=valid_mask,
    )
    tp, fp, fn, tn = (int(value) for value in count_tensor.detach().cpu().tolist())
    if valid_mask is not None:
        probabilities = probabilities.masked_fill(~binary_valid_mask(valid_mask, probabilities), 0.0)
    packed = torch.stack((probabilities, mask.detach().to(probabilities.dtype))).detach().cpu().numpy()
    probability_array = packed[0]
    target_array = packed[1] > 0.5
    prediction_array = probability_array > threshold
    return (
        {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        probability_array,
        prediction_array,
        target_array,
    )


def truncate_evaluation_batch(batch: Mapping[str, object], limit: int) -> dict[str, object]:
    """Slice every batch-aligned value so max_samples also limits loss and metrics."""

    batch_size = int(batch["img"].shape[0])
    if limit >= batch_size:
        return dict(batch)
    truncated: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            truncated[key] = value[:limit]
        elif isinstance(value, list) and len(value) == batch_size:
            truncated[key] = value[:limit]
        elif isinstance(value, tuple) and len(value) == batch_size:
            truncated[key] = value[:limit]
        else:
            truncated[key] = value
    return truncated


def make_deadhesion_evaluator(config, loader, *, split: str):
    dataset_root = _infer_dataset_root(config, loader)
    class_name = getattr(config, "isaid_transfer_class_name", None)
    if class_name:
        from geoseg.metrics.isaid_transfer import CachedISAIDTransferEvaluator

        dataset = getattr(loader, "dataset", None)
        return CachedISAIDTransferEvaluator(
            dataset_root,
            class_name=str(class_name),
            split=split,
            boundary_width=2,
            boundary_tolerance=2,
            min_overlap_pixels=8,
            min_overlap_ratio=0.05,
            manifest_path=getattr(dataset, "manifest_path", None),
            cache_root=getattr(dataset, "cache_root", None),
        )
    return CachedDeadhesionEvaluator(
        dataset_root,
        split=split,
        boundary_width=2,
        boundary_tolerance=2,
        min_overlap_pixels=8,
        min_overlap_ratio=0.05,
    )


@torch.no_grad()
def evaluate(
    config,
    checkpoint_path: Path,
    split: str,
    threshold: float,
    out_dir: Path,
    save_predictions: bool,
    max_samples: int,
    measure: bool,
    strict: bool,
    save_probabilities: bool = False,
    probability_dtype: str = "float16",
    separate_background: bool = False,
) -> dict[str, object]:
    validate_threshold_policy(split=split, threshold=threshold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = config.net.to(device)
    checkpoint_metadata = load_checkpoint_into_net(net, checkpoint_path, strict=strict)
    net.eval()
    loader = config.test_loader if split == "test" and hasattr(config, "test_loader") else config.val_loader
    scope_batch_counts: dict[str, int] | None = None
    if separate_background:
        loader, scope_batch_counts = separate_background_batches(loader)
    if probability_dtype not in {"float16", "float32"}:
        raise ValueError(f"Unsupported probability dtype: {probability_dtype}")
    probability_numpy_dtype = np.float16 if probability_dtype == "float16" else np.float32
    deadhesion_evaluator = make_deadhesion_evaluator(config, loader, split=split)
    instance_segmentation = bool(getattr(config, "instance_segmentation", False))
    instance_metric = None
    if instance_segmentation:
        from torchmetrics.detection import MeanAveragePrecision

        instance_metric = MeanAveragePrecision(iou_type="segm").to(device)
    counts = empty_binary_counts()
    cohort_counts = {
        "positive": empty_binary_counts(),
        "background": empty_binary_counts(),
    }
    cohort_sample_counts = {"positive": 0, "background": 0}
    first_batch: torch.Tensor | None = None
    loss_total = 0.0
    loss_batches = 0
    loss = getattr(config, "loss", None)
    seen = 0
    for batch in loader:
        if max_samples:
            remaining = max_samples - seen
            if remaining <= 0:
                break
            batch = truncate_evaluation_batch(batch, remaining)
        image = batch["img"].to(device)
        mask = batch["gt_semantic_seg"].to(device)
        if first_batch is None:
            first_batch = image.detach().cpu()
        prediction, loss_kwargs = predict_with_optional_model_targets(
            net,
            loss,
            image,
            batch,
        )
        if can_compute_evaluation_loss(loss, loss_kwargs):
            loss_value = loss(
                prediction,
                mask,
                **loss_kwargs,
            )
            if torch.isfinite(loss_value.detach()):
                loss_total += float(loss_value.detach().cpu().item())
                loss_batches += 1
        logits = get_main_logits(prediction)
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"Nonfinite inference logits in {split} at sample {seen}")
        if logits.shape[-2:] != mask.shape[-2:]:
            logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        structure_instances = None
        probabilities_override = None
        if instance_segmentation:
            if not isinstance(prediction, Mapping):
                raise TypeError("Instance evaluation requires named query outputs")
            from geoseg.losses.mask2former_instance import instance_metric_targets
            from geoseg.models.mask2former_instance import (
                postprocess_mask2former_instances,
                union_instance_masks,
            )

            target_size = tuple(mask.shape[-2:])
            structure_instances = postprocess_mask2former_instances(
                prediction,
                size=target_size,
                score_threshold=float(
                    getattr(config, "instance_score_threshold", threshold)
                ),
                mask_probability_threshold=float(
                    getattr(config, "instance_mask_threshold", 0.5)
                ),
                max_instances=int(getattr(config, "instance_num_queries", 100)),
                valid_masks=loss_kwargs.get("valid_masks"),
            )
            probabilities_override = union_instance_masks(
                structure_instances,
                size=target_size,
                device=image.device,
            ).float()
            ap_predictions = postprocess_mask2former_instances(
                prediction,
                size=target_size,
                score_threshold=None,
                mask_probability_threshold=float(
                    getattr(config, "instance_mask_threshold", 0.5)
                ),
                max_instances=int(getattr(config, "instance_num_queries", 100)),
                valid_masks=loss_kwargs.get("valid_masks"),
            )
            ap_targets = instance_metric_targets(
                loss_kwargs["instance_maps"],
                max_instances=int(getattr(config, "instance_num_queries", 100)),
                valid_masks=loss_kwargs.get("valid_masks"),
            )
            assert instance_metric is not None
            instance_metric.update(ap_predictions, ap_targets)
        batch_counts, probabilities, preds, masks = prepare_binary_batch_outputs(
            logits,
            mask,
            threshold=threshold,
            probabilities_override=probabilities_override,
            valid_mask=batch.get("valid_mask"),
        )
        stored_probabilities = (
            torch.sigmoid(logits).detach().cpu().numpy()
            if instance_segmentation
            else probabilities
        )
        add_binary_counts(counts, batch_counts)
        raw_background = batch.get("is_background")
        if raw_background is not None:
            background_flags = torch.as_tensor(
                raw_background, device=logits.device, dtype=torch.bool
            ).flatten()
            if int(background_flags.numel()) != int(logits.shape[0]):
                raise ValueError("is_background must contain one flag per evaluation sample")
            probability_tensor = (
                probabilities_override
                if probabilities_override is not None
                else torch.sigmoid(logits)
            )
            valid_tensor = batch.get("valid_mask")
            for cohort, selector in (
                ("positive", ~background_flags),
                ("background", background_flags),
            ):
                if not bool(selector.any()):
                    continue
                cohort_sample_counts[cohort] += int(selector.sum().item())
                selected_valid = (
                    torch.as_tensor(valid_tensor, device=logits.device)[selector]
                    if valid_tensor is not None
                    else None
                )
                cohort_tensor = binary_confusion_tensor(
                    logits[selector],
                    mask[selector],
                    threshold=threshold,
                    probabilities=probability_tensor[selector],
                    valid_mask=selected_valid,
                )
                tp, fp, fn, tn = (
                    int(value) for value in cohort_tensor.detach().cpu().tolist()
                )
                add_binary_counts(
                    cohort_counts[cohort],
                    {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                )
        ids = batch.get("tile_id", batch.get("img_id", [f"sample_{seen + idx:06d}" for idx in range(image.shape[0])]))
        if structure_instances is not None:
            deadhesion_evaluator.update_instance_batch(
                structure_instances,
                masks,
                [str(item) for item in ids],
            )
        else:
            deadhesion_evaluator.update_batch(preds, masks, [str(item) for item in ids])
        for idx in range(image.shape[0]):
            tile_id = str(ids[idx])
            pred = np.squeeze(preds[idx]).astype(bool)
            gt = np.squeeze(masks[idx]).astype(bool)
            probability = np.squeeze(stored_probabilities[idx]).astype(
                np.float32, copy=False
            )
            if save_predictions:
                save_prediction_mask(_prediction_path(out_dir, split, tile_id), pred)
            if save_probabilities:
                probability_path = _probability_path(out_dir, split, tile_id)
                probability_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(
                    probability_path,
                    probability.astype(probability_numpy_dtype, copy=False),
                    allow_pickle=False,
                )
            seen += 1
            if max_samples and seen >= max_samples:
                break
        if max_samples and seen >= max_samples:
            break

    binary_stats = binary_global_stats(counts)
    cohort_stats = {
        name: binary_global_stats(values)
        for name, values in cohort_counts.items()
        if cohort_sample_counts[name] > 0
    }
    deadhesion_summary = deadhesion_evaluator.summary()
    deadhesion_summary["settings"]["threshold"] = threshold
    deadhesion_overall = dict(deadhesion_summary.get("overall", {}))
    metrics = {**binary_stats, **deadhesion_overall}
    metrics["background_false_positive_rate"] = (
        cohort_stats["background"]["bg_fp_rate"]
        if "background" in cohort_stats
        else binary_stats["bg_fp_rate"]
    )
    if instance_metric is not None:
        instance_stats = instance_metric.compute()
        metrics.update(
            {
                "instance_ap": float(instance_stats["map"].detach().cpu()),
                "instance_ap50": float(instance_stats["map_50"].detach().cpu()),
                "instance_ap75": float(instance_stats["map_75"].detach().cpu()),
                "instance_ar100": float(instance_stats["mar_100"].detach().cpu()),
            }
        )
    if loss_batches:
        metrics[f"{split}_loss"] = loss_total / loss_batches
    efficiency = measure_efficiency(net, first_batch, device) if measure and first_batch is not None else {"params_m": sum(p.numel() for p in net.parameters()) / 1e6}
    metadata = getattr(config, "result_metadata", {"model": getattr(config, "model_name", ""), "backbone": "", "pretrain": ""})
    result_row = build_result_row(metadata, seed=int(getattr(config, "seed", 42)), split=split, metrics=metrics, efficiency=efficiency)
    out_dir.mkdir(parents=True, exist_ok=True)
    deadhesion_rows = list(deadhesion_evaluator.rows)
    _write_csv(out_dir / f"{split}_sample_metrics.csv", _legacy_sample_rows(deadhesion_rows))
    _write_deadhesion_outputs(out_dir, split, deadhesion_rows, deadhesion_summary)
    write_result_rows(out_dir / f"{split}_summary.csv", [result_row])
    resolved_checkpoint = checkpoint_path.resolve()
    summary = {
        "checkpoint": str(resolved_checkpoint),
        "checkpoint_sha256": checkpoint_sha256(resolved_checkpoint),
        "split": split,
        "threshold": threshold,
        "sample_count": seen,
        "metrics": metrics,
        "cohorts": cohort_stats,
        "deadhesion": deadhesion_summary,
        "checkpoint_metadata": checkpoint_metadata,
        "strict_load": bool(strict),
        "evaluation_protocol": {
            "inference_precision": "fp32",
            "autocast_enabled": False,
            "separate_background_batches": bool(separate_background),
            "scope_batch_counts": scope_batch_counts,
            "probability_dtype": probability_dtype if save_probabilities else None,
            "saved_probability_kind": (
                "continuous_semantic_union_sigmoid"
                if save_probabilities and instance_segmentation
                else "sigmoid"
                if save_probabilities
                else None
            ),
            "prediction_kind": (
                "query_instances" if instance_segmentation else "connected_components"
            ),
            "instance_num_queries": (
                int(getattr(config, "instance_num_queries", 100))
                if instance_segmentation
                else None
            ),
            "instance_score_threshold": (
                float(getattr(config, "instance_score_threshold", 0.5))
                if instance_segmentation
                else None
            ),
            "instance_mask_threshold": (
                float(getattr(config, "instance_mask_threshold", 0.5))
                if instance_segmentation
                else None
            ),
            "instance_ap_score_threshold": None,
        },
        "provenance": {
            "architecture": str(getattr(config, "model_name", "")),
            "model": str(metadata.get("model", "")),
            "seed": int(getattr(config, "seed", 42)),
            "treatment": str(getattr(config, "treatment", "")),
            "run_name": str(getattr(config, "run_name", getattr(config, "weights_name", ""))),
            "params_m": result_row.get("params_m", ""),
            "pretrained_revision": str(
                getattr(config.net, "pretrained_revision", "")
            ),
            "pretrained_sha256": str(
                getattr(config.net, "pretrained_sha256", "")
            ),
        },
        "result_row": coerce_result_row(result_row),
    }
    (out_dir / f"{split}_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    with checkpoint_evaluation_environment(args.config):
        experiment = load_experiment(args.config, build=True)
    request = EvaluationRequest(
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        split=args.split,
        threshold=args.threshold,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        num_workers=args.num_workers,
        save_predictions=args.save_predictions,
        save_probabilities=args.save_probabilities,
        probability_dtype=args.probability_dtype,
        separate_background_batches=args.separate_background_batches,
        max_samples=args.max_samples,
        measure_efficiency=args.measure_efficiency,
        strict=args.strict,
    )
    summary = evaluate_checkpoint(experiment, request)
    print(json.dumps(_json_ready(summary.result_row), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
