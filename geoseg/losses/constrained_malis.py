from __future__ import annotations

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import threading

import numpy as np
import torch
import torch.nn.functional as F


MALIS_SOURCE_COMMIT = "8386d68c87c0a07691c4c0a11b835e0bb33a93db"
_BUILD_LOCK = threading.Lock()
_EXECUTOR_LOCK = threading.Lock()
_EXECUTORS: dict[int, ThreadPoolExecutor] = {}


def _get_executor(worker_count: int) -> ThreadPoolExecutor:
    with _EXECUTOR_LOCK:
        executor = _EXECUTORS.get(worker_count)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="geoseg-malis",
            )
            _EXECUTORS[worker_count] = executor
        return executor


def _native_source() -> Path:
    return Path(__file__).resolve().parent / "third_party" / "malis" / "malis_native.cpp"


def _native_library_path() -> Path:
    source = _native_source()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache_root = Path(
        os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    )
    return cache_root / "geoseg" / "malis" / digest / "libgeoseg_malis.so"


def _compile_native_library(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f".{os.getpid()}.tmp.so")
    command = [
        "g++",
        "-O3",
        "-march=native",
        "-std=c++17",
        "-shared",
        "-fPIC",
        str(_native_source()),
        "-o",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        os.replace(temporary, target)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "failed to compile the bundled Constrained MALIS compatibility library: "
            + (exc.stderr or exc.stdout or str(exc))
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def _native_function():
    target = _native_library_path()
    if not target.is_file():
        with _BUILD_LOCK:
            if not target.is_file():
                _compile_native_library(target)
    library = ctypes.CDLL(str(target))
    function = library.geoseg_malis_loss_weights
    pointer_u64 = ctypes.POINTER(ctypes.c_uint64)
    pointer_f32 = ctypes.POINTER(ctypes.c_float)
    function.argtypes = [
        ctypes.c_uint64,
        pointer_u64,
        ctypes.c_uint64,
        pointer_u64,
        pointer_u64,
        pointer_f32,
        ctypes.c_int,
        pointer_u64,
    ]
    function.restype = None
    function._library_reference = library
    return function


@lru_cache(maxsize=1)
def _native_sample_function():
    library = _native_function()._library_reference
    function = library.geoseg_constrained_malis_sample
    pointer_u64 = ctypes.POINTER(ctypes.c_uint64)
    pointer_u8 = ctypes.POINTER(ctypes.c_uint8)
    pointer_f32 = ctypes.POINTER(ctypes.c_float)
    pointer_i32 = ctypes.POINTER(ctypes.c_int)
    function.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        pointer_f32,
        pointer_u64,
        pointer_u8,
        pointer_f32,
        pointer_f32,
        pointer_u64,
        pointer_u64,
        pointer_i32,
    ]
    function.restype = None
    function._library_reference = library
    return function


@lru_cache(maxsize=8)
def _edge_nodes(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = int(height), int(width)
    nodes = np.arange(height * width, dtype=np.uint64).reshape(height, width)
    invalid = np.iinfo(np.uint64).max
    first = np.stack((nodes, nodes), axis=0)
    second = np.full_like(first, invalid)
    second[0, :, :-1] = nodes[:, 1:]
    second[1, :-1, :] = nodes[1:, :]
    return np.ascontiguousarray(first), np.ascontiguousarray(second)


def instance_affinity_targets(
    instance_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(instance_map, dtype=np.uint64)
    if labels.ndim != 2:
        raise ValueError(f"MALIS instance map must be 2-D, got {labels.shape}")
    height, width = labels.shape
    targets = np.zeros((2, height, width), dtype=np.float32)
    known = np.zeros((2, height, width), dtype=bool)
    first_h, second_h = labels[:, :-1], labels[:, 1:]
    first_v, second_v = labels[:-1, :], labels[1:, :]
    targets[0, :, :-1] = (
        (first_h > 0) & (first_h == second_h)
    ).astype(np.float32)
    targets[1, :-1, :] = (
        (first_v > 0) & (first_v == second_v)
    ).astype(np.float32)
    known[0, :, :-1] = (first_h > 0) | (second_h > 0)
    known[1, :-1, :] = (first_v > 0) | (second_v > 0)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != labels.shape:
            raise ValueError(f"MALIS valid mask mismatch: {valid.shape} vs {labels.shape}")
        known[0, :, :-1] &= valid[:, :-1] & valid[:, 1:]
        known[1, :-1, :] &= valid[:-1, :] & valid[1:, :]
    return targets, known


def _raw_malis_weights(
    segmentation: np.ndarray,
    edge_weights: np.ndarray,
    *,
    positive_pass: bool,
) -> np.ndarray:
    labels = np.ascontiguousarray(segmentation, dtype=np.uint64)
    affinities = np.ascontiguousarray(edge_weights, dtype=np.float32)
    if affinities.ndim != 3 or affinities.shape[0] != 2:
        raise ValueError(f"MALIS affinities must be (2,H,W), got {affinities.shape}")
    if labels.shape != affinities.shape[1:]:
        raise ValueError("MALIS segmentation and affinity shapes do not match")
    node_first, node_second = _edge_nodes(*labels.shape)
    output = np.zeros(affinities.size, dtype=np.uint64)
    function = _native_function()
    function(
        ctypes.c_uint64(labels.size),
        labels.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ctypes.c_uint64(affinities.size),
        node_first.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        node_second.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        affinities.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(1 if positive_pass else 0),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    )
    return output.reshape(affinities.shape)


def constrained_malis_weights(
    predicted_affinities: np.ndarray,
    instance_map: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    affinity_targets: np.ndarray | None = None,
    known_affinities: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the official normalized positive and negative MALIS passes."""
    prediction = np.asarray(predicted_affinities, dtype=np.float32)
    if affinity_targets is None or known_affinities is None:
        targets, known = instance_affinity_targets(instance_map, valid_mask)
    else:
        targets = np.asarray(affinity_targets, dtype=np.float32)
        known = np.asarray(known_affinities, dtype=bool)
    if prediction.shape != targets.shape:
        raise ValueError(f"MALIS prediction mismatch: {prediction.shape} vs {targets.shape}")
    labels = np.asarray(instance_map, dtype=np.uint64)
    instance_count = int(np.count_nonzero(np.unique(labels) > 0))
    zero = np.zeros_like(prediction, dtype=np.float32)
    if instance_count == 0:
        return zero.copy(), zero.copy()
    result: list[np.ndarray] = []
    for positive in (False, True):
        if not positive and instance_count < 2:
            result.append(zero.copy())
            continue
        constrained = prediction.copy()
        constraint_edges = known & (targets == float(not positive))
        constrained[constraint_edges] = float(not positive)
        weights = _raw_malis_weights(
            labels,
            constrained,
            positive_pass=positive,
        ).astype(np.float32)
        weights[targets == float(not positive)] = 0.0
        weights[~known] = 0.0
        total = float(weights.sum(dtype=np.float64))
        if total > 0.0:
            weights /= total
        result.append(weights)
    return result[0], result[1]


def constrained_malis_loss(
    affinity_logits: torch.Tensor,
    instance_maps: torch.Tensor,
    *,
    valid_masks: torch.Tensor | None = None,
    cpu_workers: int = 6,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if affinity_logits.ndim != 4 or affinity_logits.shape[1] != 2:
        raise ValueError(
            f"Constrained MALIS requires affinity logits (B,2,H,W), got {tuple(affinity_logits.shape)}"
        )
    if instance_maps.ndim == 3:
        instance_maps = instance_maps.unsqueeze(1)
    if instance_maps.ndim != 4 or instance_maps.shape[1] != 1:
        raise ValueError("Constrained MALIS requires instance maps (B,1,H,W)")
    logits = affinity_logits.float()
    if logits.shape[-2:] != instance_maps.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=instance_maps.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    probabilities = torch.sigmoid(logits)
    batch_size, _, height, width = probabilities.shape
    if batch_size == 0:
        return logits.sum() * 0.0, {
            "malis_active_samples": 0,
            "malis_positive_edges": 0,
            "malis_negative_edges": 0,
            "malis_cpu_workers": 1,
        }

    detached = np.ascontiguousarray(probabilities.detach().cpu().numpy(), dtype=np.float32)
    labels = np.ascontiguousarray(instance_maps.detach().cpu().numpy()[:, 0], dtype=np.uint64)
    valid_cpu = None
    if valid_masks is not None:
        valid = valid_masks
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.shape[-2:] != instance_maps.shape[-2:]:
            valid = F.interpolate(valid.float(), size=instance_maps.shape[-2:], mode="nearest")
        valid_cpu = np.ascontiguousarray(valid.detach().cpu().numpy()[:, 0] > 0.5, dtype=np.uint8)

    targets_np = np.zeros((batch_size, 2, height, width), dtype=np.float32)
    weights_np = np.zeros((batch_size, 2, height, width), dtype=np.float32)
    active_indices = [i for i in range(batch_size) if np.any(labels[i] > 0)]
    worker_count = max(1, min(int(cpu_workers), max(1, len(active_indices))))

    sample_fn = _native_sample_function()
    pointer_u64 = ctypes.POINTER(ctypes.c_uint64)
    pointer_u8 = ctypes.POINTER(ctypes.c_uint8)
    pointer_f32 = ctypes.POINTER(ctypes.c_float)

    def run_sample(index: int) -> tuple[int, int, int]:
        pos_edges = ctypes.c_uint64(0)
        neg_edges = ctypes.c_uint64(0)
        is_active = ctypes.c_int(0)
        valid_ptr = (
            valid_cpu[index].ctypes.data_as(pointer_u8)
            if valid_cpu is not None
            else None
        )
        sample_fn(
            ctypes.c_uint64(height),
            ctypes.c_uint64(width),
            detached[index].ctypes.data_as(pointer_f32),
            labels[index].ctypes.data_as(pointer_u64),
            valid_ptr,
            targets_np[index].ctypes.data_as(pointer_f32),
            weights_np[index].ctypes.data_as(pointer_f32),
            ctypes.byref(pos_edges),
            ctypes.byref(neg_edges),
            ctypes.byref(is_active),
        )
        return int(is_active.value), int(pos_edges.value), int(neg_edges.value)

    if worker_count == 1:
        stats = [run_sample(index) for index in active_indices]
    else:
        executor = _get_executor(worker_count)
        stats = list(executor.map(run_sample, active_indices))

    active_samples = sum(item[0] for item in stats)
    positive_pairs = sum(item[1] for item in stats)
    negative_pairs = sum(item[2] for item in stats)

    weights_t = torch.from_numpy(weights_np).to(
        device=probabilities.device, dtype=probabilities.dtype
    )
    targets_t = torch.from_numpy(targets_np).to(
        device=probabilities.device, dtype=probabilities.dtype
    )
    loss = (weights_t * (probabilities - targets_t).square()).sum(dim=(1, 2, 3)).mean()
    return loss, {
        "malis_active_samples": active_samples,
        "malis_positive_edges": positive_pairs,
        "malis_negative_edges": negative_pairs,
        "malis_cpu_workers": worker_count,
    }


def constrained_malis_batch_weights(
    predicted_affinities: np.ndarray,
    instance_maps: np.ndarray,
    *,
    valid_masks: np.ndarray | None = None,
    cpu_workers: int = 6,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    """Compute independent per-image MALIS targets/weights on bounded CPU threads."""
    predictions = np.asarray(predicted_affinities, dtype=np.float32)
    labels = np.asarray(instance_maps, dtype=np.uint64)
    if predictions.ndim != 4 or predictions.shape[1] != 2:
        raise ValueError("batched MALIS affinities must be (B,2,H,W)")
    if labels.ndim != 3 or labels.shape != (predictions.shape[0], *predictions.shape[2:]):
        raise ValueError("batched MALIS labels must be (B,H,W) and align with affinities")
    valid = None if valid_masks is None else np.asarray(valid_masks, dtype=bool)
    if valid is not None and valid.shape != labels.shape:
        raise ValueError("batched MALIS valid masks do not align with labels")
    batch_size = int(predictions.shape[0])
    zero = np.zeros_like(predictions[0], dtype=np.float32)
    computed: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = [None] * batch_size
    active_indices: list[int] = []
    for index in range(batch_size):
        if not np.any(labels[index] > 0):
            target, _known = instance_affinity_targets(
                labels[index], None if valid is None else valid[index]
            )
            computed[index] = (target, zero.copy(), zero.copy())
        else:
            active_indices.append(index)

    def compute_sample(index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_np, _known = instance_affinity_targets(
            labels[index], None if valid is None else valid[index]
        )
        negative, positive = constrained_malis_weights(
            predictions[index],
            labels[index],
            valid_mask=None if valid is None else valid[index],
            affinity_targets=target_np,
            known_affinities=_known,
        )
        return target_np, negative, positive

    worker_count = max(1, min(int(cpu_workers), max(1, len(active_indices))))
    if worker_count == 1:
        active_results = [compute_sample(index) for index in active_indices]
    else:
        # ctypes.CDLL releases the GIL while the native MST routine runs, so
        # independent images can use separate CPU cores without changing the
        # per-image graph or reduction order. Keep this bounded: each worker
        # owns full-resolution MST scratch buffers.
        executor = _get_executor(worker_count)
        active_results = list(executor.map(compute_sample, active_indices))
    for index, result in zip(active_indices, active_results, strict=True):
        computed[index] = result
    if any(result is None for result in computed):
        raise AssertionError("MALIS batch worker left an uncomputed sample")
    return [result for result in computed if result is not None], worker_count


__all__ = [
    "MALIS_SOURCE_COMMIT",
    "constrained_malis_loss",
    "constrained_malis_batch_weights",
    "constrained_malis_weights",
    "instance_affinity_targets",
]
