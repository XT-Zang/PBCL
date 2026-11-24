from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def checkpoint_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("pretrained checkpoint must contain a state-dict mapping")
    for key in ("state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def normalize_checkpoint_key(key: str) -> str:
    normalized = str(key)
    prefixes = ("module.", "model.", "backbone.", "encoder.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized.removeprefix(prefix)
                changed = True
                break
    return normalized


def inflate_stem_weight(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.ndim != 4 or target.ndim != 4:
        raise RuntimeError("detail-backbone stem weights must be four-dimensional")
    if source.shape[0] != target.shape[0] or source.shape[2:] != target.shape[2:]:
        raise RuntimeError(
            "detail-backbone stem shape mismatch: "
            f"checkpoint={tuple(source.shape)} model={tuple(target.shape)}"
        )
    if source.shape[1] == target.shape[1]:
        return source
    if source.shape[1] != 3 or target.shape[1] != 4:
        raise RuntimeError(
            "cannot adapt detail-backbone stem from "
            f"{source.shape[1]} to {target.shape[1]} channels"
        )
    return torch.cat((source, source.mean(dim=1, keepdim=True)), dim=1)


def load_pruned_backbone_checkpoint(
    module: nn.Module,
    checkpoint_path: str | Path,
    *,
    stem_key: str,
    ignored_prefixes: tuple[str, ...],
    allowed_missing_prefixes: tuple[str, ...] = (),
    source_transform: (
        Callable[
            [Mapping[str, torch.Tensor]],
            Mapping[str, torch.Tensor],
        ]
        | None
    ) = None,
    optional_target_suffixes: tuple[str, ...] = (
        "relative_position_index",
        "num_batches_tracked",
    ),
) -> dict[str, list[str]]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"detail-backbone checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    source = {
        normalize_checkpoint_key(str(key)): value
        for key, value in checkpoint_state_dict(checkpoint).items()
        if isinstance(value, torch.Tensor)
    }
    if source_transform is not None:
        source = dict(source_transform(source))
    target = module.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    unexpected: list[str] = []
    for key, value in source.items():
        if key in target:
            candidate = inflate_stem_weight(value, target[key]) if key == stem_key else value
            if candidate.shape != target[key].shape:
                raise RuntimeError(
                    f"detail-backbone shape mismatch for {key}: "
                    f"checkpoint={tuple(candidate.shape)} model={tuple(target[key].shape)}"
                )
            loadable[key] = candidate
        elif key.startswith(ignored_prefixes):
            ignored.append(key)
        else:
            unexpected.append(key)

    optional = {
        key
        for key in target
        if any(key.endswith(suffix) for suffix in optional_target_suffixes)
    }
    all_missing = set(target) - set(loadable) - optional
    initialized = {
        key for key in all_missing if key.startswith(allowed_missing_prefixes)
    }
    missing = sorted(all_missing - initialized)
    if unexpected:
        raise RuntimeError(
            "unexpected detail-backbone checkpoint keys: "
            + ", ".join(sorted(unexpected))
        )
    if missing:
        raise RuntimeError(
            "missing detail-backbone checkpoint keys: " + ", ".join(missing)
        )
    loaded_from_source = set(loadable)
    for key in optional | initialized:
        loadable.setdefault(key, target[key])
    module.load_state_dict(loadable, strict=True)
    return {
        "loaded": sorted(loaded_from_source),
        "ignored": sorted(ignored),
        "initialized": sorted(initialized),
        "missing": [],
        "unexpected": [],
    }


__all__ = [
    "checkpoint_state_dict",
    "inflate_stem_weight",
    "load_pruned_backbone_checkpoint",
    "normalize_checkpoint_key",
]
