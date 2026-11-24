from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .aquadataset_wprf import WPRF_SOURCE_COMMIT
from .vegasdenseclose import NATIVE_SIZE, PADDED_SIZE, SPLIT_COUNTS


VEGAS_WPRF_SCHEMA_VERSION = 1
VEGAS_WPRF_TARGET_ROOT = "instances/wprf_matched_v1"
VEGAS_WPRF_PAIR_SALT = "vegasdenseclose-wprf-matched-v1"


def vegas_wprf_parameters() -> dict[str, object]:
    return {
        "source_commit": WPRF_SOURCE_COMMIT,
        "source": "augmented_instance_union",
        "native_size": NATIVE_SIZE,
        "padded_size": PADDED_SIZE,
        "padding": [0, 0, 22, 22],
        "padding_excluded_by_valid_mask": True,
        "dynamic_after_augmentation_and_padding": True,
        "phi": "closing_3x3_then_zhang_suen_thinning",
        "grid_stride": 4,
        "neighborhood": 8,
        "scales": [1, 2, 4, 8, 16],
        "gamma": [0.2] * 5,
        "sources_per_scale": 32,
        "pair_salt": VEGAS_WPRF_PAIR_SALT,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(root: Path, split: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "splits" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_vegas_wprf_manifest(
    cache_root: str | Path,
    *,
    instance_target_root: str | Path = "instances/targets",
    target_root: str | Path = VEGAS_WPRF_TARGET_ROOT,
    splits: Sequence[str] = ("train", "val", "test"),
) -> dict[str, Any]:
    root = Path(cache_root).expanduser().resolve()
    instance_root = Path(instance_target_root)
    if not instance_root.is_absolute():
        instance_root = root / instance_root
    output = Path(target_root)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    for split in splits:
        records = _records(root, split)
        split_counts[split] = len(records)
        if len(records) != SPLIT_COUNTS[split]:
            raise ValueError(f"Vegas WPRF split count mismatch for {split}")
        for record in records:
            tile_id = str(record["tile_id"])
            source = instance_root / f"{tile_id}.npz"
            if not source.is_file():
                raise FileNotFoundError(source)
            rows.append(
                {
                    "split": split,
                    "tile_id": tile_id,
                    "source_instance_sha256": _sha256(source),
                }
            )
    manifest = {
        "schema_version": VEGAS_WPRF_SCHEMA_VERSION,
        "method": "vegas_matched_wprf_dynamic_targets",
        "parameters": vegas_wprf_parameters(),
        "cache_root": str(root),
        "instance_target_root": str(instance_root),
        "target_root": str(output),
        "split_counts": split_counts,
        "target_count": len(rows),
        "sources": rows,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".manifest-", suffix=".json.tmp", dir=output
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output / "manifest.json")
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def audit_vegas_wprf_manifest(
    cache_root: str | Path,
    *,
    target_root: str | Path = VEGAS_WPRF_TARGET_ROOT,
) -> dict[str, Any]:
    root = Path(cache_root).expanduser().resolve()
    output = Path(target_root)
    if not output.is_absolute():
        output = root / output
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VEGAS_WPRF_SCHEMA_VERSION:
        raise ValueError("unsupported Vegas WPRF manifest schema")
    if manifest.get("parameters") != vegas_wprf_parameters():
        raise ValueError("Vegas WPRF manifest parameters do not match the locked contract")
    rows = manifest.get("sources")
    if not isinstance(rows, list) or len(rows) != sum(SPLIT_COUNTS.values()):
        raise ValueError("Vegas WPRF source coverage is incomplete")
    instance_root = Path(str(manifest["instance_target_root"]))
    for row in rows:
        source = instance_root / f"{row['tile_id']}.npz"
        if _sha256(source) != row["source_instance_sha256"]:
            raise ValueError(f"Vegas WPRF source hash mismatch: {source}")
    return {
        "manifest": str(manifest_path),
        "target_count": len(rows),
        "split_counts": dict(manifest["split_counts"]),
        "parameters": vegas_wprf_parameters(),
        "valid": True,
    }


__all__ = [
    "VEGAS_WPRF_PAIR_SALT",
    "VEGAS_WPRF_TARGET_ROOT",
    "audit_vegas_wprf_manifest",
    "build_vegas_wprf_manifest",
    "vegas_wprf_parameters",
]
