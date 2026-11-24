from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


EWC_SCHEMA_VERSION = 1
EWC_W0 = 10.0
EWC_SIGMA = 7.5
EWC_POWER = 2.0
EWC_TARGET_ROOT = "instance_targets/ewc_v1"
EWC_MANIFEST_NAME = "manifest.json"


def ewc_parameters() -> dict[str, object]:
    return {
        "w0": EWC_W0,
        "sigma": EWC_SIGMA,
        "power": EWC_POWER,
        "class_weight": "per_tile_inverse_frequency",
        "single_instance": "d1_only",
        "negative_tile": "unit_weight",
        "semantic_labels_modified": False,
    }


def _instance_boundaries(instance_map: np.ndarray) -> list[np.ndarray]:
    from scipy import ndimage as ndi

    labels = np.asarray(instance_map)
    if labels.ndim != 2:
        raise ValueError(f"instance_map must be 2D, got {labels.shape}")
    if np.any(labels < 0):
        raise ValueError("instance_map must contain non-negative ids")
    boundaries: list[np.ndarray] = []
    structure = ndi.generate_binary_structure(2, 1)
    for instance_id in np.unique(labels):
        if int(instance_id) == 0:
            continue
        mask = labels == instance_id
        eroded = ndi.binary_erosion(mask, structure=structure, border_value=0)
        boundary = mask & ~eroded
        if np.any(boundary):
            boundaries.append(boundary)
    return boundaries


def ewc_weight_map(
    instance_map: np.ndarray,
    semantic_mask: np.ndarray,
    *,
    w0: float = EWC_W0,
    sigma: float = EWC_SIGMA,
    power: float = EWC_POWER,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build the loss-only EWC map without changing semantic labels.

    The map follows Jiwani et al.'s exponentially weighted boundary objective.
    Distances are computed to boundaries of distinct ground-truth instances.
    For a single instance, the official implementation's d1-only form is used.
    """

    from scipy import ndimage as ndi

    labels = np.asarray(instance_map)
    semantic = np.asarray(semantic_mask, dtype=bool)
    if labels.shape != semantic.shape:
        raise ValueError(
            f"instance/semantic shapes differ: {labels.shape} != {semantic.shape}"
        )
    if not math.isfinite(float(w0)) or float(w0) <= 0.0:
        raise ValueError("w0 must be finite and positive")
    if not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(float(power)) or float(power) <= 0.0:
        raise ValueError("power must be finite and positive")

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != labels.shape:
            raise ValueError("EWC valid mask must match the supervision shape")
        labels = np.where(valid_mask, labels, 0)
    boundaries = _instance_boundaries(labels)
    if not boundaries:
        return np.ones(labels.shape, dtype=np.float32)

    nearest = np.full(labels.shape, np.inf, dtype=np.float64)
    second = np.full(labels.shape, np.inf, dtype=np.float64)
    for boundary in boundaries:
        distance = ndi.distance_transform_edt(~boundary)
        becomes_nearest = distance < nearest
        second = np.where(becomes_nearest, nearest, np.minimum(second, distance))
        nearest = np.minimum(nearest, distance)

    distance_sum = nearest if len(boundaries) == 1 else nearest + second
    border = float(w0) * np.exp(
        -np.square(distance_sum) / (2.0 * float(sigma) ** 2)
    )
    foreground_fraction = float(
        semantic.mean() if valid_mask is None else semantic[valid_mask].mean()
    )
    class_weight = np.where(
        semantic,
        1.0 - foreground_fraction,
        foreground_fraction,
    )
    raw = class_weight + border
    if valid_mask is not None:
        raw[~valid_mask] = 0.0
    maximum = float(raw.max(initial=0.0))
    normalized = raw / maximum if maximum > 0.0 else np.zeros_like(raw)
    weight = np.exp(normalized) ** float(power)
    result = np.asarray(weight, dtype=np.float32)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("EWC weight map contains non-finite or non-positive values")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-",
        suffix=".json.tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _records(dataset_root: Path, split: str) -> list[dict[str, Any]]:
    path = dataset_root / "splits" / f"{split}.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _record_shape(record: Mapping[str, Any]) -> tuple[int, int]:
    """Return the native supervision shape before any runtime padding."""

    height = int(
        record.get(
            "valid_height",
            record.get("height", record.get("tile_size", 512)),
        )
    )
    width = int(
        record.get(
            "valid_width",
            record.get("width", record.get("tile_size", 512)),
        )
    )
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid native supervision shape {(height, width)}")
    return height, width


def _semantic_mask(
    dataset_root: Path,
    record: Mapping[str, Any],
    instance_map: np.ndarray,
) -> np.ndarray:
    shape = tuple(instance_map.shape)
    semantic = record.get("semantic")
    if not semantic:
        # Vegas manifests intentionally omit a redundant semantic file.  Its
        # semantic target is defined by the same canonical instance raster.
        return np.asarray(instance_map) > 0
    from geoseg.datasets.aquav3_dataset import _read_mask

    return _read_mask(dataset_root / str(semantic), shape).astype(bool, copy=False)


def _target_path(target_root: Path, tile_id: str) -> Path:
    return target_root / f"{tile_id}.npz"


def _write_ewc_target(path: Path, weight: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-",
        suffix=".npz.tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                ewc_weight=weight.astype(np.float32),
                ewc_schema_version=np.asarray(EWC_SCHEMA_VERSION, dtype=np.int16),
                ewc_w0=np.asarray(EWC_W0, dtype=np.float32),
                ewc_sigma=np.asarray(EWC_SIGMA, dtype=np.float32),
                ewc_power=np.asarray(EWC_POWER, dtype=np.float32),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_ewc_target(path: Path) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "ewc_weight",
                "ewc_schema_version",
                "ewc_w0",
                "ewc_sigma",
                "ewc_power",
            }
            missing = required - set(payload.files)
            if missing:
                raise ValueError(f"missing metadata fields {sorted(missing)}")
            schema = int(np.asarray(payload["ewc_schema_version"]).item())
            parameters = (
                float(np.asarray(payload["ewc_w0"]).item()),
                float(np.asarray(payload["ewc_sigma"]).item()),
                float(np.asarray(payload["ewc_power"]).item()),
            )
            stored_weight = np.asarray(payload["ewc_weight"])
            if stored_weight.dtype != np.float32:
                raise ValueError(
                    f"ewc_weight dtype must be float32, got {stored_weight.dtype}"
                )
            weight = stored_weight.copy()
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(
            f"stale or invalid EWC target {path}; regenerate with --overwrite: {exc}"
        ) from exc
    expected = (EWC_W0, EWC_SIGMA, EWC_POWER)
    if schema != EWC_SCHEMA_VERSION or any(
        not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-7)
        for actual, wanted in zip(parameters, expected, strict=True)
    ):
        raise ValueError(
            f"EWC target parameter mismatch in {path}: "
            f"schema={schema}, parameters={parameters}, expected={expected}; "
            "regenerate with --overwrite"
        )
    return weight


def build_ewc_targets(
    dataset_root: str | Path,
    *,
    instance_target_root: str | Path = "instance_targets/pbcl_gap5",
    target_root: str | Path = EWC_TARGET_ROOT,
    splits: Sequence[str] = ("train", "val", "test"),
    overwrite: bool = False,
    progress_every: int = 0,
    use_manifest_valid_mask: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    instance_root = Path(instance_target_root)
    if not instance_root.is_absolute():
        instance_root = dataset_root / instance_root
    output_root = Path(target_root)
    if not output_root.is_absolute():
        output_root = dataset_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / EWC_MANIFEST_NAME
    previous_by_tile: dict[str, Mapping[str, Any]] = {}
    if manifest_path.is_file():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid existing EWC manifest {manifest_path}; "
                f"regenerate with --overwrite: {exc}"
            ) from exc
        if previous_manifest.get("source_integrity") == "per_target_sha256_v1":
            if bool(previous_manifest.get("use_manifest_valid_mask")) != use_manifest_valid_mask and not overwrite:
                raise ValueError("EWC valid-mask policy changed; regenerate with --overwrite")
            previous_rows = previous_manifest.get("targets")
            if not isinstance(previous_rows, list):
                raise ValueError(f"invalid existing EWC manifest targets: {manifest_path}")
            previous_by_tile = {
                str(row["tile_id"]): row
                for row in previous_rows
                if isinstance(row, Mapping) and "tile_id" in row
            }

    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    split_manifest_sha256: dict[str, str] = {}
    seen_tile_ids: set[str] = set()
    processed = 0
    total = sum(len(_records(dataset_root, split)) for split in splits)
    for split in splits:
        records = _records(dataset_root, split)
        split_counts[str(split)] = len(records)
        split_manifest_sha256[str(split)] = _sha256(
            dataset_root / "splits" / f"{split}.jsonl"
        )
        for record in records:
            tile_id = str(record["tile_id"])
            if tile_id in seen_tile_ids:
                raise ValueError(f"duplicate EWC tile_id across splits: {tile_id}")
            seen_tile_ids.add(tile_id)
            destination = _target_path(output_root, tile_id)
            source_instance_sha256: str | None = None
            source_semantic_sha256: str | None = None
            source: Path | None = None
            if not bool(record.get("is_background", False)):
                source = instance_root / f"{tile_id}.npz"
                if not source.is_file():
                    raise FileNotFoundError(
                        f"missing instance target for EWC tile {tile_id}: {source}"
                    )
                source_instance_sha256 = _sha256(source)
                semantic_path = record.get("semantic")
                if semantic_path:
                    semantic_source = dataset_root / str(semantic_path)
                    if not semantic_source.is_file():
                        raise FileNotFoundError(
                            f"missing semantic target for EWC tile {tile_id}: "
                            f"{semantic_source}"
                        )
                    source_semantic_sha256 = _sha256(semantic_source)
            if destination.exists() and not overwrite:
                previous = previous_by_tile.get(tile_id)
                if previous is not None and (
                    previous.get("source_instance_sha256")
                    != source_instance_sha256
                    or previous.get("source_semantic_sha256")
                    != source_semantic_sha256
                ):
                    raise ValueError(
                        f"EWC source changed for {tile_id}; regenerate with --overwrite"
                    )
                weight = _load_ewc_target(destination)
            else:
                valid_mask = None
                if use_manifest_valid_mask:
                    from geoseg.datasets.aquav3_dataset import read_manifest_valid_mask

                    valid_mask = read_manifest_valid_mask(dataset_root / str(record["image"]), record)
                if bool(record.get("is_background", False)):
                    height, width = _record_shape(record) if valid_mask is None else valid_mask.shape
                    weight = np.ones((height, width), dtype=np.float32)
                else:
                    assert source is not None
                    with np.load(source, allow_pickle=False) as payload:
                        instance_map = np.asarray(payload["instance_map"], dtype=np.int64)
                    semantic = _semantic_mask(
                        dataset_root,
                        record,
                        instance_map,
                    )
                    weight = ewc_weight_map(instance_map, semantic, valid_mask=valid_mask)
                _write_ewc_target(destination, weight)
            rows.append(
                {
                    "split": str(split),
                    "tile_id": tile_id,
                    "path": str(destination),
                    "shape": list(weight.shape),
                    "dtype": str(weight.dtype),
                    "minimum": float(weight.min()),
                    "maximum": float(weight.max()),
                    "sha256": _sha256(destination),
                    "source_instance_sha256": source_instance_sha256,
                    "source_semantic_sha256": source_semantic_sha256,
                }
            )
            processed += 1
            if progress_every > 0 and (
                processed % int(progress_every) == 0 or processed == total
            ):
                print(f"prepared {processed}/{total} EWC targets", flush=True)

    manifest = {
        "schema_version": EWC_SCHEMA_VERSION,
        "method": "exponentially_weighted_boundary_cross_entropy_loss_only",
        "parameters": ewc_parameters(),
        "source_integrity": "per_target_sha256_v1",
        "dataset_root": str(dataset_root),
        "instance_target_root": str(instance_root),
        "target_root": str(output_root),
        "split_counts": split_counts,
        "split_manifest_sha256": split_manifest_sha256,
        "target_count": len(rows),
        "use_manifest_valid_mask": use_manifest_valid_mask,
        "targets": rows,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def audit_ewc_targets(
    dataset_root: str | Path,
    *,
    target_root: str | Path = EWC_TARGET_ROOT,
    splits: Sequence[str] = ("train", "val", "test"),
    use_manifest_valid_mask: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    root = Path(target_root)
    if not root.is_absolute():
        root = dataset_root / root
    manifest_path = root / EWC_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("use_manifest_valid_mask")) != use_manifest_valid_mask:
        raise ValueError("EWC manifest valid-mask policy does not match the dataset")
    if manifest.get("schema_version") != EWC_SCHEMA_VERSION:
        raise ValueError("unsupported EWC target manifest schema")
    if manifest.get("parameters") != ewc_parameters():
        raise ValueError(
            "EWC manifest parameters do not match the locked ewc_v1 contract"
        )
    if manifest.get("method") != "exponentially_weighted_boundary_cross_entropy_loss_only":
        raise ValueError("EWC manifest method does not match the locked contract")
    source_integrity = manifest.get("source_integrity")
    if source_integrity not in {None, "per_target_sha256_v1"}:
        raise ValueError(f"unsupported EWC source integrity policy: {source_integrity}")
    instance_root = Path(str(manifest.get("instance_target_root", ""))).expanduser()
    if source_integrity == "per_target_sha256_v1":
        if Path(str(manifest.get("dataset_root", ""))).expanduser().resolve() != dataset_root:
            raise ValueError("EWC manifest dataset_root does not match the audited dataset")
        if Path(str(manifest.get("target_root", ""))).expanduser().resolve() != root.resolve():
            raise ValueError("EWC manifest target_root does not match the audited target root")
        if not instance_root.is_absolute():
            raise ValueError("EWC manifest instance_target_root must be absolute")

    expected: dict[str, tuple[str, tuple[int, int], Mapping[str, Any]]] = {}
    expected_split_counts: dict[str, int] = {}
    expected_split_hashes: dict[str, str] = {}
    for split in splits:
        split_path = dataset_root / "splits" / f"{split}.jsonl"
        records = _records(dataset_root, split)
        expected_split_counts[split] = len(records)
        expected_split_hashes[split] = _sha256(split_path)
        for record in records:
            tile_id = str(record["tile_id"])
            if tile_id in expected:
                raise ValueError(f"duplicate EWC tile_id across splits: {tile_id}")
            shape = _record_shape(record)
            if use_manifest_valid_mask:
                import rasterio

                with rasterio.open(dataset_root / str(record["image"])) as image:
                    shape = image.shape
            expected[tile_id] = (split, shape, record)
    rows = manifest.get("targets")
    if not isinstance(rows, list):
        raise TypeError("EWC manifest targets must be a list")
    if any(not isinstance(row, Mapping) or "tile_id" not in row for row in rows):
        raise TypeError("EWC manifest target rows must be mappings with tile_id")
    row_tile_ids = [str(row["tile_id"]) for row in rows]
    if len(row_tile_ids) != len(set(row_tile_ids)):
        raise ValueError("EWC manifest contains duplicate tile_id rows")
    by_tile = {str(row["tile_id"]): row for row in rows}
    if set(by_tile) != set(expected):
        missing = sorted(set(expected) - set(by_tile))
        extra = sorted(set(by_tile) - set(expected))
        raise ValueError(f"EWC target coverage mismatch: missing={missing}, extra={extra}")
    if manifest.get("target_count") != len(expected) or len(rows) != len(expected):
        raise ValueError("EWC manifest target_count does not match target coverage")
    if manifest.get("split_counts") != expected_split_counts:
        raise ValueError("EWC manifest split_counts do not match dataset manifests")
    if source_integrity == "per_target_sha256_v1" and (
        manifest.get("split_manifest_sha256") != expected_split_hashes
    ):
        raise ValueError("EWC source split manifests changed; regenerate with --overwrite")
    physical_targets = {
        path.stem
        for path in root.glob("*.npz")
        if path.is_file()
    }
    if physical_targets != set(expected):
        missing = sorted(set(expected) - physical_targets)
        extra = sorted(physical_targets - set(expected))
        raise ValueError(
            f"EWC physical target coverage mismatch: missing={missing}, extra={extra}"
        )

    for tile_id, (split, expected_shape, record) in expected.items():
        row = by_tile[tile_id]
        if row.get("split") != split:
            raise ValueError(f"wrong split for EWC target {tile_id}")
        path = _target_path(root, tile_id)
        if Path(str(row.get("path", ""))).expanduser().resolve() != path.resolve():
            raise ValueError(f"wrong path for EWC target {tile_id}")
        if not path.is_file() or _sha256(path) != row.get("sha256"):
            raise ValueError(f"missing or hash-mismatched EWC target: {path}")
        weight = _load_ewc_target(path)
        if weight.dtype != np.float32 or weight.shape != expected_shape:
            raise ValueError(f"invalid EWC target dtype/shape: {path}")
        if not np.all(np.isfinite(weight)) or np.any(weight <= 0.0):
            raise ValueError(f"invalid EWC target values: {path}")
        if np.any(weight < 1.0 - 1e-6) or np.any(
            weight > math.exp(EWC_POWER) + 1e-5
        ):
            raise ValueError(f"EWC target values exceed the locked range: {path}")
        if bool(record.get("is_background", False)) and not np.array_equal(
            weight,
            np.ones(expected_shape, dtype=np.float32),
        ):
            raise ValueError(f"background EWC target is not unit weight: {path}")
        if row.get("shape") != list(expected_shape) or row.get("dtype") != "float32":
            raise ValueError(f"invalid EWC manifest dtype/shape row: {tile_id}")
        try:
            row_minimum = float(row["minimum"])
            row_maximum = float(row["maximum"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid EWC manifest value range: {tile_id}") from exc
        if not math.isclose(row_minimum, float(weight.min()), abs_tol=1e-7) or not math.isclose(
            row_maximum,
            float(weight.max()),
            abs_tol=1e-7,
        ):
            raise ValueError(f"EWC manifest value range mismatch: {tile_id}")
        if source_integrity == "per_target_sha256_v1":
            expected_instance_hash = row.get("source_instance_sha256")
            expected_semantic_hash = row.get("source_semantic_sha256")
            if bool(record.get("is_background", False)):
                if expected_instance_hash is not None or expected_semantic_hash is not None:
                    raise ValueError(
                        f"background EWC target has unexpected source hashes: {tile_id}"
                    )
            else:
                source = instance_root / f"{tile_id}.npz"
                if not source.is_file() or _sha256(source) != expected_instance_hash:
                    raise ValueError(
                        f"stale or hash-mismatched EWC source instance: {source}; "
                        "regenerate with --overwrite"
                    )
                semantic_value = record.get("semantic")
                if semantic_value:
                    semantic_source = dataset_root / str(semantic_value)
                    if (
                        not semantic_source.is_file()
                        or _sha256(semantic_source) != expected_semantic_hash
                    ):
                        raise ValueError(
                            f"stale or hash-mismatched EWC semantic source: "
                            f"{semantic_source}; regenerate with --overwrite"
                        )
                elif expected_semantic_hash is not None:
                    raise ValueError(
                        f"EWC target has an unexpected semantic source hash: {tile_id}"
                    )
    return {
        "manifest": str(manifest_path),
        "target_count": len(by_tile),
        "split_counts": dict(manifest["split_counts"]),
        "parameters": dict(manifest["parameters"]),
        "valid": True,
    }
