from __future__ import annotations

import importlib.metadata
import hashlib
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Mapping

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from geoseg.experiments.posix_child_runtime import (
    TEMP_ENV_NAMES,
    build_posix_child_environment,
    preflight_posix_child_runtime,
)

from .experiment import ResolvedExperiment


DEPENDENCY_POLICY: Mapping[str, str] = {
    "torch": ">=2.12,<2.13",
    "torchvision": ">=0.27,<0.28",
    "pytorch-lightning": ">=2.6,<2.7",
    "timm": ">=1.0,<1.1",
    "transformers": ">=5.10,<5.11",
    "segmentation-models-pytorch": ">=0.5,<0.6",
    "rasterio": ">=1.4,<1.5",
    "tifffile": ">=2026.3,<2027",
    "Pillow": ">=12,<13",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_aquadataset_data_contract(
    experiment: ResolvedExperiment,
) -> dict[str, object]:
    if experiment.spec.data.get("dataset") != "aquadataset":
        return {"status": "not_applicable"}
    import numpy as np
    import tifffile

    root = Path(str(experiment.spec.data["root"])).expanduser().resolve()
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"AquaDateset metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_version = str(experiment.spec.data["dataset_version"])
    if metadata.get("version") != expected_version:
        raise RuntimeError(
            f"AquaDateset version mismatch: expected {expected_version}, got {metadata.get('version')}"
        )
    normalization_hash = experiment.spec.data.get("normalization_sha256")
    if normalization_hash and _sha256_file(root / "normalization.json") != normalization_hash:
        raise RuntimeError("AquaDateset normalization hash mismatch")
    split_counts: dict[str, int] = {}
    all_records: list[dict[str, object]] = []
    for split, expected_count in dict(experiment.spec.data["split_counts"]).items():
        path = root / "splits" / f"{split}.jsonl"
        actual_hash = _sha256_file(path)
        expected_hash = str(experiment.spec.data["split_hashes"][split])
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"AquaDateset {split} hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(records) != int(expected_count):
            raise RuntimeError(
                f"AquaDateset {split} count mismatch: expected {expected_count}, got {len(records)}"
            )
        split_counts[str(split)] = len(records)
        all_records.extend(records)
    sample_path = root / str(all_records[0]["image"])
    with tifffile.TiffFile(sample_path) as dataset:
        series = dataset.series[0]
        height, width, bands = series.shape
        sample = {
            "bands": int(bands),
            "width": int(width),
            "height": int(height),
            "dtypes": [str(series.dtype)] * int(bands),
        }
    if sample != {
        "bands": 4,
        "width": 512,
        "height": 512,
        "dtypes": ["uint16", "uint16", "uint16", "uint16"],
    }:
        raise RuntimeError(f"AquaDateset sample raster contract mismatch: {sample}")

    pbcl_report: dict[str, object] | None = None
    target_root_value = experiment.spec.data.get("instance_target_root")
    if target_root_value:
        target_root = root / str(target_root_value)
        positive_records = [
            record
            for record in all_records
            if not bool(record.get("is_background", False))
        ]
        missing = 0
        pair_count = 0
        bucket_counts = {0: 0, 1: 0, 2: 0}
        query_capacity = int(experiment.spec.model.get("num_queries", 0))
        max_instances = 0
        max_instance_tile = ""
        for record in positive_records:
            path = target_root / f"{record['tile_id']}.npz"
            if not path.is_file():
                missing += 1
                continue
            with np.load(path, allow_pickle=False) as payload:
                pairs = np.asarray(payload["instance_pairs"], dtype=np.int64)
                if query_capacity > 0:
                    instance_map = np.asarray(payload["instance_map"], dtype=np.int64)
                    instance_count = int(np.count_nonzero(np.unique(instance_map) > 0))
                    if instance_count > max_instances:
                        max_instances = instance_count
                        max_instance_tile = str(record["tile_id"])
            pair_count += len(pairs)
            for value in pairs[:, 2] if len(pairs) else ():
                bucket_counts[int(value)] += 1
        if missing:
            raise RuntimeError(
                f"AquaDateset PBCL targets are missing for {missing} positive tiles"
            )
        if query_capacity > 0 and max_instances > query_capacity:
            raise RuntimeError(
                "AquaDateset instance query capacity is insufficient: "
                f"max_instances={max_instances} at {max_instance_tile}, "
                f"num_queries={query_capacity}"
            )
        pbcl_report = {
            "positive_tiles": len(positive_records),
            "missing_targets": missing,
            "pairs": pair_count,
            "bucket_counts": bucket_counts,
        }
        if query_capacity > 0:
            pbcl_report.update(
                instance_query_capacity=query_capacity,
                max_instances_per_tile=max_instances,
                max_instance_tile=max_instance_tile,
            )
    ewc_report: dict[str, object] | None = None
    ewc_root_value = experiment.spec.data.get("ewc_weight_root")
    if ewc_root_value:
        from geoseg.datasets.aquadataset_ewc import audit_ewc_targets

        ewc_report = audit_ewc_targets(
            root,
            target_root=str(ewc_root_value),
            splits=experiment.spec.data.get("ewc_weight_splits", ("train", "val", "test")),
            use_manifest_valid_mask=bool(experiment.spec.data.get("use_manifest_valid_mask", False)),
        )
        loss_spec = dict(experiment.spec.optimization["loss"])
        locked = dict(ewc_report["parameters"])
        configured = {
            "w0": float(loss_spec.get("ewc_w0", float("nan"))),
            "sigma": float(loss_spec.get("ewc_sigma", float("nan"))),
            "power": float(loss_spec.get("ewc_power", float("nan"))),
            "class_weight": loss_spec.get("ewc_class_weight"),
            "single_instance": loss_spec.get("ewc_single_instance"),
            "semantic_labels_modified": False,
        }
        for name in ("w0", "sigma", "power"):
            if not math.isclose(
                float(configured[name]),
                float(locked[name]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"EWC config/target parameter mismatch for {name}: "
                    f"{configured[name]} != {locked[name]}"
                )
        for name in ("class_weight", "single_instance"):
            if configured[name] != locked[name]:
                raise RuntimeError(
                    f"EWC config/target parameter mismatch for {name}: "
                    f"{configured[name]!r} != {locked[name]!r}"
                )
    wprf_report: dict[str, object] | None = None
    wprf_root_value = experiment.spec.data.get("wprf_target_root")
    if wprf_root_value:
        from geoseg.datasets.aquadataset_wprf import audit_wprf_targets

        wprf_report = audit_wprf_targets(root, target_root=str(wprf_root_value))
        loss_spec = dict(experiment.spec.optimization["loss"])
        locked = dict(wprf_report["parameters"])
        configured = {
            "source_commit": loss_spec.get("source_commit"),
            "grid_stride": int(loss_spec.get("wprf_grid_stride", -1)),
            "scales": list(loss_spec.get("wprf_scales", [])),
            "gamma": list(loss_spec.get("wprf_gamma", [])),
            "sources_per_scale": int(
                loss_spec.get("wprf_sources_per_scale", -1)
            ),
        }
        for name, value in configured.items():
            if value != locked[name]:
                raise RuntimeError(
                    f"WPRF config/target parameter mismatch for {name}: "
                    f"{value!r} != {locked[name]!r}"
                )
    return {
        "dataset_version": expected_version,
        "split_counts": split_counts,
        "sample": sample,
        "pbcl": pbcl_report,
        "ewc": ewc_report,
        "wprf": wprf_report,
    }


def validate_vegasdenseclose_data_contract(
    experiment: ResolvedExperiment,
) -> dict[str, object]:
    if experiment.spec.data.get("dataset") != "vegasdenseclosepbcl_v1":
        return {"status": "not_applicable"}
    import csv

    from geoseg.datasets.vegasdenseclose import (
        BAND_INDICES_ZERO_BASED,
        DATASET_VERSION,
        NATIVE_SIZE,
        NORMALIZE_MEAN,
        NORMALIZE_STD,
        PADDED_SIZE,
        SPLIT_COUNTS,
        validate_cache,
    )

    data = experiment.spec.data
    if str(data["dataset_version"]) != DATASET_VERSION:
        raise RuntimeError("Vegas dataset version mismatch in experiment spec")
    if list(data["band_indices"]) != list(BAND_INDICES_ZERO_BASED):
        raise RuntimeError("Vegas selected-band mismatch in experiment spec")
    if list(data["normalize_mean"]) != list(NORMALIZE_MEAN):
        raise RuntimeError("Vegas normalization mean mismatch in experiment spec")
    if list(data["normalize_std"]) != list(NORMALIZE_STD):
        raise RuntimeError("Vegas normalization std mismatch in experiment spec")
    expected_input = (512, "resize") if data.get("native_eval_targets") else (PADDED_SIZE, "pad")
    if int(data["native_size"]) != NATIVE_SIZE or (
        int(data["image_size"]), str(data.get("resize_mode", "pad"))
    ) != expected_input:
        raise RuntimeError("Vegas input geometry mismatch in experiment spec")
    report = validate_cache(
        protocol_root=data["protocol_root"],
        cache_root=data["root"],
        validate_targets=False,
    )
    if dict(report["split_counts"]) != SPLIT_COUNTS:
        raise RuntimeError("Vegas cache split counts do not match the locked protocol")
    target_root = Path(str(data["root"])) / "instances" / "targets"
    target_count = sum(1 for _ in target_root.glob("*.npz"))
    expected_targets = sum(SPLIT_COUNTS.values())
    if target_count != expected_targets:
        raise RuntimeError(
            f"Vegas target count mismatch: expected {expected_targets}, got {target_count}"
        )
    ewc_report: dict[str, object] | None = None
    ewc_root_value = data.get("ewc_weight_root")
    if ewc_root_value:
        from geoseg.datasets.aquadataset_ewc import audit_ewc_targets

        ewc_report = audit_ewc_targets(
            data["root"],
            target_root=str(ewc_root_value),
        )
        if int(ewc_report["target_count"]) != expected_targets:
            raise RuntimeError(
                "Vegas EWC target coverage does not match the locked protocol"
            )
        loss_spec = dict(experiment.spec.optimization["loss"])
        if loss_spec.get("name") != "SSepNetEWCLoss" and loss_spec.get("method") != "ewc":
            raise RuntimeError("Vegas EWC targets are configured for a non-EWC loss")
        locked = dict(ewc_report["parameters"])
        configured = {
            "w0": float(loss_spec.get("ewc_w0", float("nan"))),
            "sigma": float(loss_spec.get("ewc_sigma", float("nan"))),
            "power": float(loss_spec.get("ewc_power", float("nan"))),
            "class_weight": loss_spec.get("ewc_class_weight"),
            "single_instance": loss_spec.get("ewc_single_instance"),
        }
        for name in ("w0", "sigma", "power"):
            if not math.isclose(
                float(configured[name]),
                float(locked[name]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"Vegas EWC config/target mismatch for {name}: "
                    f"{configured[name]} != {locked[name]}"
                )
        for name in ("class_weight", "single_instance"):
            if configured[name] != locked[name]:
                raise RuntimeError(
                    f"Vegas EWC config/target mismatch for {name}: "
                    f"{configured[name]!r} != {locked[name]!r}"
                )
    wprf_report: dict[str, object] | None = None
    wprf_root_value = data.get("wprf_target_root")
    if wprf_root_value:
        from geoseg.datasets.vegasdenseclose_wprf import (
            audit_vegas_wprf_manifest,
        )

        if data.get("wprf_dynamic_from_instance") is not True:
            raise RuntimeError("Vegas WPRF must use dynamic augmented-instance targets")
        wprf_report = audit_vegas_wprf_manifest(
            data["root"], target_root=str(wprf_root_value)
        )
        if int(wprf_report["target_count"]) != expected_targets:
            raise RuntimeError("Vegas WPRF source coverage is incomplete")
        loss_spec = dict(experiment.spec.optimization["loss"])
        locked = dict(wprf_report["parameters"])
        configured = {
            "source_commit": loss_spec.get("source_commit"),
            "grid_stride": int(loss_spec.get("wprf_grid_stride", -1)),
            "neighborhood": int(loss_spec.get("wprf_neighbors", -1)),
            "scales": list(loss_spec.get("wprf_scales", [])),
            "gamma": list(loss_spec.get("wprf_gamma", [])),
            "sources_per_scale": int(
                loss_spec.get("wprf_sources_per_scale", -1)
            ),
        }
        for name, value in configured.items():
            if value != locked[name]:
                raise RuntimeError(
                    f"Vegas WPRF config/target mismatch for {name}: "
                    f"{value!r} != {locked[name]!r}"
                )
    query_capacity = int(experiment.spec.model.get("num_queries", 0))
    max_instances = 0
    max_instance_tile = ""
    if query_capacity:
        with Path(str(data["protocol_root"]), "tile_metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                if row["split"] not in SPLIT_COUNTS:
                    continue
                count = int(row["instance_count"])
                if count > max_instances:
                    max_instances = count
                    max_instance_tile = str(row["tile_id"])
        if max_instances > query_capacity:
            raise RuntimeError(
                f"Vegas instance query capacity {query_capacity} is below "
                f"max_instances={max_instances} at {max_instance_tile}"
            )
    return {
        **report,
        "target_count": target_count,
        "max_instances_per_tile": max_instances if query_capacity else None,
        "instance_query_capacity": query_capacity if query_capacity else None,
        "ewc": ewc_report,
        "wprf": wprf_report,
    }


def validate_isaid_transfer_data_contract(
    experiment: ResolvedExperiment,
) -> dict[str, object]:
    if experiment.spec.data.get("dataset") != "isaid_transfer":
        return {"status": "not_applicable"}
    import numpy as np

    root = Path(str(experiment.spec.data["root"])).expanduser().resolve()
    provenance_path = root / "provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError(f"iSAID transfer provenance is missing: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_version = str(experiment.spec.data["dataset_version"])
    if provenance.get("dataset_version") != expected_version:
        raise RuntimeError(
            f"iSAID transfer version mismatch: expected {expected_version}, "
            f"got {provenance.get('dataset_version')}"
        )
    class_name = str(experiment.spec.data["class_name"])
    class_provenance = dict(provenance.get("classes", {}).get(class_name, {}))
    if not class_provenance:
        raise RuntimeError(
            f"iSAID transfer class is missing from provenance: {class_name}"
        )
    normalization = dict(class_provenance["normalization"])
    if list(experiment.spec.data["normalize_mean"]) != list(normalization["mean"]):
        raise RuntimeError(f"iSAID transfer mean mismatch for {class_name}")
    if list(experiment.spec.data["normalize_std"]) != list(normalization["std"]):
        raise RuntimeError(f"iSAID transfer std mismatch for {class_name}")

    if experiment.spec.data.get("dataset_family") == "isaid_dota_vegas12":
        from geoseg.datasets.isaid_vegas12_cache import (
            CACHE_VERSION,
            INSTANCE_VIEW,
            cache_manifest_path,
        )

        cache_root = Path(str(experiment.spec.data["cache_root"])).expanduser().resolve()
        metadata_path = cache_root / "metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError(
                f"iSAID Vegas12 cache is missing: {metadata_path}; "
                "run scripts/prepare_isaid_vegas12.py --build"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_version") != CACHE_VERSION:
            raise RuntimeError(
                f"iSAID Vegas12 cache version mismatch: {metadata.get('cache_version')}"
            )
        expected_original = dict(provenance.get("manifest_sha256", {}))
        if dict(metadata.get("original_sha256", {})) != expected_original:
            raise RuntimeError("iSAID Vegas12 cache no longer matches immutable v1 hashes")

        split_counts: dict[str, int] = {}
        parent_sets: dict[str, set[str]] = {}
        selected_views: dict[str, str] = {}
        for split in ("train", "val", "test"):
            view = str(experiment.spec.data.get(f"{split}_view", split))
            selected_views[split] = view
            nested_split = split if view == INSTANCE_VIEW else None
            path = cache_manifest_path(
                cache_root,
                class_name,
                view,
                split=nested_split,
            )
            record = metadata["views"][class_name][view]
            if view == INSTANCE_VIEW:
                record = record["splits"][split]
            actual_hash = _sha256_file(path)
            if actual_hash != str(record["sha256"]):
                raise RuntimeError(f"iSAID Vegas12 derived view hash mismatch: {path}")
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if len(rows) != int(record["records"]):
                raise RuntimeError(f"iSAID Vegas12 derived view count mismatch: {path}")
            split_counts[split] = len(rows)
            parent_sets[split] = {
                f"{row['parent_source_split']}:{row['parent_image']}" for row in rows
            }
        overlap = (
            (parent_sets["train"] & parent_sets["val"])
            | (parent_sets["train"] & parent_sets["test"])
            | (parent_sets["val"] & parent_sets["test"])
        )
        if overlap:
            raise RuntimeError(
                f"iSAID Vegas12 parent leakage detected: {sorted(overlap)[:10]}"
            )
        query_capacity = int(experiment.spec.model.get("num_queries", 0))
        declared_max = experiment.spec.data.get("max_instances_per_tile")
        max_instances = int(declared_max) if declared_max is not None else None
        if selected_views["train"] == INSTANCE_VIEW:
            observed_max = int(metadata["views"][class_name][INSTANCE_VIEW]["max_instances"])
            if max_instances is not None and observed_max != max_instances:
                raise RuntimeError(
                    f"iSAID Vegas12 instance max mismatch: {observed_max} != {max_instances}"
                )
            max_instances = observed_max
        if max_instances is not None and query_capacity < max_instances:
            raise RuntimeError(
                "iSAID Vegas12 instance query capacity is insufficient: "
                f"max_instances={max_instances}, num_queries={query_capacity}"
            )
        return {
            "dataset_version": expected_version,
            "cache_version": CACHE_VERSION,
            "class_name": class_name,
            "views": selected_views,
            "split_counts": split_counts,
            "parent_overlap": sorted(overlap),
            "instance_query_capacity": query_capacity or None,
            "max_instances_per_tile": max_instances,
        }

    split_counts: dict[str, int] = {}
    parent_sets: dict[str, set[str]] = {}
    all_records: list[dict[str, object]] = []
    for split, expected_count in dict(experiment.spec.data["split_counts"]).items():
        path = root / "classes" / class_name / "splits" / f"{split}.jsonl"
        actual_hash = _sha256_file(path)
        expected_hash = str(experiment.spec.data["split_hashes"][split])
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"iSAID transfer {class_name}/{split} hash mismatch: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(records) != int(expected_count):
            raise RuntimeError(
                f"iSAID transfer {class_name}/{split} count mismatch: "
                f"expected {expected_count}, got {len(records)}"
            )
        split_counts[str(split)] = len(records)
        parent_sets[str(split)] = {
            f"{record['parent_source_split']}:{record['parent_image']}"
            for record in records
        }
        all_records.extend(records)
    overlap: set[str] = set()
    split_names = sorted(parent_sets)
    for first_pos, first in enumerate(split_names):
        for second in split_names[first_pos + 1 :]:
            overlap.update(parent_sets[first] & parent_sets[second])
    if overlap:
        raise RuntimeError(
            f"iSAID transfer parent leakage detected: {sorted(overlap)[:10]}"
        )

    positive_records = [
        record for record in all_records if not bool(record["is_background"])
    ]
    missing_targets = 0
    pair_count = 0
    bucket_counts = {0: 0, 1: 0, 2: 0}
    diagnostic_pairs = 0
    for record in positive_records:
        path = root / str(record["target"])
        if not path.is_file():
            missing_targets += 1
            continue
        with np.load(path, allow_pickle=False) as payload:
            instance_map = np.asarray(payload["instance_map"], dtype=np.int64)
            pairs = np.asarray(payload["instance_pairs"], dtype=np.int64)
            diagnostic = np.asarray(payload["diagnostic_pairs_6_10"], dtype=np.int64)
        if instance_map.shape != (512, 512):
            raise RuntimeError(
                f"Invalid iSAID transfer target shape {instance_map.shape}: {path}"
            )
        if pairs.ndim != 2 or pairs.shape[1] != 3:
            raise RuntimeError(
                f"Invalid iSAID transfer pair shape {pairs.shape}: {path}"
            )
        present = set(int(value) for value in np.unique(instance_map) if int(value) > 0)
        referenced = (
            set(int(value) for value in pairs[:, :2].reshape(-1))
            if len(pairs)
            else set()
        )
        if not referenced.issubset(present):
            raise RuntimeError(f"iSAID transfer pair references missing ids: {path}")
        pair_count += len(pairs)
        diagnostic_pairs += len(diagnostic)
        for value in pairs[:, 2] if len(pairs) else ():
            bucket = int(value)
            if bucket not in bucket_counts:
                raise RuntimeError(
                    f"Invalid iSAID transfer pair bucket {bucket}: {path}"
                )
            bucket_counts[bucket] += 1
    if missing_targets:
        raise RuntimeError(
            f"iSAID transfer targets are missing for {missing_targets} positive tiles"
        )
    return {
        "dataset_version": expected_version,
        "class_name": class_name,
        "split_counts": split_counts,
        "parent_overlap": sorted(overlap),
        "positive_tiles": len(positive_records),
        "missing_targets": missing_targets,
        "pairs": pair_count,
        "bucket_counts": bucket_counts,
        "diagnostic_pairs_6_10": diagnostic_pairs,
    }


def installed_dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in DEPENDENCY_POLICY:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


def validate_dependency_versions(versions: Mapping[str, str] | None = None) -> None:
    actual = dict(versions or installed_dependency_versions())
    errors: list[str] = []
    for distribution, requirement in DEPENDENCY_POLICY.items():
        raw_version = actual.get(distribution, "missing")
        if raw_version == "missing":
            errors.append(f"{distribution} is not installed; required {requirement}")
            continue
        try:
            compatible = Version(raw_version) in SpecifierSet(requirement)
        except Exception as exc:
            errors.append(f"{distribution} version {raw_version!r} is invalid: {exc}")
            continue
        if not compatible:
            errors.append(
                f"{distribution}=={raw_version} is incompatible; required {requirement}"
            )
    if errors:
        raise RuntimeError("Dependency preflight failed:\n- " + "\n- ".join(errors))


def collect_environment_report() -> dict[str, object]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    accelerator: dict[str, object] = {
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported())
        if cuda_available
        else False,
        "device_count": int(torch.cuda.device_count()),
    }
    if cuda_available:
        accelerator["devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return {
        "schema_version": 1,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "versions": {
            "python": platform.python_version(),
            **installed_dependency_versions(),
        },
        "accelerator": accelerator,
    }


def write_environment_report(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            collect_environment_report(), ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def activate_posix_worker_environment() -> dict[str, str]:
    """Apply the same POSIX-safe temp variables that worker preflight validates."""

    child_environment = build_posix_child_environment(os.environ)
    for name in TEMP_ENV_NAMES:
        os.environ[name] = child_environment[name]
    return child_environment


def apply_runtime_optimizations(experiment: ResolvedExperiment) -> ResolvedExperiment:
    """Apply only explicitly enabled, output-sensitive runtime options."""

    if experiment.components is None or experiment.config is None:
        raise ValueError(
            "Runtime optimizations require load_experiment(..., build=True)"
        )
    import torch

    runtime = experiment.spec.runtime
    model = experiment.components.model
    if model is None:
        raise ValueError("Resolved experiment has no model to optimize")
    channels_last = bool(runtime.get("channels_last", False))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if bool(runtime.get("compile", False)):
        model = torch.compile(model)
    if bool(runtime.get("cudnn_benchmark", False)):
        torch.backends.cudnn.benchmark = True
    experiment.config.net = model
    experiment.config.channels_last = channels_last
    return replace(
        experiment,
        components=replace(experiment.components, model=model),
    )


def _validate_data_splits(experiment: ResolvedExperiment) -> dict[str, str]:
    root = Path(str(experiment.spec.data["root"])).expanduser()
    if not root.is_dir():
        raise RuntimeError(
            f"AquaV3 data root does not exist or is not a directory: {root}"
        )
    checks: dict[str, str] = {}
    shared_manifest = root / "manifest.jsonl"
    class_name = str(experiment.spec.data.get("class_name", ""))
    for logical_name, split_name in experiment.spec.data["splits"].items():
        split_manifest = (
            root / "classes" / class_name / "splits" / f"{split_name}.jsonl"
            if experiment.spec.data.get("dataset") == "isaid_transfer"
            else root / "splits" / f"{split_name}.jsonl"
        )
        if not split_manifest.is_file() and not shared_manifest.is_file():
            raise RuntimeError(
                f"Missing {logical_name} split manifest: expected {split_manifest} or {shared_manifest}"
            )
        checks[str(logical_name)] = "ok"
    return checks


def _validate_output_directory(experiment: ResolvedExperiment) -> str:
    output = Path(str(experiment.spec.artifacts["weights_path"])).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(prefix=".geoseg-write-probe-", dir=output):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Output directory is not writable: {output}: {exc}"
        ) from exc
    return str(output.resolve())


def _validate_accelerator(
    experiment: ResolvedExperiment, environment: Mapping[str, object]
) -> dict[str, object]:
    accelerator = dict(environment["accelerator"])
    devices = experiment.spec.runtime.get("devices", "auto")
    precision = str(experiment.spec.runtime.get("precision", "32"))
    if devices != "cpu" and not accelerator["cuda_available"]:
        raise RuntimeError(f"CUDA is unavailable but runtime.devices={devices!r}")
    if "bf16" in precision and not accelerator["bf16_supported"]:
        raise RuntimeError(f"precision={precision!r} requires CUDA bf16 support")
    return accelerator


def _validate_pretrained_availability(
    experiment: ResolvedExperiment,
) -> dict[str, object]:
    if not bool(experiment.spec.model.get("pretrained", False)):
        return {"status": "not_required"}
    model_key = str(experiment.spec.model["key"])
    checked: list[str]
    if model_key == "vmamba_fpn":
        from importlib.util import find_spec

        extension = find_spec("selective_scan_cuda_oflex")
        if extension is None or extension.origin is None:
            raise RuntimeError("VMamba-T requires the SM120 selective-scan CUDA extension")
        checked = [extension.origin]
    elif model_key == "dcswin_base":
        from geoseg.pretrained import resolve_pretrained_model_path

        checked = [str(resolve_pretrained_model_path("geoseg-stseg-base"))]
    elif model_key == "hrnet_swin_b_ocr_boundary_pbcl":
        from geoseg.pretrained import resolve_pretrained_model_path

        checked = [
            str(resolve_pretrained_model_path("geoseg-stseg-base")),
            str(resolve_pretrained_model_path("hrnetv2-w32-imagenet")),
        ]
    elif model_key == "hrnet_swinv2_b_ocr_boundary_pbcl":
        from geoseg.pretrained import resolve_pretrained_model_path

        checked = [
            str(resolve_pretrained_model_path("torchgeo-swin-v2-b")),
            str(resolve_pretrained_model_path("hrnetv2-w32-imagenet")),
        ]
    elif model_key == "hrnet_vit_b_ocr_boundary_pbcl":
        from geoseg.pretrained import resolve_pretrained_model_path

        checked = [
            str(resolve_pretrained_model_path("torchgeo-vit-base-sentinel2-mae")),
            str(resolve_pretrained_model_path("hrnetv2-w32-imagenet")),
        ]
    elif model_key in {"unetplusplus", "deeplabv3plus"}:
        from scripts.preflight_aquav3_hf_baselines import (
            default_smp_baselines,
            preflight_smp_baselines,
        )

        label = "UNet++" if model_key == "unetplusplus" else "DeepLabV3+"
        baseline = next(item for item in default_smp_baselines() if item.name == label)
        checked = preflight_smp_baselines([baseline])
    elif model_key == "swinunet":
        from scripts.preflight_aquav3_hf_baselines import preflight_swinunet_baseline

        checked = preflight_swinunet_baseline()
    elif model_key == "mask2former_swin_t_instance":
        from geoseg.pretrained import resolve_pretrained_model_path

        checked = [
            str(
                resolve_pretrained_model_path(
                    "mask2former-swin-tiny-coco-instance"
                )
            )
        ]
    elif model_key in {"segformer_b2", "upernet_swin_t", "mask2former_swin_t"}:
        from scripts.preflight_aquav3_hf_baselines import (
            default_hf_baselines,
            preflight_hf_baselines,
        )

        labels = {
            "segformer_b2": "SegFormer-B2",
            "upernet_swin_t": "UPerNet-Swin-T",
            "mask2former_swin_t": "Mask2Former-Swin-T",
        }
        baseline = next(
            item for item in default_hf_baselines() if item.name == labels[model_key]
        )
        checked = preflight_hf_baselines([baseline])
    elif model_key in {"unetformer", "abcnet"}:
        from geoseg.models.aquav3_external import build_compare_model

        model = build_compare_model(
            model_key, in_chans=4, num_classes=1, pretrained=True
        )
        del model
        checked = [model_key]
    elif model_key.startswith("ssepnet"):
        from geoseg.pretrained import resolve_pretrained_model_path

        checked = [
            str(resolve_pretrained_model_path(key))
            for key in experiment.spec.model.get("pretrained_keys", ())
        ]
    else:
        raise RuntimeError(
            f"No pretrained availability validator is registered for model {model_key!r}"
        )

    result: dict[str, object] = {
        "status": "ok",
        "source": str(experiment.spec.model.get("pretrained_source", "")),
        "checked": checked,
    }
    expected_artifacts = dict(experiment.spec.model.get("pretrained_artifacts", {}))
    if expected_artifacts:
        actual_hashes: dict[str, str] = {}
        for key, artifact in expected_artifacts.items():
            path = Path(str(artifact["path"]))
            if not path.is_file():
                raise FileNotFoundError(
                    f"Pretrained payload {key!r} is missing at {path}"
                )
            expected_value = artifact["sha256"]
            if expected_value in (None, ""):
                raise RuntimeError(
                    f"Pretrained artifact hash is not locked for {key}; "
                    "compute and register its SHA-256 before training"
                )
            expected_hash = str(expected_value)
            actual_hash = _sha256_file(path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Pretrained artifact hash mismatch for {key}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
            actual_hashes[str(key)] = actual_hash
        result["sha256"] = actual_hashes
    if experiment.config is None:
        return result
    checks: dict[str, str] = {}
    for name in ("pretrained_ckpt_path", "resume_ckpt_path"):
        value = getattr(experiment.config, name, None)
        if value:
            path = Path(str(value)).expanduser()
            if not path.is_file():
                raise RuntimeError(f"Configured {name} does not exist: {path}")
            checks[name] = str(path.resolve())
    if checks:
        result["checkpoint_paths"] = checks
    return result


def preflight_experiment(
    experiment: ResolvedExperiment,
    *,
    report_path: str | Path | None = None,
    check_dependencies: bool = True,
    check_accelerator: bool = True,
    check_worker_runtime: bool = True,
    check_pretrained: bool = True,
) -> Path:
    """Validate an experiment before construction and persist an actionable report."""

    if not experiment.config_path.is_file():
        raise RuntimeError(f"Config does not exist: {experiment.config_path}")
    environment = collect_environment_report()
    checks: dict[str, object] = {
        "config": {
            "path": str(experiment.config_path),
            "built": experiment.components is not None,
        },
        "data_splits": _validate_data_splits(experiment),
        "aquadataset_contract": validate_aquadataset_data_contract(experiment),
        "vegasdenseclose_contract": validate_vegasdenseclose_data_contract(
            experiment
        ),
        "output_directory": _validate_output_directory(experiment),
        "pretrained": _validate_pretrained_availability(experiment)
        if check_pretrained
        else "skipped",
    }
    if check_dependencies:
        validate_dependency_versions(environment["versions"])
        checks["dependencies"] = "ok"
    else:
        checks["dependencies"] = "skipped"
    if check_accelerator:
        checks["accelerator"] = _validate_accelerator(experiment, environment)
    else:
        checks["accelerator"] = "skipped"
    if check_worker_runtime:
        preflight_posix_child_runtime(activate_posix_worker_environment())
        checks["worker_runtime"] = "ok"
    else:
        checks["worker_runtime"] = "skipped"

    payload = {**environment, "status": "ok", "checks": checks}
    target = (
        Path(report_path)
        if report_path is not None
        else Path(str(experiment.spec.artifacts["weights_path"])) / "preflight.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target
