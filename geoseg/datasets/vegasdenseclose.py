from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from geoseg.datasets.paths import DATASETS_ROOT


DATASET_NAME = "VegasDenseClosePBCL_v1"
DATASET_VERSION = "spacenet2-vegas-dense-close-v1"
_VEGAS_ENV = os.environ.get("VEGAS_ROOT")
PROTOCOL_ROOT = (
    Path(_VEGAS_ENV).expanduser().resolve()
    if _VEGAS_ENV
    else DATASETS_ROOT / "SpaceNet2" / DATASET_NAME
)
SOURCE_ROOT = Path(
    os.environ.get("VEGAS_SOURCE_ROOT") or str(DATASETS_ROOT / "SpaceNet2" / "Vegas")
).expanduser().resolve()
CACHE_ROOT = PROTOCOL_ROOT / "_dataset_cache"
EWC_TARGET_ROOT = "instances/ewc_v1"
NATIVE_SIZE = 650
PADDED_SIZE = 672

# WorldView multispectral native order is Coastal, Blue, Green, Yellow, Red,
# Red Edge, NIR1, NIR2. Keep the established GeoSeg B/G/R/NIR input contract.
BAND_NAMES = ("B", "G", "R", "NIR1")
BAND_INDICES_ZERO_BASED = (1, 2, 4, 6)
BAND_INDICES_RASTERIO = tuple(index + 1 for index in BAND_INDICES_ZERO_BASED)
RAW_SCALE = 2047.0
NORMALIZE_MEAN = (
    0.21361868299521208,
    0.28357311705749805,
    0.2037835906844101,
    0.26128937004691777,
)
NORMALIZE_STD = (
    0.08266655238646263,
    0.13001066432460615,
    0.11424660999428865,
    0.125842682571778,
)
NORMALIZATION_PIXEL_COUNT_PER_CHANNEL = 1_061_742_500
SPLIT_COUNTS = {"train": 2513, "val": 621, "test": 568}
SOURCE_SPLIT_HASHES = {
    "train": "dadaa11a8bda6d8190ae30712fab8170072f8426f404b60f6416a6818c9153fe",
    "val": "739180bf804295151c194ec3f157c54d7f056d1a7afdd7878d945d512d219d78",
    "test": "5f04de526730573215dd2338e3353bdc38f86d0ed3c66abb383bbf877500eec4",
}
PAIR_BUCKET_CODES = {"touching": 0, "1_2": 1, "3_5": 2, "6_10": 3}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _polygonal_geometry(geometry: Any):
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons = []
        for part in geometry.geoms:
            if isinstance(part, Polygon):
                polygons.append(part)
            elif isinstance(part, MultiPolygon):
                polygons.extend(part.geoms)
        if not polygons:
            return None
        return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
    return None


def rasterize_instance_map(
    features: Sequence[Mapping[str, Any]],
    *,
    out_shape: tuple[int, int],
    transform: Any,
) -> np.ndarray:
    from rasterio.features import rasterize
    from shapely import make_valid
    from shapely.geometry import shape

    shapes: list[tuple[Mapping[str, Any], int]] = []
    instance_id = 0
    for feature in features:
        geometry_data = feature.get("geometry")
        if not isinstance(geometry_data, Mapping):
            continue
        polygonal = _polygonal_geometry(shape(geometry_data))
        if polygonal is None:
            continue
        if not polygonal.is_valid:
            polygonal = _polygonal_geometry(make_valid(polygonal))
        if polygonal is None or polygonal.is_empty:
            continue
        instance_id += 1
        shapes.append((polygonal.__geo_interface__, instance_id))
    if not shapes:
        return np.zeros(out_shape, dtype=np.int32)
    return rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )


def instance_pairs(label_map: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    labels = np.asarray(label_map, dtype=np.int32)
    instance_ids = np.unique(labels)
    instance_ids = instance_ids[instance_ids > 0]
    rows: list[tuple[int, int, int]] = []
    height, width = labels.shape
    for first_id in instance_ids:
        ys, xs = np.nonzero(labels == first_id)
        padding = 13
        y_min = max(0, int(ys.min()) - padding)
        y_max = min(height, int(ys.max()) + padding + 1)
        x_min = max(0, int(xs.min()) - padding)
        x_max = min(width, int(xs.max()) + padding + 1)
        crop = labels[y_min:y_max, x_min:x_max]
        first = crop == first_id
        distances = ndi.distance_transform_edt(~first)
        candidates = np.unique(crop[(crop > first_id) & (distances < 12.0)])
        for second_id in candidates:
            second = crop == second_id
            if np.any(first & second):
                gap = 0
            else:
                center_distance = float(np.min(distances[second]))
                gap = max(0, int(math.floor(center_distance + 1e-9)) - 1)
            if gap == 0:
                bucket = "touching"
            elif gap <= 2:
                bucket = "1_2"
            elif gap <= 5:
                bucket = "3_5"
            elif gap <= 10:
                bucket = "6_10"
            else:
                continue
            rows.append((int(first_id), int(second_id), PAIR_BUCKET_CODES[bucket]))
    return np.asarray(rows, dtype=np.int32).reshape(-1, 3)


def canonicalize_instance_map(label_map: np.ndarray) -> np.ndarray:
    labels = np.asarray(label_map, dtype=np.int32)
    present = np.unique(labels)
    present = present[present > 0]
    if not len(present):
        return np.zeros_like(labels, dtype=np.int32)
    lookup = np.zeros(int(present[-1]) + 1, dtype=np.int32)
    lookup[present] = np.arange(1, len(present) + 1, dtype=np.int32)
    return lookup[labels]


def _load_protocol_rows(protocol_root: Path) -> dict[str, dict[str, str]]:
    with (protocol_root / "tile_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = {str(row["tile_id"]): dict(row) for row in csv.DictReader(handle)}
    expected = sum(SPLIT_COUNTS.values()) + 149
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} protocol rows, found {len(rows)}")
    return rows


def _split_ids(protocol_root: Path, split: str) -> list[str]:
    path = protocol_root / "splits" / f"{split}.txt"
    actual_hash = sha256_file(path)
    if actual_hash != SOURCE_SPLIT_HASHES[split]:
        raise RuntimeError(
            f"{split} split hash mismatch: expected {SOURCE_SPLIT_HASHES[split]}, got {actual_hash}"
        )
    tile_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tile_ids) != SPLIT_COUNTS[split]:
        raise RuntimeError(
            f"{split} split count mismatch: expected {SPLIT_COUNTS[split]}, got {len(tile_ids)}"
        )
    return tile_ids


def _target_problem(path: Path, expected: Mapping[str, str]) -> str | None:
    if not path.is_file():
        return "missing"
    try:
        with np.load(path, allow_pickle=False) as payload:
            label_map = np.asarray(payload["instance_map"], dtype=np.int32)
            pbcl_pairs = np.asarray(payload["instance_pairs"], dtype=np.int32)
            all_pairs = np.asarray(payload["all_instance_pairs"], dtype=np.int32)
    except Exception as exc:
        return f"unreadable: {exc}"
    if label_map.shape != (NATIVE_SIZE, NATIVE_SIZE):
        return f"instance_map shape is {label_map.shape}"
    if pbcl_pairs.ndim != 2 or pbcl_pairs.shape[1:] != (3,):
        return f"instance_pairs shape is {pbcl_pairs.shape}"
    if all_pairs.ndim != 2 or all_pairs.shape[1:] != (3,):
        return f"all_instance_pairs shape is {all_pairs.shape}"
    counts = Counter(int(value) for value in all_pairs[:, 2])
    expected_counts = {
        0: int(expected["touching_pairs"]),
        1: int(expected["gap_1_2_pairs"]),
        2: int(expected["gap_3_5_pairs"]),
        3: int(expected["gap_6_10_pairs"]),
    }
    if counts != Counter({key: value for key, value in expected_counts.items() if value}):
        return f"pair counts are {dict(counts)}, expected {expected_counts}"
    present = np.unique(label_map)
    instance_count = int(np.count_nonzero(present > 0))
    if instance_count != int(expected["instance_count"]):
        return f"instance count is {instance_count}, expected {expected['instance_count']}"
    return None


def _source_path(value: str | Path) -> Path:
    """Map copied protocol paths onto the configured local dataset index."""
    path = Path(value).expanduser()
    marker = ("SpaceNet2", "Vegas")
    parts = path.parts
    for index in range(len(parts) - 1):
        if tuple(parts[index : index + 2]) == marker:
            candidate = SOURCE_ROOT.joinpath(*parts[index + 2 :])
            if candidate.is_file():
                return candidate.resolve()
    candidate = path if path.is_absolute() else SOURCE_ROOT / path
    if not candidate.is_file():
        raise FileNotFoundError(f"Vegas source file is missing: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(SOURCE_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"Vegas source path is outside the configured dataset root: {resolved}"
        ) from exc
    return resolved


def _write_target(path: Path, row: Mapping[str, str]) -> None:
    import rasterio

    image_path = _source_path(row["image_path"])
    label_path = _source_path(row["label_path"])
    geojson = json.loads(label_path.read_text(encoding="utf-8"))
    with rasterio.open(image_path) as dataset:
        if (dataset.count, dataset.height, dataset.width) != (8, NATIVE_SIZE, NATIVE_SIZE):
            raise RuntimeError(
                f"Unexpected raster shape for {image_path}: "
                f"{(dataset.count, dataset.height, dataset.width)}"
            )
        label_map = canonicalize_instance_map(rasterize_instance_map(
            geojson.get("features") or (),
            out_shape=(NATIVE_SIZE, NATIVE_SIZE),
            transform=dataset.transform,
        ))
    all_pairs = instance_pairs(label_map)
    pbcl_pairs = all_pairs[all_pairs[:, 2] <= 2].copy()
    problem_counts = Counter(int(value) for value in all_pairs[:, 2])
    expected_counts = {
        0: int(row["touching_pairs"]),
        1: int(row["gap_1_2_pairs"]),
        2: int(row["gap_3_5_pairs"]),
        3: int(row["gap_6_10_pairs"]),
    }
    if problem_counts != Counter({key: value for key, value in expected_counts.items() if value}):
        raise RuntimeError(
            f"{row['tile_id']}: pair-count mismatch {dict(problem_counts)} != {expected_counts}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".npz.tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                instance_map=label_map.astype(np.int32, copy=False),
                instance_pairs=pbcl_pairs.astype(np.int32, copy=False),
                all_instance_pairs=all_pairs.astype(np.int32, copy=False),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_record(row: Mapping[str, str]) -> dict[str, object]:
    return {
        "tile_id": str(row["tile_id"]),
        "image": str(_source_path(row["image_path"])),
        "semantic": None,
        "is_background": int(row["instance_count"]) == 0,
        "valid_height": NATIVE_SIZE,
        "valid_width": NATIVE_SIZE,
        "tile_size": PADDED_SIZE,
        "scene_id": str(row["macroblock_id"]),
        "species_domain": (
            "dense_close" if str(row["dense_candidate"]).lower() == "true" else "other"
        ),
    }


def _write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(dict(record), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for record in records
    )
    path.write_text(payload, encoding="utf-8")


def prepare_cache(
    *,
    protocol_root: str | Path = PROTOCOL_ROOT,
    cache_root: str | Path = CACHE_ROOT,
    progress_every: int = 50,
) -> dict[str, object]:
    protocol_root = Path(protocol_root).expanduser().resolve()
    cache_root = Path(cache_root).expanduser().resolve()
    rows = _load_protocol_rows(protocol_root)
    split_ids = {split: _split_ids(protocol_root, split) for split in SPLIT_COUNTS}
    cache_root.mkdir(parents=True, exist_ok=True)
    target_root = cache_root / "instances" / "targets"
    selected_ids = [tile_id for split in ("train", "val", "test") for tile_id in split_ids[split]]
    for index, tile_id in enumerate(selected_ids, start=1):
        row = rows[tile_id]
        target_path = target_root / f"{tile_id}.npz"
        problem = _target_problem(target_path, row)
        if problem is not None:
            _write_target(target_path, row)
            problem = _target_problem(target_path, row)
            if problem is not None:
                raise RuntimeError(f"{target_path}: {problem}")
        if progress_every > 0 and (index % progress_every == 0 or index == len(selected_ids)):
            print(f"prepared {index}/{len(selected_ids)} targets", flush=True)

    for split, tile_ids in split_ids.items():
        _write_jsonl(
            cache_root / "splits" / f"{split}.jsonl",
            (_manifest_record(rows[tile_id]) for tile_id in tile_ids),
        )
    manifest_hashes = {
        split: sha256_file(cache_root / "splits" / f"{split}.jsonl")
        for split in SPLIT_COUNTS
    }
    metadata = {
        "dataset": DATASET_NAME,
        "version": DATASET_VERSION,
        "protocol_root": str(protocol_root),
        "source_root": str(SOURCE_ROOT.resolve()),
        "native_size": NATIVE_SIZE,
        "padded_size": PADDED_SIZE,
        "padding": {"left": 0, "top": 0, "right": 22, "bottom": 22},
        "padding_excluded_by_valid_mask": True,
        "source_band_count": 8,
        "selected_bands": list(BAND_NAMES),
        "selected_band_indices_zero_based": list(BAND_INDICES_ZERO_BASED),
        "selected_band_indices_rasterio": list(BAND_INDICES_RASTERIO),
        "raw_clip_min": 0.0,
        "raw_clip_max": RAW_SCALE,
        "raw_offset": 0.0,
        "raw_scale": RAW_SCALE,
        "normalize_mean": list(NORMALIZE_MEAN),
        "normalize_std": list(NORMALIZE_STD),
        "normalization_pixel_count_per_channel": NORMALIZATION_PIXEL_COUNT_PER_CHANNEL,
        "normalization_scope": "all native train pixels including source zeros",
        "split_counts": dict(SPLIT_COUNTS),
        "source_split_sha256": dict(SOURCE_SPLIT_HASHES),
        "manifest_sha256": manifest_hashes,
        "target_count": len(selected_ids),
        "buffer_excluded": 149,
    }
    (cache_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def validate_cache(
    *,
    protocol_root: str | Path = PROTOCOL_ROOT,
    cache_root: str | Path = CACHE_ROOT,
    validate_targets: bool = True,
) -> dict[str, object]:
    protocol_root = Path(protocol_root).expanduser().resolve()
    cache_root = Path(cache_root).expanduser().resolve()
    metadata_path = cache_root / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"Vegas dataset cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset") != DATASET_NAME or metadata.get("version") != DATASET_VERSION:
        raise RuntimeError(f"Vegas dataset cache identity mismatch: {metadata_path}")
    if list(metadata.get("selected_band_indices_zero_based", ())) != list(
        BAND_INDICES_ZERO_BASED
    ):
        raise RuntimeError("Vegas selected-band contract mismatch")
    if list(metadata.get("normalize_mean", ())) != list(NORMALIZE_MEAN):
        raise RuntimeError("Vegas normalization mean mismatch")
    if list(metadata.get("normalize_std", ())) != list(NORMALIZE_STD):
        raise RuntimeError("Vegas normalization std mismatch")
    expected_protocol_root = protocol_root.resolve()
    expected_source_root = SOURCE_ROOT.expanduser().resolve()
    metadata_protocol_root = Path(
        str(metadata.get("protocol_root", ""))
    ).expanduser().resolve()
    metadata_source_root = Path(
        str(metadata.get("source_root", ""))
    ).expanduser().resolve()
    if metadata_protocol_root != expected_protocol_root:
        raise RuntimeError(
            "Vegas cache protocol_root does not match the configured local dataset index"
        )
    if metadata_source_root != expected_source_root:
        raise RuntimeError(
            "Vegas cache source_root does not match the configured local dataset index"
        )
    rows = _load_protocol_rows(protocol_root)
    checked_targets = 0
    for split in SPLIT_COUNTS:
        tile_ids = _split_ids(protocol_root, split)
        manifest = cache_root / "splits" / f"{split}.jsonl"
        expected_hash = str(metadata["manifest_sha256"][split])
        if not manifest.is_file() or sha256_file(manifest) != expected_hash:
            raise RuntimeError(f"Vegas {split} cache manifest hash mismatch")
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        if [str(record["tile_id"]) for record in records] != tile_ids:
            raise RuntimeError(f"Vegas {split} cache manifest order/content mismatch")
        for record in records:
            image_path = Path(str(record.get("image", ""))).expanduser()
            try:
                image_path.resolve().relative_to(expected_source_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Vegas {split} manifest points outside the configured local dataset index: "
                    f"{image_path}"
                ) from exc
            if not image_path.is_file():
                raise RuntimeError(f"Vegas {split} manifest image is missing: {image_path}")
        if validate_targets:
            for tile_id in tile_ids:
                target_path = cache_root / "instances" / "targets" / f"{tile_id}.npz"
                problem = _target_problem(target_path, rows[tile_id])
                if problem is not None:
                    raise RuntimeError(f"{target_path}: {problem}")
                checked_targets += 1
    return {
        "status": "ok",
        "cache_root": str(cache_root),
        "split_counts": dict(SPLIT_COUNTS),
        "target_count": checked_targets if validate_targets else int(metadata["target_count"]),
        "manifest_sha256": dict(metadata["manifest_sha256"]),
    }


__all__ = [
    "BAND_INDICES_ZERO_BASED",
    "BAND_NAMES",
    "CACHE_ROOT",
    "DATASET_NAME",
    "DATASET_VERSION",
    "EWC_TARGET_ROOT",
    "NATIVE_SIZE",
    "NORMALIZATION_PIXEL_COUNT_PER_CHANNEL",
    "NORMALIZE_MEAN",
    "NORMALIZE_STD",
    "PADDED_SIZE",
    "PROTOCOL_ROOT",
    "RAW_SCALE",
    "SOURCE_ROOT",
    "SOURCE_SPLIT_HASHES",
    "SPLIT_COUNTS",
    "instance_pairs",
    "canonicalize_instance_map",
    "prepare_cache",
    "rasterize_instance_map",
    "sha256_file",
    "validate_cache",
]
