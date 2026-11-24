from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt


WPRF_SCHEMA_VERSION = 1
WPRF_TARGET_ROOT = "instance_targets/wprf_matched_v1"
WPRF_SOURCE_COMMIT = "418e362b440df08e72ae6da54abeec6559a8f442"
WPRF_GRID_STRIDE = 4
WPRF_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)
WPRF_SCALES: tuple[int, ...] = (1, 2, 4, 8, 16)
WPRF_SOURCES_PER_SCALE = 32
WPRF_PAIR_SALT = "aquadataset-wprf-matched-v1"
WPRF_IMAGE_SIZE = 512


def wprf_parameters() -> dict[str, object]:
    return {
        "source_commit": WPRF_SOURCE_COMMIT,
        "phi": "closing_3x3_then_zhang_suen_thinning",
        "grid_stride": WPRF_GRID_STRIDE,
        "neighborhood": [list(value) for value in WPRF_OFFSETS],
        "scales": list(WPRF_SCALES),
        "gamma": [0.2] * len(WPRF_SCALES),
        "sources_per_scale": WPRF_SOURCES_PER_SCALE,
        "pair_salt": WPRF_PAIR_SALT,
    }


def _dilate3(binary: np.ndarray) -> np.ndarray:
    source = np.asarray(binary, dtype=bool)
    padded = np.pad(source, 1, mode="constant", constant_values=False)
    result = np.zeros_like(source)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            result |= padded[
                1 + dy : 1 + dy + source.shape[0],
                1 + dx : 1 + dx + source.shape[1],
            ]
    return result


def _erode3(binary: np.ndarray) -> np.ndarray:
    source = np.asarray(binary, dtype=bool)
    padded = np.pad(source, 1, mode="constant", constant_values=False)
    result = np.ones_like(source)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            result &= padded[
                1 + dy : 1 + dy + source.shape[0],
                1 + dx : 1 + dx + source.shape[1],
            ]
    return result


def _neighbors8(image: np.ndarray) -> tuple[np.ndarray, ...]:
    padded = np.pad(image, 1, mode="constant", constant_values=False)
    return (
        padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:],
        padded[2:, 2:], padded[2:, 1:-1], padded[2:, :-2],
        padded[1:-1, :-2], padded[:-2, :-2],
    )


def _thin_iteration(image: np.ndarray, phase: int) -> np.ndarray:
    p2, p3, p4, p5, p6, p7, p8, p9 = _neighbors8(image)
    count = sum(value.astype(np.uint8) for value in (p2, p3, p4, p5, p6, p7, p8, p9))
    transitions = sum(
        ((~first) & second).astype(np.uint8)
        for first, second in (
            (p2, p3), (p3, p4), (p4, p5), (p5, p6),
            (p6, p7), (p7, p8), (p8, p9), (p9, p2),
        )
    )
    selected = image & (count >= 2) & (count <= 6) & (transitions == 1)
    if phase == 0:
        return selected & ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
    return selected & ~(p2 & p4 & p8) & ~(p2 & p6 & p8)


def phi_support(mask: np.ndarray) -> np.ndarray:
    image = _erode3(_dilate3(np.asarray(mask) > 0))
    if not image.any():
        return image
    while True:
        first = _thin_iteration(image, 0)
        image[first] = False
        second = _thin_iteration(image, 1)
        image[second] = False
        if not first.any() and not second.any():
            return image


def _project_occupancy(mask: np.ndarray, stride: int = WPRF_GRID_STRIDE) -> np.ndarray:
    height, width = mask.shape
    if height % stride or width % stride:
        raise ValueError(f"WPRF target shape {mask.shape} is not divisible by stride {stride}")
    return mask.reshape(height // stride, stride, width // stride, stride).max(axis=(1, 3)).astype(bool)


def _connected_components(support: np.ndarray) -> tuple[np.ndarray, int]:
    support = np.asarray(support, dtype=bool)
    result = np.zeros(support.shape, dtype=np.int32)
    current = 0
    for y, x in np.argwhere(support):
        if result[y, x] != 0:
            continue
        current += 1
        result[y, x] = current
        stack = [(int(y), int(x))]
        while stack:
            cy, cx = stack.pop()
            for dy, dx in WPRF_OFFSETS:
                ny, nx = cy + dy, cx + dx
                if (
                    0 <= ny < support.shape[0]
                    and 0 <= nx < support.shape[1]
                    and support[ny, nx]
                    and result[ny, nx] == 0
                ):
                    result[ny, nx] = current
                    stack.append((ny, nx))
    return result, current


def _dense_components(
    support: np.ndarray,
    components: np.ndarray,
    union: np.ndarray,
) -> np.ndarray:
    if not support.any() or not union.any():
        return np.zeros_like(components)
    nearest = distance_transform_edt(
        ~support, return_distances=False, return_indices=True
    )
    propagated = components[nearest[0], nearest[1]]
    result = np.zeros_like(components)
    result[union] = propagated[union]
    result[support] = components[support]
    return result


def _adjacency(support: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    coordinates = np.argwhere(support).astype(np.int32)
    index = -np.ones(support.shape, dtype=np.int32)
    if len(coordinates):
        index[coordinates[:, 0], coordinates[:, 1]] = np.arange(len(coordinates))
    neighbors: list[list[int]] = [[] for _ in range(len(coordinates))]
    for node, (y, x) in enumerate(coordinates.tolist()):
        for dy, dx in WPRF_OFFSETS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < support.shape[0] and 0 <= nx < support.shape[1]:
                other = int(index[ny, nx])
                if other >= 0:
                    neighbors[node].append(other)
    return coordinates, neighbors


def _bfs(neighbors: list[list[int]], source: int, limit: int) -> np.ndarray:
    distance = np.full(len(neighbors), -1, dtype=np.int16)
    distance[source] = 0
    queue: deque[int] = deque((source,))
    while queue:
        node = queue.popleft()
        if distance[node] >= limit:
            continue
        for other in neighbors[node]:
            if distance[other] < 0:
                distance[other] = distance[node] + 1
                queue.append(other)
    return distance


def _sample_pairs(
    support: np.ndarray,
    components: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    coordinates, neighbors = _adjacency(support)
    empty = {
        f"{kind}_{scale}": np.zeros((0, 4), dtype=np.int16)
        for scale in WPRF_SCALES
        for kind in ("pos", "neg")
    }
    if len(coordinates) == 0:
        return empty
    component_of_node = components[coordinates[:, 0], coordinates[:, 1]]
    rng = np.random.default_rng(int(seed))
    result: dict[str, np.ndarray] = {}
    maximum_scale = max(WPRF_SCALES)
    for scale in WPRF_SCALES:
        positive: list[list[int]] = []
        negative: list[list[int]] = []
        attempts = 0
        while len(positive) < WPRF_SOURCES_PER_SCALE and attempts < 2000:
            attempts += 1
            source = int(rng.integers(0, len(coordinates)))
            component = int(component_of_node[source])
            distance = _bfs(neighbors, source, maximum_scale)
            lower = (scale + 1) // 2
            candidates = np.flatnonzero(
                (component_of_node == component)
                & (distance >= lower)
                & (distance <= scale)
            )
            if not len(candidates):
                continue
            target = int(rng.choice(candidates))
            sy, sx = coordinates[source]
            ty, tx = coordinates[target]
            positive.append([int(sy), int(sx), int(ty), int(tx)])

            different = np.flatnonzero(component_of_node != component)
            if len(different):
                delta = coordinates[different] - coordinates[source]
                squared = np.square(delta).sum(axis=1)
                near = different[squared <= scale * scale]
                negative_target = (
                    int(rng.choice(near))
                    if len(near)
                    else int(different[int(np.argmin(squared))])
                )
                ny, nx = coordinates[negative_target]
                negative.append([int(sy), int(sx), int(ny), int(nx)])

            far = np.flatnonzero(
                (component_of_node == component)
                & ((distance < 0) | (distance > scale))
            )
            far = far[far != source]
            if len(far):
                negative_target = int(rng.choice(far))
                ny, nx = coordinates[negative_target]
                negative.append([int(sy), int(sx), int(ny), int(nx)])
        result[f"pos_{scale}"] = np.asarray(positive, dtype=np.int16).reshape(-1, 4)
        result[f"neg_{scale}"] = np.asarray(negative, dtype=np.int16).reshape(-1, 4)
    return result


def build_wprf_target(union_mask: np.ndarray, *, pair_seed: int) -> dict[str, np.ndarray]:
    union_mask = np.asarray(union_mask, dtype=bool)
    support = _project_occupancy(phi_support(union_mask))
    union = _project_occupancy(union_mask)
    components, _ = _connected_components(support)
    dense = _dense_components(support, components, union)
    return {
        "support": support.astype(np.uint8),
        "union": union.astype(np.uint8),
        "components": components.astype(np.int32),
        "dense_components": dense.astype(np.int32),
        **_sample_pairs(support, components, seed=pair_seed),
    }


def _resize_union_to_locked_grid(union: np.ndarray) -> np.ndarray:
    if union.shape == (WPRF_IMAGE_SIZE, WPRF_IMAGE_SIZE):
        return np.asarray(union, dtype=bool)
    import cv2

    resized = cv2.resize(
        np.asarray(union, dtype=np.uint8),
        (WPRF_IMAGE_SIZE, WPRF_IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wprf_pair_seed(tile_id: str, *, salt: str = WPRF_PAIR_SALT) -> int:
    digest = hashlib.sha256(f"{salt}\0{tile_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _records(root: Path, split: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "splits" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".npz.tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                **payload,
                schema_version=np.asarray(WPRF_SCHEMA_VERSION, dtype=np.int16),
                source_commit=np.asarray(WPRF_SOURCE_COMMIT),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_wprf_target(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        if int(np.asarray(payload["schema_version"]).item()) != WPRF_SCHEMA_VERSION:
            raise ValueError(f"unsupported WPRF target schema: {path}")
        if str(np.asarray(payload["source_commit"]).item()) != WPRF_SOURCE_COMMIT:
            raise ValueError(f"WPRF target source commit mismatch: {path}")
        required = {"support", "dense_components"} | {
            f"{kind}_{scale}" for kind in ("pos", "neg") for scale in WPRF_SCALES
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"WPRF target missing {sorted(missing)}: {path}")
        return {key: np.asarray(payload[key]).copy() for key in required}


def build_wprf_targets(
    dataset_root: str | Path,
    *,
    instance_target_root: str | Path = "instance_targets/pbcl_gap5",
    target_root: str | Path = WPRF_TARGET_ROOT,
    splits: Sequence[str] = ("train", "val", "test"),
    overwrite: bool = False,
    progress_every: int = 0,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    instance_root = Path(instance_target_root)
    if not instance_root.is_absolute():
        instance_root = root / instance_root
    output = Path(target_root)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    total = sum(len(_records(root, split)) for split in splits)
    processed = 0
    for split in splits:
        records = _records(root, split)
        split_counts[split] = len(records)
        for record in records:
            tile_id = str(record["tile_id"])
            destination = output / f"{tile_id}.npz"
            source_hash = None
            if destination.is_file() and not overwrite:
                target = load_wprf_target(destination)
            else:
                if bool(record.get("is_background", False)):
                    height = int(record.get("valid_height", record.get("tile_size", 512)))
                    width = int(record.get("valid_width", record.get("tile_size", 512)))
                    union = np.zeros((height, width), dtype=bool)
                else:
                    source = instance_root / f"{tile_id}.npz"
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    source_hash = _sha256(source)
                    with np.load(source, allow_pickle=False) as payload:
                        union = np.asarray(payload["instance_map"]) > 0
                target = build_wprf_target(
                    _resize_union_to_locked_grid(union),
                    pair_seed=wprf_pair_seed(tile_id),
                )
                _atomic_npz(destination, target)
            rows.append(
                {
                    "split": split,
                    "tile_id": tile_id,
                    "sha256": _sha256(destination),
                    "source_instance_sha256": source_hash,
                    "support_pixels": int(target["support"].sum()),
                    "pair_counts": {
                        str(scale): {
                            "positive": int(target[f"pos_{scale}"].shape[0]),
                            "negative": int(target[f"neg_{scale}"].shape[0]),
                        }
                        for scale in WPRF_SCALES
                    },
                }
            )
            processed += 1
            if progress_every and (processed % progress_every == 0 or processed == total):
                print(f"prepared {processed}/{total} WPRF targets", flush=True)
    manifest = {
        "schema_version": WPRF_SCHEMA_VERSION,
        "method": "matched_wprf_affinity_edge_reachability",
        "parameters": wprf_parameters(),
        "dataset_root": str(root),
        "instance_target_root": str(instance_root),
        "target_root": str(output),
        "split_counts": split_counts,
        "target_count": len(rows),
        "targets": rows,
    }
    manifest_path = output / "manifest.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".manifest-", suffix=".json.tmp", dir=output
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def audit_wprf_targets(
    dataset_root: str | Path,
    *,
    target_root: str | Path = WPRF_TARGET_ROOT,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    targets = Path(target_root)
    if not targets.is_absolute():
        targets = root / targets
    manifest_path = targets / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != WPRF_SCHEMA_VERSION:
        raise ValueError("unsupported WPRF manifest schema")
    if manifest.get("parameters") != wprf_parameters():
        raise ValueError("WPRF manifest parameters do not match the locked contract")
    rows = manifest.get("targets")
    if not isinstance(rows, list) or len(rows) != int(manifest.get("target_count", -1)):
        raise ValueError("WPRF manifest target list is invalid")
    for row in rows:
        path = targets / f"{row['tile_id']}.npz"
        load_wprf_target(path)
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"WPRF target hash mismatch: {path}")
    return {
        "manifest": str(manifest_path),
        "target_count": len(rows),
        "split_counts": dict(manifest["split_counts"]),
        "parameters": wprf_parameters(),
        "valid": True,
    }


__all__ = [
    "WPRF_TARGET_ROOT",
    "WPRF_SCALES",
    "audit_wprf_targets",
    "build_wprf_target",
    "build_wprf_targets",
    "load_wprf_target",
    "phi_support",
    "wprf_parameters",
    "wprf_pair_seed",
]
