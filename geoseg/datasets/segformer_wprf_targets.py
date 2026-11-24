"""Paper-grid WPRF targets for augmented semantic masks."""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from .aquadataset_wprf import (
    WPRF_GRID_STRIDE, WPRF_OFFSETS, WPRF_SCALES, WPRF_SOURCES_PER_SCALE,
    WPRF_SOURCE_COMMIT, _atomic_npz, _project_occupancy,
)

UPSTREAM = Path(
    os.environ.get("WPRF_UPSTREAM_ROOT")
    or os.environ.get("GEOSEG_WPRF_UPSTREAM")
    or "third_party/WPRF_open"
)


@lru_cache(maxsize=1)
def author_target_functions():
    if not (UPSTREAM / "src/wprf/pairs.py").is_file():
        raise FileNotFoundError(
            f"WPRF author source {WPRF_SOURCE_COMMIT} expected at {UPSTREAM}. "
            "Clone https://github.com/d0g3/WPRF_open at commit "
            f"{WPRF_SOURCE_COMMIT} and set WPRF_UPSTREAM_ROOT (or GEOSEG_WPRF_UPSTREAM)."
        )
    sys.path.insert(0, str(UPSTREAM / "src"))
    from wprf.pairs import sample_multiscale_pairs
    from wprf.phi import phi_support

    return sample_multiscale_pairs, phi_support


def row_major_dense_components(support, components, union):
    dense = np.zeros_like(components)
    dense[support] = components[support]
    points = np.argwhere(support)
    queries = np.argwhere(union & ~support)
    if not len(points) or not len(queries):
        return dense
    if len(points) == 1:
        dense[tuple(queries.T)] = components[tuple(points[0])]
        return dense
    tree = cKDTree(points)
    distances, candidates = tree.query(queries, k=2)
    nearest = candidates[:, 0].copy()
    tied = np.flatnonzero(distances[:, 0] == distances[:, 1])
    for index, matches in zip(
        tied, tree.query_ball_point(queries[tied], np.nextafter(distances[tied, 0], np.inf)),
        strict=True,
    ):
        nearest[index] = min(matches)
    dense[tuple(queries.T)] = components[tuple(points[nearest].T)]
    return dense


def build_semantic_target(mask, valid, seed):
    sample_pairs, phi = author_target_functions()
    union = _project_occupancy(mask & valid)
    stride = WPRF_GRID_STRIDE
    grid_valid = valid.reshape(
        valid.shape[0] // stride, stride, valid.shape[1] // stride, stride,
    ).all(axis=(1, 3))
    union &= grid_valid
    support = _project_occupancy(phi(mask & valid, l_prune=0)) & grid_valid
    components, _ = label(support, structure=np.ones((3, 3), dtype=np.uint8))
    dense = row_major_dense_components(support, components, union)
    pairs = sample_pairs(
        gt_support=support, gt_component_id=components, gt_union_omega=union,
        offsets=WPRF_OFFSETS, k_list=WPRF_SCALES, r_list=WPRF_SCALES,
        num_sources_per_k=WPRF_SOURCES_PER_SCALE, seed=seed,
    )
    result = {"support": support, "dense_components": dense}
    for scale in WPRF_SCALES:
        result[f"pos_{scale}"] = pairs.pos_pairs[scale].astype(np.int16)
        result[f"neg_{scale}"] = pairs.neg_pairs_struct[scale].astype(np.int16)
    return result


def cached_semantic_target(root, mask, valid, seed):
    digest = hashlib.sha256()
    digest.update(f"{seed}:{mask.shape}:".encode())
    digest.update(mask.tobytes())
    digest.update(valid.tobytes())
    key = digest.hexdigest()
    path = Path(root) / key[:2] / f"{key}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            return dict(data)
    result = build_semantic_target(mask, valid, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(path, result)
    return result
