"""Derived PBCL targets for the frozen Aqua geographic holdout."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation


FOUR_NEIGHBORS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def four_contact_flags(label_map: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    flags = np.zeros(len(pairs), dtype=bool)
    for index, (first, second, bucket) in enumerate(pairs):
        if bucket == 0:
            flags[index] = np.any(
                binary_dilation(label_map == first, structure=FOUR_NEIGHBORS)
                & (label_map == second)
            )
    return flags


def filter_instance_pairs(label_map: np.ndarray, pairs: np.ndarray, policy: str) -> np.ndarray:
    if policy == "all":
        return pairs
    if policy == "noncontact":
        return pairs[pairs[:, 2] > 0]
    flags = four_contact_flags(label_map, pairs)
    if policy == "no_four_contact":
        return pairs[~flags]
    if policy == "four_contact":
        return pairs[flags]
    raise ValueError(f"Unknown pair policy: {policy}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_instance_targets(root: Path) -> dict:
    from geoseg.metrics.deadhesion import (
        _instance_label_map, load_coco_index, rasterize_coco_instances,
    )

    target_root = root / "instance_targets" / "pbcl_gap5"
    target_root.mkdir(parents=True, exist_ok=True)
    coco_path = root / "instances" / "annotations.coco.json"
    coco = load_coco_index(coco_path)
    names = {"touching": 0, "1_2": 1, "3_5": 2, "6_10": 3}
    summary = {"coco_sha256": sha256_file(coco_path), "splits": {}, "files": {}}
    for split in ("train", "val", "test"):
        rows = [json.loads(line) for line in (root / "splits" / f"{split}.jsonl").read_text().splitlines()]
        counts = Counter()
        for row in rows:
            if row.get("is_background", False):
                continue
            tile_id = row["tile_id"]
            path = target_root / f"{tile_id}.npz"
            masks = rasterize_coco_instances(coco[tile_id], (512, 512))
            label_map, _, raw_pairs = _instance_label_map(masks)
            all_pairs = np.array(
                [[first + 1, second + 1, names[bucket]] for first, second, bucket in raw_pairs],
                dtype=np.int32,
            ).reshape(-1, 3)
            pairs = all_pairs[all_pairs[:, 2] < 3]
            direct = four_contact_flags(label_map, pairs)
            counts["positive_tiles"] += 1
            counts["four_contact"] += int(direct.sum())
            counts["diagonal_contact"] += int(((pairs[:, 2] == 0) & ~direct).sum())
            counts["gap_1_2"] += int((pairs[:, 2] == 1).sum())
            counts["gap_3_5"] += int((pairs[:, 2] == 2).sum())
            if path.exists():
                with np.load(path, allow_pickle=False) as existing:
                    if not np.array_equal(existing["instance_map"], label_map) or not np.array_equal(existing["instance_pairs"], pairs):
                        raise RuntimeError(f"Derived target changed: {path}")
            else:
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, instance_map=label_map, instance_pairs=pairs,
                                    all_instance_pairs=all_pairs, pair_four_contact=direct)
                temporary.replace(path)
            summary["files"][tile_id] = sha256_file(path)
        summary["splits"][split] = dict(counts)
        print(json.dumps({"split": split, **dict(counts)}), flush=True)
    expected = {"four_contact": 16, "diagonal_contact": 9, "gap_1_2": 66, "gap_3_5": 196}
    for key, value in expected.items():
        if summary["splits"]["train"].get(key, 0) != value:
            raise RuntimeError(f"Rebuilt Train {key} differs from audit: {summary['splits']['train']}")
    (target_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
