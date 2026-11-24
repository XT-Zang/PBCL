"""Build the fixed two-sensor AquaDateset geographic holdout on existing assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import shutil
import tempfile

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import transform_geom
from rasterio.windows import Window
from shapely import get_coordinates, make_valid
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union


VERSION = "aquav3-geographic-holdout-v1"
CRS = "EPSG:32651"
NORMAL = np.array([np.cos(np.deg2rad(30)), 0.5])
LOWER_CUT = 2091546.7995778362
UPPER_CUT = 2105049.445366734
HALF_GAP = 512
SPLITS = ("train", "val", "test")
EXPECTED = {
    "train": (1768, 911, 243, 90),
    "val": (879, 280, 355, 55),
    "test": (563, 339, 162, 59),
    "excluded": (338, 244),
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def project(geometry, crs: str):
    return geometry if crs == CRS else shape(transform_geom(crs, CRS, mapping(geometry)))


def limits(geometry) -> tuple[float, float]:
    projected = get_coordinates(geometry) @ NORMAL
    return float(projected.min()), float(projected.max())


def zone(low: float, high: float) -> str:
    if high <= LOWER_CUT - HALF_GAP:
        return "val"
    if low >= LOWER_CUT + HALF_GAP and high <= UPPER_CUT - HALF_GAP:
        return "test"
    if low >= UPPER_CUT + HALF_GAP:
        return "train"
    return "excluded"


def assign_tiles(source: Path, records: list[dict], audit: list[dict], group_by_source: dict):
    by_id = {row["tile_id"]: row for row in audit}
    if len(records) != 3548 or len(by_id) != len(audit) or set(by_id) != {r["tile_id"] for r in records}:
        raise ValueError("The source manifest does not match the audited 3,548 tiles")
    footprints = {}
    tile_groups = defaultdict(set)
    group_parts = defaultdict(list)
    for record in records:
        tile = record["tile_id"]
        row = by_id[tile]
        width, height = int(row["valid_width"]), int(row["valid_height"])
        if (row["scene_id"], row["split"], width, height) != (
            record["scene_id"], record["split"], record["valid_width"], record["valid_height"]
        ) or int(row["nodata_pixels_in_valid_rect"]) != 0:
            raise ValueError(f"Valid footprint differs from the source audit: {tile}")
        affine = Affine(*json.loads(row["transform"])[:6])
        corners = [(0, 0), (width, 0), (width, height), (0, height)]
        footprint = project(Polygon([affine * point for point in corners]), row["crs"])
        footprints[tile] = footprint
        if record["vector"]:
            vector = json.loads((source / record["vector"]).read_text(encoding="utf-8"))
            source_crs = vector.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
            for feature in vector["features"]:
                group = group_by_source[feature["properties"]["source_instance_id"]]
                tile_groups[tile].add(group)
                geometry = project(shape(feature["geometry"]), source_crs)
                if not geometry.is_valid:
                    geometry = make_valid(geometry)
                clipped = geometry.intersection(footprint)
                if not clipped.is_empty:
                    group_parts[group].append(clipped)

    group_shapes = {key: unary_union(parts) for key, parts in group_parts.items()}
    group_bounds = {key: limits(geometry) for key, geometry in group_shapes.items()}
    buckets = defaultdict(list)
    for record in records:
        tile = record["tile_id"]
        low, high = limits(footprints[tile])
        initial = zone(low, high)
        for group in tile_groups[tile]:
            group_low, group_high = group_bounds[group]
            low, high = min(low, group_low), max(high, group_high)
        selected = zone(low, high)
        entry = dict(record, split=selected, historical_split=record["split"],
                     spatial_split=f"geographic_holdout_{selected}",
                     historical_spatial_split=record["spatial_split"])
        if selected == "excluded":
            entry["exclusion_reason"] = (
                "valid_footprint_crosses_cut_or_buffer" if initial == "excluded"
                else "observed_object_group_crosses_cut_or_buffer"
            )
        buckets[selected].append(entry)

    checks = []
    input_union = {}
    object_union = {}
    groups_in_split = {}
    for split in SPLITS:
        tile_ids = [record["tile_id"] for record in buckets[split]]
        input_union[split] = unary_union([footprints[tile] for tile in tile_ids])
        groups_in_split[split] = set().union(*(tile_groups[tile] for tile in tile_ids))
        object_union[split] = unary_union([group_shapes[g] for g in sorted(groups_in_split[split])])
    for left, right in combinations(SPLITS, 2):
        a, b = input_union[left], input_union[right]
        shared = len(groups_in_split[left] & groups_in_split[right])
        overlap = a.intersection(b).area
        cross_ab = object_union[left].intersection(b).area
        cross_ba = object_union[right].intersection(a).area
        distance = a.distance(b)
        if shared or overlap or cross_ab or cross_ba or distance < 2 * HALF_GAP:
            raise ValueError(f"Geographic isolation failed for {left}/{right}")
        checks.append({"splits": [left, right], "shared_object_groups": shared,
                       "valid_footprint_overlap_m2": overlap, "minimum_distance_m": distance,
                       "left_objects_overlap_right_input_m2": cross_ab,
                       "right_objects_overlap_left_input_m2": cross_ba})
    return buckets, checks


def summarize(buckets, pair_rows, group_by_source):
    rows_by_tile = defaultdict(list)
    for row in pair_rows:
        rows_by_tile[row["tile_id"]].append(row)
    gap_codes = {"touching": 0, "1_2": 1, "3_5": 2}
    summary = {}
    for split, records in buckets.items():
        positives = sum(r["sample_role"] == "positive" for r in records)
        pairs = {}
        sensors = Counter()
        scenes = Counter()
        for record in records:
            scenes[record["scene_id"]] += 1
            sensors[record["scene_id"][:3]] += 1
            for row in rows_by_tile[record["tile_id"]]:
                key = tuple(sorted((row["first_source_id"], row["second_source_id"])))
                pairs[key] = min(pairs.get(key, 3), gap_codes[row["gap_bucket"]])
        near = {key: gap for key, gap in pairs.items() if gap}
        groups = {tuple(sorted(group_by_source[s] for s in key)) for key in near}
        observed = (len(records), positives, len(near), len(groups))
        if observed[:len(EXPECTED[split])] != EXPECTED[split]:
            raise ValueError(f"Fixed split statistics changed for {split}: {observed}")
        summary[split] = {
            "tiles": len(records), "positive": positives, "background": len(records) - positives,
            "sensors": dict(sorted(sensors.items())), "scenes": dict(sorted(scenes.items())),
            "historical_splits": dict(sorted(Counter(r["historical_split"] for r in records).items())),
            "source_pairs_1_5": len(near), "object_group_pairs_1_5": len(groups),
            "source_pairs_by_sensor": {
                sensor: {name: sum(key[0].startswith(sensor) and gap == code for key, gap in near.items())
                         for name, code in (("1_2", 1), ("3_5", 2))}
                for sensor in sorted(sensors)
            },
        }
    summary["excluded"]["reasons"] = dict(sorted(Counter(r["exclusion_reason"] for r in buckets["excluded"]).items()))
    return summary


def normalization(source: Path, train: list[dict]) -> dict:
    hist = np.zeros((4, 65536), dtype=np.int64)
    for i, record in enumerate(train, start=1):
        with rasterio.open(source / record["image"]) as raster:
            if (raster.height, raster.width, raster.count, raster.dtypes) != (512, 512, 4, ("uint16",) * 4):
                raise ValueError(f"Raster contract changed: {record['tile_id']}")
            window = Window(0, 0, record["valid_width"], record["valid_height"])
            valid = raster.dataset_mask(window=window) > 0
            if not valid.all():
                raise ValueError(f"GDAL valid mask differs from the audited footprint: {record['tile_id']}")
            pixels = raster.read(window=window)
        for band, values in enumerate(pixels):
            hist[band] += np.bincount(values.ravel(), minlength=65536)
        if i % 500 == 0:
            print(f"Normalization: {i}/{len(train)} train tiles", flush=True)
    count = int(hist[0].sum())
    transformed = (np.clip(np.arange(65536), 100, 60000) - 100) / 59900
    mean = hist @ transformed / count
    std = np.sqrt(hist @ (transformed ** 2) / count - mean ** 2)
    return {"source_split": "train", "valid_pixels_per_band": count,
            "validity": "manifest valid rectangle intersect GDAL dataset mask",
            "bands": ["B", "G", "R", "NIR"], "clip_min": 100, "clip_max": 60000,
            "raw_offset": 100, "raw_scale": 59900, "mean": mean.tolist(), "std": std.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, evidence, output = (p.resolve() for p in (args.source, args.evidence, args.output))
    if output.exists() or output.is_relative_to(source):
        raise FileExistsError(f"Use a new output directory outside the original dataset: {output}")

    receipt_path = output.parent / "aquav3-sceneholdout-v1-backup" / "backup_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for relative, expected in receipt["sha256"].items():
        if digest(receipt_path.parent / relative) != expected:
            raise ValueError(f"Original backup hash mismatch: {relative}")
        if not relative.startswith(("geoseg_source/", "original_builder/")) and digest(source / relative) != expected:
            raise ValueError(f"Source data no longer match the original backup: {relative}")

    source_manifest = source / "manifest.jsonl"
    with source_manifest.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    inputs = {
        "footprints": evidence / "20260911T161307Z/tile_footprints.csv",
        "object_groups": evidence / "20260912T013822Z/source_group_membership.csv",
        "near_pairs": evidence / "20260912T002132Z/gt_near_pairs.csv",
    }
    group_by_source = {r["source_instance_id"]: r["source_group"] for r in read_csv(inputs["object_groups"])}
    buckets, checks = assign_tiles(source, records, read_csv(inputs["footprints"]), group_by_source)
    summary = summarize(buckets, read_csv(inputs["near_pairs"]), group_by_source)
    print(f"Fixed split verified: {[(s, len(buckets[s])) for s in (*SPLITS, 'excluded')]}", flush=True)
    norm = normalization(source, buckets["train"])

    stage = Path(tempfile.mkdtemp(prefix=".geographic-", dir=output.parent))
    try:
        for directory in ("splits", "instances"):
            (stage / directory).mkdir()
        for entries in buckets.values():
            for entry in entries:
                for field in ("image", "semantic", "vector"):
                    if entry[field]:
                        asset = source / entry[field]
                        if not asset.is_file():
                            raise FileNotFoundError(asset)
                        entry[field] = os.path.relpath(asset, output)
        for split in (*SPLITS, "excluded"):
            write_jsonl(stage / "splits" / f"{split}.jsonl", buckets[split])
            (stage / "splits" / f"{split}.txt").write_text(
                "".join(f"{r['tile_id']}\n" for r in buckets[split]), encoding="utf-8"
            )
        retained = [r for split in SPLITS for r in buckets[split]]
        write_jsonl(stage / "manifest.jsonl", retained)
        positive = {r["tile_id"]: r["image"] for r in retained if not r["is_background"]}
        coco = json.loads((source / "instances/annotations.coco.json").read_text(encoding="utf-8"))
        coco["images"] = [dict(r, file_name=positive[r["tile_id"]]) for r in coco["images"] if r["tile_id"] in positive]
        if {r["tile_id"] for r in coco["images"]} != set(positive):
            raise ValueError("Retained positive tiles do not match the COCO index")
        image_ids = {r["id"] for r in coco["images"]}
        coco["annotations"] = [r for r in coco["annotations"] if r["image_id"] in image_ids]
        write_json(stage / "instances/annotations.coco.json", coco)
        write_json(stage / "normalization.json", norm)
        write_json(stage / "metadata.json", {
            "dataset_name": "AquaDateset", "version": VERSION, "parent_version": receipt["dataset_version"],
            "source_root": str(source), "task": "binary_semantic_segmentation_with_instance_aware_training_targets",
            "split_counts": {s: len(buckets[s]) for s in SPLITS},
            "split_hashes": {s: digest(stage / "splits" / f"{s}.jsonl") for s in SPLITS},
            "normalization_sha256": digest(stage / "normalization.json"),
        })
        write_json(stage / "split_audit.json", {
            "crs": CRS, "normal_angle_from_east_degrees": 30,
            "cuts_m": [LOWER_CUT, UPPER_CUT], "half_buffer_m": HALF_GAP,
            "zones_in_increasing_projection_order": ["val", "test", "train"],
            "summary": {s: summary[s] for s in (*SPLITS, "excluded")}, "isolation": checks,
        })
        shutil.copyfile(Path(__file__), stage / "build_aquadataset_geographic_holdout.py")
        write_json(stage / "provenance.json", {
            "original_backup_receipt": str(receipt_path), "original_manifest_sha256": digest(source_manifest),
            "evidence": {name: {"path": str(path), "sha256": digest(path)} for name, path in inputs.items()},
            "files_sha256": {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob("*")) if p.is_file()},
        })
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(output, flush=True)


if __name__ == "__main__":
    main()
