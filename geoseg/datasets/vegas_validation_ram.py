"""Select a verified, shared tmpfs copy for the Vegas factorial validation loader."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import warnings

from geoseg.runtime.specs import ExperimentSpec


def with_vegas_validation_ram(spec: ExperimentSpec) -> ExperimentSpec:
    if spec.provenance.get("family") != "external_factorial":
        return spec
    marker = Path(spec.artifacts["weights_path"]).parents[1] / "validation_ram_cache.json"
    if not marker.is_file():
        return spec
    try:
        approved = json.loads(marker.read_text())
        cache = Path(approved["path"]).resolve()
        if cache.parent != Path("/dev/shm"):
            raise ValueError("cache is outside /dev/shm")
        inventory = json.loads((cache / "inventory.json").read_text())
        if approved["manifest_sha256"] != inventory["manifest_sha256"] or (
            inventory["manifest_sha256"] != spec.data["split_hashes"]["val"]
        ):
            raise ValueError("cache manifest differs from the current validation split")
        if len(inventory["files"]) != 2 * spec.data["split_counts"]["val"]:
            raise ValueError("cache does not cover all validation images and targets")
        for relative, (source, size, mtime) in inventory["files"].items():
            path = Path(relative)
            if len(path.parts) != 2 or path.parts[0] not in {"images", "targets"}:
                raise ValueError(f"invalid cache entry: {relative}")
            original = Path(source).stat()
            if (original.st_size, original.st_mtime_ns) != (size, mtime) or (cache / path).stat().st_size != size:
                raise ValueError(f"stale or incomplete cache entry: {relative}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        warnings.warn(f"Vegas validation RAM cache unavailable, using original files: {error}", stacklevel=2)
        return spec
    return replace(spec, data={**spec.data, "validation_ram_cache": str(cache)})
