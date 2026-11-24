#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoseg.datasets.aquadataset_ewc import (  # noqa: E402
    audit_ewc_targets,
    build_ewc_targets,
)
from geoseg.datasets.vegasdenseclose import (  # noqa: E402
    CACHE_ROOT,
    EWC_TARGET_ROOT,
    PROTOCOL_ROOT,
    validate_cache,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build or audit native-resolution VegasDenseClosePBCL_v1 EWC "
            "weight maps."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--protocol-root", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument(
        "--instance-target-root",
        default="instances/targets",
    )
    parser.add_argument("--target-root", default=EWC_TARGET_ROOT)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args(argv)
    if args.overwrite and not args.build:
        parser.error("--overwrite requires --build")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.expanduser().resolve()
    validate_cache(
        protocol_root=args.protocol_root,
        cache_root=dataset_root,
        validate_targets=bool(args.build),
    )
    built: dict[str, object] | None = None
    if args.build:
        manifest = build_ewc_targets(
            dataset_root,
            instance_target_root=args.instance_target_root,
            target_root=args.target_root,
            overwrite=bool(args.overwrite),
            progress_every=max(0, int(args.progress_every)),
        )
        built = {
            "target_count": manifest["target_count"],
            "split_counts": manifest["split_counts"],
            "target_root": manifest["target_root"],
        }
    audit = audit_ewc_targets(dataset_root, target_root=args.target_root)
    print(
        json.dumps(
            {"built": built, "audit": audit},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
