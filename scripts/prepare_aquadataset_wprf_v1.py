#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoseg.datasets.aquadataset_wprf import (  # noqa: E402
    audit_wprf_targets,
    build_wprf_targets,
)
from geoseg.experiments.aquadataset_catalog import AQUADATASET_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or audit the locked AquaDateset WPRF matched target cache."
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=Path(AQUADATASET_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.overwrite and not args.build:
        raise SystemExit("--overwrite requires --build")
    if args.build:
        manifest = build_wprf_targets(
            args.dataset_root,
            overwrite=bool(args.overwrite),
            progress_every=100,
        )
        print(
            f"BUILT WPRF targets={manifest['target_count']} "
            f"root={manifest['target_root']}"
        )
    report = audit_wprf_targets(args.dataset_root)
    print(f"VALID WPRF targets={report['target_count']} manifest={report['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
