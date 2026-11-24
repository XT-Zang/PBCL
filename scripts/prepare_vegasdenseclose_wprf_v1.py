#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoseg.datasets.vegasdenseclose import CACHE_ROOT  # noqa: E402
from geoseg.datasets.vegasdenseclose_wprf import (  # noqa: E402
    audit_vegas_wprf_manifest,
    build_vegas_wprf_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or audit the locked Vegas dynamic-WPRF source manifest."
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    args = parser.parse_args()
    if args.build:
        manifest = build_vegas_wprf_manifest(args.cache_root)
        print(
            f"BUILT Vegas WPRF sources={manifest['target_count']} "
            f"root={manifest['target_root']}"
        )
    report = audit_vegas_wprf_manifest(args.cache_root)
    print(
        f"VALID Vegas WPRF sources={report['target_count']} "
        f"manifest={report['manifest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
