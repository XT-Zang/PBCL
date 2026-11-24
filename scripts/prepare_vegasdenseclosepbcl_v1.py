#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from geoseg.datasets.vegasdenseclose import (
    CACHE_ROOT,
    PROTOCOL_ROOT,
    prepare_cache,
    validate_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or validate the resumable VegasDenseClosePBCL_v1 training cache."
    )
    parser.add_argument("--protocol-root", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--check", action="store_true", help="Read-only full cache validation")
    parser.add_argument("--quick-check", action="store_true", help="Validate manifests without loading every target")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check or args.quick_check:
        result = validate_cache(
            protocol_root=args.protocol_root,
            cache_root=args.cache_root,
            validate_targets=not args.quick_check,
        )
    else:
        prepare_cache(
            protocol_root=args.protocol_root,
            cache_root=args.cache_root,
            progress_every=args.progress_every,
        )
        result = validate_cache(
            protocol_root=args.protocol_root,
            cache_root=args.cache_root,
            validate_targets=True,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
