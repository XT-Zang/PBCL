#!/usr/bin/env python3
"""Unified CLI for Pairwise Bottleneck Connectivity Loss (PBCL) experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
CONFIGS_ROOT = REPO_ROOT / "configs"
DEFAULT_SEED = 42

# Short aliases for the most commonly inspected / run experiments.
ALIASES: dict[str, str] = {
    "pbcl": "aquadataset/full",
    "pbcl-dino": "aquadataset/full",
    "dinov3-full": "aquadataset/full",
    "dinov3-pbcl": "aquadataset/pbcl_only",
    "dinov3-boundary": "aquadataset/boundary",
    "dinov3-base": "aquadataset/base",
    "segformer-full": "aquadataset/segformer_b2_full",
    "segformer-boundary": "aquadataset/segformer_b2_boundary",
    "vegas-segformer-full": "vegas/segformer_b2__full",
    "vegas-segformer-boundary": "vegas/segformer_b2__boundary",
}


def discover_configs(dataset: str | None = None) -> list[tuple[str, Path]]:
    datasets = (dataset,) if dataset else ("aquadataset", "vegas")
    items: list[tuple[str, Path]] = []
    for ds in datasets:
        folder = CONFIGS_ROOT / ds
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.py")):
            if path.name.startswith("__"):
                continue
            items.append((f"{ds}/{path.stem}", path))
    return items


def resolve_config(target: str) -> Path:
    normalized = ALIASES.get(target.strip().lower(), target.strip())
    direct = Path(normalized)
    if direct.is_file():
        return direct.resolve()
    if not normalized.endswith(".py"):
        candidate = CONFIGS_ROOT / f"{normalized}.py"
        if candidate.is_file():
            return candidate.resolve()
        # Allow bare treatment name if unique in aquadataset/
        aqua_candidate = CONFIGS_ROOT / "aquadataset" / f"{normalized}.py"
        if aqua_candidate.is_file():
            return aqua_candidate.resolve()
    raise SystemExit(
        f"Unknown experiment or config path: {target!r}. "
        "Run `python run.py list` to see all available experiments."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PBCL: Pairwise Bottleneck Connectivity Loss experiment runner",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List available experiment configurations")
    list_cmd.add_argument(
        "--dataset",
        choices=("aquadataset", "vegas"),
        default=None,
        help="Filter by dataset",
    )
    list_cmd.add_argument(
        "--query",
        type=str,
        default="",
        help="Case-insensitive substring filter (e.g. segformer, cmalis, wprf, split, vmamba)",
    )

    for name, help_text in (
        ("inspect", "Print the resolved experiment specification as JSON"),
        ("preflight", "Validate environment, dataset protocol, targets, and pretrained weights"),
        ("train", "Train an experiment configuration"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument(
            "experiment",
            nargs="?",
            default="aquadataset/full",
            help="Experiment name (e.g. aquadataset/full, aquadataset/cmalis, vegas/segformer_b2__full) or config path",
        )
        cmd.add_argument(
            "--seed",
            type=int,
            choices=(42, 43, 44),
            default=DEFAULT_SEED,
            help="Reproducibility seed (default: %(default)s)",
        )

    eval_cmd = sub.add_parser("eval", help="Evaluate a trained checkpoint on val or test")
    eval_cmd.add_argument("experiment", help="Experiment name or config path")
    eval_cmd.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path (.ckpt)")
    eval_cmd.add_argument("--split", choices=("val", "test"), default="test")
    eval_cmd.add_argument("--seed", type=int, choices=(42, 43, 44), default=DEFAULT_SEED)
    eval_cmd.add_argument("--output-dir", type=Path, default=None)

    sub.add_parser(
        "verify-results",
        help="Verify all paper tables (Tables II-VI) against bundled results/",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list":
        configs = discover_configs(args.dataset)
        query = args.query.strip().lower()
        shown = 0
        for name, _ in configs:
            if query and query not in name.lower():
                continue
            print(name)
            shown += 1
        print(f"\nTotal: {shown} configuration(s).", file=sys.stderr)
        return 0

    if args.command == "verify-results":
        from scripts.verify_results import verify_all

        verify_all(verbose=True)
        return 0

    config_path = resolve_config(args.experiment)

    if args.command == "inspect":
        from geoseg.runtime import load_experiment

        experiment = load_experiment(config_path, build=False, seed=args.seed)
        print(experiment.spec.to_json(), end="")
        return 0

    if args.command == "preflight":
        from geoseg.runtime import load_experiment, preflight_experiment

        experiment = load_experiment(config_path, build=False, seed=args.seed)
        report = preflight_experiment(experiment)
        print(json.dumps(report, indent=2, default=str))
        return 0

    if args.command == "train":
        from train_supervision import run_training

        run_training(config_path, seed=args.seed)
        return 0

    if args.command == "eval":
        import subprocess

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_aquav3_checkpoint.py"),
            "--config",
            str(config_path),
            "--checkpoint",
            str(args.checkpoint),
            "--split",
            args.split,
            "--seed",
            str(args.seed),
        ]
        if args.output_dir is not None:
            cmd.extend(["--output-dir", str(args.output_dir)])
        return subprocess.call(cmd, cwd=REPO_ROOT)

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
