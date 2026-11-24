from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m geoseg", description="GeoSeg experiment runtime")
    subcommands = parser.add_subparsers(dest="command", required=True)

    for name in ("preflight", "inspect", "train"):
        command = subcommands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--seed", type=int, choices=(42, 43, 44), default=None)
    preflight = subcommands.choices["preflight"]
    preflight.add_argument("--report", type=Path, default=None)
    train = subcommands.choices["train"]
    train.add_argument("--compile", action="store_true")
    train.add_argument("--channels-last", action="store_true")
    train.add_argument("--cudnn-benchmark", action="store_true")
    train.add_argument("--persistent-workers", action="store_true")
    train.add_argument("--prefetch-factor", type=int, default=None)
    train.add_argument("--skip-gdal-directory-scan", action="store_true")
    train.add_argument("--async-writes", action="store_true")
    train.add_argument("--max-epochs", type=int, default=None)
    train.add_argument("--weights-path", type=Path, default=None)
    train.add_argument("--log-name", type=str, default=None)
    train.add_argument("--logger-version", type=int, default=None)
    train.add_argument("--resume-checkpoint", type=Path, default=None)
    train.add_argument("--train-batch-size", type=int, default=None)
    train.add_argument("--accumulate-grad-batches", type=int, default=None)

    evaluate = subcommands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--seed", type=int, choices=(42, 43, 44), default=None)
    evaluate.add_argument("--checkpoint", type=Path, default=None)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--threshold", type=float, default=0.5)
    evaluate.add_argument("--out-dir", type=Path, default=None)
    evaluate.add_argument("--num-workers", type=int, default=None)
    evaluate.add_argument("--save-predictions", action="store_true")
    evaluate.add_argument("--save-probabilities", action="store_true")
    evaluate.add_argument("--probability-dtype", choices=("float16", "float32"), default="float16")
    evaluate.add_argument("--separate-background-batches", action="store_true")
    evaluate.add_argument("--max-samples", type=int, default=0)
    evaluate.add_argument("--measure-efficiency", action="store_true")
    evaluate.add_argument("--strict", action="store_true")

    pretrained_index = subcommands.add_parser("pretrained-index")
    pretrained_index.add_argument("--root", type=Path, default=None)
    pretrained_index.add_argument("--workspace-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "inspect":
        from geoseg.runtime import load_experiment

        experiment = load_experiment(args.config, build=False, seed=args.seed)
        print(experiment.spec.to_json(), end="")
        return 0

    if args.command == "preflight":
        from geoseg.runtime import load_experiment, preflight_experiment

        experiment = load_experiment(args.config, build=False, seed=args.seed)
        report = preflight_experiment(experiment, report_path=args.report)
        print(report)
        return 0

    if args.command == "train":
        from train_supervision import run_training

        run_training(
            args.config,
            seed=args.seed,
            runtime_options={
                "compile": bool(args.compile),
                "channels_last": bool(args.channels_last),
                "cudnn_benchmark": bool(args.cudnn_benchmark),
                "persistent_workers": bool(args.persistent_workers),
                "prefetch_factor": args.prefetch_factor,
                "skip_gdal_directory_scan": args.skip_gdal_directory_scan,
                "async_writes": bool(args.async_writes),
            },
            training_overrides={
                "max_epochs": args.max_epochs,
                "weights_path": args.weights_path,
                "log_name": args.log_name,
                "logger_version": args.logger_version,
                "resume_checkpoint": args.resume_checkpoint,
                "train_batch_size": args.train_batch_size,
                "accumulate_grad_batches": args.accumulate_grad_batches,
            },
        )
        return 0

    if args.command == "evaluate":
        from geoseg.evaluation import EvaluationRequest, evaluate_checkpoint
        from geoseg.runtime import load_experiment
        from scripts.evaluate_aquav3_checkpoint import checkpoint_evaluation_environment

        with checkpoint_evaluation_environment(args.config):
            experiment = load_experiment(args.config, build=True, seed=args.seed)
        request = EvaluationRequest(
            checkpoint=args.checkpoint,
            split=args.split,
            threshold=args.threshold,
            out_dir=args.out_dir,
            num_workers=args.num_workers,
            save_predictions=args.save_predictions,
            save_probabilities=args.save_probabilities,
            probability_dtype=args.probability_dtype,
            separate_background_batches=args.separate_background_batches,
            max_samples=args.max_samples,
            measure_efficiency=args.measure_efficiency,
            strict=args.strict,
        )
        summary = evaluate_checkpoint(experiment, request)
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    if args.command == "pretrained-index":
        from geoseg.pretrained import index_pretrained_models

        manifest = index_pretrained_models(root=args.root, workspace_root=args.workspace_root)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")
