try:
    import rasterio  # noqa: F401  # Load GDAL/libtiff before PyTorch/OpenCV bind an incompatible libjpeg.
except ImportError:
    rasterio = None

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from geoseg.engine.callbacks import (
    CheckpointCallbackFactory,
    ComparisonEpochTiming,
    EpochStateCheckpoint,
    FinalStateCheckpoint,
    FiniteTrainingLoss,
    MultiMetricBestCheckpoint,
    PilotStateCheckpoint,
    TrainingManifestCallback,
)
from geoseg.engine.task import SegmentationTask
from geoseg.experiments.reproducibility import seed_experiment
from geoseg.runtime import apply_runtime_optimizations, load_experiment, preflight_experiment


# Public compatibility name used by existing inference scripts and checkpoints.
Supervision_Train = SegmentationTask

_AQUAV3_CONFIG_DIRS = {
    "aquadataset",
    "vegas",
    "pbcl_revision",
    "segformer_loss_comparison",
    "vegasdenseclosepbcl_v1",
}


def seed_everything(seed):
    seed_experiment(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=None)
    return parser.parse_args()


def write_training_protocol_manifest(config, config_path: str | Path) -> Path:
    if not hasattr(config, "training_protocol"):
        raise AttributeError("config has no training_protocol")
    payload = dict(config.training_protocol)
    payload["config_path"] = str(config_path)
    path = Path(config.weights_path) / "training_protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def build_checkpoint_callbacks(config):
    if hasattr(config, "checkpoint_monitors"):
        callbacks = [
            MultiMetricBestCheckpoint(
                directory=config.weights_path,
                filename=config.weights_name,
                plain_filenames=bool(getattr(config, "plain_checkpoint_names", False)),
                monitors=dict(config.checkpoint_monitors),
                save_weights_only=bool(
                    getattr(config, "checkpoint_save_weights_only", True)
                ),
                start_epoch=int(getattr(config, "checkpoint_start_epoch", 10)),
            )
        ]
        if bool(getattr(config, "epoch_state_checkpoint", False)):
            callbacks.append(
                EpochStateCheckpoint(
                    config.weights_path,
                    plain_filename=bool(getattr(config, "plain_checkpoint_names", False)),
                    start_epoch=int(
                        getattr(config, "epoch_state_checkpoint_start_epoch", 1)
                    ),
                    every_n_epochs=int(
                        getattr(
                            config,
                            "epoch_state_checkpoint_every_n_epochs",
                            1,
                        )
                    ),
                )
            )
        if bool(getattr(config, "save_final_state", False)):
            callbacks.append(FinalStateCheckpoint(config.weights_path))
        if bool(getattr(config, "plain_checkpoint_names", False)):
            callbacks.append(ComparisonEpochTiming(config.weights_path))
        if bool(getattr(config, "guard_finite_training_loss", False)):
            callbacks.append(FiniteTrainingLoss())
        if bool(getattr(config, "distillation_spec", {}).get("enabled", False)):
            callbacks.append(
                PilotStateCheckpoint(
                    config.weights_path,
                    pilot_epochs=int(getattr(config, "round2_pilot_epochs", 8)),
                )
            )
        return callbacks
    return CheckpointCallbackFactory.from_config(config).build()


def build_trainer_kwargs(config, experiment=None) -> dict:
    callbacks = build_checkpoint_callbacks(config)
    if experiment is not None:
        callbacks.append(TrainingManifestCallback(experiment))
    logger = CSVLogger(
        os.environ.get("GEOSEG_LIGHTNING_LOG_ROOT", "lightning_logs"),
        name=config.log_name,
        version=getattr(config, "logger_version", None),
    )
    if getattr(config, "resume_ckpt_path", None):
        metrics_path = Path(logger.log_dir) / "metrics.csv"
        if metrics_path.is_file():
            existing_metrics = metrics_path.read_bytes()
            with metrics_path.open(newline="", encoding="utf-8") as handle:
                fieldnames = list(csv.DictReader(handle).fieldnames or ())
            experiment_writer = logger.experiment
            temporary = metrics_path.with_name(f".{metrics_path.name}.resume.tmp")
            temporary.write_bytes(existing_metrics)
            os.replace(temporary, metrics_path)
            experiment_writer.metrics_keys = fieldnames
    kwargs = {
        "devices": config.gpus,
        "max_epochs": config.max_epoch,
        "accelerator": "auto",
        "check_val_every_n_epoch": config.check_val_every_n_epoch,
        "callbacks": callbacks,
        "strategy": "auto",
        "logger": logger,
    }
    if hasattr(config, "checkpoint_monitors"):
        kwargs["enable_checkpointing"] = False
    for name in ("precision", "gradient_clip_val", "accumulate_grad_batches", "num_sanity_val_steps"):
        if hasattr(config, name):
            kwargs[name] = getattr(config, name)
    return kwargs


def _apply_training_options(experiment, options: dict | None):
    if not options:
        return experiment
    prefetch = options.get("prefetch_factor")
    if prefetch is not None and int(prefetch) <= 0:
        raise ValueError("prefetch_factor must be positive")
    spec = replace(
        experiment.spec,
        data={
            **experiment.spec.data,
            **({"skip_gdal_directory_scan": True} if options.get("skip_gdal_directory_scan") else {}),
        },
        runtime={
            **experiment.spec.runtime,
            "compile": bool(options.get("compile", False)),
            "channels_last": bool(options.get("channels_last", False)),
            "cudnn_benchmark": bool(options.get("cudnn_benchmark", False)),
        },
        loader={
            **experiment.spec.loader,
            "persistent_workers": bool(options.get("persistent_workers", False)),
            "prefetch_factor": int(prefetch) if prefetch is not None else None,
        },
        artifacts={
            **experiment.spec.artifacts,
            "async_writes": bool(options.get("async_writes", False)),
        },
    )
    experiment.components.datamodule.spec = spec
    experiment.config.async_writes = bool(spec.artifacts["async_writes"])
    return replace(experiment, spec=spec)


def apply_training_overrides(
    experiment,
    *,
    max_epochs: int | None = None,
    weights_path: str | Path | None = None,
    log_name: str | None = None,
    logger_version: int | None = None,
    resume_checkpoint: str | Path | None = None,
    train_batch_size: int | None = None,
    accumulate_grad_batches: int | None = None,
):
    """Return an unbuilt experiment with explicit, isolated run overrides."""

    if experiment.components is not None:
        raise ValueError("Training overrides must be applied before model construction")
    if max_epochs is not None and int(max_epochs) <= 0:
        raise ValueError("max_epochs must be positive")

    spec = experiment.spec
    runtime = dict(spec.runtime)
    optimization = dict(spec.optimization)
    scheduler = dict(optimization["scheduler"])
    artifacts = dict(spec.artifacts)
    loader = dict(spec.loader)

    if max_epochs is not None:
        runtime["max_epochs"] = int(max_epochs)
        if (
            "schedule_epochs" in scheduler
            and spec.provenance.get("family")
            != "ssepnet_multiscale_dinov3_v2"
        ):
            scheduler["schedule_epochs"] = int(max_epochs)
        optimization["scheduler"] = scheduler
    if weights_path is not None:
        output = Path(weights_path).expanduser()
        artifacts["weights_path"] = str(output)
        artifacts["val_prediction_dir"] = str(output / "val_predictions")
    if log_name is not None:
        if not str(log_name).strip():
            raise ValueError("log_name must be non-empty")
        artifacts["log_name"] = str(log_name)
    if logger_version is not None:
        if int(logger_version) < 0:
            raise ValueError("logger_version must be non-negative")
        artifacts["logger_version"] = int(logger_version)
    if resume_checkpoint is not None:
        checkpoint = Path(resume_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        runtime["resume_checkpoint"] = str(checkpoint)
        if (
            spec.provenance.get("family") == "ssepnet_multiscale_dinov3_v2"
            and logger_version is None
        ):
            artifacts["logger_version"] = 1
    if train_batch_size is not None:
        if int(train_batch_size) <= 0:
            raise ValueError("train_batch_size must be positive")
        loader["train_batch_size"] = int(train_batch_size)
    if accumulate_grad_batches is not None:
        if int(accumulate_grad_batches) <= 0:
            raise ValueError("accumulate_grad_batches must be positive")
        optimization["accumulate_grad_batches"] = int(accumulate_grad_batches)
    return replace(
        experiment,
        spec=replace(
            spec,
            loader=loader,
            runtime=runtime,
            optimization=optimization,
            artifacts=artifacts,
        ),
    )


def run_training(
    config_path: str | Path,
    *,
    seed: int | None = None,
    runtime_options: dict | None = None,
    training_overrides: dict | None = None,
) -> None:
    path = Path(config_path).expanduser().resolve()
    if path.parent.name not in _AQUAV3_CONFIG_DIRS:
        if (
            runtime_options and any(value not in (None, False) for value in runtime_options.values())
        ) or (
            training_overrides and any(value is not None for value in training_overrides.values())
        ):
            raise ValueError("Shared runtime overrides are currently supported only for AquaV3 configs")
        config = py2cfg(path)
        seed_everything(int(getattr(config, "seed", 42)))
        if hasattr(config, "training_protocol"):
            write_training_protocol_manifest(config, path)
        model = Supervision_Train(config)
        if getattr(config, "pretrained_ckpt_path", None):
            model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)
        trainer = pl.Trainer(**build_trainer_kwargs(config))
        trainer.fit(model=model, ckpt_path=getattr(config, "resume_ckpt_path", None))
        return

    unresolved = apply_training_overrides(
        load_experiment(path, build=False, seed=seed),
        **(training_overrides or {}),
    )
    preflight_experiment(unresolved)
    from geoseg.datasets.vegas_validation_ram import with_vegas_validation_ram
    from geoseg.runtime.aquav3_builder import build_aquav3_components

    unresolved = replace(unresolved, spec=with_vegas_validation_ram(unresolved.spec))
    spec, components = build_aquav3_components(unresolved.spec)
    experiment = replace(unresolved, spec=spec, components=components)
    experiment = _apply_training_options(experiment, runtime_options)
    config = experiment.config
    seed_everything(int(getattr(config, "seed", 42)))
    experiment = apply_runtime_optimizations(experiment)
    config = experiment.config
    experiment.write_resolved_config()
    if hasattr(config, "training_protocol"):
        write_training_protocol_manifest(config, path)

    task_class = Supervision_Train
    if bool(getattr(config, "distillation_spec", {}).get("enabled", False)):
        from geoseg.engine.distillation import DistilledSegmentationTask

        task_class = DistilledSegmentationTask
    model = task_class(config)
    if getattr(config, "pretrained_ckpt_path", None):
        model = task_class.load_from_checkpoint(config.pretrained_ckpt_path, config=config)
    trainer = pl.Trainer(**build_trainer_kwargs(config, experiment=experiment))
    trainer.fit(model=model, ckpt_path=getattr(config, "resume_ckpt_path", None))


def main():
    args = get_args()
    run_training(args.config_path, seed=args.seed)


if __name__ == "__main__":
    main()
