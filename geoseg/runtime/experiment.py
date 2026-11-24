from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Any

from .specs import ExperimentSpec, make_aquav3_spec


_CAMPAIGN_BY_DIR = {
    "aquav3": "baseline",
    "aquav3_tuned": "tuned",
    "aquav3_compare": "compare",
    "aquav3_compare_full_structure": "compare_full_structure",
    "aquav3_pbcl_crossarch": "pbcl_crossarch",
    "aquav3_ssepnet": "ssepnet",
}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExperimentComponents:
    config: Any
    model: Any
    loss: Any
    optimizer: Any
    scheduler: Any
    datamodule: Any


@dataclass(frozen=True)
class ResolvedExperiment:
    spec: ExperimentSpec
    config_path: Path
    components: ExperimentComponents | None = None

    @property
    def config(self) -> Any | None:
        return self.components.config if self.components is not None else None

    def write_resolved_config(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else Path(self.spec.artifacts["weights_path"]) / "resolved_config.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.spec.to_json(), encoding="utf-8")
        return target


def _identity_from_path(config_path: Path) -> tuple[str, str, int, str | None]:
    try:
        campaign = _CAMPAIGN_BY_DIR[config_path.parent.name]
    except KeyError as exc:
        raise ValueError(f"Unsupported declarative config directory: {config_path.parent.name}") from exc

    stem = config_path.stem
    seed_match = re.search(r"_seed(\d+)$", stem)
    seed = int(seed_match.group(1)) if seed_match else 42
    if seed_match:
        stem = stem[: seed_match.start()]

    treatment = None
    if campaign == "pbcl_crossarch":
        for suffix in ("_base", "_pbcl"):
            if stem.endswith(suffix):
                treatment = suffix[1:]
                stem = stem[: -len(suffix)]
                break
    return campaign, stem, seed, treatment


def _spec_from_legacy(spec: ExperimentSpec, config: Any) -> ExperimentSpec:
    def value(name: str, default: Any) -> Any:
        return getattr(config, name, default)

    runtime = {
        **spec.runtime,
        "seed": int(value("seed", spec.runtime["seed"])),
        "max_epochs": int(value("max_epoch", spec.runtime["max_epochs"])),
        "precision": str(value("precision", spec.runtime["precision"])),
        "devices": value("gpus", spec.runtime["devices"]),
        "check_val_every_n_epoch": int(value("check_val_every_n_epoch", 1)),
        "num_sanity_val_steps": int(value("num_sanity_val_steps", 0)),
        "compile": bool(value("torch_compile", value("compile", spec.runtime["compile"]))),
        "channels_last": bool(value("channels_last", spec.runtime["channels_last"])),
        "cudnn_benchmark": bool(value("cudnn_benchmark", spec.runtime["cudnn_benchmark"])),
    }
    train_loader = value("train_loader", None)
    val_loader = value("val_loader", None)
    test_loader = value("test_loader", None)
    loader = {
        **spec.loader,
        "train_batch_size": int(value("train_batch_size", spec.loader["train_batch_size"])),
        "val_batch_size": int(value("val_batch_size", spec.loader["val_batch_size"])),
        "test_batch_size": int(value("test_batch_size", spec.loader["test_batch_size"])),
        "train_workers": int(getattr(train_loader, "num_workers", spec.loader["train_workers"])),
        "eval_workers": int(getattr(val_loader, "num_workers", spec.loader["eval_workers"])),
        "pin_memory": bool(getattr(train_loader, "pin_memory", spec.loader["pin_memory"])),
        "drop_last": bool(getattr(train_loader, "drop_last", spec.loader["drop_last"])),
        "sampler": type(getattr(train_loader, "sampler", None)).__name__ if train_loader is not None else "",
        "collate": getattr(getattr(train_loader, "collate_fn", None), "__name__", ""),
        "test_workers": int(getattr(test_loader, "num_workers", spec.loader["eval_workers"])),
    }
    checkpoint = {
        **spec.checkpoint,
        "monitor": str(value("monitor", spec.checkpoint["monitor"])),
        "mode": str(value("monitor_mode", spec.checkpoint["mode"])),
        "save_top_k": int(value("save_top_k", spec.checkpoint["save_top_k"])),
        "every_n_epochs": int(value("checkpoint_every_n_epochs", spec.checkpoint["every_n_epochs"])),
        "start_epoch": int(
            value("checkpoint_start_epoch", spec.checkpoint.get("start_epoch", 10))
        ),
        "save_last": False,
        "declared_save_last": bool(value("save_last", spec.checkpoint.get("declared_save_last", False))),
    }
    metrics = {
        **spec.metrics,
        "binary": bool(value("binary_segmentation", spec.metrics["binary"])),
        "threshold": float(value("metric_threshold", spec.metrics["threshold"])),
        "deadhesion_validation": bool(value("deadhesion_validation", spec.metrics["deadhesion_validation"])),
    }
    artifacts = {
        **spec.artifacts,
        "run_name": str(value("weights_name", spec.artifacts["run_name"])),
        "weights_path": str(value("weights_path", spec.artifacts["weights_path"])),
        "log_name": str(value("log_name", spec.artifacts["log_name"])),
        "save_val_predictions": bool(value("save_val_predictions", spec.artifacts["save_val_predictions"])),
        "val_prediction_dir": str(value("val_prediction_dir", spec.artifacts["val_prediction_dir"])),
    }
    train_dataset = value("train_dataset", getattr(train_loader, "dataset", None))
    data = {
        **spec.data,
        "root": str(value("data_root", getattr(train_dataset, "root", spec.data["root"]))),
        "image_size": int(value("image_size", getattr(train_dataset, "image_size", spec.data["image_size"]))),
        "band_indices": list(getattr(train_dataset, "band_indices", spec.data["band_indices"])),
        "instance_target_root": (
            str(getattr(train_dataset, "instance_target_root"))
            if getattr(train_dataset, "instance_target_root", None) is not None
            else spec.data["instance_target_root"]
        ),
        "ewc_weight_root": (
            str(getattr(train_dataset, "ewc_weight_root"))
            if getattr(train_dataset, "ewc_weight_root", None) is not None
            else spec.data.get("ewc_weight_root")
        ),
        "wprf_target_root": (
            str(getattr(train_dataset, "wprf_target_root"))
            if getattr(train_dataset, "wprf_target_root", None) is not None
            else spec.data.get("wprf_target_root")
        ),
    }

    net = value("net", None)
    model = {
        **spec.model,
        "key": str(value("model_name", spec.model["key"])),
        "class_name": type(net).__name__ if net is not None else "",
        "parameter_count": int(sum(parameter.numel() for parameter in net.parameters()))
        if net is not None and hasattr(net, "parameters")
        else 0,
    }
    training_protocol = value("training_protocol", {})
    if isinstance(training_protocol, dict):
        model["pretrained_source"] = str(training_protocol.get("pretrain_source", ""))

    optimizer = value("optimizer", None)
    optimizer_group = optimizer.param_groups[0] if optimizer is not None and optimizer.param_groups else {}
    scheduler = value("lr_scheduler", None)
    loss = value("loss", None)
    optimization = {
        **spec.optimization,
        "loss": {
            **dict(spec.optimization["loss"]),
            "name": type(loss).__name__ if loss is not None else dict(spec.optimization["loss"])["name"],
        },
        "optimizer": {
            "name": type(optimizer).__name__ if optimizer is not None else str(spec.optimization["optimizer"]["name"]),
            "lr": float(value("lr", optimizer_group.get("lr", spec.optimization["optimizer"]["lr"]))),
            "weight_decay": float(
                value("weight_decay", optimizer_group.get("weight_decay", spec.optimization["optimizer"]["weight_decay"]))
            ),
            "parameter_groups": len(getattr(optimizer, "param_groups", [])),
        },
        "scheduler": {
            "name": type(scheduler).__name__ if scheduler is not None else str(spec.optimization["scheduler"]["name"]),
            "warmup_epochs": int(value("warmup_epochs", spec.optimization["scheduler"]["warmup_epochs"])),
            "min_lr_ratio": float(value("min_lr_ratio", spec.optimization["scheduler"]["min_lr_ratio"])),
            "max_epochs": runtime["max_epochs"],
        },
        "gradient_clip_val": float(value("gradient_clip_val", spec.optimization["gradient_clip_val"])),
        "accumulate_grad_batches": int(
            value("accumulate_grad_batches", spec.optimization["accumulate_grad_batches"])
        ),
    }
    provenance = {
        **spec.provenance,
        "architecture": str(value("model_name", spec.provenance["model_key"])),
        "config_source": str(spec.provenance["source_config"]),
    }
    if isinstance(training_protocol, dict):
        provenance["protocol_campaign"] = str(training_protocol.get("campaign", ""))
        provenance["pretrain_source"] = str(training_protocol.get("pretrain_source", ""))
    return replace(
        spec,
        model=model,
        data=data,
        loader=loader,
        optimization=optimization,
        runtime=runtime,
        checkpoint=checkpoint,
        metrics=metrics,
        artifacts=artifacts,
        provenance=provenance,
    )


_SEGFORMER_LOSS_ALIASES = {
    "ewc": "ewc",
    "cldice": "cldice",
    "cmalis": "cmalis",
    "wprf_gt": "wprf_gt",
    "wprf_valid": "wprf_valid",
    "boundary_split": "split",
    "split": "split",
    "bce_split": "split_paper",
    "split_paper": "split_paper",
}


def load_experiment(
    config_path: str | Path,
    *,
    build: bool = True,
    seed: int | None = None,
) -> ResolvedExperiment:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved_seed = 42 if seed is None else int(seed)

    if path.parent.name == "segformer_loss_comparison":
        from geoseg.experiments.segformer_loss_comparison import build_spec
        from .aquav3_builder import build_aquav3_components

        spec = build_spec(path.stem, resolved_seed, source_config=path)
        if not build:
            return ResolvedExperiment(spec=spec, config_path=path)
        spec, components = build_aquav3_components(spec)
        return ResolvedExperiment(spec=spec, config_path=path, components=components)

    if path.parent.name in {"pbcl_revision", "aquadataset"} and (
        path.parent.name == "pbcl_revision" or path.parent.parent.name == "configs"
    ):
        from .aquav3_builder import build_aquav3_components

        if path.stem.startswith("segformer_b2_"):
            suffix = path.stem.removeprefix("segformer_b2_")
            mapped = _SEGFORMER_LOSS_ALIASES.get(suffix)
            if mapped is not None:
                from geoseg.experiments.segformer_loss_comparison import build_spec

                spec = build_spec(f"aqua__{mapped}", resolved_seed, source_config=path)
                if not build:
                    return ResolvedExperiment(spec=spec, config_path=path)
                spec, components = build_aquav3_components(spec)
                return ResolvedExperiment(spec=spec, config_path=path, components=components)

        from geoseg.experiments.pbcl_revision import build_revision_spec

        spec = build_revision_spec(path.stem, resolved_seed, source_config=path)
        if not build:
            return ResolvedExperiment(spec=spec, config_path=path)
        spec, components = build_aquav3_components(spec)
        return ResolvedExperiment(spec=spec, config_path=path, components=components)

    if path.parent.name in {"vegas", "vegasdenseclosepbcl_v1"}:
        from .aquav3_builder import build_aquav3_components

        if path.stem.startswith("segformer_b2__"):
            suffix = path.stem.removeprefix("segformer_b2__")
            mapped = _SEGFORMER_LOSS_ALIASES.get(suffix)
            if mapped is not None:
                from geoseg.experiments.segformer_loss_comparison import build_spec

                spec = build_spec(f"vegas__{mapped}", resolved_seed, source_config=path)
                if not build:
                    return ResolvedExperiment(spec=spec, config_path=path)
                spec, components = build_aquav3_components(spec)
                return ResolvedExperiment(spec=spec, config_path=path, components=components)

        from geoseg.experiments.vegasdenseclose_catalog import (
            build_vegasdenseclose_spec,
        )

        spec = build_vegasdenseclose_spec(
            path.stem,
            resolved_seed,
            source_config=path,
        )
        if not build:
            return ResolvedExperiment(spec=spec, config_path=path)
        spec, components = build_aquav3_components(spec)
        return ResolvedExperiment(spec=spec, config_path=path, components=components)

    if path.parent.name in {
        "aquadataset",
        "aquadataset_instance_baselines",
        "aquadataset_candidates",
        "aquadataset_matched_controls",
        "aquadataset_lora_efficiency",
        "aquadataset_ocr_optimization",
        "aquadataset_dual_fpn_ocr",
        "aquadataset_multiscale_dinov3",
        "aquadataset_stage_adapter_fpn_ocr",
        "aquadataset_stage_adapter_fpn_binary",
        "aquadataset_loss_baselines",
        "aquadataset_structural_loss_baselines",
        "aquadataset_binary_stage_adapter_ablation",
        "aquadataset_stage_adapter_fpn_binary_sensitivity",
        "aquadataset_multiscale_dinov3_v2",
    }:
        from geoseg.experiments.aquadataset_catalog import build_aquadataset_spec

        spec = build_aquadataset_spec(path.stem, resolved_seed, source_config=path)
        if not build:
            return ResolvedExperiment(spec=spec, config_path=path)
        from .aquav3_builder import build_aquav3_components

        spec, components = build_aquav3_components(spec)
        return ResolvedExperiment(spec=spec, config_path=path, components=components)

    campaign, model_key, inferred_seed, treatment = _identity_from_path(path)
    resolved_seed = inferred_seed if seed is None else int(seed)
    spec = make_aquav3_spec(
        model_key,
        campaign=campaign,
        seed=resolved_seed,
        treatment=treatment,
        source_config=path,
    )
    if not build:
        return ResolvedExperiment(spec=spec, config_path=path)

    try:
        is_project_config = path.parent.parent.resolve() == (_PROJECT_ROOT / "config").resolve()
    except OSError:
        is_project_config = False
    if is_project_config:
        from .aquav3_builder import build_aquav3_components

        spec, components = build_aquav3_components(spec)
        return ResolvedExperiment(spec=spec, config_path=path, components=components)

    from tools.cfg import py2cfg

    config = py2cfg(path)
    spec = _spec_from_legacy(spec, config)
    from geoseg.datasets.aquav3_datamodule import AquaV3DataModule

    datamodule = AquaV3DataModule.from_legacy_config(config)
    components = ExperimentComponents(
        config=config,
        model=getattr(config, "net", None),
        loss=getattr(config, "loss", None),
        optimizer=getattr(config, "optimizer", None),
        scheduler=getattr(config, "lr_scheduler", None),
        datamodule=datamodule,
    )
    return ResolvedExperiment(spec=spec, config_path=path, components=components)
