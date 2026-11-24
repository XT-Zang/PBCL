from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from geoseg.datasets.vegasdenseclose import (
    BAND_INDICES_ZERO_BASED,
    BAND_NAMES,
    CACHE_ROOT,
    DATASET_NAME,
    DATASET_VERSION,
    EWC_TARGET_ROOT,
    NATIVE_SIZE,
    NORMALIZATION_PIXEL_COUNT_PER_CHANNEL,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    PADDED_SIZE,
    PROTOCOL_ROOT,
    RAW_SCALE,
    SOURCE_ROOT,
    SOURCE_SPLIT_HASHES,
    SPLIT_COUNTS,
)
from geoseg.datasets.vegasdenseclose_wprf import VEGAS_WPRF_TARGET_ROOT
from geoseg.experiments.aquadataset_catalog import (
    AQUADATASET_EXPERIMENTS,
    build_aquadataset_spec,
)
from geoseg.runtime.specs import ExperimentSpec
from geoseg.experiments.pbcl_revision import TRANSFER_ARCHITECTURES, VMAMBA_ARCHITECTURE


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get("GEOSEG_OUTPUT_ROOT", str(ROOT))).expanduser().resolve()
MAIN_TABLE_EXPERIMENT_IDS = tuple(
    recipe.experiment_id
    for recipe in AQUADATASET_EXPERIMENTS.values()
    if recipe.family == "main_table"
)
ADDITIONAL_EXPERIMENT_IDS = (
    "mask2former_swin_t_instance",
    "ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_head",
)
VEGAS_EXPERIMENT_IDS = MAIN_TABLE_EXPERIMENT_IDS + ADDITIONAL_EXPERIMENT_IDS
VEGAS_EWC_EXPERIMENT_ID = (
    "ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_boundary_ewc"
)
VEGAS_EWC_EXPERIMENT_IDS = (VEGAS_EWC_EXPERIMENT_ID,)
VEGAS_STRUCTURAL_EXPERIMENT_IDS = (
    "ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_boundary_cldice",
    "ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_boundary_cmalis",
    "ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_boundary_wprf",
)
VEGAS_LOSS_EXPERIMENT_IDS = VEGAS_EWC_EXPERIMENT_IDS + VEGAS_STRUCTURAL_EXPERIMENT_IDS
# Preserve VEGAS_EXPERIMENT_IDS as the completed historical 12-model benchmark.
# Optional loss baselines are runnable without silently changing downstream
# iSAID matrices or canonical Vegas summary coverage.
VEGAS_SUPPORTED_EXPERIMENT_IDS = VEGAS_EXPERIMENT_IDS + VEGAS_LOSS_EXPERIMENT_IDS
VEGAS_SPLIT_WEIGHTS = {
    "segformer_b2__split_bce": 0.0,
    "segformer_b2__split_loss": 0.1,
}
VEGAS_SEEDS = (42, 43, 44)
VEGAS_FACTORIAL_ARCHITECTURES = TRANSFER_ARCHITECTURES + (VMAMBA_ARCHITECTURE,)
VEGAS_FACTORIAL_VARIANTS = ("base", "boundary", "pbcl", "full")
VEGAS_FACTORIAL_EXPERIMENTS = {
    f"{architecture}__{variant}": (architecture, variant)
    for architecture in VEGAS_FACTORIAL_ARCHITECTURES
    for variant in VEGAS_FACTORIAL_VARIANTS
}
VEGAS_SSEPNET_FACTORIAL_RECIPES = {
    f"ssepnet__{variant}": ADDITIONAL_EXPERIMENT_IDS[1].removesuffix("_head") + f"_{suffix}"
    for variant, suffix in zip(VEGAS_FACTORIAL_VARIANTS, ("base", "boundary", "pbcl", "head"), strict=True)
}
OOM_SAFE_BATCH_PROFILES = {
    "mask2former_swin_t_instance": (4, 3),
}


BUNDLED_VEGAS_METADATA = (
    ROOT / "splits" / "spacenet2_vegas_dense_close_v1" / "dataset_metadata.json"
)


def _cache_metadata(cache_root: Path) -> dict[str, object]:
    path = cache_root / "metadata.json"
    bundled = False
    if not path.is_file():
        if BUNDLED_VEGAS_METADATA.is_file():
            path = BUNDLED_VEGAS_METADATA
            bundled = True
        else:
            raise RuntimeError(
                f"{DATASET_NAME} cache is not prepared: {path}. "
                "Run scripts/prepare_vegasdenseclosepbcl_v1.py first."
            )
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("dataset") != DATASET_NAME or metadata.get("version") != DATASET_VERSION:
        raise RuntimeError(f"{DATASET_NAME} cache identity mismatch: {path}")
    if not bundled:
        metadata_protocol_root = Path(
            str(metadata.get("protocol_root", ""))
        ).expanduser().resolve()
        metadata_source_root = Path(
            str(metadata.get("source_root", ""))
        ).expanduser().resolve()
        if metadata_protocol_root != PROTOCOL_ROOT.resolve():
            raise RuntimeError(f"{DATASET_NAME} cache protocol root mismatch: {path}")
        if metadata_source_root != SOURCE_ROOT.resolve():
            raise RuntimeError(f"{DATASET_NAME} cache source root mismatch: {path}")
    return metadata


def build_vegasdenseclose_spec(
    experiment_id: str,
    seed: int,
    *,
    source_config: str | Path | None = None,
) -> ExperimentSpec:
    experiment_id = str(experiment_id).strip().lower()
    seed = int(seed)
    if experiment_id in VEGAS_SPLIT_WEIGHTS:
        return _build_split_spec(experiment_id, seed, source_config=source_config)
    factorial = VEGAS_FACTORIAL_EXPERIMENTS.get(experiment_id)
    ssepnet_recipe = VEGAS_SSEPNET_FACTORIAL_RECIPES.get(experiment_id)
    if experiment_id not in VEGAS_SUPPORTED_EXPERIMENT_IDS and factorial is None and ssepnet_recipe is None:
        raise ValueError(f"Unsupported {DATASET_NAME} experiment: {experiment_id}")
    if seed not in VEGAS_SEEDS:
        raise ValueError(f"{DATASET_NAME} seed must be one of {VEGAS_SEEDS}, got {seed}")

    base = build_aquadataset_spec(
        factorial[0] if factorial else ssepnet_recipe or experiment_id,
        seed,
        source_config=source_config,
    )
    is_ewc = experiment_id in VEGAS_EWC_EXPERIMENT_IDS
    is_structural = experiment_id in VEGAS_STRUCTURAL_EXPERIMENT_IDS
    is_wprf = experiment_id.endswith("_wprf")
    cache_root = (OUTPUT_ROOT / CACHE_ROOT).resolve()
    metadata = _cache_metadata(cache_root)
    original_batch = int(base.loader["train_batch_size"])
    original_accumulate = int(base.optimization["accumulate_grad_batches"])
    train_batch, accumulate = OOM_SAFE_BATCH_PROFILES.get(
        experiment_id,
        (original_batch, original_accumulate),
    )
    run_name = f"{experiment_id}_seed{seed}"
    output_dir = (
        OUTPUT_ROOT
        / "train_out"
        / DATASET_NAME
        / str(base.provenance["family"])
        / experiment_id
        / f"seed{seed}"
    )

    spec = replace(
        base,
        data={
            **base.data,
            "dataset": "vegasdenseclosepbcl_v1",
            "root": str(cache_root),
            "protocol_root": str(PROTOCOL_ROOT.resolve()),
            "source_root": str(SOURCE_ROOT.resolve()),
            "dataset_version": DATASET_VERSION,
            "split_counts": dict(SPLIT_COUNTS),
            "split_hashes": dict(metadata["manifest_sha256"]),
            "source_split_hashes": dict(SOURCE_SPLIT_HASHES),
            "bands": list(BAND_NAMES),
            "band_indices": list(BAND_INDICES_ZERO_BASED),
            "source_band_count": 8,
            "native_size": NATIVE_SIZE,
            "image_size": PADDED_SIZE,
            "resize_mode": "pad",
            "padding": {"left": 0, "top": 0, "right": 22, "bottom": 22},
            "padding_excluded_by_valid_mask": True,
            "raw_clip_min": 0.0,
            "raw_clip_max": RAW_SCALE,
            "raw_offset": 0.0,
            "raw_scale": RAW_SCALE,
            "normalize_mean": list(NORMALIZE_MEAN),
            "normalize_std": list(NORMALIZE_STD),
            "normalization_pixel_count_per_channel": (
                NORMALIZATION_PIXEL_COUNT_PER_CHANNEL
            ),
            "instance_target_root": "instances/targets",
            "ewc_weight_root": EWC_TARGET_ROOT if is_ewc else None,
            "wprf_target_root": VEGAS_WPRF_TARGET_ROOT if is_wprf else None,
            "wprf_dynamic_from_instance": is_wprf,
        },
        loader={
            **base.loader,
            "train_batch_size": train_batch,
            "val_batch_size": 4,
            "test_batch_size": 4,
            "balanced_sampler": False,
            "collate": "aquav3",
        },
        optimization={
            **base.optimization,
            "accumulate_grad_batches": accumulate,
        },
        checkpoint={
            **base.checkpoint,
            "policy": "best_overall_validation_only_v1",
            "monitors": {"overall": "val_overall_score"},
            "primary_role": "overall",
            "monitor": "val_overall_score",
            "save_top_k": 1,
            "every_n_epochs": 0,
            "save_last": False,
            "declared_save_last": False,
            "allow_epoch_state_checkpoint": False,
            "deduplicate_same_epoch": False,
        },
        artifacts={
            **base.artifacts,
            "run_name": run_name,
            "weights_path": str(output_dir),
            "log_name": (
                f"vegasdenseclosepbcl_v1/{base.provenance['family']}/"
                f"{experiment_id}/seed{seed}"
            ),
            "save_val_predictions": False,
            "val_prediction_dir": str(output_dir / "val_predictions"),
        },
        provenance={
            **base.provenance,
            "dataset": DATASET_NAME,
            "experiment_id": experiment_id,
            "training_protocol": (
                "vegasdenseclose_4band_native650_pad672_ewc_loss_only_v1"
                if is_ewc
                else (
                    "vegasdenseclose_4band_native650_pad672_structural_loss_v1"
                    if is_structural
                    else (
                    "vegasdenseclose_mask2former_instance_aligned_padding_v3"
                    if experiment_id == "mask2former_swin_t_instance"
                    else "vegasdenseclose_4band_pad672_40ep_overall_only_historical_batch_v2"
                ))
            ),
            "source_config": str(source_config) if source_config else "",
            "batch_profile_source": (
                "vegas_oom_safe_effective_batch_12"
                if experiment_id in OOM_SAFE_BATCH_PROFILES
                else "aquadataset_recipe"
            ),
            "instance_target_coordinate_policy": (
                "padded_672_target_zero_invalid_pixels_v1"
                if experiment_id == "mask2former_swin_t_instance"
                else None
            ),
            "instance_metric_postprocess_policy": (
                "no_grad_v1"
                if experiment_id == "mask2former_swin_t_instance"
                else None
            ),
            "model_order_scope": (
                "isolated_ewc_loss_baseline"
                if is_ewc
                else "isolated_structural_loss_baseline"
                if is_structural
                else "main_table_all_plus_mask2former_instance_plus_stage_adapter_binary"
            ),
            **(
                {
                    "reference_experiment_ids": [
                        "ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_head"
                    ],
                    "comparison_scope": "same_dataset_same_physical_architecture_ewc_vs_pbcl",
                    "ewc_target_coordinate_policy": "native_650_then_pad672_unit_weight_invalid_v1",
                }
                if is_ewc
                else {}
            ),
            **(
                {
                    "reference_experiment_ids": [
                        "ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_head"
                    ],
                    "comparison_scope": "same_dataset_matched_structural_supervision",
                    "wprf_target_coordinate_policy": (
                        "dynamic_after_native_augmentation_then_pad672"
                        if is_wprf
                        else None
                    ),
                }
                if is_structural
                else {}
            ),
        },
    )
    if factorial is None and ssepnet_recipe is None:
        return spec
    architecture, variant = factorial or ("ssepnet", experiment_id.split("__", 1)[1])
    boundary = variant in {"boundary", "full"}
    pbcl = variant in {"pbcl", "full"}
    loss = dict(spec.optimization["loss"])
    if factorial is not None and (boundary or pbcl):
        loss.update(
            name="PBCLBinarySegmentationLoss",
            boundary_weight=0.6 if boundary else 0.0,
            boundary_kernel=5,
            pbcl_weight=0.1 if pbcl else 0.0,
            pbcl_band_radius=3,
            pbcl_iterations=9,
            pbcl_threshold=0.5,
            pbcl_margin=0.1,
            pbcl_neighbors=4,
        )
    optimizer = dict(spec.optimization["optimizer"])
    if boundary and architecture in {"unetplusplus", "deeplabv3plus"}:
        optimizer["encoder_prefixes"] = [f"base_model.{prefix}" for prefix in optimizer["encoder_prefixes"]]
    directory = OUTPUT_ROOT / "train_out" / DATASET_NAME / "external_factorial" / experiment_id / f"seed{seed}"
    profile = {}
    profile_path = OUTPUT_ROOT / "train_out" / DATASET_NAME / "external_factorial" / "efficiency" / "capacity.json"
    if profile_path.is_file():
        profile = json.loads(profile_path.read_text()).get(architecture, {})
    microbatch = int(profile.get("microbatch", spec.loader["train_batch_size"]))
    if microbatch not in {1, 2, 3, 4, 6, 12}:
        raise ValueError(f"Invalid Vegas microbatch for {architecture}: {microbatch}")
    return replace(
        spec,
        loader={**spec.loader, "train_batch_size": microbatch},
        model={**spec.model, **({"external_boundary_head": True} if factorial is not None and boundary else {})},
        checkpoint={
            **spec.checkpoint,
            "save_last": True,
            "declared_save_last": True,
            "save_final_state": True,
        },
        optimization={
            **spec.optimization, "loss": loss, "optimizer": optimizer,
            "accumulate_grad_batches": 12 // microbatch,
        },
        artifacts={
            **spec.artifacts, "weights_path": str(directory),
            "log_name": f"vegasdenseclosepbcl_v1/external_factorial/{experiment_id}/seed{seed}",
            "val_prediction_dir": str(directory / "val_predictions"),
            "guard_finite_training_loss": True,
        },
        provenance={
            **spec.provenance, "family": "external_factorial",
            "training_protocol": "vegas_external_pbcl_factorial_v1",
            "model_order_scope": "external_factorial",
            "transfer_architecture": architecture, "treatment": variant,
        },
    )


def _build_split_spec(
    experiment_id: str, seed: int, *, source_config: str | Path | None,
) -> ExperimentSpec:
    base = build_vegasdenseclose_spec("segformer_b2", seed, source_config=source_config)
    family = "split_paper"
    output_dir = OUTPUT_ROOT / "train_out" / DATASET_NAME / family / experiment_id / f"seed{seed}"
    data = dict(base.data)
    data.pop("padding")
    return replace(
        base,
        data={
            **data,
            "image_size": 512,
            "resize_mode": "resize",
            "native_eval_targets": True,
            "padding_excluded_by_valid_mask": False,
        },
        optimization={
            **base.optimization,
            "loss": {
                "name": "SplitSegmentationLoss",
                "split_weight": VEGAS_SPLIT_WEIGHTS[experiment_id],
                "boundary_kernel": 3,
            },
            "optimizer": {"name": "Adam", "lr": 1e-4, "weight_decay": 0.0},
            "scheduler": {"name": "constant", "warmup_epochs": 0},
            "gradient_clip_val": 0.0,
        },
        artifacts={
            **base.artifacts,
            "run_name": f"{experiment_id}_seed{seed}",
            "weights_path": str(output_dir),
            "log_name": f"vegasdenseclosepbcl_v1/{family}/{experiment_id}/seed{seed}",
            "val_prediction_dir": str(output_dir / "val_predictions"),
            "guard_finite_training_loss": True,
        },
        provenance={
            **base.provenance,
            "experiment_id": experiment_id,
            "family": family,
            "training_protocol": "vegas_split_512_adam_40ep_v1",
            "model_order_scope": family,
        },
    )


__all__ = [
    "ADDITIONAL_EXPERIMENT_IDS",
    "MAIN_TABLE_EXPERIMENT_IDS",
    "OOM_SAFE_BATCH_PROFILES",
    "OUTPUT_ROOT",
    "VEGAS_EWC_EXPERIMENT_ID",
    "VEGAS_EWC_EXPERIMENT_IDS",
    "VEGAS_LOSS_EXPERIMENT_IDS",
    "VEGAS_STRUCTURAL_EXPERIMENT_IDS",
    "VEGAS_EXPERIMENT_IDS",
    "VEGAS_SEEDS",
    "VEGAS_SUPPORTED_EXPERIMENT_IDS",
    "build_vegasdenseclose_spec",
]
