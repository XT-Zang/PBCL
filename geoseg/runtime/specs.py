from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from geoseg.datasets.paths import DATASETS_ROOT

_AQUAV3_DATA_ROOT = str(DATASETS_ROOT / "AquaV3_SPECIES_HNM_R1_4band")
_CAMPAIGN_OUT_ROOTS = {
    "baseline": "train_out/AquaV3_SPECIES_HNM_R1_4band/geoseg",
    "tuned": "train_out/AquaV3_SPECIES_HNM_R1_4band/geoseg_tuned",
    "compare": "train_out/AquaV3_SPECIES_HNM_R1_4band/main_table",
    "compare_full_structure": "train_out/AquaV3_SPECIES_HNM_R1_4band/compare_full_structure",
    "pbcl_crossarch": "train_out/AquaV3_SPECIES_HNM_R1_4band/pbcl_crossarch",
    "ssepnet": "train_out/AquaV3_SPECIES_HNM_R1_4band/ssepnet_model_port",
}

_COMPARE_RECIPES = {
    "unetplusplus": (1e-4, 5e-4, 12, 8, 16, "bf16-mixed", 3, 1.0, "imagenet/official", True),
    "deeplabv3plus": (1e-4, 5e-4, 12, 8, 16, "bf16-mixed", 3, 1.0, "imagenet/official", True),
    "swinunet": (6e-5, 5e-4, 12, 24, 24, "bf16-mixed", 5, 1.0, "imagenet/official", True),
    "segformer_b2": (6e-5, 5e-4, 12, 8, 16, "bf16-mixed", 5, 1.0, "imagenet/official", True),
    "upernet_swin_t": (6e-5, 5e-4, 12, 8, 16, "bf16-mixed", 5, 1.0, "imagenet/official", True),
    "mask2former_swin_t": (6e-5, 5e-4, 12, 8, 16, "bf16-mixed", 5, 0.5, "imagenet/official", True),
    "geoseg_a2fpn": (2e-4, 1e-4, 12, 8, 16, "bf16-mixed", 3, 1.0, "none", False),
    "geoseg_dcswin": (5e-5, 5e-4, 12, 8, 16, "bf16-mixed", 5, 0.5, "none", False),
    "unetformer": (5e-5, 5e-4, 12, 8, 16, "bf16-mixed", 5, 1.0, "swsl/official", True),
    "abcnet": (1e-4, 5e-4, 12, 8, 16, "bf16-mixed", 3, 1.0, "swsl/official", True),
}

_TUNED_RECIPES = {
    "a2fpn": (1e-4, 3, "bf16-mixed", 1.0),
    "abcnet": (1e-4, 3, "bf16-mixed", 1.0),
    "banet": (5e-5, 5, "32-true", 0.5),
    "manet": (1e-4, 3, "bf16-mixed", 1.0),
    "ftunetformer": (5e-5, 5, "bf16-mixed", 1.0),
    "pyramidmamba": (5e-5, 5, "bf16-mixed", 1.0),
    "unetformer": (5e-5, 5, "bf16-mixed", 1.0),
    "dcswin": (5e-5, 5, "32-true", 0.5),
}


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"ExperimentSpec contains non-serializable value {type(value)!r}")


@dataclass(frozen=True)
class ExperimentSpec:
    """Pure-data description of a segmentation experiment."""

    model: Mapping[str, Any]
    data: Mapping[str, Any]
    loader: Mapping[str, Any]
    optimization: Mapping[str, Any]
    runtime: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    metrics: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not str(self.model.get("key", "")).strip():
            raise ValueError("ExperimentSpec.model.key is required")
        if not str(self.data.get("dataset", "")).strip():
            raise ValueError("ExperimentSpec.data.dataset is required")
        if int(self.runtime.get("seed", -1)) < 0:
            raise ValueError("ExperimentSpec.runtime.seed must be non-negative")
        if int(self.runtime.get("max_epochs", 0)) <= 0:
            raise ValueError("ExperimentSpec.runtime.max_epochs must be positive")
        _plain(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def make_aquav3_spec(
    model_key: str,
    campaign: str,
    seed: int = 42,
    *,
    treatment: str | None = None,
    source_config: str | Path | None = None,
) -> ExperimentSpec:
    """Build a side-effect-free AquaV3 experiment description.

    Object construction remains a resolver concern, so calling this function never
    loads a dataset, pretrained checkpoint, or model implementation.
    """

    normalized_campaign = campaign.strip().lower().replace("-", "_")
    aliases = {
        "aquav3": "baseline",
        "aquav3_tuned": "tuned",
        "aquav3_compare": "compare",
        "full_structure": "compare_full_structure",
        "aquav3_compare_full_structure": "compare_full_structure",
        "aquav3_pbcl_crossarch": "pbcl_crossarch",
        "aquav3_ssepnet": "ssepnet",
    }
    normalized_campaign = aliases.get(normalized_campaign, normalized_campaign)
    if normalized_campaign not in _CAMPAIGN_OUT_ROOTS:
        raise ValueError(f"Unsupported AquaV3 campaign: {campaign}")

    key = model_key.strip().lower().replace("-", "_")
    max_epochs = 50
    lr, weight_decay = 2e-4, 1e-4
    train_batch_size, val_batch_size, test_batch_size = 12, 8, 8
    precision, warmup_epochs, gradient_clip_val = "bf16-mixed", 3, 1.0
    pretrained_source, pretrained = "none", False
    accumulate_grad_batches = 1
    seeded_loader_streams = normalized_campaign in {"compare", "pbcl_crossarch"}
    collate = "aquav3" if normalized_campaign == "pbcl_crossarch" else "default"

    ssepnet_variant = None
    if normalized_campaign == "ssepnet":
        from geoseg.models.ssepnet import get_ssepnet_variant

        ssepnet_variant = get_ssepnet_variant(key)
        max_epochs = 30
        pretrained_source, pretrained = "DINOv3 SAT-493M + HRNetV2-W32 ImageNet", True
        seeded_loader_streams = True
        collate = "aquav3"

    if normalized_campaign in {"compare", "compare_full_structure", "pbcl_crossarch"}:
        try:
            (
                lr,
                weight_decay,
                train_batch_size,
                val_batch_size,
                test_batch_size,
                precision,
                warmup_epochs,
                gradient_clip_val,
                pretrained_source,
                pretrained,
            ) = _COMPARE_RECIPES[key]
        except KeyError as exc:
            raise ValueError(f"Unsupported {normalized_campaign} model: {key}") from exc
    elif normalized_campaign == "tuned":
        try:
            lr, warmup_epochs, precision, gradient_clip_val = _TUNED_RECIPES[key]
        except KeyError as exc:
            raise ValueError(f"Unsupported tuned AquaV3 model: {key}") from exc
        weight_decay = 5e-4

    if normalized_campaign == "ssepnet":
        run_name = f"{key}_model_port"
    elif normalized_campaign == "baseline":
        run_name = f"geoseg_{key}_aquav3_4band_binary_e50"
    elif normalized_campaign == "tuned":
        run_name = f"geoseg_{key}_aquav3_4band_binary_tuned_e50"
    elif normalized_campaign == "compare":
        run_name = f"main_{key}_aquav3_4band_binary_e50_seed{int(seed)}"
    elif normalized_campaign == "compare_full_structure":
        run_name = f"full_structure_{key}_aquav3_4band_binary_e50_seed{int(seed)}"
        if key == "mask2former_swin_t":
            train_batch_size, accumulate_grad_batches, seeded_loader_streams = 6, 2, False
        elif key == "swinunet":
            val_batch_size, test_batch_size, seeded_loader_streams = 24, 24, False
    else:
        if treatment not in {"base", "pbcl"}:
            raise ValueError("PBCL cross-architecture specs require treatment='base' or treatment='pbcl'")
        run_name = f"{key}_{treatment}_seed{int(seed)}"
    output_dir = str(Path(_CAMPAIGN_OUT_ROOTS[normalized_campaign]) / run_name)
    loss_name = "BinarySegmentationLoss"
    if normalized_campaign in {"tuned", "compare_full_structure"}:
        loss_name = "StructureAwareBinarySegmentationLoss"
    elif normalized_campaign == "pbcl_crossarch":
        loss_name = "PBCLBinarySegmentationLoss"
    elif normalized_campaign == "ssepnet":
        loss_name = "SSepNetLoss"
    log_groups = {
        "baseline": "aquav3",
        "tuned": "aquav3_tuned",
        "compare": "aquav3_compare",
        "compare_full_structure": "aquav3_compare_full_structure",
        "pbcl_crossarch": "aquav3_pbcl_crossarch",
        "ssepnet": "aquav3_ssepnet",
    }
    loss_config: dict[str, Any] = {
        "name": loss_name,
        "main_bce": 1.0,
        "main_dice": 1.0,
        "aux_bce": 0.3,
        "aux_dice": 0.3,
    }
    if normalized_campaign == "tuned":
        loss_config.update(
            main_dice=1.2,
            boundary_bce=0.1,
            boundary_dice=0.2,
            boundary_kernel=5,
        )
    elif normalized_campaign == "compare_full_structure":
        loss_config.update(
            edge_bce=0.6,
            edge_dice=0.6,
            skeleton_bce=0.4,
            skeleton_dice=0.4,
            distance_smooth_l1=0.2,
            boundary_kernel=5,
            distance_iterations=32,
            distance_kernel=3,
            skeleton_kernel=3,
        )
    elif normalized_campaign == "pbcl_crossarch":
        loss_config.update(
            pbcl_weight=0.1 if treatment == "pbcl" else 0.0,
            pbcl_band_radius=3,
            pbcl_iterations=9,
            pbcl_threshold=0.5,
            pbcl_margin=0.1,
            pbcl_neighbors=4,
        )
    elif normalized_campaign == "ssepnet":
        assert ssepnet_variant is not None
        loss_config.update(
            boundary_weight=0.6 if ssepnet_variant.boundary_enabled else 0.0,
            boundary_kernel=5,
            pbcl_weight=ssepnet_variant.pbcl_weight,
            pbcl_band_radius=3,
            pbcl_iterations=9,
            pbcl_threshold=0.5,
            pbcl_margin=0.1,
            pbcl_neighbors=4,
        )

    return ExperimentSpec(
        model={
            "key": key,
            "in_channels": 4,
            "num_classes": 1,
            "pretrained": pretrained,
            "pretrained_source": pretrained_source,
            **(
                {
                    "branch_ablation": ssepnet_variant.branch_ablation,
                    "fusion_mode": ssepnet_variant.fusion_mode,
                    "boundary_enabled": ssepnet_variant.boundary_enabled,
                    "freeze_backbones": True,
                }
                if ssepnet_variant is not None
                else {}
            ),
        },
        data={
            "dataset": "aquav3",
            "root": _AQUAV3_DATA_ROOT,
            "splits": {"train": "train", "val": "val", "test": "test"},
            "bands": ["B", "G", "R", "NIR"],
            "band_indices": [0, 1, 2, 3],
            "image_size": 512,
            "instance_target_root": (
                "instance_targets/pbcl_gap5"
                if normalized_campaign in {"pbcl_crossarch", "ssepnet"}
                else None
            ),
        },
        loader={
            "train_batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
            "test_batch_size": test_batch_size,
            "train_workers": 4,
            "eval_workers": 2,
            "pin_memory": True,
            "drop_last": True,
            "balanced_sampler": True,
            "persistent_workers": False,
            "prefetch_factor": None,
            "collate": collate,
            "sampler_generator_stream": 1 if seeded_loader_streams else None,
            "loader_generator_streams": {"train": 2, "val": 3, "test": 4} if seeded_loader_streams else {},
        },
        optimization={
            "loss": loss_config,
            "optimizer": {"name": "AdamW", "lr": lr, "weight_decay": weight_decay},
            "scheduler": {"name": "warmup_cosine", "warmup_epochs": warmup_epochs, "min_lr_ratio": 0.02},
            "gradient_clip_val": gradient_clip_val,
            "accumulate_grad_batches": accumulate_grad_batches,
        },
        runtime={
            "seed": int(seed),
            "max_epochs": max_epochs,
            "precision": precision,
            "devices": "auto",
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": 0,
            "compile": False,
            "channels_last": False,
            "cudnn_benchmark": False,
        },
        checkpoint={
            "monitor": "val_fg_iou",
            "mode": "max",
            "save_top_k": 1,
            "every_n_epochs": 0 if normalized_campaign in {"compare", "pbcl_crossarch", "ssepnet"} else 5,
            "start_epoch": 10,
            "save_last": False,
            "declared_save_last": normalized_campaign in {"baseline", "tuned"},
            "selection": "earliest_max_val_fg_iou",
        },
        metrics={
            "binary": True,
            "threshold": 0.5,
            "deadhesion_validation": normalized_campaign in {"compare", "compare_full_structure", "pbcl_crossarch", "ssepnet"},
            "boundary_width": 2,
            "boundary_tolerance": 2,
            "min_overlap_pixels": 8,
            "min_overlap_ratio": 0.05,
        },
        artifacts={
            "run_name": run_name,
            "weights_path": output_dir,
            "log_name": f"{log_groups[normalized_campaign]}/{run_name}",
            "save_val_predictions": normalized_campaign != "pbcl_crossarch",
            "val_prediction_dir": f"{output_dir}/val_predictions",
            "async_writes": False,
        },
        provenance={
            "schema_version": 1,
            "campaign": normalized_campaign,
            "model_key": key,
            "treatment": treatment or "",
            "source_config": str(source_config) if source_config else "",
            **(
                {"scope": "model_inference_port", "complete_two_stage": False}
                if normalized_campaign == "ssepnet"
                else {}
            ),
        },
    )
