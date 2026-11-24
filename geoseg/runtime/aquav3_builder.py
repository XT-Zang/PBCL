from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
import os
from typing import Any

import torch

from geoseg.datasets.aquav3_datamodule import AquaV3DataModule
from geoseg.experiments.aquav3_protocol import get_experiment
from geoseg.experiments.reproducibility import seed_experiment
from geoseg.losses import BinarySegmentationLoss, StructureAwareBinarySegmentationLoss
from geoseg.losses.binary_seg import PBCLBinarySegmentationLoss
from geoseg.models.aquav3_external import build_compare_model
from geoseg.models.aquav3_registry import build_aquav3_model
from geoseg.models.aquav3_structure import StructureMultiHeadWrapper

from .experiment import ExperimentComponents
from .specs import ExperimentSpec


class LegacyConfigView:
    """Attribute-compatible view backed by a pure spec and a lazy DataModule."""

    _LAZY_DATA = {
        "train_loader": "train_dataloader",
        "val_loader": "val_dataloader",
        "test_loader": "test_dataloader",
        "train_dataset": "train_dataset",
        "val_dataset": "val_dataset",
        "test_dataset": "test_dataset",
    }

    def __init__(self, values: dict[str, Any], datamodule: Any) -> None:
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_datamodule", datamodule)

    def __getattr__(self, name: str) -> Any:
        if name in self._values:
            return self._values[name]
        if name in {"train_loader", "val_loader", "test_loader"}:
            return getattr(self._datamodule, self._LAZY_DATA[name])()
        if name in {"train_dataset", "val_dataset", "test_dataset"}:
            stage = {"train_dataset": "fit", "val_dataset": "validate", "test_dataset": "test"}[name]
            self._datamodule.setup(stage)
            return getattr(self._datamodule, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self._values[name] = value

    def __contains__(self, name: str) -> bool:
        return name in self._values or name in self._LAZY_DATA


def _epoch_state_checkpoint_interval(
    spec: ExperimentSpec,
    environment: Mapping[str, str] | None = None,
) -> int | None:
    """Resolve the opt-in recovery checkpoint interval for this experiment."""

    if not bool(spec.checkpoint.get("allow_epoch_state_checkpoint", True)):
        return None
    source = os.environ if environment is None else environment
    raw_interval = spec.checkpoint.get("epoch_state_every_n_epochs", source.get("GEOSEG_EPOCH_STATE_CHECKPOINT_EVERY"))
    if raw_interval is None:
        return None
    interval = int(raw_interval)
    if interval <= 0:
        raise ValueError(
            "GEOSEG_EPOCH_STATE_CHECKPOINT_EVERY must be positive"
        )
    return interval


def _warmup_cosine_lambda(epoch: int, max_epoch: int, warmup_epochs: int, min_lr_ratio: float) -> float:
    current_epoch = epoch + 1
    if warmup_epochs > 0 and current_epoch <= warmup_epochs:
        return current_epoch / warmup_epochs
    progress = (current_epoch - warmup_epochs) / max(1, max_epoch - warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _resolve_pbcl_batch_settings(spec: ExperimentSpec) -> ExperimentSpec:
    if spec.provenance["campaign"] != "pbcl_crossarch":
        return spec
    batch_size = int(os.environ.get("AQUAV3_BATCH_SIZE", str(spec.loader["train_batch_size"])))
    accumulate = int(os.environ.get("AQUAV3_ACCUMULATE_GRAD_BATCHES", "1"))
    if (batch_size, accumulate) not in {(12, 1), (6, 2)}:
        raise ValueError(
            "PBCL campaign permits batch/accum only as 12/1 or 6/2, "
            f"got {batch_size}/{accumulate}"
        )
    return replace(
        spec,
        loader={**spec.loader, "train_batch_size": batch_size},
        optimization={**spec.optimization, "accumulate_grad_batches": accumulate},
    )


def _build_model(spec: ExperimentSpec) -> torch.nn.Module:
    campaign = str(spec.provenance["campaign"])
    model_key = str(spec.model["key"])
    if campaign == "aquadataset":
        if model_key == "mask2former_swin_t_instance":
            from geoseg.models.mask2former_instance import (
                build_mask2former_swin_t_instance,
            )

            return build_mask2former_swin_t_instance(
                in_chans=int(spec.model["in_channels"]),
                num_classes=int(spec.model["num_classes"]),
                pretrained=bool(spec.model["pretrained"]),
                bgrnir_pretrained_stem=True,
                num_queries=int(spec.model.get("num_queries", 100)),
                query_seed=int(spec.runtime.get("seed", 0)),
            )
        if str(spec.provenance["family"]) == "context_hybrid":
            from geoseg.models.ssepnet import build_hrnet_context_variant

            return build_hrnet_context_variant(
                model_key,
                pretrained=bool(spec.model["pretrained"]),
                freeze_backbones=bool(spec.model["freeze_backbones"]),
                activation_checkpoint=bool(spec.model["activation_checkpoint"]),
            )
        if model_key.startswith("ssepnet"):
            from geoseg.models.ssepnet import build_ssepnet_variant

            return build_ssepnet_variant(
                model_key,
                in_chans=int(spec.model["in_channels"]),
                num_classes=int(spec.model["num_classes"]),
                pretrained=bool(spec.model["pretrained"]),
                freeze_backbones=bool(spec.model["freeze_backbones"]),
                activation_checkpoint=bool(spec.model["activation_checkpoint"]),
                dinov3_adaptation=str(spec.model.get("dinov3_adaptation", "full")),
                dinov3_lora_rank=int(spec.model.get("dinov3_lora_rank", 0)),
                dinov3_lora_alpha=spec.model.get("dinov3_lora_alpha"),
                physical_pruning=bool(spec.model.get("physical_pruning", False)),
            )
        model = build_compare_model(
            model_key,
            in_chans=int(spec.model["in_channels"]),
            num_classes=int(spec.model["num_classes"]),
            pretrained=bool(spec.model["pretrained"]),
            bgrnir_pretrained_stem=True,
        )
        if spec.model.get("external_boundary_head", False):
            from geoseg.models.external_boundary import add_external_boundary_head

            with torch.random.fork_rng(devices=[]):
                model = add_external_boundary_head(model, model_key)
        if affinity := spec.model.get("comparison_affinity"):
            with torch.random.fork_rng(devices=[]):
                model.add_comparison_affinity(affinity)
        return model
    if campaign == "ssepnet":
        from geoseg.models.ssepnet import build_ssepnet_variant

        return build_ssepnet_variant(
            model_key,
            in_chans=int(spec.model["in_channels"]),
            num_classes=int(spec.model["num_classes"]),
            pretrained=bool(spec.model["pretrained"]),
            freeze_backbones=bool(spec.model["freeze_backbones"]),
        )
    if campaign in {"compare", "compare_full_structure", "pbcl_crossarch"}:
        model = build_compare_model(
            model_key,
            in_chans=int(spec.model["in_channels"]),
            num_classes=int(spec.model["num_classes"]),
            pretrained=bool(spec.model["pretrained"]),
        )
        if campaign == "compare_full_structure":
            model = StructureMultiHeadWrapper(
                model,
                in_chans=int(spec.model["in_channels"]),
                num_classes=int(spec.model["num_classes"]),
                structure_channels=64,
                dropout=0.05,
            )
        return model
    return build_aquav3_model(
        model_key,
        in_chans=int(spec.model["in_channels"]),
        num_classes=int(spec.model["num_classes"]),
        pretrained=bool(spec.model["pretrained"]),
    )


def _build_loss(spec: ExperimentSpec) -> torch.nn.Module:
    campaign = str(spec.provenance["campaign"])
    if campaign == "aquadataset":
        loss_spec = dict(spec.optimization["loss"])
        if loss_spec["name"] == "SegFormerComparisonLoss":
            from geoseg.losses.segformer_comparison import SegFormerComparisonLoss

            return SegFormerComparisonLoss(loss_spec, int(spec.data["image_size"]))
        if loss_spec["name"] == "SplitSegmentationLoss":
            from geoseg.losses.split import SplitSegmentationLoss

            return SplitSegmentationLoss(
                image_size=int(spec.data["image_size"]),
                split_weight=float(loss_spec["split_weight"]),
                boundary_kernel=int(loss_spec["boundary_kernel"]),
            )
        if loss_spec["name"] == "SSepNetSplitLoss":
            from geoseg.losses.ssepnet import SSepNetSplitLoss

            return SSepNetSplitLoss(
                image_size=int(spec.data["image_size"]),
                boundary_weight=float(loss_spec.get("boundary_weight", 0.6)),
                boundary_kernel=int(loss_spec.get("boundary_kernel", 5)),
                split_weight=float(loss_spec.get("split_weight", 0.1)),
                split_kernel=int(loss_spec.get("split_kernel", 3)),
            )
        if loss_spec["name"] == "SSepNetEWCLoss":
            from geoseg.losses.ssepnet import SSepNetEWCLoss

            return SSepNetEWCLoss(
                boundary_weight=float(loss_spec["boundary_weight"]),
                boundary_kernel=int(loss_spec["boundary_kernel"]),
                ewc_w0=float(loss_spec["ewc_w0"]),
                ewc_sigma=float(loss_spec["ewc_sigma"]),
                ewc_power=float(loss_spec["ewc_power"]),
            )
        if loss_spec["name"] == "SSepNetClDiceLoss":
            from geoseg.losses.ssepnet import SSepNetClDiceLoss

            return SSepNetClDiceLoss(
                boundary_weight=float(loss_spec["boundary_weight"]),
                boundary_kernel=int(loss_spec["boundary_kernel"]),
                cldice_alpha=float(loss_spec["cldice_alpha"]),
                cldice_iterations=int(loss_spec["cldice_iterations"]),
                cldice_smooth=float(loss_spec["cldice_smooth"]),
            )
        if loss_spec["name"] == "SSepNetConstrainedMALISLoss":
            from geoseg.losses.ssepnet import SSepNetConstrainedMALISLoss

            return SSepNetConstrainedMALISLoss(
                boundary_weight=float(loss_spec["boundary_weight"]),
                boundary_kernel=int(loss_spec["boundary_kernel"]),
                malis_weight=float(loss_spec["malis_weight"]),
                malis_cpu_workers=int(loss_spec.get("malis_cpu_workers", 6)),
            )
        if loss_spec["name"] == "SSepNetWPRFLoss":
            from geoseg.losses.ssepnet import SSepNetWPRFLoss

            return SSepNetWPRFLoss(
                boundary_weight=float(loss_spec["boundary_weight"]),
                boundary_kernel=int(loss_spec["boundary_kernel"]),
                wprf_edge_weight=float(loss_spec["wprf_edge_weight"]),
                wprf_reach_weight=float(loss_spec["wprf_reach_weight"]),
                wprf_warmup_epochs=int(loss_spec["wprf_warmup_epochs"]),
                wprf_source_batch_size=int(loss_spec["wprf_source_batch_size"]),
                wprf_activation_checkpoint=bool(
                    loss_spec["wprf_activation_checkpoint"]
                ),
                wprf_negative_support=str(loss_spec.get("wprf_negative_support", "gt")),
                wprf_upstream=bool(loss_spec.get("wprf_upstream", False)),
            )
        if loss_spec["name"] == "SSepNetLoss":
            from geoseg.losses.ssepnet import SSepNetLoss

            return SSepNetLoss(
                boundary_weight=float(loss_spec["boundary_weight"]),
                pbcl_weight=float(loss_spec["pbcl_weight"]),
                boundary_kernel=int(loss_spec["boundary_kernel"]),
                pbcl_band_radius=int(loss_spec["pbcl_band_radius"]),
                pbcl_iterations=int(loss_spec["pbcl_iterations"]),
                pbcl_threshold=float(loss_spec["pbcl_threshold"]),
                pbcl_margin=float(loss_spec["pbcl_margin"]),
                pbcl_neighbors=int(loss_spec["pbcl_neighbors"]),
                pbcl_max_pairs_per_tile=(
                    int(loss_spec["pbcl_max_pairs_per_tile"])
                    if loss_spec.get("pbcl_max_pairs_per_tile") is not None
                    else None
                ),
                pbcl_pair_sampling_salt=str(
                    loss_spec.get("pbcl_pair_sampling_salt", "pbcl-pair-sampling-v1")
                ),
            )
        if loss_spec["name"] == "DeformableMask2FormerLoss":
            from geoseg.losses.mask2former import DeformableMask2FormerLoss

            return DeformableMask2FormerLoss(
                num_queries=int(loss_spec["num_queries"]),
                class_weight=float(loss_spec["class_weight"]),
                mask_weight=float(loss_spec["mask_weight"]),
                dice_weight=float(loss_spec["dice_weight"]),
                no_object_weight=float(loss_spec["no_object_weight"]),
                auxiliary_weight=float(loss_spec["auxiliary_weight"]),
                boundary_weight=float(loss_spec["boundary_weight"]),
                pbcl_weight=float(loss_spec["pbcl_weight"]),
                boundary_kernel=int(loss_spec["boundary_kernel"]),
                pbcl_band_radius=int(loss_spec["pbcl_band_radius"]),
                pbcl_iterations=int(loss_spec["pbcl_iterations"]),
                pbcl_threshold=float(loss_spec["pbcl_threshold"]),
                pbcl_margin=float(loss_spec["pbcl_margin"]),
                pbcl_neighbors=int(loss_spec["pbcl_neighbors"]),
            )
        if loss_spec["name"] == "Mask2FormerInstanceLoss":
            from geoseg.losses.mask2former_instance import Mask2FormerInstanceLoss

            return Mask2FormerInstanceLoss(
                num_queries=int(loss_spec["num_queries"]),
            )
        if loss_spec["name"] == "PBCLBinarySegmentationLoss":
            return PBCLBinarySegmentationLoss(
                main_bce=float(loss_spec["main_bce"]),
                main_dice=float(loss_spec["main_dice"]),
                aux_bce=float(loss_spec["aux_bce"]),
                aux_dice=float(loss_spec["aux_dice"]),
                boundary_weight=float(loss_spec.get("boundary_weight", 0.0)),
                boundary_kernel=int(loss_spec.get("boundary_kernel", 5)),
                pbcl_weight=float(loss_spec["pbcl_weight"]),
                pbcl_band_radius=int(loss_spec["pbcl_band_radius"]),
                pbcl_iterations=int(loss_spec["pbcl_iterations"]),
                pbcl_threshold=float(loss_spec["pbcl_threshold"]),
                pbcl_margin=float(loss_spec["pbcl_margin"]),
                pbcl_neighbors=int(loss_spec["pbcl_neighbors"]),
                pbcl_max_pairs_per_tile=(
                    int(loss_spec["pbcl_max_pairs_per_tile"])
                    if loss_spec.get("pbcl_max_pairs_per_tile") is not None
                    else None
                ),
                pbcl_pair_sampling_salt=str(
                    loss_spec.get("pbcl_pair_sampling_salt", "pbcl-pair-sampling-v1")
                ),
            )
        return BinarySegmentationLoss(main_bce=1.0, main_dice=1.0, aux_bce=0.3, aux_dice=0.3)
    if campaign == "ssepnet":
        from geoseg.models.ssepnet import build_ssepnet_loss

        return build_ssepnet_loss(str(spec.model["key"]))
    if campaign == "tuned":
        return StructureAwareBinarySegmentationLoss(
            main_bce=1.0,
            main_dice=1.2,
            aux_bce=0.3,
            aux_dice=0.3,
            boundary_bce=0.1,
            boundary_dice=0.2,
            boundary_kernel=5,
        )
    if campaign == "compare_full_structure":
        return StructureAwareBinarySegmentationLoss(
            main_bce=1.0,
            main_dice=1.0,
            aux_bce=0.3,
            aux_dice=0.3,
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
    if campaign == "pbcl_crossarch":
        return PBCLBinarySegmentationLoss(
            main_bce=1.0,
            main_dice=1.0,
            aux_bce=0.3,
            aux_dice=0.3,
            pbcl_weight=float(spec.optimization["loss"]["pbcl_weight"]),
            pbcl_band_radius=3,
            pbcl_iterations=9,
            pbcl_threshold=0.5,
            pbcl_margin=0.1,
            pbcl_neighbors=4,
        )
    return BinarySegmentationLoss(main_bce=1.0, main_dice=1.0, aux_bce=0.3, aux_dice=0.3)


def _result_metadata(spec: ExperimentSpec) -> dict[str, Any] | None:
    campaign = str(spec.provenance["campaign"])
    if campaign == "aquadataset":
        if spec.data.get("dataset") == "isaid_transfer":
            family = str(spec.provenance.get("family", "isaid_transfer"))
            return {
                "model": str(spec.provenance["experiment_id"]),
                "family": family,
                "dataset": "iSAID/DOTA single-class",
                "class_name": str(spec.data["class_name"]),
                "treatment": str(spec.provenance.get("treatment", "")),
                "training_protocol": str(
                    spec.provenance.get(
                        "training_protocol", "isaid_rgb_singleclass_40ep_v1"
                    )
                ),
            }
        dataset_name = (
            "VegasDenseClosePBCL_v1"
            if spec.data.get("dataset") == "vegasdenseclosepbcl_v1"
            else "AquaDateset"
        )
        return {
            "model": str(spec.provenance["experiment_id"]),
            "family": str(spec.provenance["family"]),
            "dataset": dataset_name,
            "training_protocol": str(
                spec.provenance.get("training_protocol", "uniform_40ep_v1")
            ),
        }
    if campaign == "ssepnet":
        return {
            "model": "SSepNet",
            "variant": str(spec.model["key"]),
            "backbone": "DINOv3 ViT-L/16 SAT-493M + HRNetV2-W32",
            "scope": "model_inference_port",
        }
    if campaign not in {"compare", "compare_full_structure", "pbcl_crossarch"}:
        return None
    metadata = get_experiment(str(spec.model["key"])).metadata()
    if campaign == "compare_full_structure":
        metadata = dict(metadata)
        metadata["model"] = f"{metadata['model']} + Full-Structure"
    return metadata


def _legacy_training_protocol(
    spec: ExperimentSpec,
    result_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    campaign = str(spec.provenance["campaign"])
    model_key = str(spec.model["key"])
    seed = int(spec.runtime["seed"])
    if campaign == "aquadataset":
        if spec.data.get("dataset") == "isaid_transfer":
            family = str(spec.provenance.get("family", "isaid_transfer"))
            return {
                "schema_version": 1,
                "campaign": family,
                "family": family,
                "experiment_id": str(spec.provenance["experiment_id"]),
                "paired_experiment_id": str(
                    spec.provenance.get("paired_experiment_id", "")
                ),
                "class_name": str(spec.data["class_name"]),
                "treatment": str(spec.provenance.get("treatment", "")),
                "run_name": str(spec.artifacts["run_name"]),
                "seed": seed,
                "epochs": int(spec.runtime["max_epochs"]),
                "data": dict(spec.data),
                "model": dict(spec.model),
                "loss": dict(spec.optimization["loss"]),
                "optimizer": dict(spec.optimization["optimizer"]),
                "scheduler": dict(spec.optimization["scheduler"]),
                "checkpoint_selection": str(spec.checkpoint["policy"]),
                "checkpoint_policy": dict(spec.checkpoint),
                "test_threshold": 0.5,
                "test_used_for_selection": False,
            }
        return {
            "schema_version": 1,
            "campaign": "aquadataset",
            "family": str(spec.provenance["family"]),
            "experiment_id": str(spec.provenance["experiment_id"]),
            "training_protocol": str(
                spec.provenance.get("training_protocol", "uniform_40ep_v1")
            ),
            "run_name": str(spec.artifacts["run_name"]),
            "seed": seed,
            "epochs": int(spec.runtime["max_epochs"]),
            "data": dict(spec.data),
            "model": dict(spec.model),
            "loss": dict(spec.optimization["loss"]),
            "optimizer": dict(spec.optimization["optimizer"]),
            "scheduler": dict(spec.optimization["scheduler"]),
            "checkpoint_selection": str(spec.checkpoint["policy"]),
            "checkpoint_policy": dict(spec.checkpoint),
            "test_threshold": 0.5,
        }
    if campaign == "ssepnet":
        return {
            "schema_version": 1,
            "campaign": "ssepnet_model_inference_port",
            "run_name": str(spec.artifacts["run_name"]),
            "variant": model_key,
            "seed": seed,
            "complete_two_stage": False,
            "warning": (
                "This configuration exposes the paper model and downstream loss interface only; "
                "it does not reproduce the 40+30 epoch two-stage protocol."
            ),
            "model": dict(spec.model),
            "loss": dict(spec.optimization["loss"]),
            "checkpoint_selection": "earliest_max_val_fg_iou",
            "test_threshold": 0.5,
        }
    if campaign in {"baseline", "tuned"}:
        return None
    if campaign in {"compare", "compare_full_structure"}:
        protocol: dict[str, Any] = {
            "schema_version": 1,
            "campaign": "main_table_multiseed",
            "run_name": f"main_{model_key}_aquav3_4band_binary_e50_seed{seed}",
            "architecture": model_key,
            "seed": seed,
            "seed_applied_before_construction": True,
            "image_size": 512,
            "input_bands": ["B", "G", "R", "NIR"],
            "dataset_splits": {"train": "train", "val": "val", "test": "test"},
            "epochs": 50,
            "loss": {
                "name": "BinarySegmentationLoss",
                "main": {"bce": 1.0, "dice": 1.0},
                "aux_each": {"bce": 0.3, "dice": 0.3},
                "pbcl": False,
                "structure_aware": False,
            },
            "checkpoint_selection": "earliest_max_val_fg_iou",
            "checkpoint_policy": {
                "monitor": "val_fg_iou",
                "monitor_mode": "max",
                "save_top_k": 1,
                "checkpoint_every_n_epochs": int(spec.checkpoint["every_n_epochs"]),
                "checkpoint_start_epoch": int(spec.checkpoint.get("start_epoch", 10)),
                "save_last": False,
                "resume": False,
            },
            "test_threshold": 0.5,
        }
        if model_key in {"unetformer", "abcnet"}:
            from geoseg.models.aquav3_resnet18 import SWSL_RESNET18_NAME, SWSL_RESNET18_SOURCE

            protocol.update(
                {
                    "backbone": SWSL_RESNET18_NAME,
                    "pretrain_source": SWSL_RESNET18_SOURCE,
                    "input_stem_adaptation": (
                        "RGB weights reordered to BGR; NIR initialized from the RGB mean; "
                        "all four input kernels scaled by 3/4"
                    ),
                }
            )
        return protocol

    treatment = str(spec.provenance["treatment"])
    batch_size = int(spec.loader["train_batch_size"])
    accumulate = int(spec.optimization["accumulate_grad_batches"])
    run_name = str(spec.artifacts["run_name"])
    return {
        "schema_version": 1,
        "campaign": "pbcl_crossarch",
        "run_name": run_name,
        "architecture": model_key,
        "treatment": treatment,
        "seed": seed,
        "model": {
            "key": model_key,
            "pretrained": bool(spec.model["pretrained"]),
            "result_metadata": result_metadata,
        },
        "data": {
            "root": str(spec.data["root"]),
            "instance_target_root": str(spec.data["instance_target_root"]),
            "image_size": int(spec.data["image_size"]),
            "band_indices": list(spec.data["band_indices"]),
            "splits": dict(spec.data["splits"]),
            "train_augment": True,
            "sampler": "balanced_50_50",
            "generator_streams": {"sampler": 1, "train": 2, "val": 3, "test": 4},
        },
        "training": {
            "epochs": int(spec.runtime["max_epochs"]),
            "train_batch_size": batch_size,
            "val_batch_size": int(spec.loader["val_batch_size"]),
            "test_batch_size": int(spec.loader["test_batch_size"]),
            "accumulate_grad_batches": accumulate,
            "effective_train_batch_size": batch_size * accumulate,
            "precision": str(spec.runtime["precision"]),
            "gradient_clip_val": float(spec.optimization["gradient_clip_val"]),
            "warmup_epochs": int(spec.optimization["scheduler"]["warmup_epochs"]),
            "metric_threshold": float(spec.metrics["threshold"]),
        },
        "optimizer": {
            "name": "AdamW",
            "lr": float(spec.optimization["optimizer"]["lr"]),
            "weight_decay": float(spec.optimization["optimizer"]["weight_decay"]),
        },
        "loss": {
            "name": "PBCLBinarySegmentationLoss",
            "main_bce": 1.0,
            "main_dice": 1.0,
            "aux_bce": 0.3,
            "aux_dice": 0.3,
            "pbcl": {
                "weight": float(spec.optimization["loss"]["pbcl_weight"]),
                "band_radius": 3,
                "iterations": 9,
                "neighbors": 4,
                "threshold": 0.5,
                "margin": 0.1,
                "cutoff": 0.4,
            },
        },
        "checkpoint_policy": {
            "monitor": "val_fg_iou",
            "monitor_mode": "max",
            "save_top_k": 1,
            "checkpoint_every_n_epochs": 0,
            "checkpoint_start_epoch": int(spec.checkpoint.get("start_epoch", 10)),
            "save_last": False,
            "resume": False,
        },
        "output": {
            "weights_name": run_name,
            "weights_path": str(spec.artifacts["weights_path"]),
            "save_val_predictions": False,
        },
        "oom": {
            "fallback_active": batch_size == 6,
            "policy": (
                "No automatic retry. If PBCL OOMs, restart both paired treatments "
                "for that architecture with batch 6 and accumulate_grad_batches 2."
            ),
        },
    }


def build_aquav3_components(spec: ExperimentSpec) -> tuple[ExperimentSpec, ExperimentComponents]:
    """Construct AquaV3 runtime objects while keeping datasets and loaders lazy."""

    spec = _resolve_pbcl_batch_settings(spec)
    campaign = str(spec.provenance["campaign"])
    if campaign in {"compare", "compare_full_structure", "pbcl_crossarch", "ssepnet", "aquadataset"}:
        seed_experiment(int(spec.runtime["seed"]))

    model = _build_model(spec)
    loss = _build_loss(spec)
    optimizer_spec = spec.optimization["optimizer"]
    optimizer_parameter_names: dict[str, tuple[str, ...]] = {}
    if optimizer_spec.get("name") == "Adam":
        resolved_lr = float(optimizer_spec["lr"])
        optimizer = torch.optim.Adam(
            model.parameters(), lr=resolved_lr, weight_decay=float(optimizer_spec["weight_decay"])
        )
    elif campaign == "aquadataset":
        from .optimization import build_uniform_adamw

        optimizer_build = build_uniform_adamw(model, optimizer_spec)
        optimizer = optimizer_build.optimizer
        optimizer_parameter_names = dict(optimizer_build.parameter_names)
        resolved_lr = float(optimizer_spec["base_lr"])
    else:
        optimizer_parameters = (
            [parameter for parameter in model.parameters() if parameter.requires_grad]
            if campaign == "ssepnet"
            else model.parameters()
        )
        optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=float(optimizer_spec["lr"]),
            weight_decay=float(optimizer_spec["weight_decay"]),
        )
        resolved_lr = float(optimizer_spec["lr"])
    scheduler_spec = spec.optimization["scheduler"]
    if scheduler_spec.get("name") == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: _warmup_cosine_lambda(
                epoch,
                max_epoch=int(
                    scheduler_spec.get("schedule_epochs", spec.runtime["max_epochs"])
                ),
                warmup_epochs=int(scheduler_spec["warmup_epochs"]),
                min_lr_ratio=float(scheduler_spec["min_lr_ratio"]),
            ),
        )
    semantic_revision_baseline = (
        "revision_treatment" in spec.provenance
        and isinstance(loss, BinarySegmentationLoss)
        and not loss.requires_instance_supervision
        and not spec.metrics.get("instance_segmentation", False)
    )
    datamodule = AquaV3DataModule(spec, load_fit_instance_supervision=not semantic_revision_baseline)
    result_metadata = _result_metadata(spec)
    training_protocol = _legacy_training_protocol(spec, result_metadata)
    if campaign == "aquadataset" and training_protocol is not None:
        training_protocol["optimizer_parameter_names"] = optimizer_parameter_names
        training_protocol["pretrained_artifacts"] = dict(spec.model["pretrained_artifacts"])
    model_name = str(spec.model["key"])
    if campaign == "compare_full_structure":
        model_name = f"full_structure_{model_name}"
    values: dict[str, Any] = {
        "max_epoch": int(spec.runtime["max_epochs"]),
        "early_stopping_patience": 0,
        "ignore_index": 255,
        "train_batch_size": int(spec.loader["train_batch_size"]),
        "val_batch_size": int(spec.loader["val_batch_size"]),
        "test_batch_size": int(spec.loader["test_batch_size"]),
        "accumulate_grad_batches": int(spec.optimization["accumulate_grad_batches"]),
        "effective_train_batch_size": int(spec.loader["train_batch_size"])
        * int(spec.optimization["accumulate_grad_batches"]),
        "lr": resolved_lr,
        "weight_decay": float(optimizer_spec["weight_decay"]),
        "warmup_epochs": int(scheduler_spec["warmup_epochs"]),
        "num_classes": int(spec.model["num_classes"]),
        "classes": ("background", "foreground"),
        "weights_name": str(spec.artifacts["run_name"]),
        "weights_path": str(spec.artifacts["weights_path"]),
        "test_weights_name": str(spec.artifacts["run_name"]),
        "log_name": str(spec.artifacts["log_name"]),
        "monitor": str(spec.checkpoint["monitor"]),
        "monitor_mode": str(spec.checkpoint["mode"]),
        "save_top_k": int(spec.checkpoint["save_top_k"]),
        "save_last": bool(spec.checkpoint.get("declared_save_last", spec.checkpoint["save_last"])),
        "check_val_every_n_epoch": int(spec.runtime["check_val_every_n_epoch"]),
        "checkpoint_every_n_epochs": int(spec.checkpoint["every_n_epochs"]),
        "checkpoint_start_epoch": int(spec.checkpoint.get("start_epoch", 10)),
        "pretrained_ckpt_path": None,
        "resume_ckpt_path": spec.runtime.get("resume_checkpoint"),
        "gpus": spec.runtime["devices"],
        "precision": str(spec.runtime["precision"]),
        "gradient_clip_val": float(spec.optimization["gradient_clip_val"]),
        "num_sanity_val_steps": int(spec.runtime["num_sanity_val_steps"]),
        "binary_segmentation": bool(spec.metrics["binary"]),
        "instance_segmentation": bool(
            spec.metrics.get("instance_segmentation", False)
        ),
        "metric_threshold": float(spec.metrics["threshold"]),
        "instance_score_threshold": float(
            spec.metrics.get("instance_score_threshold", 0.5)
        ),
        "instance_mask_threshold": float(
            spec.metrics.get("instance_mask_threshold", 0.5)
        ),
        "instance_ap_score_threshold": spec.metrics.get(
            "instance_ap_score_threshold"
        ),
        "instance_num_queries": int(
            spec.metrics.get("instance_num_queries", 100)
        ),
        "prediction_kind": str(
            spec.metrics.get("prediction_kind", "connected_components")
        ),
        "save_val_predictions": bool(spec.artifacts["save_val_predictions"]),
        "val_prediction_dir": str(spec.artifacts["val_prediction_dir"]),
        "async_writes": bool(spec.artifacts["async_writes"]),
        "deadhesion_validation": bool(spec.metrics["deadhesion_validation"]),
        "deadhesion_split": "val",
        "deadhesion_instances_coco": None,
        "deadhesion_boundary_width": int(spec.metrics["boundary_width"]),
        "deadhesion_boundary_tolerance": int(spec.metrics["boundary_tolerance"]),
        "deadhesion_min_overlap_pixels": int(spec.metrics["min_overlap_pixels"]),
        "deadhesion_min_overlap_ratio": float(spec.metrics["min_overlap_ratio"]),
        "data_root": str(spec.data["root"]),
        "seed": int(spec.runtime["seed"]),
        "model_name": model_name,
        "net": model,
        "loss": loss,
        "use_aux_loss": False,
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "pretrain_policy": str(spec.model.get("pretrained_source", "none")),
        "channels_last": bool(spec.runtime.get("channels_last", False)),
    }
    if "logger_version" in spec.artifacts:
        values["logger_version"] = int(spec.artifacts["logger_version"])
    if spec.checkpoint.get("plain_checkpoint_names"):
        values["plain_checkpoint_names"] = True
    epoch_state_interval = _epoch_state_checkpoint_interval(spec)
    if spec.checkpoint.get("save_final_state", False):
        values["save_final_state"] = True
    if spec.artifacts.get("guard_finite_training_loss", False):
        values["guard_finite_training_loss"] = True
    if epoch_state_interval is not None:
        values["epoch_state_checkpoint"] = True
        values["epoch_state_checkpoint_start_epoch"] = 1
        values["epoch_state_checkpoint_every_n_epochs"] = epoch_state_interval
    distillation = spec.model.get("distillation")
    if isinstance(distillation, Mapping) and bool(distillation.get("enabled", False)):
        teacher_key = str(distillation["teacher"])
        teacher_artifact = dict(spec.model["pretrained_artifacts"])[teacher_key]
        values["distillation_spec"] = {
            **dict(distillation),
            "teacher_taps": list(spec.model["teacher_taps"]),
            "feature_loss_weights": list(spec.model["feature_loss_weights"]),
        }
        values["distillation_teacher_checkpoint"] = str(teacher_artifact["path"])
        values["logger_version"] = int(spec.artifacts.get("logger_version", 0))
        values["round2_pilot_epochs"] = 8
    if spec.data.get("dataset") == "isaid_transfer":
        values["isaid_transfer_class_name"] = str(spec.data["class_name"])
    if result_metadata is not None:
        values["result_metadata"] = result_metadata
    if training_protocol is not None:
        values["training_protocol"] = training_protocol
    if campaign == "pbcl_crossarch":
        values["run_name"] = str(spec.artifacts["run_name"])
        values["treatment"] = str(spec.provenance["treatment"])
        values["pbcl_weight"] = float(spec.optimization["loss"]["pbcl_weight"])
    if campaign == "compare_full_structure":
        values["structure_variant"] = "full_structure_multi_head"
    if campaign == "ssepnet":
        values["training_scope"] = "model_inference_port"
        values["complete_two_stage"] = False
    if campaign == "aquadataset":
        values["optimizer_parameter_names"] = optimizer_parameter_names
        values["training_scope"] = (
            str(
                spec.provenance.get(
                    "training_protocol", "isaid_rgb_singleclass_40ep_v1"
                )
            )
            if spec.data.get("dataset") == "isaid_transfer"
            else str(spec.provenance.get("training_protocol", "uniform_40ep_v1"))
        )
        values["checkpoint_monitors"] = dict(spec.checkpoint["monitors"])
        values["checkpoint_primary_role"] = str(spec.checkpoint["primary_role"])
        values["checkpoint_save_weights_only"] = bool(spec.checkpoint["save_weights_only"])
        values["overall_score_weights"] = dict(spec.checkpoint["overall_score"]["weights"])

    config = LegacyConfigView(values, datamodule)
    spec = replace(
        spec,
        model={
            **spec.model,
            "class_name": type(model).__name__,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        },
    )
    components = ExperimentComponents(
        config=config,
        model=model,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        datamodule=datamodule,
    )
    return spec, components
