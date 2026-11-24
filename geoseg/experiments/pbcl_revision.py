"""Matched R1 treatments on the spatially partitioned AquaDataset holdout."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

from geoseg.experiments.aquadataset_catalog import build_aquadataset_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_SPLIT_ROOT = REPO_ROOT / "splits" / "aquadataset_geographic_holdout_v1"


def resolve_data_root() -> Path:
    configured = (
        os.environ.get("AQUADATASET_ROOT")
        or os.environ.get("GEOSEG_REVISION_DATA_ROOT")
    )
    if configured:
        return Path(configured).expanduser().resolve()
    return (REPO_ROOT / "data" / "AquaDataset").resolve()


def resolve_output_root() -> Path:
    configured = (
        os.environ.get("GEOSEG_REVISION_OUTPUT_ROOT")
        or os.environ.get("PBCL_OUTPUT_ROOT")
    )
    if configured:
        return Path(configured).expanduser().resolve()
    return (REPO_ROOT / "outputs" / "aquadataset").resolve()


def resolve_comparison_root() -> Path:
    configured = os.environ.get("GEOSEG_REVISION_COMPARISON_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return resolve_output_root()


DATA_ROOT = resolve_data_root()
OUTPUT_ROOT = resolve_output_root()
COMPARISON_ROOT = resolve_comparison_root()

COMPARISON_TREATMENTS = ("ewc", "cldice", "cmalis", "bce_split", "boundary_split")
EXTERNAL_TREATMENTS = (
    "unetplusplus",
    "segformer_b2",
    "mask2former_swin_t",
    "dcswin_base",
    "unetformer",
    "mask2former_swin_t_instance",
)
SUPPLEMENTAL_EXTERNAL_TREATMENTS = (
    "a2fpn",
    "abcnet",
    "deeplabv3plus",
    "swinunet",
    "upernet_swin_t",
)
TRANSFER_ARCHITECTURES = (
    "unetplusplus",
    "deeplabv3plus",
    "segformer_b2",
    "unetformer",
    "dcswin_base",
    "mask2former_swin_t",
)
TRANSFER_TREATMENTS = {
    f"{architecture}_{variant}": (architecture, variant)
    for architecture in TRANSFER_ARCHITECTURES
    for variant in ("pbcl", "full")
}
TRANSFER_BOUNDARY_TREATMENTS = {
    f"{architecture}_boundary": (architecture, "boundary")
    for architecture in TRANSFER_ARCHITECTURES
}
VMAMBA_ARCHITECTURE = "vmamba_fpn"
VMAMBA_TREATMENTS = (
    VMAMBA_ARCHITECTURE,
    "vmamba_fpn_pbcl",
    "vmamba_fpn_boundary",
    "vmamba_fpn_full",
)
VMAMBA_MICROBATCHES = (6, 4, 3, 2, 1)
REFERENCE = "ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary"
TREATMENTS = (
    "base",
    "boundary",
    "full",
    "pbcl_only",
    "no_four_contact",
    "noncontact",
    "four_contact",
    "wprf_gt",
    "wprf_valid",
)


def _load_holdout_metadata(data_root: Path) -> tuple[dict, dict]:
    meta_path = data_root / "metadata.json"
    norm_path = data_root / "normalization.json"
    if not meta_path.is_file() or not norm_path.is_file():
        meta_path = BUNDLED_SPLIT_ROOT / "metadata.json"
        norm_path = BUNDLED_SPLIT_ROOT / "normalization.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    norm = json.loads(norm_path.read_text(encoding="utf-8"))
    if metadata["version"] != "aquav3-geographic-holdout-v1":
        raise ValueError("Revision requires the audited geographic holdout")
    return metadata, norm


def build_revision_spec(treatment: str, seed: int, *, source_config: str | Path | None = None):
    data_root = resolve_data_root()
    out_root = resolve_output_root()
    comp_root = resolve_comparison_root()
    vmamba_capacity = comp_root / "external/preflight/vmamba_capacity.json"

    comparison = treatment in COMPARISON_TREATMENTS
    vmamba_variants = {
        method: (VMAMBA_ARCHITECTURE, variant)
        for method, variant in zip(VMAMBA_TREATMENTS, ("base", "pbcl", "boundary", "full"), strict=True)
    }
    transfer = (TRANSFER_TREATMENTS | TRANSFER_BOUNDARY_TREATMENTS | vmamba_variants).get(treatment)
    external = transfer is not None or treatment in EXTERNAL_TREATMENTS + SUPPLEMENTAL_EXTERNAL_TREATMENTS
    if treatment not in TREATMENTS and not (comparison or external):
        raise ValueError(f"Unknown revision treatment: {treatment}")

    base_id = f"{REFERENCE}_boundary"
    if treatment in {"full", "no_four_contact", "noncontact", "four_contact"}:
        base_id = f"{REFERENCE}_head"
    if treatment in {"base", "pbcl_only", "bce_split"}:
        base_id = f"{REFERENCE}_{'pbcl' if treatment == 'pbcl_only' else 'base'}"
    if treatment.startswith("wprf_"):
        base_id = f"{REFERENCE}_boundary_wprf"
    if treatment in {"ewc", "cldice", "cmalis"}:
        base_id = f"{REFERENCE}_boundary_{treatment}"
    if transfer is not None:
        base_id, variant = transfer
    elif external:
        base_id = treatment

    spec = build_aquadataset_spec(base_id, seed, source_config=source_config)
    metadata, norm = _load_holdout_metadata(data_root)
    policy = treatment if treatment in {"no_four_contact", "noncontact", "four_contact"} else "all"
    output_root = comp_root / "external" if external else comp_root if comparison else out_root
    output = output_root / "train" / treatment / f"seed{seed}"
    loss = dict(spec.optimization["loss"])
    model = spec.model
    loader = {**spec.loader, "collate": "aquav3"}
    optimization = dict(spec.optimization)
    optimizer = spec.optimization["optimizer"]

    if treatment == "bce_split":
        loss = {
            "name": "SplitSegmentationLoss",
            "mask_bce_weight": 0.9,
            "split_weight": 0.1,
            "boundary_kernel": 3,
            "paper_equation": 15,
        }
    elif treatment == "boundary_split":
        loss = {
            **loss,
            "name": "SSepNetSplitLoss",
            "boundary_weight": 0.6,
            "boundary_kernel": 5,
            "split_weight": 0.1,
            "split_kernel": 3,
        }

    if transfer is not None:
        boundary = variant in {"boundary", "full"}
        loss.update(
            name="PBCLBinarySegmentationLoss",
            pbcl_weight=0.1 if variant in {"pbcl", "full"} else 0.0,
            pbcl_band_radius=3,
            pbcl_iterations=9,
            pbcl_threshold=0.5,
            pbcl_margin=0.1,
            pbcl_neighbors=4,
            boundary_weight=0.6 if boundary else 0.0,
            boundary_kernel=5,
        )
        model = {**model, "external_boundary_head": boundary}
        if boundary and base_id in {"unetplusplus", "deeplabv3plus"}:
            optimizer = {
                **optimizer,
                "encoder_prefixes": [
                    f"base_model.{prefix}" for prefix in optimizer["encoder_prefixes"]
                ],
            }
    if base_id == VMAMBA_ARCHITECTURE:
        batch = (
            int(json.loads(vmamba_capacity.read_text())["microbatch"])
            if vmamba_capacity.is_file() else VMAMBA_MICROBATCHES[0]
        )
        if batch not in VMAMBA_MICROBATCHES:
            raise ValueError(f"Unsupported VMamba microbatch: {batch}")
        loader.update(train_batch_size=batch, val_batch_size=batch, test_batch_size=batch)
        optimization["accumulate_grad_batches"] = 12 // batch
        model = {
            **model,
            "activation_checkpoint": True,
            "input_stem_adaptation": "rgb_to_bgr_plus_nir_rgb_mean",
            "pretrained_source": "VMamba-T V2 official ImageNet",
        }
    if treatment.startswith("wprf_"):
        loss["wprf_negative_support"] = "valid" if treatment == "wprf_valid" else "gt"
        loss["wprf_upstream"] = True
    ewc_data = (
        {"ewc_weight_root": str(comp_root / "cache" / "ewc"), "ewc_weight_splits": ["train", "val"]}
        if treatment == "ewc"
        else {}
    )
    checkpoint = spec.checkpoint
    if comparison or external:
        checkpoint = {
            **checkpoint,
            "monitors": {"overall": checkpoint["monitors"]["overall"]},
            "allow_epoch_state_checkpoint": False,
            "save_final_state": True,
        }
    return replace(
        spec,
        model=model,
        data={
            **spec.data,
            "root": str(data_root),
            "dataset_version": metadata["version"],
            "split_counts": metadata["split_counts"],
            "split_hashes": metadata["split_hashes"],
            "normalize_mean": norm["mean"],
            "normalize_std": norm["std"],
            "normalization_sha256": metadata["normalization_sha256"],
            "normalization_pixel_count_per_channel": norm["valid_pixels_per_band"],
            "use_manifest_valid_mask": True,
            "instance_pair_policy": policy,
            "instance_target_root": "instance_targets/pbcl_gap5",
            **ewc_data,
        },
        loader=loader,
        optimization={**optimization, "loss": loss, "optimizer": optimizer},
        checkpoint=checkpoint,
        artifacts={
            **spec.artifacts,
            "weights_path": str(output),
            "run_name": f"{treatment}_seed{seed}",
            "log_name": f"aquadataset/{treatment}/seed{seed}",
            "logger_version": 0,
            **({"guard_finite_training_loss": True} if external else {}),
            "val_prediction_dir": str(output / "val_predictions"),
        },
        provenance={
            **spec.provenance,
            "revision": "JSTARS-2026-05520",
            "revision_treatment": treatment,
            **({"dcswin_attention_numerics": "fp32_safe_l2_v1"} if base_id == "dcswin_base" else {}),
            **({
                "experiment_id": treatment,
                "base_experiment_id": base_id,
                "transfer_architecture": base_id,
                "transfer_variant": variant,
            } if transfer is not None else {}),
            **({"linear_attention_numerics": "fp32_safe_l2_v1"} if treatment in {"a2fpn", "abcnet"} else {}),
            "evaluation_scope": "geographic_holdout_within_existing_acquisitions",
        },
    )
