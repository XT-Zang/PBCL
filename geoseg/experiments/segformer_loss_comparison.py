"""Fixed SegFormer-B2 loss comparisons on AquaDataset and SpaceNet 2 Vegas."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

from geoseg.datasets.aquadataset_wprf import WPRF_SOURCE_COMMIT


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_output_root() -> Path:
    configured = (
        os.environ.get("GEOSEG_SEGFORMER_LOSS_ROOT")
        or os.environ.get("PBCL_OUTPUT_ROOT")
    )
    if configured:
        return Path(configured).expanduser().resolve()
    return (REPO_ROOT / "outputs" / "segformer_loss_comparison").resolve()


OUTPUT = resolve_output_root()
METHODS = ("split", "split_paper", "wprf_gt", "wprf_valid", "ewc", "cldice", "cmalis")
DATASETS = ("aqua", "vegas")
SEEDS = (42, 43, 44)


def build_spec(name, seed, *, source_config):
    dataset, method = name.split("__", 1)
    if dataset not in DATASETS or method not in METHODS:
        raise ValueError(f"Unknown comparison: {name}")

    output_root = resolve_output_root()
    if method == "split_paper":
        if dataset == "aqua":
            from geoseg.experiments.pbcl_revision import build_revision_spec
            base = build_revision_spec("segformer_b2", seed, source_config=source_config)
        else:
            from geoseg.experiments.vegasdenseclose_catalog import build_vegasdenseclose_spec
            base = build_vegasdenseclose_spec("segformer_b2__base", seed, source_config=source_config)
        directory = output_root / "train" / dataset / method / f"seed{seed}"
        loss = {
            "name": "SplitSegmentationLoss",
            "mask_bce_weight": 0.9,
            "split_weight": 0.1,
            "boundary_kernel": 3,
            "paper_equation": 15,
        }
        return replace(
            base,
            optimization={**base.optimization, "loss": loss},
            checkpoint={
                **base.checkpoint,
                "save_last": True,
                "declared_save_last": True,
                "allow_epoch_state_checkpoint": True,
                "epoch_state_every_n_epochs": 1,
                "save_final_state": True,
                "plain_checkpoint_names": True,
            },
            artifacts={
                **base.artifacts,
                "weights_path": str(directory),
                "run_name": f"{name}_seed{seed}",
                "log_name": f"{name}/seed{seed}",
                "val_prediction_dir": str(directory / "val_predictions"),
            },
            provenance={
                **base.provenance,
                "experiment_id": name,
                "family": "segformer_loss_comparison",
                "comparison_method": method,
            },
        )

    if dataset == "aqua":
        from geoseg.experiments.pbcl_revision import build_revision_spec
        base = build_revision_spec("segformer_b2_boundary", seed, source_config=source_config)
    else:
        from geoseg.experiments.vegasdenseclose_catalog import build_vegasdenseclose_spec
        base = build_vegasdenseclose_spec("segformer_b2__boundary", seed, source_config=source_config)

    directory = output_root / "train" / dataset / method / f"seed{seed}"
    loss = {**base.optimization["loss"], "name": "SegFormerComparisonLoss", "method": method}
    data = dict(base.data)
    model = dict(base.model)
    if method == "ewc":
        loss.update(
            ewc_w0=10.0,
            ewc_sigma=7.5,
            ewc_power=2.0,
            ewc_class_weight="per_tile_inverse_frequency",
            ewc_single_instance="d1_only",
        )
        data.update(
            ewc_weight_root=str(output_root / "cache" / dataset / "ewc"),
            ewc_weight_splits=["train", "val"] if dataset == "aqua" else ["train", "val", "test"],
        )
    elif method == "cldice":
        loss.update(cldice_alpha=0.5, cldice_iterations=10, cldice_smooth=1.0)
    elif method == "split":
        loss.update(split_weight=0.1, split_kernel=3)
    elif method == "cmalis":
        loss.update(malis_weight=1.0, malis_cpu_workers=12)
        model.update(comparison_affinity="cmalis")
    else:
        loss.update(
            source_commit=WPRF_SOURCE_COMMIT,
            wprf_upstream=True,
            wprf_negative_support=method.removeprefix("wprf_"),
            wprf_warmup_epochs=4,
            wprf_edge_weight=1.0,
            wprf_reach_weight=1.0,
            wprf_grid_stride=4,
            wprf_neighbors=8,
            wprf_scales=[1, 2, 4, 8, 16],
            wprf_gamma=[0.2] * 5,
            wprf_sources_per_scale=32,
            wprf_source_batch_size=32,
            wprf_activation_checkpoint=True,
            wprf_pair_radius="r_k=k",
            wprf_phi_prune=0,
            wprf_epsilon=1e-6,
            wprf_vna_ties="row_major",
        )
        model.update(comparison_affinity="wprf")
        data.update(wprf_semantic_cache=str(output_root / "cache" / dataset / "wprf_v1"))
    if base.loader["train_batch_size"] != 12 or base.optimization["accumulate_grad_batches"] != 1:
        raise ValueError("SegFormer-B2 comparison requires the original batch-12 protocol")
    return replace(
        base,
        model=model,
        data=data,
        optimization={**base.optimization, "loss": loss},
        checkpoint={
            **base.checkpoint,
            "save_last": True,
            "declared_save_last": True,
            "allow_epoch_state_checkpoint": True,
            "epoch_state_every_n_epochs": 1,
            "save_final_state": True,
            "plain_checkpoint_names": True,
        },
        artifacts={
            **base.artifacts,
            "weights_path": str(directory),
            "run_name": f"{name}_seed{seed}",
            "log_name": f"{name}/seed{seed}",
            "val_prediction_dir": str(directory / "val_predictions"),
        },
        provenance={
            **base.provenance,
            "experiment_id": name,
            "family": "segformer_loss_comparison",
            "comparison_method": method,
        },
    )
