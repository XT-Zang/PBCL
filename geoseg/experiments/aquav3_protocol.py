from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


RESULT_SCHEMA = (
    "model",
    "backbone",
    "pretrain",
    "seed",
    "split",
    "fg_iou",
    "dice",
    "precision",
    "recall",
    "miou",
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "boundary_iou",
    "skeleton_f1",
    "merge_rate",
    "split_rate",
    "close_pair_recall",
    "close_pair_separation_recall",
    "object_count_abs_error",
    "normalized_object_count_abs_error",
    "count_nmae",
    "pq",
    "pq_tp",
    "pq_fp",
    "pq_fn",
    "close_pair_touching_total",
    "close_pair_touching_separated",
    "close_pair_touching_separation_recall",
    "close_pair_1_2_total",
    "close_pair_1_2_separated",
    "close_pair_1_2_separation_recall",
    "close_pair_3_5_total",
    "close_pair_3_5_separated",
    "close_pair_3_5_separation_recall",
    "close_pair_6_10_total",
    "close_pair_6_10_separated",
    "close_pair_6_10_separation_recall",
    "close_pair_1_5_total",
    "close_pair_1_5_separated",
    "close_pair_1_5_separation_recall",
    "params_m",
    "flops_g",
    "ms_per_img",
    "gpu_mem_mb",
)


@dataclass(frozen=True)
class AquaV3Experiment:
    key: str
    model: str
    backbone: str
    pretrain: str
    source: str
    group: str
    config_path: str | None = None
    notes: str = ""

    def metadata(self) -> dict[str, str]:
        return {"model": self.model, "backbone": self.backbone, "pretrain": self.pretrain}


MAIN_TABLE_MODELS = (
    AquaV3Experiment(
        key="unetplusplus",
        model="UNet++",
        backbone="EfficientNet-B4",
        pretrain="imagenet/official",
        source="geoseg",
        group="main",
        config_path="config/aquav3_compare/unetplusplus.py",
    ),
    AquaV3Experiment(
        key="deeplabv3plus",
        model="DeepLabV3+",
        backbone="ResNet-50",
        pretrain="imagenet/official",
        source="geoseg",
        group="main",
        config_path="config/aquav3_compare/deeplabv3plus.py",
    ),
    AquaV3Experiment(
        key="swinunet",
        model="Swin-Unet",
        backbone="Swin-T symmetric encoder-decoder",
        pretrain="imagenet/official",
        source="geoseg",
        group="main",
        config_path="config/aquav3_compare/swinunet.py",
        notes="Original symmetric Swin-Unet adapted to 512x512 four-band AquaV3 inputs with dynamic window padding.",
    ),
    AquaV3Experiment(
        key="hrnet_ocr",
        model="HRNet-OCR",
        backbone="HRNet-W32+OCR",
        pretrain="imagenet/official",
        source="geoseg",
        group="main",
        notes="External HRNet-W32+OCR reference.",
    ),
    AquaV3Experiment(
        key="segformer_b2",
        model="SegFormer-B2",
        backbone="MiT-B2",
        pretrain="imagenet/official",
        source="geoseg",
        group="main",
        config_path="config/aquav3_compare/segformer_b2.py",
    ),
    AquaV3Experiment(
        key="upernet_swin_t",
        model="UPerNet-Swin-T",
        backbone="Swin-T",
        pretrain="imagenet/official",
        source="geoseg",
        group="main",
        config_path="config/aquav3_compare/upernet_swin_t.py",
    ),
    AquaV3Experiment(
        key="mask2former_swin_t",
        model="Mask2Former-Swin-T",
        backbone="Swin-T",
        pretrain="imagenet/official",
        source="geoseg",
        group="main",
        config_path="config/aquav3_compare/mask2former_swin_t.py",
    ),
    AquaV3Experiment(
        key="geoseg_a2fpn",
        model="GeoSeg-A2FPN",
        backbone="A2FPN",
        pretrain="none",
        source="geoseg",
        group="main",
        config_path="config/aquav3_compare/geoseg_a2fpn.py",
    ),
    AquaV3Experiment(
        key="dinov3_hrnet_raftsepnet",
        model="DINOv3-HRNet / RaftSepNet",
        backbone="DINOv3-ViT-L/16 + HRNet-W32",
        pretrain="foundation/self-supervised",
        source="geoseg",
        group="proposed",
        notes="Reference hybrid DINOv3+HRNet baseline.",
    ),
)


PENDING_MAIN_TABLE_MODELS = (
    AquaV3Experiment(
        key="unetformer",
        model="UNetFormer",
        backbone="SWSL ResNet-18",
        pretrain="swsl/official",
        source="geoseg",
        group="main_pending",
        config_path="config/aquav3_compare/unetformer.py",
        notes="Pending seed42 formal retraining and reviewer-facing result admission.",
    ),
    AquaV3Experiment(
        key="abcnet",
        model="ABCNet",
        backbone="SWSL ResNet-18 context path",
        pretrain="swsl/official",
        source="geoseg",
        group="main_pending",
        config_path="config/aquav3_compare/abcnet.py",
        notes="Pending seed42 formal retraining and reviewer-facing result admission.",
    ),
)


APPENDIX_GEOSEG_MODELS = (
    AquaV3Experiment("geoseg_a2fpn", "GeoSeg-A2FPN", "A2FPN", "none", "geoseg", "appendix", "config/aquav3/a2fpn.py"),
    AquaV3Experiment("geoseg_abcnet", "GeoSeg-ABCNet", "ABCNet", "none", "geoseg", "appendix", "config/aquav3/abcnet.py"),
    AquaV3Experiment("geoseg_banet", "GeoSeg-BANet", "BANet", "none", "geoseg", "appendix", "config/aquav3/banet.py"),
    AquaV3Experiment("geoseg_dcswin", "GeoSeg-DCSwin", "DCSwin-S", "none", "geoseg", "appendix", "config/aquav3/dcswin.py"),
    AquaV3Experiment(
        "geoseg_ftunetformer",
        "GeoSeg-FTUNetFormer",
        "FTUNetFormer",
        "none",
        "geoseg",
        "appendix",
        "config/aquav3/ftunetformer.py",
    ),
    AquaV3Experiment("geoseg_manet", "GeoSeg-MANet", "MANet", "none", "geoseg", "appendix", "config/aquav3/manet.py"),
    AquaV3Experiment(
        "geoseg_pyramidmamba",
        "GeoSeg-PyramidMamba",
        "PyramidMamba",
        "none",
        "geoseg",
        "appendix",
        "config/aquav3/pyramidmamba.py",
    ),
    AquaV3Experiment(
        "geoseg_unetformer",
        "GeoSeg-UNetFormer",
        "UNetFormer",
        "none",
        "geoseg",
        "appendix",
        "config/aquav3/unetformer.py",
    ),
)


def iter_experiments() -> tuple[AquaV3Experiment, ...]:
    preferred_keys = {model.key for model in MAIN_TABLE_MODELS + PENDING_MAIN_TABLE_MODELS}
    return (
        MAIN_TABLE_MODELS
        + PENDING_MAIN_TABLE_MODELS
        + tuple(item for item in APPENDIX_GEOSEG_MODELS if item.key not in preferred_keys)
    )


def get_experiment(key: str) -> AquaV3Experiment:
    for experiment in iter_experiments():
        if experiment.key == key:
            return experiment
    raise KeyError(f"Unknown AquaV3 experiment: {key}")


def empty_result_row(metadata: Mapping[str, object] | None = None) -> dict[str, object]:
    row = {key: "" for key in RESULT_SCHEMA}
    if metadata:
        for key in ("model", "backbone", "pretrain"):
            if key in metadata:
                row[key] = metadata[key]
    return row


def coerce_result_row(row: Mapping[str, object]) -> dict[str, object]:
    return {key: row.get(key, "") for key in RESULT_SCHEMA}


def write_result_rows(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RESULT_SCHEMA))
        writer.writeheader()
        for row in rows:
            writer.writerow(coerce_result_row(row))
