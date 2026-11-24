from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint_forward

from .multiscale import (
    ConvGNReLU,
    PyramidBinaryHead,
    PyramidOCRHead,
    StandardPyramidDecoder,
    _resize_outputs,
)
from .multiscale_dinov3 import StageAdapterDINOv3Backbone


@dataclass(frozen=True)
class StageAdapterFPNOCRSpec:
    name: str
    stage_depths: tuple[int, int, int, int]
    stage_dims: tuple[int, int, int, int]
    feature_adapter: str = "stage_adapter"
    head_kind: str = "ocr"
    boundary_enabled: bool = True
    pbcl_weight: float = 0.1
    boundary_source: str = "raw_p2"
    decoder_channels: int = 128
    ocr_channels: int = 256
    dropout: float = 0.05
    affinity_channels: int = 0

    @property
    def output_layers(self) -> tuple[int, int, int, int]:
        total = 0
        values: list[int] = []
        for depth in self.stage_depths:
            total += int(depth)
            values.append(total - 1)
        return tuple(values)  # type: ignore[return-value]

    @property
    def active_backbones(self) -> tuple[str, ...]:
        return ("dinov3",)

    def __post_init__(self) -> None:
        if self.feature_adapter not in {"stage_adapter", "projection_resize"}:
            raise ValueError(
                f"unsupported feature adapter: {self.feature_adapter}"
            )
        if self.head_kind not in {"ocr", "binary"}:
            raise ValueError(f"unsupported head kind: {self.head_kind}")
        if self.boundary_source not in {"raw_p2", "decoded_p2"}:
            raise ValueError(
                f"unsupported Boundary source: {self.boundary_source}"
            )
        if float(self.pbcl_weight) not in {0.0, 0.1}:
            raise ValueError("PBCL ablation weight must be 0.0 or 0.1")
        if int(self.affinity_channels) not in {0, 2, 8}:
            raise ValueError("affinity channels must be 0, 2, or 8")
        if self.affinity_channels and self.head_kind != "binary":
            raise ValueError("training-only affinity heads require the Binary head")
        if self.feature_adapter == "projection_resize" and self.head_kind != "binary":
            raise ValueError(
                "the projection-resize matched control requires the Binary head"
            )


_MODEL_NAME = "ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6"
_PROJECTION_RESIZE_MODEL_NAME = (
    "ssepnet_ms_dinov3_projection_resize_fpn_ocr_6_6_6_6_binary_head"
)
_BASE_KWARGS = {
    "stage_depths": (6, 6, 6, 6),
    "stage_dims": (128, 256, 512, 512),
}
_SPECS = {
    _MODEL_NAME: StageAdapterFPNOCRSpec(
        name=_MODEL_NAME,
        **_BASE_KWARGS,
    ),
    f"{_MODEL_NAME}_base": StageAdapterFPNOCRSpec(
        name=f"{_MODEL_NAME}_base",
        boundary_enabled=False,
        pbcl_weight=0.0,
        **_BASE_KWARGS,
    ),
    f"{_MODEL_NAME}_boundary": StageAdapterFPNOCRSpec(
        name=f"{_MODEL_NAME}_boundary",
        pbcl_weight=0.0,
        **_BASE_KWARGS,
    ),
    f"{_MODEL_NAME}_pbcl": StageAdapterFPNOCRSpec(
        name=f"{_MODEL_NAME}_pbcl",
        boundary_enabled=False,
        **_BASE_KWARGS,
    ),
    f"{_MODEL_NAME}_binary_head": StageAdapterFPNOCRSpec(
        name=f"{_MODEL_NAME}_binary_head",
        head_kind="binary",
        **_BASE_KWARGS,
    ),
    f"{_MODEL_NAME}_decoded_p2_boundary": StageAdapterFPNOCRSpec(
        name=f"{_MODEL_NAME}_decoded_p2_boundary",
        boundary_source="decoded_p2",
        **_BASE_KWARGS,
    ),
    f"{_MODEL_NAME}_binary_base": StageAdapterFPNOCRSpec(
        name=f"{_MODEL_NAME}_binary_base",
        head_kind="binary",
        boundary_enabled=False,
        pbcl_weight=0.0,
        **_BASE_KWARGS,
    ),
    f"{_MODEL_NAME}_binary_boundary": StageAdapterFPNOCRSpec(
        name=f"{_MODEL_NAME}_binary_boundary",
        head_kind="binary",
        pbcl_weight=0.0,
        **_BASE_KWARGS,
    ),
    f"{_MODEL_NAME}_binary_pbcl": StageAdapterFPNOCRSpec(
        name=f"{_MODEL_NAME}_binary_pbcl",
        head_kind="binary",
        boundary_enabled=False,
        **_BASE_KWARGS,
    ),
    f"{_MODEL_NAME}_binary_decoded_p2_boundary": StageAdapterFPNOCRSpec(
        name=f"{_MODEL_NAME}_binary_decoded_p2_boundary",
        head_kind="binary",
        boundary_source="decoded_p2",
        **_BASE_KWARGS,
    ),
    f"{_MODEL_NAME}_binary_boundary_cldice": StageAdapterFPNOCRSpec(
        name=f"{_MODEL_NAME}_binary_boundary_cldice",
        head_kind="binary",
        pbcl_weight=0.0,
        **_BASE_KWARGS,
    ),
    f"{_MODEL_NAME}_binary_boundary_cmalis": StageAdapterFPNOCRSpec(
        name=f"{_MODEL_NAME}_binary_boundary_cmalis",
        head_kind="binary",
        pbcl_weight=0.0,
        affinity_channels=2,
        **_BASE_KWARGS,
    ),
    f"{_MODEL_NAME}_binary_boundary_wprf": StageAdapterFPNOCRSpec(
        name=f"{_MODEL_NAME}_binary_boundary_wprf",
        head_kind="binary",
        pbcl_weight=0.0,
        affinity_channels=8,
        **_BASE_KWARGS,
    ),
    _PROJECTION_RESIZE_MODEL_NAME: StageAdapterFPNOCRSpec(
        name=_PROJECTION_RESIZE_MODEL_NAME,
        feature_adapter="projection_resize",
        head_kind="binary",
        **_BASE_KWARGS,
    ),
}

SSEPNET_STAGE_ADAPTER_FPN_OCR: Mapping[str, StageAdapterFPNOCRSpec] = (
    MappingProxyType(_SPECS)
)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def is_ssepnet_stage_adapter_fpn_ocr(name: str) -> bool:
    return _normalize_name(name) in SSEPNET_STAGE_ADAPTER_FPN_OCR


def get_ssepnet_stage_adapter_fpn_ocr(name: str) -> StageAdapterFPNOCRSpec:
    key = _normalize_name(name)
    try:
        return SSEPNET_STAGE_ADAPTER_FPN_OCR[key]
    except KeyError as exc:
        raise ValueError(
            "Unsupported Stage-Adapter FPN/OCR variant "
            f"{name!r}; choose from {', '.join(SSEPNET_STAGE_ADAPTER_FPN_OCR)}"
        ) from exc


class StageAdapterStandardFPNOCRHead(nn.Module):
    """Standard top-down FPN plus one OCR/Boundary head for real P2-P5 inputs."""

    def __init__(
        self,
        input_channels: Sequence[int],
        *,
        decoder_channels: int = 128,
        ocr_channels: int = 256,
        dropout: float = 0.05,
        boundary_enabled: bool = True,
        boundary_source: str = "raw_p2",
    ) -> None:
        super().__init__()
        if len(input_channels) != 4:
            raise ValueError("Stage-Adapter FPN/OCR head requires P2/P3/P4/P5")
        decoder_channels = int(decoder_channels)
        self.output_strides = (4, 8, 16, 32)
        self.ocr_stride = 4
        self.auxiliary_stride = 16
        self.boundary_stride = 4
        self.boundary_enabled = bool(boundary_enabled)
        if boundary_source not in {"raw_p2", "decoded_p2"}:
            raise ValueError(f"unsupported Boundary source: {boundary_source}")
        self.boundary_source = str(boundary_source)
        self.lateral_projections = nn.ModuleList(
            ConvGNReLU(int(channels), decoder_channels, 1)
            for channels in input_channels
        )
        self.decoder = StandardPyramidDecoder(
            decoder_channels, dropout=float(dropout)
        )
        self.ocr_head = PyramidOCRHead(
            decoder_channels,
            int(ocr_channels),
            float(dropout),
        )
        if not self.boundary_enabled:
            self.ocr_head.boundary = nn.Identity()

    def _project_and_decode(
        self, features: Sequence[torch.Tensor]
    ) -> tuple[
        list[torch.Tensor],
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        if len(features) != 4:
            raise ValueError(
                f"Stage-Adapter FPN/OCR requires four features, got {len(features)}"
            )
        projected = [
            projection(feature)
            for projection, feature in zip(
                self.lateral_projections, features, strict=True
            )
        ]
        return projected, self.decoder(projected)

    def forward_main(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        _projected, pyramid = self._project_and_decode(features)
        return self.ocr_head.forward_main(pyramid)

    def forward(
        self, features: Sequence[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        projected, pyramid = self._project_and_decode(features)
        # Keep the exact single OCR path, but preserve the Stage-Adapter's raw
        # lateral P2 detail for Boundary instead of feeding the semanticized
        # top-down P2 back into the edge branch.
        main, auxiliary = self.ocr_head.forward_main_and_auxiliary(pyramid)
        outputs = {
            "out": main,
            "aux": auxiliary,
        }
        if self.boundary_enabled:
            boundary_features = (
                projected[0]
                if self.boundary_source == "raw_p2"
                else pyramid[0]
            )
            outputs["edge"] = self.ocr_head.boundary(boundary_features)
        return outputs


class StageAdapterStandardFPNBinaryHead(nn.Module):
    """Matched Standard-FPN head without OCR object-context refinement."""

    def __init__(
        self,
        input_channels: Sequence[int],
        *,
        decoder_channels: int = 128,
        dropout: float = 0.05,
        boundary_enabled: bool = True,
        boundary_source: str = "raw_p2",
        affinity_channels: int = 0,
    ) -> None:
        super().__init__()
        if len(input_channels) != 4:
            raise ValueError("Stage-Adapter FPN/Binary head requires P2/P3/P4/P5")
        if boundary_source not in {"raw_p2", "decoded_p2"}:
            raise ValueError(f"unsupported Boundary source: {boundary_source}")
        decoder_channels = int(decoder_channels)
        self.output_strides = (4, 8, 16, 32)
        self.auxiliary_stride = 16
        self.boundary_stride = 4
        self.boundary_enabled = bool(boundary_enabled)
        self.boundary_source = str(boundary_source)
        self.affinity_channels = int(affinity_channels)
        self.lateral_projections = nn.ModuleList(
            ConvGNReLU(int(channels), decoder_channels, 1)
            for channels in input_channels
        )
        self.decoder = StandardPyramidDecoder(
            decoder_channels, dropout=float(dropout)
        )
        self.binary_head = PyramidBinaryHead(decoder_channels, float(dropout))
        self.affinity_head: nn.Module = (
            nn.Conv2d(
                decoder_channels,
                self.affinity_channels,
                kernel_size=3,
                padding=1,
            )
            if self.affinity_channels > 0
            else nn.Identity()
        )
        if not self.boundary_enabled:
            self.binary_head.boundary = nn.Identity()

    def _project_and_decode(
        self, features: Sequence[torch.Tensor]
    ) -> tuple[
        list[torch.Tensor],
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        if len(features) != 4:
            raise ValueError(
                f"Stage-Adapter FPN/Binary requires four features, got {len(features)}"
            )
        projected = [
            projection(feature)
            for projection, feature in zip(
                self.lateral_projections, features, strict=True
            )
        ]
        return projected, self.decoder(projected)

    def forward_main(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        _projected, pyramid = self._project_and_decode(features)
        return self.binary_head.main(pyramid[4])

    def forward(
        self, features: Sequence[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        projected, pyramid = self._project_and_decode(features)
        outputs = {
            "out": self.binary_head.main(pyramid[4]),
            "aux": self.binary_head.auxiliary(pyramid[2]),
        }
        if self.affinity_channels > 0:
            outputs["affinity"] = self.affinity_head(pyramid[4])
        if self.boundary_enabled:
            boundary_features = (
                projected[0]
                if self.boundary_source == "raw_p2"
                else pyramid[0]
            )
            outputs["edge"] = self.binary_head.boundary(boundary_features)
        return outputs


class FlatDINOv3PyramidBackbone(nn.Module):
    """Stage-Adapter-free layer taps mapped by a projection-resize baseline.

    The four flat ViT outputs remain on the native stride-16 token grid. A
    learned per-level 1x1 Conv-GN-ReLU compatibility projection changes
    channels only, followed by deterministic bilinear resizing to strides
    4/8/16/32. There is no spatial stem, gated interaction, token feedback, or
    learned layer mixing.
    """

    def __init__(
        self,
        *,
        dinov3_backbone: nn.Module,
        output_channels: Sequence[int],
        activation_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in output_channels)
        if len(channels) != 4:
            raise ValueError("flat DINOv3 pyramid requires four output widths")
        self.dinov3_backbone = dinov3_backbone
        self.activation_checkpoint = bool(activation_checkpoint)
        layer_channels = int(
            getattr(
                dinov3_backbone,
                "embed_dim",
                int(dinov3_backbone.out_channels) // 4,
            )
        )
        if layer_channels * 4 != int(dinov3_backbone.out_channels):
            raise ValueError(
                "flat DINOv3 pyramid requires four equal-width intermediate "
                f"layers, got out_channels={dinov3_backbone.out_channels}"
            )
        self.scale_projections = nn.ModuleList(
            ConvGNReLU(layer_channels, channels[index], 1)
            for index in range(4)
        )
        self.out_channels = channels
        self.out_strides = (4, 8, 16, 32)
        self.initialization_report = {
            "policy": "strict_flat_dinov3_layer_taps_plus_deterministic_resizing",
            "strict_source_blocks": len(
                getattr(
                    getattr(dinov3_backbone, "model", None),
                    "blocks",
                    (),
                )
            ),
            "projected_source_blocks": 0,
            "new_stage_modules": 0,
            "compatibility_projections": 4,
            "feature_adapter": "projection_resize",
        }

    def _forward_layers(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        def forward_layers(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
            values = self.dinov3_backbone.forward_layers(tensor)
            return tuple(values)

        trainable = self.training and any(
            parameter.requires_grad
            for parameter in self.dinov3_backbone.parameters()
        )
        if trainable and self.activation_checkpoint:
            layers = activation_checkpoint_forward(
                forward_layers,
                inputs,
                use_reentrant=False,
            )
        else:
            layers = forward_layers(inputs)
        if len(layers) != 4:
            raise ValueError(
                f"flat DINOv3 must expose four layers, got {len(layers)}"
            )
        return layers

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        height, width = (int(value) for value in inputs.shape[-2:])
        sizes = tuple(
            ((height + stride - 1) // stride, (width + stride - 1) // stride)
            for stride in self.out_strides
        )
        layers = self._forward_layers(inputs)
        return [
            F.interpolate(
                projection(layer),
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            for projection, layer, size in zip(
                self.scale_projections,
                layers,
                sizes,
                strict=True,
            )
        ]


class StageAdapterFPNSSepNet(nn.Module):
    """Controlled DINOv3 pyramid plus a matched Standard-FPN head."""

    active_backbones = ("dinov3",)

    def __init__(
        self,
        *,
        multiscale_backbone: nn.Module,
        decoder_channels: int = 128,
        ocr_channels: int = 256,
        head_kind: str = "ocr",
        boundary_enabled: bool = True,
        boundary_source: str = "raw_p2",
        num_classes: int = 1,
        freeze_backbone: bool = False,
        dropout: float = 0.05,
        affinity_channels: int = 0,
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("matched DINOv3 FPN requires binary segmentation")
        channels = tuple(int(value) for value in multiscale_backbone.out_channels)
        if len(channels) != 4:
            raise ValueError("DINOv3 pyramid must expose four feature widths")
        self.multiscale_backbone = multiscale_backbone
        self.freeze_backbone = bool(freeze_backbone)
        if head_kind == "ocr":
            self.head: nn.Module = StageAdapterStandardFPNOCRHead(
                channels,
                decoder_channels=int(decoder_channels),
                ocr_channels=int(ocr_channels),
                dropout=float(dropout),
                boundary_enabled=bool(boundary_enabled),
                boundary_source=boundary_source,
            )
        elif head_kind == "binary":
            self.head = StageAdapterStandardFPNBinaryHead(
                channels,
                decoder_channels=int(decoder_channels),
                dropout=float(dropout),
                boundary_enabled=bool(boundary_enabled),
                boundary_source=boundary_source,
                affinity_channels=int(affinity_channels),
            )
        else:
            raise ValueError(f"unsupported Stage-Adapter FPN head: {head_kind}")
        self.head_kind = str(head_kind)
        self.boundary_enabled = bool(boundary_enabled)
        self.boundary_source = str(boundary_source)
        self.training_only_affinity_head = int(affinity_channels) > 0
        if self.freeze_backbone:
            self.multiscale_backbone.requires_grad_(False)

    @property
    def pretrained_initialization_report(self) -> Mapping[str, object]:
        return dict(getattr(self.multiscale_backbone, "initialization_report", {}))

    def interaction_gate_statistics(
        self,
    ) -> tuple[tuple[float, float, float, float], ...]:
        callback = getattr(
            self.multiscale_backbone, "interaction_gate_statistics", None
        )
        return () if callback is None else callback()

    def train(self, mode: bool = True) -> "StageAdapterFPNSSepNet":
        super().train(mode)
        if self.freeze_backbone:
            self.multiscale_backbone.eval()
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        main_only: bool = False,
        disable_hrnet: bool = False,
        disable_dinov3: bool = False,
    ) -> dict[str, torch.Tensor]:
        del disable_hrnet
        features = list(self.multiscale_backbone(inputs))
        if len(features) != 4:
            raise ValueError(
                f"DINOv3 pyramid must return four features, got {len(features)}"
            )
        if disable_dinov3:
            features = [torch.zeros_like(feature) for feature in features]
        outputs = (
            {"out": self.head.forward_main(features)}
            if main_only
            else self.head(features)
        )
        return _resize_outputs(outputs, tuple(inputs.shape[-2:]))

    def forward_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs, main_only=True)["out"]


def build_ssepnet_stage_adapter_fpn_ocr(
    name: str,
    *,
    in_chans: int = 4,
    num_classes: int = 1,
    pretrained: bool = True,
    dinov3_checkpoint: str | Path | None = None,
    freeze_backbone: bool = False,
    activation_checkpoint: bool = True,
    dinov3_adaptation: str = "full",
    dinov3_lora_rank: int = 0,
    dinov3_lora_alpha: float | None = None,
) -> StageAdapterFPNSSepNet:
    spec = get_ssepnet_stage_adapter_fpn_ocr(name)
    if int(in_chans) not in {3, 4}:
        raise ValueError("Stage-Adapter FPN/OCR requires RGB or B/G/R/NIR input")
    from .dinov3 import build_dinov3_vitl16_sat493m
    from .lora import configure_dinov3_adaptation

    flat = build_dinov3_vitl16_sat493m(
        in_chans=int(in_chans),
        pretrained=bool(pretrained),
        checkpoint_path=dinov3_checkpoint,
    )
    adaptation = configure_dinov3_adaptation(
        flat,
        mode=dinov3_adaptation,
        rank=int(dinov3_lora_rank),
        alpha=dinov3_lora_alpha,
    )
    if spec.feature_adapter == "stage_adapter":
        backbone: nn.Module = StageAdapterDINOv3Backbone(
            dinov3_backbone=flat,
            stage_depths=spec.stage_depths,
            spatial_dims=spec.stage_dims,
            in_chans=int(in_chans),
            activation_checkpoint=bool(activation_checkpoint),
        )
    else:
        backbone = FlatDINOv3PyramidBackbone(
            dinov3_backbone=flat,
            output_channels=spec.stage_dims,
            activation_checkpoint=bool(activation_checkpoint),
        )
    backbone.initialization_report.update(
        {
            "architecture": (
                f"{spec.feature_adapter}_standard_fpn_{spec.head_kind}"
            ),
            "feature_adapter": spec.feature_adapter,
            "stage_depths": spec.stage_depths,
            "stage_output_layers": spec.output_layers,
            "output_strides": (4, 8, 16, 32),
            "segmentation_head": (
                "standard_fpn_single_ocr_boundary_v1"
                if spec.head_kind == "ocr"
                else "standard_fpn_binary_boundary_v1"
            ),
            "boundary_enabled": spec.boundary_enabled,
            "boundary_source": spec.boundary_source,
            "pbcl_weight": spec.pbcl_weight,
            "affinity_channels": spec.affinity_channels,
            "soft_layer_mixing": False,
            "dinov3_adaptation": getattr(adaptation, "mode", dinov3_adaptation),
        }
    )
    return StageAdapterFPNSSepNet(
        multiscale_backbone=backbone,
        decoder_channels=spec.decoder_channels,
        ocr_channels=spec.ocr_channels,
        head_kind=spec.head_kind,
        boundary_enabled=spec.boundary_enabled,
        boundary_source=spec.boundary_source,
        num_classes=int(num_classes),
        freeze_backbone=bool(freeze_backbone),
        dropout=spec.dropout,
        affinity_channels=spec.affinity_channels,
    )


def build_ssepnet_stage_adapter_fpn_ocr_loss(name: str) -> nn.Module:
    spec = get_ssepnet_stage_adapter_fpn_ocr(name)
    from geoseg.losses.ssepnet import SSepNetLoss

    return SSepNetLoss(
        boundary_weight=0.6 if spec.boundary_enabled else 0.0,
        pbcl_weight=float(spec.pbcl_weight),
    )


__all__ = [
    "SSEPNET_STAGE_ADAPTER_FPN_OCR",
    "FlatDINOv3PyramidBackbone",
    "StageAdapterStandardFPNBinaryHead",
    "StageAdapterStandardFPNOCRHead",
    "StageAdapterFPNSSepNet",
    "StageAdapterFPNOCRSpec",
    "build_ssepnet_stage_adapter_fpn_ocr",
    "build_ssepnet_stage_adapter_fpn_ocr_loss",
    "get_ssepnet_stage_adapter_fpn_ocr",
    "is_ssepnet_stage_adapter_fpn_ocr",
]
