from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint_forward

from .common import ConvBNReLU
from .lora import DinoV3AdaptationReport, configure_dinov3_adaptation
from .ocr import SSepNetOCRHead


@dataclass(frozen=True)
class SSepNetOCROptimizationSpec:
    name: str
    adapter: str
    head_input_channels: int = 256
    layer_projection_channels: int = 0


_OCR_OPTIMIZATIONS = {
    "ssepnet_dinov3_unified_ocr": SSepNetOCROptimizationSpec(
        "ssepnet_dinov3_unified_ocr",
        "unified_projection",
    ),
    "ssepnet_dinov3_layeraware_ocr": SSepNetOCROptimizationSpec(
        "ssepnet_dinov3_layeraware_ocr",
        "layeraware_progressive",
        layer_projection_channels=64,
    ),
}

SSEPNET_OCR_OPTIMIZATIONS: Mapping[str, SSepNetOCROptimizationSpec] = MappingProxyType(
    _OCR_OPTIMIZATIONS
)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def is_ssepnet_ocr_optimization(name: str) -> bool:
    return _normalize_name(name) in SSEPNET_OCR_OPTIMIZATIONS


def get_ssepnet_ocr_optimization(name: str) -> SSepNetOCROptimizationSpec:
    key = _normalize_name(name)
    try:
        return SSEPNET_OCR_OPTIMIZATIONS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported SSepNet OCR optimization {name!r}; choose from "
            f"{', '.join(SSEPNET_OCR_OPTIMIZATIONS)}"
        ) from exc


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _resize_named_outputs(
    outputs: dict[str, torch.Tensor],
    input_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    return {
        name: F.interpolate(
            value, size=input_size, mode="bilinear", align_corners=False
        )
        for name, value in outputs.items()
    }


class DepthwiseSeparableResidual(nn.Module):
    """A lightweight spatial refinement block that preserves channel width."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError(f"refinement channels must be positive, got {channels}")
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.depthwise_norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pointwise_norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        refined = self.activation(self.depthwise_norm(self.depthwise(inputs)))
        refined = self.pointwise_norm(self.pointwise(refined))
        return self.activation(inputs + refined)


class _DINOv3OCRBase(nn.Module):
    active_backbones = ("dinov3",)

    def __init__(
        self,
        *,
        dinov3_backbone: nn.Module,
        adapter_channels: int,
        ocr_channels: int,
        boundary_enabled: bool,
        num_classes: int,
        freeze_backbone: bool,
        activation_checkpoint: bool,
        dinov3_adaptation: str,
        dinov3_lora_rank: int,
        dinov3_lora_alpha: float | None,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("SSepNet OCR optimizations require num_classes=1")
        if int(adapter_channels) <= 0:
            raise ValueError("adapter_channels must be positive")
        normalized_adaptation = str(dinov3_adaptation).strip().lower()
        if bool(freeze_backbone) and normalized_adaptation != "full":
            raise ValueError(
                "freeze_backbone=True cannot be combined with targeted DINOv3 adaptation"
            )

        self.dinov3_backbone = dinov3_backbone
        self.adapter_channels = int(adapter_channels)
        self.freeze_backbone = bool(freeze_backbone)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.dinov3_adaptation = normalized_adaptation
        self.dinov3_lora_rank = int(dinov3_lora_rank)
        self.dinov3_lora_alpha = (
            None if dinov3_lora_alpha is None else float(dinov3_lora_alpha)
        )
        self.head = SSepNetOCRHead(
            self.adapter_channels,
            channels=int(ocr_channels),
            num_classes=int(num_classes),
            boundary_enabled=bool(boundary_enabled),
            prune_disabled_boundary=True,
            dropout=float(dropout),
        )

        if self.freeze_backbone:
            self.dinov3_backbone.requires_grad_(False)
            self.dinov3_adaptation_report = DinoV3AdaptationReport(
                "frozen", 0, None, (), 0
            )
        else:
            self.dinov3_adaptation_report = configure_dinov3_adaptation(
                self.dinov3_backbone,
                mode=self.dinov3_adaptation,
                rank=self.dinov3_lora_rank or 3,
                alpha=self.dinov3_lora_alpha,
            )

    def _backbone_is_trainable(self) -> bool:
        return self.training and any(
            parameter.requires_grad for parameter in self.dinov3_backbone.parameters()
        )

    def _forward_dinov3(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._backbone_is_trainable() and self.activation_checkpoint:
            return activation_checkpoint_forward(
                self.dinov3_backbone, inputs, use_reentrant=False
            )
        return self.dinov3_backbone(inputs)

    def _forward_dinov3_layers(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        def forward_layers(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
            if hasattr(self.dinov3_backbone, "forward_layers"):
                values = self.dinov3_backbone.forward_layers(tensor)
            else:
                values = self.dinov3_backbone(tensor).chunk(4, dim=1)
            return tuple(values)

        if self._backbone_is_trainable() and self.activation_checkpoint:
            layers = activation_checkpoint_forward(
                forward_layers, inputs, use_reentrant=False
            )
        else:
            layers = forward_layers(inputs)
        if len(layers) != 4:
            raise ValueError(
                f"DINOv3 must expose four intermediate layers, got {len(layers)}"
            )
        return list(layers)

    def _ocr_features(
        self,
        inputs: torch.Tensor,
        *,
        disable_dinov3: bool,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        main_only: bool = False,
        disable_dinov3: bool = False,
    ) -> dict[str, torch.Tensor]:
        features = self._ocr_features(inputs, disable_dinov3=disable_dinov3)
        outputs = (
            {"out": self.head.forward_main(features)}
            if main_only
            else self.head(features)
        )
        return _resize_named_outputs(outputs, tuple(inputs.shape[-2:]))

    def forward_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs, main_only=True)["out"]

    def train(self, mode: bool = True) -> "_DINOv3OCRBase":
        super().train(mode)
        if self.freeze_backbone or self.dinov3_adaptation in {"frozen", "lora"}:
            self.dinov3_backbone.eval()
        return self


class DINOv3UnifiedOCRSSepNet(_DINOv3OCRBase):
    """Minimal optimization: replace detail/context duplication with one 256-channel adapter."""

    def __init__(
        self,
        *,
        dinov3_backbone: nn.Module,
        boundary_enabled: bool = True,
        num_classes: int = 1,
        adapter_channels: int = 256,
        ocr_channels: int = 256,
        freeze_backbone: bool = False,
        activation_checkpoint: bool = True,
        dinov3_adaptation: str = "full",
        dinov3_lora_rank: int = 0,
        dinov3_lora_alpha: float | None = None,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(
            dinov3_backbone=dinov3_backbone,
            adapter_channels=adapter_channels,
            ocr_channels=ocr_channels,
            boundary_enabled=boundary_enabled,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            activation_checkpoint=activation_checkpoint,
            dinov3_adaptation=dinov3_adaptation,
            dinov3_lora_rank=dinov3_lora_rank,
            dinov3_lora_alpha=dinov3_lora_alpha,
            dropout=dropout,
        )
        dino_channels = int(dinov3_backbone.out_channels)
        self.unified_projection = nn.Sequential(
            ConvBNReLU(dino_channels, self.adapter_channels, 1),
            ConvBNReLU(self.adapter_channels, self.adapter_channels, 3),
        )

    def _ocr_features(
        self,
        inputs: torch.Tensor,
        *,
        disable_dinov3: bool,
    ) -> torch.Tensor:
        features = self._forward_dinov3(inputs)
        if disable_dinov3:
            features = torch.zeros_like(features)
        features = self.unified_projection(features)
        target_size = (_ceil_div(inputs.shape[-2], 4), _ceil_div(inputs.shape[-1], 4))
        return F.interpolate(
            features,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )


class DINOv3LayerAwareOCRSSepNet(_DINOv3OCRBase):
    """Recommended optimization: preserve four layer identities before progressive OCR adaptation."""

    def __init__(
        self,
        *,
        dinov3_backbone: nn.Module,
        boundary_enabled: bool = True,
        num_classes: int = 1,
        layer_projection_channels: int = 64,
        adapter_channels: int = 256,
        ocr_channels: int = 256,
        freeze_backbone: bool = False,
        activation_checkpoint: bool = True,
        dinov3_adaptation: str = "full",
        dinov3_lora_rank: int = 0,
        dinov3_lora_alpha: float | None = None,
        dropout: float = 0.05,
    ) -> None:
        layer_projection_channels = int(layer_projection_channels)
        adapter_channels = int(adapter_channels)
        if layer_projection_channels <= 0:
            raise ValueError("layer_projection_channels must be positive")
        if layer_projection_channels * 4 != adapter_channels:
            raise ValueError(
                "four layer projections must concatenate to adapter_channels, got "
                f"4 * {layer_projection_channels} != {adapter_channels}"
            )
        super().__init__(
            dinov3_backbone=dinov3_backbone,
            adapter_channels=adapter_channels,
            ocr_channels=ocr_channels,
            boundary_enabled=boundary_enabled,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            activation_checkpoint=activation_checkpoint,
            dinov3_adaptation=dinov3_adaptation,
            dinov3_lora_rank=dinov3_lora_rank,
            dinov3_lora_alpha=dinov3_lora_alpha,
            dropout=dropout,
        )
        dino_channels = int(dinov3_backbone.out_channels)
        layer_channels = int(getattr(dinov3_backbone, "embed_dim", dino_channels // 4))
        if layer_channels * 4 != dino_channels:
            raise ValueError(
                "layer-aware OCR requires four equal-width DINOv3 layers, got "
                f"out_channels={dino_channels}, layer_channels={layer_channels}"
            )
        self.layer_projection_channels = layer_projection_channels
        self.layer_projections = nn.ModuleList(
            ConvBNReLU(layer_channels, self.layer_projection_channels, 1)
            for _ in range(4)
        )
        self.fusion_refine = DepthwiseSeparableResidual(self.adapter_channels)
        self.mid_refine = DepthwiseSeparableResidual(self.adapter_channels)
        self.output_refine = DepthwiseSeparableResidual(self.adapter_channels)

    def _ocr_features(
        self,
        inputs: torch.Tensor,
        *,
        disable_dinov3: bool,
    ) -> torch.Tensor:
        layers = self._forward_dinov3_layers(inputs)
        if disable_dinov3:
            layers = [torch.zeros_like(layer) for layer in layers]
        projected = [
            projection(layer)
            for projection, layer in zip(self.layer_projections, layers, strict=True)
        ]
        features = self.fusion_refine(torch.cat(projected, dim=1))
        mid_size = (_ceil_div(inputs.shape[-2], 8), _ceil_div(inputs.shape[-1], 8))
        target_size = (_ceil_div(inputs.shape[-2], 4), _ceil_div(inputs.shape[-1], 4))
        features = self.mid_refine(
            F.interpolate(features, size=mid_size, mode="bilinear", align_corners=False)
        )
        return self.output_refine(
            F.interpolate(
                features, size=target_size, mode="bilinear", align_corners=False
            )
        )


def build_ssepnet_ocr_optimization(
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
) -> nn.Module:
    spec = get_ssepnet_ocr_optimization(name)
    from .dinov3 import build_dinov3_vitl16_sat493m

    dinov3_backbone = build_dinov3_vitl16_sat493m(
        in_chans=in_chans,
        pretrained=pretrained,
        checkpoint_path=dinov3_checkpoint,
    )
    common = {
        "dinov3_backbone": dinov3_backbone,
        "boundary_enabled": True,
        "num_classes": int(num_classes),
        "adapter_channels": int(spec.head_input_channels),
        "ocr_channels": 256,
        "freeze_backbone": bool(freeze_backbone),
        "activation_checkpoint": bool(activation_checkpoint),
        "dinov3_adaptation": dinov3_adaptation,
        "dinov3_lora_rank": int(dinov3_lora_rank),
        "dinov3_lora_alpha": dinov3_lora_alpha,
    }
    if spec.adapter == "unified_projection":
        return DINOv3UnifiedOCRSSepNet(**common)
    if spec.adapter == "layeraware_progressive":
        return DINOv3LayerAwareOCRSSepNet(
            **common,
            layer_projection_channels=int(spec.layer_projection_channels),
        )
    raise AssertionError(f"unhandled OCR adapter: {spec.adapter}")


def build_ssepnet_ocr_optimization_loss(name: str) -> nn.Module:
    get_ssepnet_ocr_optimization(name)
    from geoseg.losses.ssepnet import SSepNetLoss

    return SSepNetLoss(boundary_weight=0.6, pbcl_weight=0.1)


__all__ = [
    "SSEPNET_OCR_OPTIMIZATIONS",
    "SSepNetOCROptimizationSpec",
    "DepthwiseSeparableResidual",
    "DINOv3UnifiedOCRSSepNet",
    "DINOv3LayerAwareOCRSSepNet",
    "build_ssepnet_ocr_optimization",
    "build_ssepnet_ocr_optimization_loss",
    "get_ssepnet_ocr_optimization",
    "is_ssepnet_ocr_optimization",
]
