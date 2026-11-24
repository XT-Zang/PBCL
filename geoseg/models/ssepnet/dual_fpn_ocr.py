from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint_forward

from .lora import DinoV3AdaptationReport, configure_dinov3_adaptation
from .multiscale import (
    ConvGNReLU,
    PyramidOCRHead,
    SoftLayerMixPyramid,
    StandardPyramidDecoder,
    _resize_like,
    _resize_outputs,
)


@dataclass(frozen=True)
class DualFPNOCRSpec:
    name: str
    detail_backbone: str
    active_backbones: tuple[str, ...]
    decoder_channels: int = 128
    ocr_channels: int = 256


_SPECS = {
    "ssepnet_dinov3_fpn_ocr": DualFPNOCRSpec(
        "ssepnet_dinov3_fpn_ocr",
        "none",
        ("dinov3",),
    ),
    "ssepnet_dinov3_hrformer_s_fpn_ocr": DualFPNOCRSpec(
        "ssepnet_dinov3_hrformer_s_fpn_ocr",
        "hrformer_s",
        ("dinov3", "hrformer_s"),
    ),
    "ssepnet_dinov3_pidnet_s_pd_fpn_ocr": DualFPNOCRSpec(
        "ssepnet_dinov3_pidnet_s_pd_fpn_ocr",
        "pidnet_s_pd",
        ("dinov3", "pidnet_s_pd"),
    ),
    "ssepnet_dinov3_ddrnet23_slim_fpn_ocr": DualFPNOCRSpec(
        "ssepnet_dinov3_ddrnet23_slim_fpn_ocr",
        "ddrnet23_slim",
        ("dinov3", "ddrnet23_slim"),
    ),
}

SSEPNET_DUAL_FPN_OCR: Mapping[str, DualFPNOCRSpec] = MappingProxyType(_SPECS)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def is_ssepnet_dual_fpn_ocr(name: str) -> bool:
    return _normalize_name(name) in SSEPNET_DUAL_FPN_OCR


def get_ssepnet_dual_fpn_ocr(name: str) -> DualFPNOCRSpec:
    key = _normalize_name(name)
    try:
        return SSEPNET_DUAL_FPN_OCR[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dual FPN+OCR model {name!r}; choose from "
            f"{', '.join(SSEPNET_DUAL_FPN_OCR)}"
        ) from exc


def _pyramid_references(inputs: torch.Tensor) -> list[torch.Tensor]:
    batch, _channels, height, width = inputs.shape
    return [
        inputs.new_zeros(
            (
                batch,
                1,
                (int(height) + stride - 1) // stride,
                (int(width) + stride - 1) // stride,
            )
        )
        for stride in (4, 8, 16, 32)
    ]


class DinoAnchoredResidualGate(nn.Module):
    """Preserve the semantic tensor exactly and add a gated detail residual."""

    def __init__(
        self,
        semantic_channels: int,
        detail_channels: int,
        *,
        gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        semantic_channels = int(semantic_channels)
        self.detail_projection = ConvGNReLU(
            int(detail_channels),
            semantic_channels,
            1,
        )
        self.gate = nn.Conv2d(
            semantic_channels * 2,
            semantic_channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init_bias))
        self.last_gate_mean: torch.Tensor | None = None
        self.last_gate_statistics: torch.Tensor | None = None

    def clear_statistics(self) -> None:
        self.last_gate_mean = None
        self.last_gate_statistics = None

    def forward(
        self,
        semantic: torch.Tensor,
        detail: torch.Tensor,
    ) -> torch.Tensor:
        detail = _resize_like(self.detail_projection(detail), semantic)
        gate = torch.sigmoid(self.gate(torch.cat((semantic, detail), dim=1)))
        self.last_gate_mean = gate.detach().mean()
        self.last_gate_statistics = torch.stack(
            (
                gate.detach().mean(),
                gate.detach().std(unbiased=False),
                gate.detach().amin(),
                gate.detach().amax(),
            )
        )
        return semantic + gate * detail


class DINOv3DetailFPNOCR(nn.Module):
    """DINO-anchored semantic pyramid with optional two-scale detail residuals."""

    def __init__(
        self,
        *,
        dinov3_backbone: nn.Module,
        detail_backbone: nn.Module | None,
        detail_kind: str,
        decoder_channels: int = 128,
        ocr_channels: int = 256,
        num_classes: int = 1,
        activation_checkpoint: bool = True,
        dinov3_adaptation: str = "full",
        dinov3_lora_rank: int = 0,
        dinov3_lora_alpha: float | None = None,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("dual FPN+OCR models require binary num_classes=1")
        self.dinov3_backbone = dinov3_backbone
        self.detail_backbone = detail_backbone
        self.detail_kind = str(detail_kind)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.dinov3_adaptation = str(dinov3_adaptation).strip().lower()
        self.dinov3_lora_rank = int(dinov3_lora_rank)
        self.dinov3_lora_alpha = (
            None if dinov3_lora_alpha is None else float(dinov3_lora_alpha)
        )
        dino_channels = int(dinov3_backbone.out_channels)
        self.dino_layer_channels = int(
            getattr(dinov3_backbone, "embed_dim", dino_channels // 4)
        )
        if self.dino_layer_channels * 4 != dino_channels:
            raise ValueError(
                "dual FPN+OCR requires four equal-width DINOv3 layers, got "
                f"out_channels={dino_channels}, layer_channels={self.dino_layer_channels}"
            )
        self.dino_pyramid = SoftLayerMixPyramid(
            self.dino_layer_channels,
            int(decoder_channels),
        )
        self.decoder = StandardPyramidDecoder(int(decoder_channels), float(dropout))
        self.head = PyramidOCRHead(
            int(decoder_channels),
            int(ocr_channels),
            float(dropout),
        )
        self.fusion_blocks = nn.ModuleList()
        if detail_backbone is not None:
            channels = tuple(
                int(value) for value in getattr(detail_backbone, "detail_channels", ())
            )
            if len(channels) != 2 or any(value <= 0 for value in channels):
                raise ValueError(
                    "detail backbone must expose two positive-width features, got "
                    f"{channels}"
                )
            self.fusion_blocks.extend(
                DinoAnchoredResidualGate(
                    int(decoder_channels),
                    detail_channels,
                    gate_init_bias=-2.0,
                )
                for detail_channels in channels
            )
        self.dinov3_adaptation_report: DinoV3AdaptationReport = (
            configure_dinov3_adaptation(
                self.dinov3_backbone,
                mode=self.dinov3_adaptation,
                rank=self.dinov3_lora_rank or 3,
                alpha=self.dinov3_lora_alpha,
            )
        )

    @property
    def active_backbones(self) -> tuple[str, ...]:
        return (
            ("dinov3",)
            if self.detail_backbone is None
            else ("dinov3", self.detail_kind)
        )

    def _backbone_is_trainable(self) -> bool:
        return self.training and any(
            parameter.requires_grad for parameter in self.dinov3_backbone.parameters()
        )

    def _forward_dinov3_layers(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        def forward_layers(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
            if hasattr(self.dinov3_backbone, "forward_layers"):
                values = self.dinov3_backbone.forward_layers(tensor)
            else:
                values = self.dinov3_backbone(tensor).chunk(4, dim=1)
            return tuple(values)

        if self._backbone_is_trainable() and self.activation_checkpoint:
            layers = activation_checkpoint_forward(
                forward_layers,
                inputs,
                use_reentrant=False,
            )
        else:
            layers = forward_layers(inputs)
        if len(layers) != 4:
            raise ValueError(f"DINOv3 must expose four layers, got {len(layers)}")
        return list(layers)

    def layer_mixing_weights(self) -> torch.Tensor:
        return self.dino_pyramid.mixing_weights()

    def fusion_gate_statistics(self) -> tuple[dict[str, float], ...]:
        names = ("mean", "std", "min", "max")
        return tuple(
            {
                name: float(value.float().cpu())
                for name, value in zip(
                    names,
                    block.last_gate_statistics,
                    strict=True,
                )
            }
            for block in self.fusion_blocks
            if block.last_gate_statistics is not None
        )

    def _fused_pyramid(
        self,
        inputs: torch.Tensor,
        *,
        disable_detail: bool,
        disable_dinov3: bool,
    ) -> list[torch.Tensor]:
        layers = self._forward_dinov3_layers(inputs)
        if disable_dinov3:
            layers = [torch.zeros_like(value) for value in layers]
        references = _pyramid_references(inputs)
        semantic = self.dino_pyramid(layers, references)
        if self.detail_backbone is None or disable_detail:
            for block in self.fusion_blocks:
                block.clear_statistics()
            return semantic
        detail = list(self.detail_backbone(inputs))
        if len(detail) != 2:
            raise ValueError(
                f"detail backbone must return stride-4/8 tensors, got {len(detail)}"
            )
        return [
            self.fusion_blocks[0](semantic[0], detail[0]),
            self.fusion_blocks[1](semantic[1], detail[1]),
            semantic[2],
            semantic[3],
        ]

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        main_only: bool = False,
        disable_detail: bool = False,
        disable_dinov3: bool = False,
    ) -> dict[str, torch.Tensor]:
        pyramid = self._fused_pyramid(
            inputs,
            disable_detail=disable_detail,
            disable_dinov3=disable_dinov3,
        )
        decoded = self.decoder(pyramid)
        outputs = (
            {"out": self.head.forward_main(decoded)}
            if main_only
            else self.head(decoded)
        )
        resized = _resize_outputs(outputs, tuple(inputs.shape[-2:]))
        return {
            key: value
            for key, value in resized.items()
            if isinstance(value, torch.Tensor)
        }

    def forward_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs, main_only=True)["out"]

    def train(self, mode: bool = True) -> "DINOv3DetailFPNOCR":
        super().train(mode)
        if self.dinov3_adaptation in {"frozen", "lora"}:
            self.dinov3_backbone.eval()
        return self


def _build_detail_backbone(
    kind: str,
    *,
    in_chans: int,
    pretrained: bool,
    checkpoint_path: str | Path | None,
) -> nn.Module | None:
    if kind == "none":
        if checkpoint_path is not None:
            raise ValueError("DINO-only FPN+OCR cannot load a detail checkpoint")
        return None
    if kind == "hrformer_s":
        from .hrformer_detail import build_hrformer_s_detail

        return build_hrformer_s_detail(
            in_chans=in_chans,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
        )
    if kind == "pidnet_s_pd":
        from .pidnet_detail import build_pidnet_s_detail

        return build_pidnet_s_detail(
            in_chans=in_chans,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
        )
    if kind == "ddrnet23_slim":
        from .ddrnet_detail import build_ddrnet23_slim_detail

        return build_ddrnet23_slim_detail(
            in_chans=in_chans,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
        )
    raise AssertionError(kind)


def build_ssepnet_dual_fpn_ocr(
    name: str,
    *,
    in_chans: int = 4,
    num_classes: int = 1,
    pretrained: bool = True,
    dinov3_checkpoint: str | Path | None = None,
    detail_checkpoint: str | Path | None = None,
    activation_checkpoint: bool = True,
    dinov3_adaptation: str = "full",
    dinov3_lora_rank: int = 0,
    dinov3_lora_alpha: float | None = None,
) -> DINOv3DetailFPNOCR:
    spec = get_ssepnet_dual_fpn_ocr(name)
    from .dinov3 import build_dinov3_vitl16_sat493m

    dinov3 = build_dinov3_vitl16_sat493m(
        in_chans=in_chans,
        pretrained=pretrained,
        checkpoint_path=dinov3_checkpoint,
    )
    with torch.random.fork_rng(devices=[]):
        detail = _build_detail_backbone(
            spec.detail_backbone,
            in_chans=in_chans,
            pretrained=pretrained,
            checkpoint_path=detail_checkpoint,
        )
    return DINOv3DetailFPNOCR(
        dinov3_backbone=dinov3,
        detail_backbone=detail,
        detail_kind=spec.detail_backbone,
        decoder_channels=spec.decoder_channels,
        ocr_channels=spec.ocr_channels,
        num_classes=num_classes,
        activation_checkpoint=activation_checkpoint,
        dinov3_adaptation=dinov3_adaptation,
        dinov3_lora_rank=dinov3_lora_rank,
        dinov3_lora_alpha=dinov3_lora_alpha,
    )


def build_ssepnet_dual_fpn_ocr_loss(name: str) -> nn.Module:
    get_ssepnet_dual_fpn_ocr(name)
    from geoseg.losses.ssepnet import SSepNetLoss

    return SSepNetLoss(boundary_weight=0.6, pbcl_weight=0.1)


__all__ = [
    "DINOv3DetailFPNOCR",
    "DinoAnchoredResidualGate",
    "DualFPNOCRSpec",
    "SSEPNET_DUAL_FPN_OCR",
    "build_ssepnet_dual_fpn_ocr",
    "build_ssepnet_dual_fpn_ocr_loss",
    "get_ssepnet_dual_fpn_ocr",
    "is_ssepnet_dual_fpn_ocr",
]
