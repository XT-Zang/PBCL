from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint_forward

from .common import ConvBNReLU
from .lora import DinoV3AdaptationReport, configure_dinov3_adaptation
from .ocr import SSepNetOCRHead
from .variants import SSepNetVariantSpec


def _resize_named_outputs(
    outputs: dict[str, torch.Tensor],
    input_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    return {
        name: F.interpolate(value, size=input_size, mode="bilinear", align_corners=False)
        for name, value in outputs.items()
    }


class HRNetOnlySSepNet(nn.Module):
    def __init__(
        self,
        *,
        hrnet_backbone: nn.Module,
        boundary_enabled: bool,
        num_classes: int = 1,
        ocr_channels: int = 256,
        freeze_backbone: bool = False,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.hrnet_backbone = hrnet_backbone
        self.freeze_backbone = bool(freeze_backbone)
        self.stage_channels = tuple(int(value) for value in hrnet_backbone.stage_channels)
        if len(self.stage_channels) != 4:
            raise ValueError("SSepNet HRNet backbone must expose four stage_channels")
        self.head = SSepNetOCRHead(
            sum(self.stage_channels),
            channels=int(ocr_channels),
            num_classes=num_classes,
            boundary_enabled=boundary_enabled,
            prune_disabled_boundary=True,
            dropout=dropout,
        )
        if self.freeze_backbone:
            self.hrnet_backbone.requires_grad_(False)

    def _fused_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = list(self.hrnet_backbone(inputs))
        high_size = features[0].shape[-2:]
        return torch.cat(
            [features[0]]
            + [
                F.interpolate(feature, size=high_size, mode="bilinear", align_corners=False)
                for feature in features[1:]
            ],
            dim=1,
        )

    def forward(self, inputs: torch.Tensor, *, main_only: bool = False) -> dict[str, torch.Tensor]:
        fused = self._fused_features(inputs)
        outputs = {"out": self.head.forward_main(fused)} if main_only else self.head(fused)
        return _resize_named_outputs(outputs, tuple(inputs.shape[-2:]))

    def forward_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs, main_only=True)["out"]

    def train(self, mode: bool = True) -> "HRNetOnlySSepNet":
        super().train(mode)
        if self.freeze_backbone:
            self.hrnet_backbone.eval()
        return self


class DINOv3OnlySSepNet(nn.Module):
    def __init__(
        self,
        *,
        dinov3_backbone: nn.Module,
        boundary_enabled: bool,
        num_classes: int = 1,
        ocr_channels: int = 256,
        detail_channels: int = 32,
        context_channels: int = 320,
        freeze_backbone: bool = False,
        activation_checkpoint: bool = False,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.dinov3_backbone = dinov3_backbone
        self.freeze_backbone = bool(freeze_backbone)
        self.activation_checkpoint = bool(activation_checkpoint)
        dino_channels = int(dinov3_backbone.out_channels)
        self.detail_projection = nn.Sequential(
            ConvBNReLU(dino_channels, int(context_channels), 1),
            ConvBNReLU(int(context_channels), int(detail_channels), 3),
        )
        self.context_projection = nn.Sequential(
            ConvBNReLU(dino_channels, int(context_channels), 1),
            ConvBNReLU(int(context_channels), int(context_channels), 3),
        )
        self.head = SSepNetOCRHead(
            int(detail_channels) + int(context_channels),
            channels=int(ocr_channels),
            num_classes=num_classes,
            boundary_enabled=boundary_enabled,
            prune_disabled_boundary=True,
            dropout=dropout,
        )
        if self.freeze_backbone:
            self.dinov3_backbone.requires_grad_(False)

    def _forward_dinov3(self, inputs: torch.Tensor) -> torch.Tensor:
        trainable = self.training and any(
            parameter.requires_grad for parameter in self.dinov3_backbone.parameters()
        )
        if trainable and self.activation_checkpoint:
            return activation_checkpoint_forward(self.dinov3_backbone, inputs, use_reentrant=False)
        return self.dinov3_backbone(inputs)

    def _fused_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self._forward_dinov3(inputs)
        target_size = ((int(inputs.shape[-2]) + 3) // 4, (int(inputs.shape[-1]) + 3) // 4)
        detail = F.interpolate(
            self.detail_projection(features),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        context = F.interpolate(
            self.context_projection(features),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return torch.cat((detail, context), dim=1)

    def forward(self, inputs: torch.Tensor, *, main_only: bool = False) -> dict[str, torch.Tensor]:
        fused = self._fused_features(inputs)
        outputs = {"out": self.head.forward_main(fused)} if main_only else self.head(fused)
        return _resize_named_outputs(outputs, tuple(inputs.shape[-2:]))

    def forward_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs, main_only=True)["out"]

    def train(self, mode: bool = True) -> "DINOv3OnlySSepNet":
        super().train(mode)
        if self.freeze_backbone:
            self.dinov3_backbone.eval()
        return self


class SSepNet(nn.Module):
    FUSION_MODES = frozenset({"add", "concat", "gate", "gate_concat"})
    BRANCH_ABLATIONS = frozenset({"none", "hrnet_only", "dinov3_only"})

    def __init__(
        self,
        *,
        hrnet_backbone: nn.Module,
        dinov3_backbone: nn.Module,
        fusion_mode: str = "gate_concat",
        branch_ablation: str = "none",
        boundary_enabled: bool = True,
        num_classes: int = 1,
        ocr_channels: int = 256,
        dinov3_context_channels: int = 320,
        freeze_backbones: bool = True,
        activation_checkpoint: bool = False,
        dinov3_adaptation: str = "full",
        dinov3_lora_rank: int = 0,
        dinov3_lora_alpha: float | None = None,
        prune_disabled_boundary: bool = False,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.fusion_mode = str(fusion_mode).strip().lower()
        self.branch_ablation = str(branch_ablation).strip().lower()
        self.dinov3_adaptation = str(dinov3_adaptation).strip().lower()
        self.dinov3_lora_rank = int(dinov3_lora_rank)
        self.dinov3_lora_alpha = (
            None if dinov3_lora_alpha is None else float(dinov3_lora_alpha)
        )
        self.dinov3_adaptation_report: DinoV3AdaptationReport | None = None
        if self.fusion_mode not in self.FUSION_MODES:
            raise ValueError(
                f"Unsupported fusion_mode {fusion_mode!r}; choose from {sorted(self.FUSION_MODES)}"
            )
        if self.branch_ablation not in self.BRANCH_ABLATIONS:
            raise ValueError(
                "Unsupported branch_ablation "
                f"{branch_ablation!r}; choose from {sorted(self.BRANCH_ABLATIONS)}"
            )
        if self.branch_ablation != "none" and self.fusion_mode != "gate_concat":
            raise ValueError("single-branch ablations require the gate_concat parameter layout")
        if self.branch_ablation != "none" and self.dinov3_adaptation != "full":
            raise ValueError("DINOv3 adaptation controls require the full dual-backbone path")
        if freeze_backbones and self.dinov3_adaptation != "full":
            raise ValueError(
                "targeted DINOv3 adaptation cannot be combined with freeze_backbones=True"
            )
        if num_classes != 1:
            raise ValueError("SSepNet is a binary segmentor and requires num_classes=1")

        self.hrnet_backbone = hrnet_backbone
        self.dinov3_backbone = dinov3_backbone
        self.freeze_backbones = bool(freeze_backbones)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.boundary_enabled = bool(boundary_enabled)
        self.stage_channels = tuple(int(value) for value in hrnet_backbone.stage_channels)
        if len(self.stage_channels) != 4:
            raise ValueError("SSepNet HRNet backbone must expose four stage_channels")
        high_channels = self.stage_channels[0]
        dino_channels = int(dinov3_backbone.out_channels)
        context_channels = int(dinov3_context_channels)

        self.dinov3_to_high: nn.Module | None = None
        if self.fusion_mode in {"add", "gate", "gate_concat"}:
            self.dinov3_to_high = nn.Sequential(
                ConvBNReLU(dino_channels, context_channels, 1),
                ConvBNReLU(context_channels, high_channels, 3),
            )
        self.dinov3_high_gate: nn.Module | None = None
        if self.fusion_mode in {"gate", "gate_concat"}:
            self.dinov3_high_gate = nn.Sequential(
                nn.Conv2d(high_channels * 2, high_channels, kernel_size=1),
                nn.Sigmoid(),
            )
        self.dinov3_to_head: nn.Module | None = None
        if self.fusion_mode in {"concat", "gate_concat"}:
            self.dinov3_to_head = nn.Sequential(
                ConvBNReLU(dino_channels, context_channels, 1),
                ConvBNReLU(context_channels, context_channels, 3),
            )

        fused_channels = sum(self.stage_channels)
        if self.dinov3_to_head is not None:
            fused_channels += context_channels
        self.head = SSepNetOCRHead(
            fused_channels,
            channels=int(ocr_channels),
            num_classes=num_classes,
            boundary_enabled=self.boundary_enabled,
            prune_disabled_boundary=prune_disabled_boundary,
            dropout=dropout,
        )
        if self.freeze_backbones:
            self.hrnet_backbone.requires_grad_(False)
            self.dinov3_backbone.requires_grad_(False)
        elif self.branch_ablation == "hrnet_only":
            self.dinov3_backbone.requires_grad_(False)
            for module in (self.dinov3_to_high, self.dinov3_high_gate, self.dinov3_to_head):
                if module is not None:
                    module.requires_grad_(False)
        elif self.branch_ablation == "dinov3_only":
            self.hrnet_backbone.requires_grad_(False)
        if not self.freeze_backbones and self.branch_ablation == "none":
            self.dinov3_adaptation_report = configure_dinov3_adaptation(
                self.dinov3_backbone,
                mode=self.dinov3_adaptation,
                rank=self.dinov3_lora_rank or 3,
                alpha=self.dinov3_lora_alpha,
            )

    @classmethod
    def from_variant(
        cls,
        spec: SSepNetVariantSpec,
        *,
        in_chans: int,
        num_classes: int,
        pretrained: bool,
        hrnet_checkpoint: str | Path | None,
        dinov3_checkpoint: str | Path | None,
        freeze_backbones: bool,
        activation_checkpoint: bool = False,
        dinov3_adaptation: str = "full",
        dinov3_lora_rank: int = 0,
        dinov3_lora_alpha: float | None = None,
        prune_disabled_boundary: bool = False,
    ) -> "SSepNet":
        from .dinov3 import build_dinov3_vitl16_sat493m
        from .hrnet import build_hrnetv2_w32

        return cls(
            hrnet_backbone=build_hrnetv2_w32(
                in_chans=in_chans,
                pretrained=pretrained,
                checkpoint_path=hrnet_checkpoint,
            ),
            dinov3_backbone=build_dinov3_vitl16_sat493m(
                in_chans=in_chans,
                pretrained=pretrained,
                checkpoint_path=dinov3_checkpoint,
            ),
            fusion_mode=spec.fusion_mode,
            branch_ablation=spec.branch_ablation,
            boundary_enabled=spec.boundary_enabled,
            num_classes=num_classes,
            ocr_channels=256,
            dinov3_context_channels=320,
            freeze_backbones=freeze_backbones,
            activation_checkpoint=activation_checkpoint,
            dinov3_adaptation=dinov3_adaptation,
            dinov3_lora_rank=dinov3_lora_rank,
            dinov3_lora_alpha=dinov3_lora_alpha,
            prune_disabled_boundary=prune_disabled_boundary,
        )

    def _forward_dinov3(self, inputs: torch.Tensor) -> torch.Tensor:
        trainable = self.training and any(
            parameter.requires_grad for parameter in self.dinov3_backbone.parameters()
        )
        if trainable and self.activation_checkpoint:
            return activation_checkpoint_forward(self.dinov3_backbone, inputs, use_reentrant=False)
        return self.dinov3_backbone(inputs)

    @staticmethod
    def _zero_hrnet_features(inputs: torch.Tensor, channels: tuple[int, ...]) -> list[torch.Tensor]:
        batch, _, height, width = inputs.shape
        return [
            inputs.new_zeros(
                (
                    batch,
                    stage_channels,
                    (height + stride - 1) // stride,
                    (width + stride - 1) // stride,
                )
            )
            for stage_channels, stride in zip(channels, (4, 8, 16, 32), strict=True)
        ]

    def _fused_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.branch_ablation == "dinov3_only":
            hrnet_features = self._zero_hrnet_features(inputs, self.stage_channels)
        else:
            hrnet_features = list(self.hrnet_backbone(inputs))
        high_size = hrnet_features[0].shape[-2:]

        dino_features: torch.Tensor | None = None
        if self.branch_ablation != "hrnet_only":
            dino_features = self._forward_dinov3(inputs)
            if self.fusion_mode in {"add", "gate", "gate_concat"}:
                assert self.dinov3_to_high is not None
                dino_detail = self.dinov3_to_high(dino_features)
                dino_detail = F.interpolate(
                    dino_detail,
                    size=high_size,
                    mode="bilinear",
                    align_corners=False,
                )
                if self.fusion_mode == "add":
                    hrnet_features[0] = hrnet_features[0] + dino_detail
                else:
                    assert self.dinov3_high_gate is not None
                    gate = self.dinov3_high_gate(torch.cat((hrnet_features[0], dino_detail), dim=1))
                    hrnet_features[0] = hrnet_features[0] + gate * dino_detail

        fused = [hrnet_features[0]]
        fused.extend(
            F.interpolate(feature, size=high_size, mode="bilinear", align_corners=False)
            for feature in hrnet_features[1:]
        )
        if self.dinov3_to_head is not None:
            if dino_features is None:
                context_channels = self.dinov3_to_head[-1][0].out_channels
                fused.append(inputs.new_zeros((inputs.shape[0], context_channels, *high_size)))
            else:
                context = self.dinov3_to_head(dino_features)
                fused.append(F.interpolate(context, size=high_size, mode="bilinear", align_corners=False))
        return torch.cat(fused, dim=1)

    def forward(self, inputs: torch.Tensor, *, main_only: bool = False) -> dict[str, torch.Tensor]:
        input_size = inputs.shape[-2:]
        fused = self._fused_features(inputs)
        outputs = {"out": self.head.forward_main(fused)} if main_only else self.head(fused)
        return _resize_named_outputs(outputs, tuple(input_size))

    def forward_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs, main_only=True)["out"]

    def train(self, mode: bool = True) -> "SSepNet":
        super().train(mode)
        if self.freeze_backbones:
            self.hrnet_backbone.eval()
            self.dinov3_backbone.eval()
        elif self.dinov3_adaptation in {"frozen", "lora"}:
            self.dinov3_backbone.eval()
        return self
