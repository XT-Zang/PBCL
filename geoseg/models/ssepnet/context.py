from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from geoseg.losses.ssepnet import SSepNetLoss

from .common import ConvBNReLU
from .ocr import SSepNetOCRHead


class HRNetContextOCR(nn.Module):
    """Gate+Concat HRNet-context binary OCR model with training-only Boundary output."""

    def __init__(
        self,
        *,
        hrnet_backbone: nn.Module,
        context_backbone: nn.Module,
        freeze_backbones: bool = False,
        activation_checkpoint: bool = True,
        num_classes: int = 1,
        context_channels: int = 320,
        ocr_channels: int = 256,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("HRNetContextOCR requires binary num_classes=1")
        self.hrnet_backbone = hrnet_backbone
        self.context_backbone = context_backbone
        self.freeze_backbones = bool(freeze_backbones)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.stage_channels = tuple(int(value) for value in hrnet_backbone.stage_channels)
        if self.stage_channels != (32, 64, 128, 256):
            raise ValueError(f"HRNetContextOCR requires HRNetV2-W32 channels, got {self.stage_channels}")
        high_channels = self.stage_channels[0]
        context_in = int(context_backbone.out_channels)
        self.context_to_high = nn.Sequential(
            ConvBNReLU(context_in, context_channels, 1),
            ConvBNReLU(context_channels, high_channels, 3),
        )
        self.context_gate = nn.Sequential(
            nn.Conv2d(high_channels * 2, high_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.context_to_head = nn.Sequential(
            ConvBNReLU(context_in, context_channels, 1),
            ConvBNReLU(context_channels, context_channels, 3),
        )
        self.head = SSepNetOCRHead(
            sum(self.stage_channels) + context_channels,
            channels=ocr_channels,
            num_classes=1,
            boundary_enabled=True,
            dropout=dropout,
        )
        if self.freeze_backbones:
            self.hrnet_backbone.requires_grad_(False)
            self.context_backbone.requires_grad_(False)

    def _context_features(self, inputs: torch.Tensor) -> torch.Tensor:
        trainable = self.training and any(
            parameter.requires_grad for parameter in self.context_backbone.parameters()
        )
        if trainable and self.activation_checkpoint:
            return activation_checkpoint(self.context_backbone, inputs, use_reentrant=False)
        return self.context_backbone(inputs)

    def _fused_features(self, inputs: torch.Tensor) -> torch.Tensor:
        hrnet_features = list(self.hrnet_backbone(inputs))
        high_size = hrnet_features[0].shape[-2:]
        context = self._context_features(inputs)
        detail = self.context_to_high[0](context)
        detail = F.interpolate(detail, size=high_size, mode="bilinear", align_corners=False)
        detail = self.context_to_high[1](detail)
        gate = self.context_gate(torch.cat((hrnet_features[0], detail), dim=1))
        hrnet_features[0] = hrnet_features[0] + gate * detail
        fused = [hrnet_features[0]]
        fused.extend(
            F.interpolate(feature, size=high_size, mode="bilinear", align_corners=False)
            for feature in hrnet_features[1:]
        )
        context_head = self.context_to_head[0](context)
        context_head = F.interpolate(context_head, size=high_size, mode="bilinear", align_corners=False)
        fused.append(self.context_to_head[1](context_head))
        return torch.cat(fused, dim=1)

    def forward(self, inputs: torch.Tensor, *, main_only: bool = False) -> dict[str, torch.Tensor]:
        input_size = inputs.shape[-2:]
        fused = self._fused_features(inputs)
        outputs = {"out": self.head.forward_main(fused)} if main_only else self.head(fused)
        return {
            name: F.interpolate(value, size=input_size, mode="bilinear", align_corners=False)
            for name, value in outputs.items()
        }

    def forward_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs, main_only=True)["out"]

    def train(self, mode: bool = True) -> "HRNetContextOCR":
        super().train(mode)
        if self.freeze_backbones:
            self.hrnet_backbone.eval()
            self.context_backbone.eval()
        return self


def build_context_hybrid_loss() -> SSepNetLoss:
    return SSepNetLoss(boundary_weight=0.6, pbcl_weight=0.1)


def build_hrnet_context_variant(
    name: str,
    *,
    pretrained: bool = True,
    freeze_backbones: bool = False,
    activation_checkpoint: bool = True,
) -> HRNetContextOCR:
    from .context_backbones import build_context_backbone
    from .hrnet import build_hrnetv2_w32

    return HRNetContextOCR(
        hrnet_backbone=build_hrnetv2_w32(in_chans=4, pretrained=pretrained),
        context_backbone=build_context_backbone(name, pretrained=pretrained),
        freeze_backbones=freeze_backbones,
        activation_checkpoint=activation_checkpoint,
    )
