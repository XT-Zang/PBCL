from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aquav3_outputs import get_main_logits


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class StructureMultiHeadWrapper(nn.Module):
    """Add full-structure auxiliary heads around an arbitrary binary segmentor."""

    def __init__(
        self,
        base_model: nn.Module,
        in_chans: int = 4,
        num_classes: int = 1,
        structure_channels: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if in_chans <= 0:
            raise ValueError("in_chans must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if structure_channels <= 0:
            raise ValueError("structure_channels must be positive")
        self.base_model = base_model
        self.in_chans = int(in_chans)
        self.num_classes = int(num_classes)
        self.structure_channels = int(structure_channels)
        self.feature_adapter = nn.Sequential(
            ConvBNReLU(self.in_chans + self.num_classes, self.structure_channels, 3),
            ConvBNReLU(self.structure_channels, self.structure_channels, 3),
            nn.Dropout2d(float(dropout)),
        )
        self.aux_head = nn.Conv2d(self.structure_channels, self.num_classes, kernel_size=1)
        self.edge_head = nn.Sequential(
            ConvBNReLU(self.structure_channels, self.structure_channels, 3),
            nn.Conv2d(self.structure_channels, 1, kernel_size=1),
        )
        self.skeleton_head = nn.Sequential(
            ConvBNReLU(self.structure_channels, self.structure_channels, 3),
            ConvBNReLU(self.structure_channels, self.structure_channels, 3),
            nn.Conv2d(self.structure_channels, 1, kernel_size=1),
        )
        self.distance_head = nn.Sequential(
            ConvBNReLU(self.structure_channels, self.structure_channels, 3),
            ConvBNReLU(self.structure_channels, self.structure_channels, 3),
            nn.Conv2d(self.structure_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        base_outputs = self.base_model(x)
        main = get_main_logits(base_outputs)
        main_features = main
        if main_features.shape[-2:] != x.shape[-2:]:
            main_features = F.interpolate(main_features, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if main_features.shape[1] != self.num_classes:
            main_features = main_features[:, : self.num_classes]
        features = self.feature_adapter(torch.cat([x, main_features], dim=1))
        outputs = {
            "out": main,
            "aux": self.aux_head(features),
            "edge": self.edge_head(features),
            "skeleton": self.skeleton_head(features),
            "distance": self.distance_head(features),
        }
        if isinstance(base_outputs, Mapping):
            for key, value in base_outputs.items():
                outputs.setdefault(key, value)
        return outputs
