from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBNReLU


class OCRGather(nn.Module):
    def forward(self, features: torch.Tensor, auxiliary_logits: torch.Tensor) -> torch.Tensor:
        foreground = torch.sigmoid(auxiliary_logits).flatten(2)
        probabilities = torch.cat((1.0 - foreground, foreground), dim=1)
        probabilities = probabilities / probabilities.sum(dim=2, keepdim=True).clamp_min(1e-6)
        flattened_features = features.flatten(2).transpose(1, 2)
        context = torch.bmm(probabilities, flattened_features)
        return context.transpose(1, 2).unsqueeze(-1)


class ObjectAttentionBlock(nn.Module):
    def __init__(self, channels: int, key_channels: int) -> None:
        super().__init__()
        self.query = nn.Conv2d(channels, key_channels, kernel_size=1, bias=False)
        self.key = nn.Conv2d(channels, key_channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.project = ConvBNReLU(channels, channels, 1)
        self.key_channels = int(key_channels)

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        query = self.query(features).flatten(2).transpose(1, 2)
        key = self.key(context).flatten(2)
        value = self.value(context).flatten(2).transpose(1, 2)
        attention = F.softmax(torch.bmm(query, key) / math.sqrt(self.key_channels), dim=-1)
        pixels = torch.bmm(attention, value).transpose(1, 2).reshape(batch, channels, height, width)
        return self.project(pixels)


class SSepNetOCRHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int = 256,
        num_classes: int = 1,
        boundary_enabled: bool = True,
        prune_disabled_boundary: bool = False,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("SSepNet is a binary segmentor and requires num_classes=1")
        self.in_channels = int(in_channels)
        self.boundary_enabled = bool(boundary_enabled)
        self.shared = ConvBNReLU(self.in_channels, channels, 3)
        self.auxiliary = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, num_classes, kernel_size=1),
        )
        self.gather = OCRGather()
        self.object_attention = ObjectAttentionBlock(channels, max(32, channels // 2))
        self.bottleneck = nn.Sequential(
            ConvBNReLU(channels * 2, channels, 3),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.boundary: nn.Module | None
        if self.boundary_enabled or not bool(prune_disabled_boundary):
            self.boundary = nn.Sequential(
                ConvBNReLU(channels, max(32, channels // 2), 3),
                nn.Conv2d(max(32, channels // 2), 1, kernel_size=1),
            )
            if not self.boundary_enabled:
                self.boundary.requires_grad_(False)
        else:
            self.boundary = None

    def _shared_forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared(inputs)
        auxiliary_logits = self.auxiliary(features)
        object_context = self.object_attention(features, self.gather(features, auxiliary_logits))
        refined = self.bottleneck(torch.cat((features, object_context), dim=1))
        return features, auxiliary_logits, refined

    def forward_main(self, inputs: torch.Tensor) -> torch.Tensor:
        _, _, refined = self._shared_forward(inputs)
        return self.classifier(refined)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features, auxiliary_logits, refined = self._shared_forward(inputs)
        outputs = {
            "out": self.classifier(refined),
            "aux": auxiliary_logits,
        }
        if self.boundary_enabled:
            assert self.boundary is not None
            outputs["edge"] = self.boundary(features)
        return outputs
