from __future__ import annotations

import torch
import torch.nn as nn

from .aquav3_structure import ConvBNReLU


def build_boundary_head(in_channels: int, *, downsample: int = 1) -> nn.Sequential:
    layers = []
    if downsample > 1:
        layers.append(nn.AvgPool2d(downsample, stride=downsample))
    layers.extend((ConvBNReLU(in_channels, 32), nn.Conv2d(32, 1, kernel_size=1)))
    return nn.Sequential(*layers)


class SMPBoundaryModel(nn.Module):
    """Supervise the final decoder feature of a binary SMP model."""

    def __init__(self, base_model: nn.Module, *, downsample: int) -> None:
        super().__init__()
        self.base_model = base_model
        channels = base_model.segmentation_head[0].in_channels
        self.boundary_head = build_boundary_head(channels, downsample=downsample)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        model = self.base_model
        model.check_input_shape(image)
        features = model.decoder(model.encoder(image))
        return {"out": model.segmentation_head(features), "edge": self.boundary_head(features)}


def add_external_boundary_head(model: nn.Module, model_key: str) -> nn.Module:
    if model_key in {"unetplusplus", "deeplabv3plus"}:
        return SMPBoundaryModel(model, downsample=2 if model_key == "unetplusplus" else 1)
    if model_key in {"unetformer", "dcswin_base"}:
        channels = model.decoder.segmentation_head[0][0].in_channels
        model.decoder.boundary_head = build_boundary_head(channels)
    elif model_key == "segformer_b2":
        model.boundary_head = build_boundary_head(model.model.config.hidden_sizes[0])
    elif model_key == "mask2former_swin_t":
        model.boundary_head = build_boundary_head(model.model.config.mask_feature_size)
    elif model_key == "vmamba_fpn":
        model.boundary_head = build_boundary_head(model.fpn_channels)
    else:
        raise ValueError(f"No boundary feature specified for {model_key!r}")
    return model
