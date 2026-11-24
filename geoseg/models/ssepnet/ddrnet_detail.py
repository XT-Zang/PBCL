from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .detail_pretrained import load_pruned_backbone_checkpoint
from .multiscale import ConvGNReLU


_BN_MOMENTUM = 0.1


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class _DDRBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        no_relu: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels, momentum=_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels, momentum=_BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = bool(no_relu)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.downsample is None else self.downsample(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs)) + residual
        return outputs if self.no_relu else self.relu(outputs)


class _DDRBottleneck(nn.Module):
    expansion = 2

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        no_relu: bool = True,
    ) -> None:
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=_BN_MOMENTUM)
        self.conv2 = _conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels, momentum=_BN_MOMENTUM)
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = bool(no_relu)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.downsample is None else self.downsample(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.relu(self.bn2(self.conv2(outputs)))
        outputs = self.bn3(self.conv3(outputs)) + residual
        return outputs if self.no_relu else self.relu(outputs)


class _DDRNet23SlimPrunedEncoder(nn.Module):
    """Official DDRNet-23-slim pruned after its final high-resolution block."""

    def __init__(self, in_chans: int = 4) -> None:
        super().__init__()
        if in_chans not in {3, 4}:
            raise ValueError("DDRNet-23-slim detail backbone requires 3 or 4 channels")
        planes = 32
        highres_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(_DDRBasicBlock, planes, planes, 2)
        self.layer2 = self._make_layer(
            _DDRBasicBlock, planes, planes * 2, 2, stride=2
        )
        self.layer3 = self._make_layer(
            _DDRBasicBlock, planes * 2, planes * 4, 2, stride=2
        )
        self.layer4 = self._make_layer(
            _DDRBasicBlock, planes * 4, planes * 8, 2, stride=2
        )
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, highres_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_channels, momentum=_BN_MOMENTUM),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, highres_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_channels, momentum=_BN_MOMENTUM),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(
                highres_channels,
                planes * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(planes * 4, momentum=_BN_MOMENTUM),
        )
        self.layer3_ = self._make_layer(
            _DDRBasicBlock, planes * 2, highres_channels, 2
        )
        self.layer4_ = self._make_layer(
            _DDRBasicBlock, highres_channels, highres_channels, 2
        )
        self.layer5_ = self._make_layer(
            _DDRBottleneck, highres_channels, highres_channels, 1
        )
        self._initialize()

    @staticmethod
    def _make_layer(
        block: type[_DDRBasicBlock] | type[_DDRBottleneck],
        in_channels: int,
        channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    channels * block.expansion, momentum=_BN_MOMENTUM
                ),
            )
        layers: list[nn.Module] = [
            block(in_channels, channels, stride, downsample)
        ]
        current_channels = channels * block.expansion
        for index in range(1, blocks):
            layers.append(
                block(
                    current_channels,
                    channels,
                    no_relu=index == blocks - 1,
                )
            )
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.conv1(inputs)
        stride4 = self.layer1(outputs)
        stride8_low = self.layer2(self.relu(stride4))

        low16_raw = self.layer3(self.relu(stride8_low))
        high8 = self.layer3_(self.relu(stride8_low))
        low16 = low16_raw + self.down3(self.relu(high8))
        high8 = high8 + F.interpolate(
            self.compression3(self.relu(low16_raw)),
            size=high8.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        low32_raw = self.layer4(self.relu(low16))
        high8 = self.layer4_(self.relu(high8))
        high8 = high8 + F.interpolate(
            self.compression4(self.relu(low32_raw)),
            size=high8.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high8 = self.layer5_(self.relu(high8))
        return stride4, high8


class DDRNet23SlimDetailBackbone(nn.Module):
    detail_channels = (128, 128)
    pretrained_key = "ddrnet23-slim-imagenet"

    def __init__(
        self,
        *,
        in_chans: int = 4,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.encoder = _DDRNet23SlimPrunedEncoder(in_chans=in_chans)
        self.stride8_projection = ConvGNReLU(128, 128, 1)
        self.stride4_projection = ConvGNReLU(32, 128, 1)
        self.stride4_refine = ConvGNReLU(256, 128, 3)
        self.pretrained_report: dict[str, list[str]] | None = None
        if checkpoint_path is not None:
            self.pretrained_report = load_pruned_backbone_checkpoint(
                self.encoder,
                checkpoint_path,
                stem_key="conv1.0.weight",
                ignored_prefixes=(
                    "layer5.",
                    "down4.",
                    "spp.",
                    "seghead_extra.",
                    "final_layer.",
                    "classifier.",
                    "fc.",
                    "down5.",
                    "last_layer.",
                    "linear.",
                ),
            )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shallow, high = self.encoder(inputs)
        stride8 = self.stride8_projection(high)
        stride4 = self.stride4_refine(
            torch.cat(
                (
                    self.stride4_projection(shallow),
                    F.interpolate(
                        stride8,
                        size=shallow.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                ),
                dim=1,
            )
        )
        return stride4, stride8


def build_ddrnet23_slim_detail(
    *,
    in_chans: int = 4,
    pretrained: bool = True,
    checkpoint_path: str | Path | None = None,
) -> DDRNet23SlimDetailBackbone:
    if pretrained:
        if checkpoint_path is None:
            from geoseg.pretrained import resolve_pretrained_model_path

            checkpoint_path = resolve_pretrained_model_path(
                DDRNet23SlimDetailBackbone.pretrained_key
            )
    elif checkpoint_path is not None:
        raise ValueError("checkpoint_path requires pretrained=True")
    return DDRNet23SlimDetailBackbone(
        in_chans=in_chans,
        checkpoint_path=checkpoint_path,
    )


__all__ = ["DDRNet23SlimDetailBackbone", "build_ddrnet23_slim_detail"]
