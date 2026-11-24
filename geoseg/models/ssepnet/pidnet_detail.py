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


class _PIDBasicBlock(nn.Module):
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


class _PIDBottleneck(nn.Module):
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


class _PagFM(nn.Module):
    def __init__(self, channels: int, mid_channels: int) -> None:
        super().__init__()
        self.f_x = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )

    def forward(self, detail: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context_query = F.interpolate(
            self.f_y(context),
            size=detail.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        detail_key = self.f_x(detail)
        similarity = torch.sigmoid(
            torch.sum(detail_key * context_query, dim=1, keepdim=True)
        )
        context = F.interpolate(
            context,
            size=detail.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return (1.0 - similarity) * detail + similarity * context


class _PIDNetSPrunedEncoder(nn.Module):
    """Official PIDNet-S trunk pruned after its P/D feature extractors."""

    def __init__(self, in_chans: int = 4) -> None:
        super().__init__()
        if in_chans not in {3, 4}:
            raise ValueError("PIDNet-S detail backbone requires 3 or 4 input channels")
        planes = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(_PIDBasicBlock, planes, planes, 2)
        self.layer2 = self._make_layer(
            _PIDBasicBlock, planes, planes * 2, 2, stride=2
        )
        self.layer3 = self._make_layer(
            _PIDBasicBlock, planes * 2, planes * 4, 3, stride=2
        )
        self.layer4 = self._make_layer(
            _PIDBasicBlock, planes * 4, planes * 8, 3, stride=2
        )

        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=_BN_MOMENTUM),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=_BN_MOMENTUM),
        )
        self.pag3 = _PagFM(planes * 2, planes)
        self.pag4 = _PagFM(planes * 2, planes)
        self.layer3_ = self._make_layer(
            _PIDBasicBlock, planes * 2, planes * 2, 2
        )
        self.layer4_ = self._make_layer(
            _PIDBasicBlock, planes * 2, planes * 2, 2
        )
        self.layer5_ = self._make_layer(
            _PIDBottleneck, planes * 2, planes * 2, 1
        )

        self.layer3_d = self._make_single_layer(
            _PIDBasicBlock, planes * 2, planes
        )
        self.layer4_d = self._make_layer(_PIDBottleneck, planes, planes, 1)
        self.diff3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM),
        )
        self.diff4 = nn.Sequential(
            nn.Conv2d(
                planes * 8,
                planes * 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(planes * 2, momentum=_BN_MOMENTUM),
        )
        self.layer5_d = self._make_layer(
            _PIDBottleneck, planes * 2, planes * 2, 1
        )
        self._initialize()

    @staticmethod
    def _make_layer(
        block: type[_PIDBasicBlock] | type[_PIDBottleneck],
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

    @staticmethod
    def _make_single_layer(
        block: type[_PIDBasicBlock],
        in_channels: int,
        channels: int,
    ) -> nn.Module:
        downsample: nn.Module | None = None
        if in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    channels * block.expansion,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    channels * block.expansion, momentum=_BN_MOMENTUM
                ),
            )
        return block(in_channels, channels, downsample=downsample, no_relu=True)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.conv1(inputs)
        stride4 = self.layer1(shared)
        context = self.relu(self.layer2(self.relu(stride4)))
        detail = self.layer3_(context)
        derivative = self.layer3_d(context)

        context = self.relu(self.layer3(context))
        detail = self.pag3(detail, self.compression3(context))
        derivative = derivative + F.interpolate(
            self.diff3(context),
            size=derivative.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        context = self.relu(self.layer4(context))
        detail = self.layer4_(self.relu(detail))
        derivative = self.layer4_d(self.relu(derivative))
        detail = self.pag4(detail, self.compression4(context))
        derivative = derivative + F.interpolate(
            self.diff4(context),
            size=derivative.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        detail = self.layer5_(self.relu(detail))
        derivative = self.layer5_d(self.relu(derivative))
        return stride4, detail, derivative


class PIDNetSDetailBackbone(nn.Module):
    detail_channels = (128, 128)
    pretrained_key = "pidnet-s-imagenet"

    def __init__(
        self,
        *,
        in_chans: int = 4,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.encoder = _PIDNetSPrunedEncoder(in_chans=in_chans)
        self.stride8_refine = ConvGNReLU(256, 128, 3)
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
                    "spp.",
                    "dfm.",
                    "seghead_p.",
                    "seghead_d.",
                    "final_layer.",
                    "classifier.",
                    "fc.",
                    "down5.",
                    "last_layer.",
                    "linear.",
                ),
                allowed_missing_prefixes=(
                    "layer3_d.",
                    "layer4_d.",
                    "layer5_d.",
                    "diff3.",
                    "diff4.",
                ),
            )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shallow, proportional, derivative = self.encoder(inputs)
        stride8 = self.stride8_refine(
            torch.cat((proportional, derivative), dim=1)
        )
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


def build_pidnet_s_detail(
    *,
    in_chans: int = 4,
    pretrained: bool = True,
    checkpoint_path: str | Path | None = None,
) -> PIDNetSDetailBackbone:
    if pretrained:
        if checkpoint_path is None:
            from geoseg.pretrained import resolve_pretrained_model_path

            checkpoint_path = resolve_pretrained_model_path(
                PIDNetSDetailBackbone.pretrained_key
            )
    elif checkpoint_path is not None:
        raise ValueError("checkpoint_path requires pretrained=True")
    return PIDNetSDetailBackbone(
        in_chans=in_chans,
        checkpoint_path=checkpoint_path,
    )


__all__ = ["PIDNetSDetailBackbone", "build_pidnet_s_detail"]
