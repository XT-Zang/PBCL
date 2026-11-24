from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            _conv3x3(in_channels, out_channels, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(
                _conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.relu(outputs + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = _conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = _conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = _conv1x1(channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(
                _conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.relu(self.bn2(self.conv2(outputs)))
        outputs = self.bn3(self.conv3(outputs))
        return self.relu(outputs + residual)


def _make_layer(
    block: type[BasicBlock] | type[Bottleneck],
    in_channels: int,
    channels: int,
    blocks: int,
) -> nn.Sequential:
    layers: list[nn.Module] = [block(in_channels, channels)]
    current_channels = channels * block.expansion
    layers.extend(block(current_channels, channels) for _ in range(1, blocks))
    return nn.Sequential(*layers)


class HighResolutionModule(nn.Module):
    def __init__(self, channels: tuple[int, ...], num_blocks: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_branches = len(channels)
        self.branches = nn.ModuleList(
            [_make_layer(BasicBlock, value, value, num_blocks) for value in channels]
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_fuse_layers(self) -> nn.ModuleList:
        all_layers: list[nn.ModuleList] = []
        for target_index, target_channels in enumerate(self.channels):
            target_layers: list[nn.Module] = []
            for source_index, source_channels in enumerate(self.channels):
                if source_index == target_index:
                    target_layers.append(nn.Identity())
                elif source_index > target_index:
                    target_layers.append(
                        nn.Sequential(
                            _conv1x1(source_channels, target_channels),
                            nn.BatchNorm2d(target_channels),
                        )
                    )
                else:
                    downsample_layers: list[nn.Module] = []
                    in_channels = source_channels
                    for step in range(target_index - source_index):
                        is_last = step == target_index - source_index - 1
                        out_channels = target_channels if is_last else in_channels
                        operations: list[nn.Module] = [
                            _conv3x3(in_channels, out_channels, stride=2),
                            nn.BatchNorm2d(out_channels),
                        ]
                        if not is_last:
                            operations.append(nn.ReLU(inplace=True))
                        downsample_layers.append(nn.Sequential(*operations))
                        in_channels = out_channels
                    target_layers.append(nn.Sequential(*downsample_layers))
            all_layers.append(nn.ModuleList(target_layers))
        return nn.ModuleList(all_layers)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        branch_outputs = [branch(inputs[index]) for index, branch in enumerate(self.branches)]
        fused_outputs: list[torch.Tensor] = []
        for target_index in range(self.num_branches):
            fused = self.fuse_layers[target_index][0](branch_outputs[0])
            for source_index in range(1, self.num_branches):
                contribution = self.fuse_layers[target_index][source_index](
                    branch_outputs[source_index]
                )
                if source_index > target_index:
                    contribution = F.interpolate(
                        contribution,
                        size=branch_outputs[target_index].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                fused = fused + contribution
            fused_outputs.append(self.relu(fused))
        return fused_outputs


class HRNetV2W32(nn.Module):
    """Canonical HRNetV2-W32 feature backbone used by SSepNet."""

    stage_channels = (32, 64, 128, 256)

    def __init__(self, in_chans: int = 4) -> None:
        super().__init__()
        if in_chans not in {3, 4}:
            raise ValueError("SSepNet HRNet requires 3-channel RGB or 4-channel B/G/R/NIR input")
        self.conv1 = _conv3x3(in_chans, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = _conv3x3(64, 64, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = _make_layer(Bottleneck, 64, 64, 4)

        self.transition1 = self._make_transition((256,), self.stage_channels[:2])
        self.stage2 = self._make_stage(self.stage_channels[:2], modules=1)
        self.transition2 = self._make_transition(self.stage_channels[:2], self.stage_channels[:3])
        self.stage3 = self._make_stage(self.stage_channels[:3], modules=4)
        self.transition3 = self._make_transition(self.stage_channels[:3], self.stage_channels)
        self.stage4 = self._make_stage(self.stage_channels, modules=3)

    @staticmethod
    def _make_stage(channels: tuple[int, ...], modules: int) -> nn.Sequential:
        return nn.Sequential(*(HighResolutionModule(channels, num_blocks=4) for _ in range(modules)))

    @staticmethod
    def _make_transition(
        previous_channels: tuple[int, ...],
        current_channels: tuple[int, ...],
    ) -> nn.ModuleList:
        layers: list[nn.Module] = []
        for index, out_channels in enumerate(current_channels):
            if index < len(previous_channels):
                in_channels = previous_channels[index]
                layers.append(
                    _ConvBNReLU(in_channels, out_channels)
                    if in_channels != out_channels
                    else nn.Identity()
                )
                continue
            operations: list[nn.Module] = []
            in_channels = previous_channels[-1]
            for step in range(index + 1 - len(previous_channels)):
                last_step = step == index - len(previous_channels)
                next_channels = out_channels if last_step else in_channels
                operations.append(_ConvBNReLU(in_channels, next_channels, stride=2))
                in_channels = next_channels
            layers.append(nn.Sequential(*operations))
        return nn.ModuleList(layers)

    @staticmethod
    def _apply_transition(
        inputs: list[torch.Tensor],
        transition: nn.ModuleList,
    ) -> list[torch.Tensor]:
        return [
            layer(inputs[index] if index < len(inputs) else inputs[-1])
            for index, layer in enumerate(transition)
        ]

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.relu(self.bn2(self.conv2(outputs)))
        outputs = self.layer1(outputs)
        stage_inputs = self._apply_transition([outputs], self.transition1)
        stage_outputs = self.stage2(stage_inputs)
        stage_inputs = self._apply_transition(stage_outputs, self.transition2)
        stage_outputs = self.stage3(stage_inputs)
        stage_inputs = self._apply_transition(stage_outputs, self.transition3)
        return list(self.stage4(stage_inputs))


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("HRNetV2 checkpoint must contain a state-dict mapping")
    for key in ("state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def inflate_hrnet_stem_weight(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.ndim != 4 or target.ndim != 4:
        raise RuntimeError("HRNet conv1 weights must be four-dimensional")
    if source.shape[0] != target.shape[0] or source.shape[2:] != target.shape[2:]:
        raise RuntimeError(
            f"HRNet conv1 shape mismatch: checkpoint={tuple(source.shape)} model={tuple(target.shape)}"
        )
    if source.shape[1] == target.shape[1]:
        return source
    if source.shape[1] != 3 or target.shape[1] != 4:
        raise RuntimeError(
            f"Cannot adapt HRNet conv1 from {source.shape[1]} to {target.shape[1]} channels"
        )
    return torch.cat((source, source.mean(dim=1, keepdim=True)), dim=1)


def load_hrnetv2_w32_checkpoint(
    backbone: HRNetV2W32,
    checkpoint_path: str | Path,
) -> dict[str, list[str]]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HRNetV2-W32 pretrained checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    source = {
        str(key).removeprefix("module."): value
        for key, value in _checkpoint_state_dict(checkpoint).items()
        if isinstance(value, torch.Tensor)
    }
    target = backbone.state_dict()
    ignored_prefixes = (
        "classifier.",
        "head.",
        "fc.",
        "last_layer.",
        "incre_modules.",
        "downsamp_modules.",
        "final_layer.",
    )
    loadable: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    unexpected: list[str] = []
    for key, value in source.items():
        if key in target:
            candidate = inflate_hrnet_stem_weight(value, target[key]) if key == "conv1.weight" else value
            if candidate.shape != target[key].shape:
                raise RuntimeError(
                    f"HRNetV2-W32 shape mismatch for {key}: "
                    f"checkpoint={tuple(candidate.shape)} model={tuple(target[key].shape)}"
                )
            loadable[key] = candidate
        elif key.startswith(ignored_prefixes):
            ignored.append(key)
        else:
            unexpected.append(key)
    missing = sorted(set(target) - set(loadable))
    if unexpected:
        raise RuntimeError("unexpected HRNetV2-W32 checkpoint keys: " + ", ".join(sorted(unexpected)))
    if missing:
        raise RuntimeError("missing HRNetV2-W32 backbone keys: " + ", ".join(missing))
    backbone.load_state_dict(loadable, strict=True)
    return {"loaded": sorted(loadable), "ignored": sorted(ignored), "missing": [], "unexpected": []}


def build_hrnetv2_w32(
    *,
    in_chans: int = 4,
    pretrained: bool = True,
    checkpoint_path: str | Path | None = None,
) -> HRNetV2W32:
    if in_chans not in {3, 4}:
        raise ValueError("SSepNet HRNet requires 3-channel RGB or 4-channel B/G/R/NIR input")
    backbone = HRNetV2W32(in_chans=in_chans)
    if pretrained:
        if checkpoint_path is None:
            from geoseg.pretrained import resolve_pretrained_model_path

            checkpoint_path = resolve_pretrained_model_path("hrnetv2-w32-imagenet")
        load_hrnetv2_w32_checkpoint(backbone, checkpoint_path)
    elif checkpoint_path is not None:
        raise ValueError("checkpoint_path requires pretrained=True")
    return backbone
