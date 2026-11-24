from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .detail_pretrained import load_pruned_backbone_checkpoint
from .multiscale import ConvGNReLU


_BN_MOMENTUM = 0.1


def _convert_hrformer_checkpoint(
    source: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        if key.endswith(".attn.attn.qkv.weight"):
            prefix = key.removesuffix("qkv.weight")
            query, key_weight, value_weight = value.chunk(3, dim=0)
            converted[prefix + "q_proj.weight"] = query
            converted[prefix + "k_proj.weight"] = key_weight
            converted[prefix + "v_proj.weight"] = value_weight
            continue
        if key.endswith(".attn.attn.qkv.bias"):
            prefix = key.removesuffix("qkv.bias")
            query, key_bias, value_bias = value.chunk(3, dim=0)
            converted[prefix + "q_proj.bias"] = query
            converted[prefix + "k_proj.bias"] = key_bias
            converted[prefix + "v_proj.bias"] = value_bias
            continue
        normalized = key.replace(".attn.attn.proj.", ".attn.attn.out_proj.")
        normalized = normalized.replace(".ffn.", ".mlp.")
        converted[normalized] = value
    return converted


class _HRTBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=_BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels, momentum=_BN_MOMENTUM)
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.downsample is None else self.downsample(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.relu(self.bn2(self.conv2(outputs)))
        outputs = self.bn3(self.conv3(outputs))
        return self.relu(outputs + residual)


class _WindowMultiheadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("HRFormer channels must be divisible by attention heads")
        self.embed_dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.window_size = [int(window_size), int(window_size)]
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        table_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads)
        )
        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        flattened = coordinates.flatten(1)
        relative = flattened[:, :, None] - flattened[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        self.register_buffer(
            "relative_position_index",
            relative.sum(-1),
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = inputs.shape
        query = self.q_proj(inputs).reshape(
            batch, tokens, self.num_heads, self.head_dim
        )
        key = self.k_proj(inputs).reshape(
            batch, tokens, self.num_heads, self.head_dim
        )
        value = self.v_proj(inputs).reshape(
            batch, tokens, self.num_heads, self.head_dim
        )
        query = query.permute(0, 2, 1, 3) * self.scale
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attention = query @ key.transpose(-2, -1)
        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        relative_bias = relative_bias.reshape(tokens, tokens, self.num_heads)
        attention = attention + relative_bias.permute(2, 0, 1).unsqueeze(0)
        attention = attention.softmax(dim=-1)
        outputs = attention @ value
        outputs = outputs.transpose(1, 2).reshape(batch, tokens, channels)
        return self.out_proj(outputs)


class _LocalWindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 7) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.attn_type = "isa_local"
        self.with_rpe = True
        self.attn = _WindowMultiheadAttention(dim, num_heads, window_size)

    def forward(
        self,
        inputs: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch, tokens, channels = inputs.shape
        if tokens != height * width:
            raise ValueError("HRFormer token count does not match feature size")
        features = inputs.reshape(batch, height, width, channels)
        window = self.window_size
        padded_height = math.ceil(height / window) * window
        padded_width = math.ceil(width / window) * window
        pad_height = padded_height - height
        pad_width = padded_width - width
        pad_top = pad_height // 2
        pad_left = pad_width // 2
        features = F.pad(
            features,
            (
                0,
                0,
                pad_left,
                pad_width - pad_left,
                pad_top,
                pad_height - pad_top,
            ),
        )
        rows = padded_height // window
        columns = padded_width // window
        windows = (
            features.reshape(batch, rows, window, columns, window, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch * rows * columns, window * window, channels)
        )
        windows = self.attn(windows)
        features = (
            windows.reshape(batch, rows, columns, window, window, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, padded_height, padded_width, channels)
        )
        return features[
            :,
            pad_top : pad_top + height,
            pad_left : pad_left + width,
        ].reshape(batch, tokens, channels)


class _HRTMlpDWBN(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.act1 = nn.GELU()
        self.norm1 = nn.BatchNorm2d(hidden_channels)
        self.dw3x3 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.act2 = nn.GELU()
        self.norm2 = nn.BatchNorm2d(hidden_channels)
        self.fc2 = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.act3 = nn.GELU()
        self.norm3 = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout(0.0)

    def forward(
        self,
        inputs: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch, tokens, channels = inputs.shape
        features = inputs.permute(0, 2, 1).reshape(
            batch, channels, height, width
        )
        features = self.act1(self.norm1(self.fc1(features)))
        features = self.act2(self.norm2(self.dw3x3(features)))
        features = self.drop(features)
        features = self.act3(self.norm3(self.fc2(features)))
        features = self.drop(features)
        return features.flatten(2).transpose(1, 2)


class _HRTBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = int(channels)
        self.out_dim = int(channels)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.attn_type = "isa_local"
        self.ffn_type = "conv_mlp"
        self.mlp_ratio = float(mlp_ratio)
        self.attn = _LocalWindowAttention(channels, num_heads, window_size)
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.drop_path = nn.Identity()
        self.mlp = _HRTMlpDWBN(channels, int(channels * mlp_ratio))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        tokens = inputs.flatten(2).transpose(1, 2)
        tokens = tokens + self.drop_path(
            self.attn(self.norm1(tokens), height, width)
        )
        tokens = tokens + self.drop_path(
            self.mlp(self.norm2(tokens), height, width)
        )
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class _HRTModule(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...],
        heads: tuple[int, ...],
        blocks_per_branch: int = 2,
        window_size: int = 7,
        output_branches: int | None = None,
    ) -> None:
        super().__init__()
        self.num_branches = len(channels)
        self.output_branches = (
            self.num_branches if output_branches is None else int(output_branches)
        )
        if not 1 <= self.output_branches <= self.num_branches:
            raise ValueError("HRFormer output_branches is outside the branch range")
        self.num_inchannels = list(channels)
        self.multi_scale_output = True
        self.branches = nn.ModuleList(
            nn.Sequential(
                *(
                    _HRTBlock(channel, head, window_size)
                    for _ in range(blocks_per_branch)
                )
            )
            for channel, head in zip(channels, heads, strict=True)
        )
        self.fuse_layers = self._make_fuse_layers(channels)
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _downsample(source_channels: int, target_channels: int, steps: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        current = source_channels
        for index in range(steps):
            output = target_channels if index == steps - 1 else current
            operations: list[nn.Module] = [
                nn.Conv2d(
                    current,
                    current,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=current,
                    bias=False,
                ),
                nn.BatchNorm2d(current, momentum=_BN_MOMENTUM),
                nn.Conv2d(current, output, kernel_size=1, bias=False),
                nn.BatchNorm2d(output, momentum=_BN_MOMENTUM),
            ]
            if index != steps - 1:
                operations.append(nn.ReLU(inplace=False))
            layers.append(nn.Sequential(*operations))
            current = output
        return nn.Sequential(*layers)

    def _make_fuse_layers(self, channels: tuple[int, ...]) -> nn.ModuleList:
        all_layers: list[nn.ModuleList] = []
        for target_index, target_channels in enumerate(
            channels[: self.output_branches]
        ):
            target_layers: list[nn.Module] = []
            for source_index, source_channels in enumerate(channels):
                if source_index == target_index:
                    target_layers.append(nn.Identity())
                elif source_index > target_index:
                    target_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                source_channels,
                                target_channels,
                                kernel_size=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                target_channels, momentum=_BN_MOMENTUM
                            ),
                        )
                    )
                else:
                    target_layers.append(
                        self._downsample(
                            source_channels,
                            target_channels,
                            target_index - source_index,
                        )
                    )
            all_layers.append(nn.ModuleList(target_layers))
        return nn.ModuleList(all_layers)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        branches = [
            branch(value)
            for branch, value in zip(self.branches, inputs, strict=True)
        ]
        outputs: list[torch.Tensor] = []
        for target_index in range(self.output_branches):
            fused = branches[target_index]
            for source_index in range(self.num_branches):
                if source_index == target_index:
                    continue
                contribution = self.fuse_layers[target_index][source_index](
                    branches[source_index]
                )
                if source_index > target_index:
                    contribution = F.interpolate(
                        contribution,
                        size=branches[target_index].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                fused = fused + contribution
            outputs.append(self.relu(fused))
        return outputs


class _HRFormerSPrunedEncoder(nn.Module):
    stage_channels = (32, 64, 128, 256)

    def __init__(self, in_chans: int = 4) -> None:
        super().__init__()
        if in_chans not in {3, 4}:
            raise ValueError("HRFormer-S detail backbone requires 3 or 4 channels")
        self.conv1 = nn.Conv2d(
            in_chans, 64, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64, momentum=_BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(64, momentum=_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_bottleneck_layer(64, 64, blocks=2)
        self.transition1 = self._make_transition((256,), self.stage_channels[:2])
        self.stage2 = nn.Sequential(
            _HRTModule(self.stage_channels[:2], (1, 2))
        )
        self.transition2 = self._make_transition(
            self.stage_channels[:2], self.stage_channels[:3]
        )
        self.stage3 = nn.Sequential(
            *(_HRTModule(self.stage_channels[:3], (1, 2, 4)) for _ in range(4))
        )
        self.transition3 = self._make_transition(
            self.stage_channels[:3], self.stage_channels
        )
        self.stage4 = nn.Sequential(
            _HRTModule(self.stage_channels, (1, 2, 4, 8)),
            _HRTModule(
                self.stage_channels,
                (1, 2, 4, 8),
                output_branches=2,
            ),
        )
        self._initialize()

    @staticmethod
    def _make_bottleneck_layer(
        in_channels: int,
        channels: int,
        *,
        blocks: int,
    ) -> nn.Sequential:
        downsample = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels * _HRTBottleneck.expansion,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                channels * _HRTBottleneck.expansion,
                momentum=_BN_MOMENTUM,
            ),
        )
        layers: list[nn.Module] = [
            _HRTBottleneck(in_channels, channels, downsample=downsample)
        ]
        layers.extend(
            _HRTBottleneck(
                channels * _HRTBottleneck.expansion,
                channels,
            )
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    @staticmethod
    def _make_transition(
        previous_channels: tuple[int, ...],
        current_channels: tuple[int, ...],
    ) -> nn.ModuleList:
        transitions: list[nn.Module] = []
        for index, output_channels in enumerate(current_channels):
            if index < len(previous_channels):
                input_channels = previous_channels[index]
                transitions.append(
                    nn.Sequential(
                        nn.Conv2d(
                            input_channels,
                            output_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(
                            output_channels, momentum=_BN_MOMENTUM
                        ),
                        nn.ReLU(inplace=True),
                    )
                    if input_channels != output_channels
                    else nn.Identity()
                )
                continue
            operations: list[nn.Module] = []
            input_channels = previous_channels[-1]
            for step in range(index + 1 - len(previous_channels)):
                output = (
                    output_channels
                    if step == index - len(previous_channels)
                    else input_channels
                )
                operations.append(
                    nn.Sequential(
                        nn.Conv2d(
                            input_channels,
                            output,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(output, momentum=_BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    )
                )
                input_channels = output
            transitions.append(nn.Sequential(*operations))
        return nn.ModuleList(transitions)

    @staticmethod
    def _apply_transition(
        previous: list[torch.Tensor],
        transitions: nn.ModuleList,
    ) -> list[torch.Tensor]:
        return [
            transition(previous[index] if index < len(previous) else previous[-1])
            for index, transition in enumerate(transitions)
        ]

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.relu(self.bn2(self.conv2(outputs)))
        outputs = self.layer1(outputs)
        branches = self.stage2(self._apply_transition([outputs], self.transition1))
        branches = self.stage3(self._apply_transition(branches, self.transition2))
        return list(self.stage4(self._apply_transition(branches, self.transition3)))


class HRFormerSDetailBackbone(nn.Module):
    detail_channels = (128, 128)
    pretrained_key = "hrformer-s-imagenet"

    def __init__(
        self,
        *,
        in_chans: int = 4,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.encoder = _HRFormerSPrunedEncoder(in_chans=in_chans)
        self.stride8_projection = ConvGNReLU(64, 128, 1)
        self.stride4_projection = ConvGNReLU(32, 128, 1)
        self.stride4_refine = ConvGNReLU(256, 128, 3)
        self.pretrained_report: dict[str, list[str]] | None = None
        if checkpoint_path is not None:
            self.pretrained_report = load_pruned_backbone_checkpoint(
                self.encoder,
                checkpoint_path,
                stem_key="conv1.weight",
                ignored_prefixes=(
                    "incre_modules.",
                    "downsamp_modules.",
                    "final_layer.",
                    "classifier.",
                    "head.",
                    "fc.",
                    "stage4.1.fuse_layers.2.",
                    "stage4.1.fuse_layers.3.",
                ),
                source_transform=_convert_hrformer_checkpoint,
            )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(inputs)
        stride8 = self.stride8_projection(features[1])
        stride4 = self.stride4_refine(
            torch.cat(
                (
                    self.stride4_projection(features[0]),
                    F.interpolate(
                        stride8,
                        size=features[0].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                ),
                dim=1,
            )
        )
        return stride4, stride8


def build_hrformer_s_detail(
    *,
    in_chans: int = 4,
    pretrained: bool = True,
    checkpoint_path: str | Path | None = None,
) -> HRFormerSDetailBackbone:
    if pretrained:
        if checkpoint_path is None:
            from geoseg.pretrained import resolve_pretrained_model_path

            checkpoint_path = resolve_pretrained_model_path(
                HRFormerSDetailBackbone.pretrained_key
            )
    elif checkpoint_path is not None:
        raise ValueError("checkpoint_path requires pretrained=True")
    return HRFormerSDetailBackbone(
        in_chans=in_chans,
        checkpoint_path=checkpoint_path,
    )


__all__ = ["HRFormerSDetailBackbone", "build_hrformer_s_detail"]
