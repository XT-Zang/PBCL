from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint_forward

from .multiscale import _resize_like, _resize_outputs
from .ocr import OCRGather
from .third_party.dinov3.vit import (
    RopePositionEmbedding,
    SelfAttentionBlock,
    _initialize_module,
)


@dataclass(frozen=True)
class MultiscaleDINOv3Spec:
    name: str
    architecture: str
    stage_depths: tuple[int, int, int, int]
    stage_dims: tuple[int, int, int, int]
    stage_heads: tuple[int, int, int, int]
    pretrained_policy: str
    window_size: int = 8
    dynamic_base_depths: tuple[int, int, int, int] | None = None

    @property
    def output_layers(self) -> tuple[int, int, int, int]:
        total = 0
        values: list[int] = []
        for depth in self.stage_depths:
            total += int(depth)
            values.append(total - 1)
        return tuple(values)  # type: ignore[return-value]

    @property
    def active_backbones(self) -> tuple[str, ...]:
        return ("dinov3",)


_STAGE_ADAPTER_DIMS = (128, 256, 512, 512)
_HIERARCHICAL_DIMS = (256, 384, 768, 1024)
_HIERARCHICAL_HEADS = (4, 6, 12, 16)

_SPECS = {
    "ssepnet_ms_dinov3_stage_adapter_6_6_6_6": MultiscaleDINOv3Spec(
        "ssepnet_ms_dinov3_stage_adapter_6_6_6_6",
        "stage_adapter",
        (6, 6, 6, 6),
        _STAGE_ADAPTER_DIMS,
        (16, 16, 16, 16),
        "strict_flat_dinov3_semantic_anchor_plus_gated_spatial_detail",
    ),
    "ssepnet_ms_dinov3_poolhier_6_6_6_6": MultiscaleDINOv3Spec(
        "ssepnet_ms_dinov3_poolhier_6_6_6_6",
        "pool_hierarchy",
        (6, 6, 6, 6),
        (128, 1024, 1024, 1024),
        (16, 16, 16, 16),
        "strict_flat_dinov3_blocks_with_gated_spatial_transitions",
    ),
    "ssepnet_ms_dinov3_hier_6_6_6_6": MultiscaleDINOv3Spec(
        "ssepnet_ms_dinov3_hier_6_6_6_6",
        "hierarchy",
        (6, 6, 6, 6),
        _HIERARCHICAL_DIMS,
        _HIERARCHICAL_HEADS,
        "complete_head_projection_from_flat_dinov3",
    ),
    "ssepnet_ms_dinov3_hier_6_6_8_4": MultiscaleDINOv3Spec(
        "ssepnet_ms_dinov3_hier_6_6_8_4",
        "hierarchy",
        (6, 6, 8, 4),
        _HIERARCHICAL_DIMS,
        _HIERARCHICAL_HEADS,
        "complete_head_projection_from_flat_dinov3",
    ),
    "ssepnet_ms_dinov3_hier_8_6_6_4": MultiscaleDINOv3Spec(
        "ssepnet_ms_dinov3_hier_8_6_6_4",
        "hierarchy",
        (8, 6, 6, 4),
        _HIERARCHICAL_DIMS,
        _HIERARCHICAL_HEADS,
        "complete_head_projection_from_flat_dinov3",
    ),
    "ssepnet_ms_dinov3_dynamic_depth": MultiscaleDINOv3Spec(
        "ssepnet_ms_dinov3_dynamic_depth",
        "dynamic_hierarchy",
        (8, 6, 6, 4),
        _HIERARCHICAL_DIMS,
        _HIERARCHICAL_HEADS,
        "complete_head_projection_with_content_routed_extra_blocks",
        dynamic_base_depths=(4, 4, 6, 4),
    ),
}

SSEPNET_MULTISCALE_DINOV3: Mapping[str, MultiscaleDINOv3Spec] = MappingProxyType(_SPECS)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def is_ssepnet_multiscale_dinov3(name: str) -> bool:
    return _normalize_name(name) in SSEPNET_MULTISCALE_DINOV3


def get_ssepnet_multiscale_dinov3(name: str) -> MultiscaleDINOv3Spec:
    key = _normalize_name(name)
    try:
        return SSEPNET_MULTISCALE_DINOV3[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported multiscale DINOv3 variant {name!r}; choose from "
            f"{', '.join(SSEPNET_MULTISCALE_DINOV3)}"
        ) from exc


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(channels), eps=1e-5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(inputs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ConvGNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = int(kernel_size) // 2
        super().__init__(
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                stride=int(stride),
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(int(out_channels)), int(out_channels)),
            nn.GELU(),
        )


class SpatialPyramidStem(nn.Module):
    def __init__(self, in_chans: int, dims: Sequence[int]) -> None:
        super().__init__()
        if len(dims) != 4:
            raise ValueError("spatial pyramid stem requires four channel widths")
        widths = tuple(int(value) for value in dims)
        hidden = max(32, widths[0] // 2)
        self.stem = nn.Sequential(
            ConvGNAct(in_chans, hidden, stride=2),
            ConvGNAct(hidden, widths[0], stride=2),
        )
        self.downsamples = nn.ModuleList(
            ConvGNAct(widths[index], widths[index + 1], stride=2) for index in range(3)
        )
        self.out_channels = widths

    def first(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.stem(inputs)

    def downsample(self, stage: int, inputs: torch.Tensor) -> torch.Tensor:
        return self.downsamples[int(stage)](inputs)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        outputs = [self.first(inputs)]
        for index in range(3):
            outputs.append(self.downsample(index, outputs[-1]))
        return outputs


class DepthwiseDownsample(nn.Sequential):
    def __init__(self, channels: int) -> None:
        channels = int(channels)
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )


def _run_token_block(
    block: SelfAttentionBlock,
    tokens: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor],
    *,
    activation_checkpoint: bool,
) -> torch.Tensor:
    if activation_checkpoint and tokens.requires_grad:

        def forward(value: torch.Tensor) -> torch.Tensor:
            return block(value, rope)

        return activation_checkpoint_forward(forward, tokens, use_reentrant=False)
    return block(tokens, rope)


@dataclass(frozen=True)
class _WindowMetadata:
    batch: int
    height: int
    width: int
    padded_height: int
    padded_width: int
    shift: int
    window_size: int


def _partition_windows(
    features: torch.Tensor,
    *,
    window_size: int,
    shift: int,
) -> tuple[torch.Tensor, _WindowMetadata]:
    batch, channels, height, width = features.shape
    window_size = int(window_size)
    shift = int(shift)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    pad_top = shift
    pad_left = shift
    padded_height = height + pad_top
    padded_width = width + pad_left
    pad_bottom = (-padded_height) % window_size
    pad_right = (-padded_width) % window_size
    mode = "replicate" if height > 0 and width > 0 else "constant"
    padded = F.pad(
        features,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode=mode,
    )
    padded_height, padded_width = padded.shape[-2:]
    windows = (
        padded.permute(0, 2, 3, 1)
        .reshape(
            batch,
            padded_height // window_size,
            window_size,
            padded_width // window_size,
            window_size,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window_size * window_size, channels)
    )
    return windows, _WindowMetadata(
        batch=batch,
        height=height,
        width=width,
        padded_height=padded_height,
        padded_width=padded_width,
        shift=shift,
        window_size=window_size,
    )


def _reverse_windows(
    windows: torch.Tensor,
    metadata: _WindowMetadata,
) -> torch.Tensor:
    channels = int(windows.shape[-1])
    size = int(metadata.window_size)
    features = (
        windows.reshape(
            metadata.batch,
            metadata.padded_height // size,
            metadata.padded_width // size,
            size,
            size,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(
            metadata.batch,
            metadata.padded_height,
            metadata.padded_width,
            channels,
        )
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    top = int(metadata.shift)
    left = int(metadata.shift)
    return features[:, :, top : top + metadata.height, left : left + metadata.width]


def _normalize_feature(features: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    batch, channels, height, width = features.shape
    normalized = norm(features.flatten(2).transpose(1, 2))
    return (
        normalized.transpose(1, 2).reshape(batch, channels, height, width).contiguous()
    )


def _run_feature_block(
    block: SelfAttentionBlock,
    rope_embed: RopePositionEmbedding,
    features: torch.Tensor,
    *,
    window_size: int | None,
    shift: int,
    activation_checkpoint: bool,
) -> torch.Tensor:
    if window_size is None:
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        tokens = _run_token_block(
            block,
            tokens,
            rope_embed(height=height, width=width),
            activation_checkpoint=activation_checkpoint,
        )
        return (
            tokens.transpose(1, 2).reshape(batch, channels, height, width).contiguous()
        )
    windows, metadata = _partition_windows(
        features,
        window_size=int(window_size),
        shift=int(shift),
    )
    windows = _run_token_block(
        block,
        windows,
        rope_embed(height=int(window_size), width=int(window_size)),
        activation_checkpoint=activation_checkpoint,
    )
    return _reverse_windows(windows, metadata)


class DinoStage(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        depth: int,
        window_size: int | None,
        activation_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.window_size = None if window_size is None else int(window_size)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.rope_embed = RopePositionEmbedding(int(dim), num_heads=int(heads))
        self.blocks = nn.ModuleList(
            SelfAttentionBlock(int(dim), int(heads), ffn_ratio=4.0)
            for _ in range(int(depth))
        )
        self.blocks.apply(_initialize_module)
        self.output_norm = nn.LayerNorm(int(dim), eps=1e-5)

    def ordered_blocks(self) -> tuple[SelfAttentionBlock, ...]:
        return tuple(self.blocks)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for index, block in enumerate(self.blocks):
            shift = (
                0
                if self.window_size is None or index % 2 == 0
                else self.window_size // 2
            )
            features = _run_feature_block(
                block,
                self.rope_embed,
                features,
                window_size=self.window_size,
                shift=shift,
                activation_checkpoint=self.activation_checkpoint and self.training,
            )
        return _normalize_feature(features, self.output_norm)


class DynamicDinoStage(nn.Module):
    """Shared-window base depth with differentiable train-time and sparse eval routing."""

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        base_depth: int,
        extra_depth: int,
        window_size: int,
        activation_checkpoint: bool,
        routing_temperature: float = 0.15,
    ) -> None:
        super().__init__()
        if int(extra_depth) <= 0:
            raise ValueError("dynamic stage requires at least one routed block")
        self.window_size = int(window_size)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.routing_temperature = float(routing_temperature)
        self.rope_embed = RopePositionEmbedding(int(dim), num_heads=int(heads))
        self.base_blocks = nn.ModuleList(
            SelfAttentionBlock(int(dim), int(heads), ffn_ratio=4.0)
            for _ in range(int(base_depth))
        )
        self.extra_blocks = nn.ModuleList(
            SelfAttentionBlock(int(dim), int(heads), ffn_ratio=4.0)
            for _ in range(int(extra_depth))
        )
        self.base_blocks.apply(_initialize_module)
        self.extra_blocks.apply(_initialize_module)
        self.router = nn.Linear(int(dim), 1)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.register_buffer(
            "routing_thresholds",
            torch.linspace(0.35, 0.65, steps=int(extra_depth)),
            persistent=True,
        )
        self.output_norm = nn.LayerNorm(int(dim), eps=1e-5)
        self.last_routing_means: torch.Tensor | None = None
        self.last_active_fractions: torch.Tensor | None = None

    def ordered_blocks(self) -> tuple[SelfAttentionBlock, ...]:
        return tuple(self.base_blocks) + tuple(self.extra_blocks)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for index, block in enumerate(self.base_blocks):
            features = _run_feature_block(
                block,
                self.rope_embed,
                features,
                window_size=self.window_size,
                shift=0 if index % 2 == 0 else self.window_size // 2,
                activation_checkpoint=self.activation_checkpoint and self.training,
            )
        windows, metadata = _partition_windows(
            features,
            window_size=self.window_size,
            shift=0,
        )
        difficulty = torch.sigmoid(self.router(windows.mean(dim=1))).squeeze(-1)
        rope = self.rope_embed(height=self.window_size, width=self.window_size)
        routing_means: list[torch.Tensor] = []
        active_fractions: list[torch.Tensor] = []
        for index, block in enumerate(self.extra_blocks):
            threshold = self.routing_thresholds[index].to(difficulty.dtype)
            if self.training:
                gate = torch.sigmoid(
                    (difficulty - threshold) / self.routing_temperature
                )
                candidate = _run_token_block(
                    block,
                    windows,
                    rope,
                    activation_checkpoint=self.activation_checkpoint,
                )
                windows = windows + gate[:, None, None] * (candidate - windows)
                routing_means.append(gate.detach().mean())
                active_fractions.append((gate.detach() >= 0.5).float().mean())
            else:
                active = difficulty >= threshold
                routed = windows.clone()
                if bool(active.any()):
                    routed[active] = block(windows[active], rope)
                windows = routed
                routing_means.append(active.float().mean())
                active_fractions.append(active.float().mean())
        self.last_routing_means = torch.stack(routing_means)
        self.last_active_fractions = torch.stack(active_fractions)
        features = _reverse_windows(windows, metadata)
        return _normalize_feature(features, self.output_norm)


class DinoSpatialInteraction(nn.Module):
    def __init__(
        self,
        dino_dim: int,
        spatial_dim: int,
        *,
        detail_gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.dino_to_spatial = ConvGNAct(int(dino_dim), int(spatial_dim), kernel_size=1)
        self.spatial_to_dino = nn.Conv2d(int(spatial_dim), int(dino_dim), 1, bias=False)
        self.detail_gate = nn.Conv2d(int(spatial_dim) * 2, int(spatial_dim), 1)
        nn.init.zeros_(self.detail_gate.weight)
        nn.init.constant_(self.detail_gate.bias, float(detail_gate_init_bias))
        self.dino_gate = nn.Parameter(torch.zeros(()))
        self.last_detail_gate_statistics: torch.Tensor | None = None

    def forward(
        self,
        spatial: torch.Tensor,
        tokens: torch.Tensor,
        normalized_patch: torch.Tensor,
        *,
        patch_size: tuple[int, int],
        prefix_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_height, patch_width = patch_size
        batch = int(spatial.shape[0])
        dino_channels = int(normalized_patch.shape[-1])
        dino_map = (
            normalized_patch.transpose(1, 2)
            .reshape(batch, dino_channels, patch_height, patch_width)
            .contiguous()
        )
        semantic_anchor = F.interpolate(
            self.dino_to_spatial(dino_map),
            size=spatial.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        detail_gate = torch.sigmoid(
            self.detail_gate(torch.cat((semantic_anchor, spatial), dim=1))
        )
        detached = detail_gate.detach()
        self.last_detail_gate_statistics = torch.stack(
            (
                detached.mean(),
                detached.std(unbiased=False),
                detached.amin(),
                detached.amax(),
            )
        )
        spatial = semantic_anchor + detail_gate * spatial
        injected_dino = (
            F.adaptive_avg_pool2d(
                self.spatial_to_dino(spatial),
                output_size=(patch_height, patch_width),
            )
            .flatten(2)
            .transpose(1, 2)
        )
        prefix = tokens[:, :prefix_tokens]
        patches = tokens[:, prefix_tokens:]
        patches = patches + torch.tanh(self.dino_gate) * injected_dino
        return spatial, torch.cat((prefix, patches), dim=1)


class StageAdapterDINOv3Backbone(nn.Module):
    def __init__(
        self,
        *,
        dinov3_backbone: nn.Module,
        stage_depths: Sequence[int],
        spatial_dims: Sequence[int] = _STAGE_ADAPTER_DIMS,
        in_chans: int = 4,
        activation_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if len(stage_depths) != 4 or sum(int(value) for value in stage_depths) != len(
            dinov3_backbone.model.blocks
        ):
            raise ValueError(
                "stage adapter depths must partition every flat DINOv3 block"
            )
        self.dinov3_backbone = dinov3_backbone
        self.stage_depths = tuple(int(value) for value in stage_depths)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.spatial_stem = SpatialPyramidStem(in_chans, spatial_dims)
        dino_dim = int(dinov3_backbone.model.embed_dim)
        self.interactions = nn.ModuleList(
            DinoSpatialInteraction(dino_dim, int(spatial_dim))
            for spatial_dim in spatial_dims
        )
        self.out_channels = tuple(int(value) for value in spatial_dims)
        self.out_strides = (4, 8, 16, 32)
        self.initialization_report = {
            "policy": "strict_flat_dinov3_semantic_anchor_plus_gated_spatial_detail",
            "strict_source_blocks": len(dinov3_backbone.model.blocks),
            "projected_source_blocks": 0,
            "new_stage_modules": 4,
        }

    def interaction_gate_statistics(
        self,
    ) -> tuple[tuple[float, float, float, float], ...]:
        return tuple(
            tuple(
                float(value) for value in interaction.last_detail_gate_statistics.cpu()
            )
            for interaction in self.interactions
            if interaction.last_detail_gate_statistics is not None
        )

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        model = self.dinov3_backbone.model
        tokens, (patch_height, patch_width) = model._prepare_tokens(inputs)
        prefix_tokens = int(tokens.shape[1] - patch_height * patch_width)
        rope = model.rope_embed(height=patch_height, width=patch_width)
        spatial = self.spatial_stem.first(inputs)
        outputs: list[torch.Tensor] = []
        block_index = 0
        for stage_index, depth in enumerate(self.stage_depths):
            for _ in range(depth):
                tokens = _run_token_block(
                    model.blocks[block_index],
                    tokens,
                    rope,
                    activation_checkpoint=self.activation_checkpoint and self.training,
                )
                block_index += 1
            normalized = model.norm(tokens)[:, prefix_tokens:]
            spatial, tokens = self.interactions[stage_index](
                spatial,
                tokens,
                normalized,
                patch_size=(patch_height, patch_width),
                prefix_tokens=prefix_tokens,
            )
            outputs.append(spatial)
            if stage_index < 3:
                spatial = self.spatial_stem.downsample(stage_index, spatial)
        return outputs


class PoolHierarchyDINOv3Backbone(nn.Module):
    def __init__(
        self,
        *,
        dinov3_backbone: nn.Module,
        stage_depths: Sequence[int],
        in_chans: int = 4,
        spatial_dims: Sequence[int] = _STAGE_ADAPTER_DIMS,
        window_size: int = 8,
        activation_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if len(stage_depths) != 4 or sum(int(value) for value in stage_depths) != len(
            dinov3_backbone.model.blocks
        ):
            raise ValueError(
                "pool hierarchy depths must partition every flat DINOv3 block"
            )
        self.dinov3_backbone = dinov3_backbone
        self.stage_depths = tuple(int(value) for value in stage_depths)
        self.window_size = int(window_size)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.spatial_stem = SpatialPyramidStem(in_chans, spatial_dims)
        dino_dim = int(dinov3_backbone.model.embed_dim)
        self.p3_projection = nn.Conv2d(int(spatial_dims[1]), dino_dim, 1, bias=False)
        self.transitions = nn.ModuleList(
            DepthwiseDownsample(dino_dim) for _ in range(2)
        )
        self.p3_prior_gate = SemanticResidualGate(dino_dim, gate_init_bias=-2.0)
        self.p4_patch_gate = SemanticResidualGate(dino_dim, gate_init_bias=-2.0)
        self.out_channels = (int(spatial_dims[0]), dino_dim, dino_dim, dino_dim)
        self.out_strides = (4, 8, 16, 32)
        self.initialization_report = {
            "policy": "strict_flat_dinov3_blocks_with_gated_spatial_transitions",
            "strict_source_blocks": len(dinov3_backbone.model.blocks),
            "projected_source_blocks": 0,
            "new_stage_modules": 4,
        }

    def _run_stage(
        self,
        features: torch.Tensor,
        *,
        start: int,
        depth: int,
        windowed: bool,
    ) -> torch.Tensor:
        model = self.dinov3_backbone.model
        for offset in range(int(depth)):
            window_size = self.window_size if windowed else None
            shift = 0 if not windowed or offset % 2 == 0 else self.window_size // 2
            features = _run_feature_block(
                model.blocks[int(start) + offset],
                model.rope_embed,
                features,
                window_size=window_size,
                shift=shift,
                activation_checkpoint=self.activation_checkpoint and self.training,
            )
        return features

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        model = self.dinov3_backbone.model
        p2 = self.spatial_stem.first(inputs)
        p3_prior = self.spatial_stem.downsample(0, p2)
        patch4 = model.patch_embed(inputs).permute(0, 3, 1, 2).contiguous()
        p3 = self.p3_prior_gate(
            F.interpolate(
                patch4,
                size=p3_prior.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ),
            self.p3_projection(p3_prior),
        )
        starts = (
            0,
            self.stage_depths[0],
            sum(self.stage_depths[:2]),
            sum(self.stage_depths[:3]),
        )
        p3 = self._run_stage(
            p3,
            start=starts[0],
            depth=self.stage_depths[0],
            windowed=True,
        )
        p3_out = _normalize_feature(p3, model.norm)
        p4 = self.transitions[0](p3)
        if p4.shape[-2:] != patch4.shape[-2:]:
            patch4 = F.interpolate(
                patch4, size=p4.shape[-2:], mode="bilinear", align_corners=False
            )
        p4 = self._run_stage(
            self.p4_patch_gate(p4, patch4),
            start=starts[1],
            depth=self.stage_depths[1],
            windowed=False,
        )
        p4_out = _normalize_feature(p4, model.norm)
        p5 = self._run_stage(
            self.transitions[1](p4),
            start=starts[2],
            depth=self.stage_depths[2],
            windowed=False,
        )
        p5 = self._run_stage(
            p5,
            start=starts[3],
            depth=self.stage_depths[3],
            windowed=False,
        )
        p5_out = _normalize_feature(p5, model.norm)
        return [p2, p3_out, p4_out, p5_out]


class HierarchicalPatchStem(nn.Module):
    def __init__(self, in_chans: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(int(in_chans), int(out_channels), kernel_size=4, stride=4)
        self.norm = LayerNorm2d(int(out_channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(inputs))


class HierarchicalDINOv3Backbone(nn.Module):
    def __init__(
        self,
        *,
        in_chans: int = 4,
        stage_depths: Sequence[int] = (6, 6, 6, 6),
        stage_dims: Sequence[int] = _HIERARCHICAL_DIMS,
        stage_heads: Sequence[int] = _HIERARCHICAL_HEADS,
        window_size: int = 8,
        activation_checkpoint: bool = True,
        dynamic_base_depths: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if not (len(stage_depths) == len(stage_dims) == len(stage_heads) == 4):
            raise ValueError("hierarchical DINOv3 requires four stages")
        depths = tuple(int(value) for value in stage_depths)
        dims = tuple(int(value) for value in stage_dims)
        heads = tuple(int(value) for value in stage_heads)
        if sum(depths) != 24:
            raise ValueError(
                f"hierarchical DINOv3 must retain 24 blocks, got {sum(depths)}"
            )
        if any(dim % (4 * head) != 0 for dim, head in zip(dims, heads, strict=True)):
            raise ValueError(
                "each stage dimension must be divisible by four times its head count"
            )
        self.stage_depths = depths
        self.stage_heads = heads
        self.out_channels = dims
        self.out_strides = (4, 8, 16, 32)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.stem = HierarchicalPatchStem(in_chans, dims[0])
        self.downsamples = nn.ModuleList(
            ConvGNAct(dims[index], dims[index + 1], stride=2) for index in range(3)
        )
        base_depths = (
            None
            if dynamic_base_depths is None
            else tuple(int(value) for value in dynamic_base_depths)
        )
        if base_depths is not None and len(base_depths) != 4:
            raise ValueError("dynamic_base_depths must define four stages")
        stages: list[nn.Module] = []
        for index, (depth, dim, head) in enumerate(
            zip(depths, dims, heads, strict=True)
        ):
            window = int(window_size) if index < 2 else None
            base_depth = depth if base_depths is None else base_depths[index]
            extra_depth = depth - int(base_depth)
            if extra_depth > 0:
                if window is None:
                    raise ValueError(
                        "routed extra blocks are only supported on window stages"
                    )
                stages.append(
                    DynamicDinoStage(
                        dim=dim,
                        heads=head,
                        base_depth=base_depth,
                        extra_depth=extra_depth,
                        window_size=window,
                        activation_checkpoint=activation_checkpoint,
                    )
                )
            else:
                stages.append(
                    DinoStage(
                        dim=dim,
                        heads=head,
                        depth=depth,
                        window_size=window,
                        activation_checkpoint=activation_checkpoint,
                    )
                )
        self.stages = nn.ModuleList(stages)
        self.initialization_report: dict[str, object] = {
            "policy": "random",
            "strict_source_blocks": 0,
            "projected_source_blocks": 0,
            "new_stage_modules": 4,
        }

    def ordered_blocks(self) -> tuple[SelfAttentionBlock, ...]:
        blocks: list[SelfAttentionBlock] = []
        for stage in self.stages:
            blocks.extend(stage.ordered_blocks())
        return tuple(blocks)

    def routing_statistics(self) -> tuple[dict[str, tuple[float, ...]], ...]:
        values: list[dict[str, tuple[float, ...]]] = []
        for stage in self.stages:
            if not isinstance(stage, DynamicDinoStage):
                continue
            if stage.last_routing_means is None or stage.last_active_fractions is None:
                continue
            values.append(
                {
                    "mean_gate": tuple(
                        float(value) for value in stage.last_routing_means.cpu()
                    ),
                    "active_fraction": tuple(
                        float(value) for value in stage.last_active_fractions.cpu()
                    ),
                }
            )
        return tuple(values)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        features = self.stem(inputs)
        outputs: list[torch.Tensor] = []
        for index, stage in enumerate(self.stages):
            features = stage(features)
            outputs.append(features)
            if index < 3:
                features = self.downsamples[index](features)
        return outputs


def _uniform_indices(
    source_size: int, target_size: int, *, device: torch.device
) -> torch.Tensor:
    if int(target_size) > int(source_size):
        raise ValueError(
            f"cannot project {source_size} channels into wider {target_size} channels"
        )
    if int(target_size) == int(source_size):
        return torch.arange(int(source_size), device=device)
    return (
        torch.linspace(
            0,
            int(source_size) - 1,
            steps=int(target_size),
            device=device,
        )
        .round()
        .long()
    )


def _complete_head_indices(
    *,
    source_heads: int,
    target_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if int(target_heads) > int(source_heads):
        raise ValueError(
            f"cannot select {target_heads} complete heads from {source_heads} heads"
        )
    head_ids = (
        torch.linspace(
            0,
            int(source_heads) - 1,
            steps=int(target_heads),
            device=device,
        )
        .round()
        .long()
    )
    if int(torch.unique(head_ids).numel()) != int(target_heads):
        raise ValueError("complete-head selection produced duplicate source heads")
    channels = torch.cat(
        tuple(
            torch.arange(
                int(head_id) * int(head_dim),
                (int(head_id) + 1) * int(head_dim),
                device=device,
            )
            for head_id in head_ids
        )
    )
    return channels, tuple(int(value) for value in head_ids.cpu())


def _copy_projected_block(
    target: SelfAttentionBlock, source: SelfAttentionBlock
) -> tuple[bool, tuple[int, ...]]:
    target_dim = int(target.norm1.weight.numel())
    source_dim = int(source.norm1.weight.numel())
    device = source.norm1.weight.device
    source_heads = int(source.attn.num_heads)
    target_heads = int(target.attn.num_heads)
    source_head_dim = source_dim // source_heads
    target_head_dim = target_dim // target_heads
    if source_head_dim != target_head_dim:
        raise ValueError(
            f"head-aware projection requires matching head dimensions, got "
            f"{source_head_dim} and {target_head_dim}"
        )
    channels, selected_heads = _complete_head_indices(
        source_heads=source_heads,
        target_heads=target_heads,
        head_dim=source_head_dim,
        device=device,
    )
    source_hidden = int(source.mlp.fc1.out_features)
    target_hidden = int(target.mlp.fc1.out_features)
    hidden = _uniform_indices(source_hidden, target_hidden, device=device)
    qkv_rows = torch.cat(
        tuple(channels + branch * source_dim for branch in range(3)),
        dim=0,
    )
    with torch.no_grad():
        for target_norm, source_norm in (
            (target.norm1, source.norm1),
            (target.norm2, source.norm2),
        ):
            target_norm.weight.copy_(source_norm.weight.index_select(0, channels))
            target_norm.bias.copy_(source_norm.bias.index_select(0, channels))
        target.attn.qkv.weight.copy_(
            source.attn.qkv.weight.index_select(0, qkv_rows).index_select(1, channels)
        )
        if target.attn.qkv.bias is not None and source.attn.qkv.bias is not None:
            target.attn.qkv.bias.copy_(source.attn.qkv.bias.index_select(0, qkv_rows))
        target_mask = getattr(target.attn.qkv, "bias_mask", None)
        source_mask = getattr(source.attn.qkv, "bias_mask", None)
        if target_mask is not None and source_mask is not None:
            target_mask.copy_(source_mask.index_select(0, qkv_rows))
        target.attn.proj.weight.copy_(
            source.attn.proj.weight.index_select(0, channels).index_select(1, channels)
        )
        target.attn.proj.bias.copy_(source.attn.proj.bias.index_select(0, channels))
        target.mlp.fc1.weight.copy_(
            source.mlp.fc1.weight.index_select(0, hidden).index_select(1, channels)
        )
        target.mlp.fc1.bias.copy_(source.mlp.fc1.bias.index_select(0, hidden))
        target.mlp.fc2.weight.copy_(
            source.mlp.fc2.weight.index_select(0, channels).index_select(1, hidden)
        )
        target.mlp.fc2.bias.copy_(source.mlp.fc2.bias.index_select(0, channels))
        target.ls1.gamma.copy_(source.ls1.gamma.index_select(0, channels))
        target.ls2.gamma.copy_(source.ls2.gamma.index_select(0, channels))
    return target_dim == source_dim, selected_heads


def initialize_hierarchical_from_flat_dinov3(
    target: HierarchicalDINOv3Backbone,
    source_backbone: nn.Module,
) -> dict[str, object]:
    source = source_backbone.model
    target_blocks = target.ordered_blocks()
    source_blocks = tuple(source.blocks)
    if len(target_blocks) != len(source_blocks):
        raise ValueError(
            f"projected initialization requires equal depth, got "
            f"{len(target_blocks)} and {len(source_blocks)}"
        )
    exact = 0
    selected_heads_by_block: list[tuple[int, ...]] = []
    for target_block, source_block in zip(target_blocks, source_blocks, strict=True):
        is_exact, selected_heads = _copy_projected_block(target_block, source_block)
        exact += int(is_exact)
        selected_heads_by_block.append(selected_heads)
    with torch.no_grad():
        source_weight = source.patch_embed.proj.weight
        source_head_dim = int(source.embed_dim) // int(source.num_heads)
        stem_channels, _stem_head_ids = _complete_head_indices(
            source_heads=int(source.num_heads),
            target_heads=int(target.stage_heads[0]),
            head_dim=source_head_dim,
            device=source_weight.device,
        )
        selected = source_weight.index_select(0, stem_channels)
        resized = F.adaptive_avg_pool2d(selected, target.stem.proj.kernel_size)
        area_ratio = (
            source.patch_embed.proj.kernel_size[0]
            * source.patch_embed.proj.kernel_size[1]
            / (target.stem.proj.kernel_size[0] * target.stem.proj.kernel_size[1])
        )
        target.stem.proj.weight.copy_(resized * float(area_ratio))
        if (
            target.stem.proj.bias is not None
            and source.patch_embed.proj.bias is not None
        ):
            target.stem.proj.bias.copy_(
                source.patch_embed.proj.bias.index_select(0, stem_channels)
            )
        for stage, target_heads in zip(target.stages, target.stage_heads, strict=True):
            indices, _stage_head_ids = _complete_head_indices(
                source_heads=int(source.num_heads),
                target_heads=int(target_heads),
                head_dim=source_head_dim,
                device=source.norm.weight.device,
            )
            stage.output_norm.weight.copy_(source.norm.weight.index_select(0, indices))
            stage.output_norm.bias.copy_(source.norm.bias.index_select(0, indices))
            if stage.rope_embed.periods.shape == source.rope_embed.periods.shape:
                stage.rope_embed.periods.copy_(source.rope_embed.periods)
    report: dict[str, object] = {
        "policy": "uniform_channel_projection_from_flat_dinov3",
        "strict_source_blocks": exact,
        "projected_source_blocks": len(target_blocks) - exact,
        "source_blocks": len(source_blocks),
        "patch_embed": "area_resampled_complete_head_channel_groups",
        "head_dim_preserved": True,
        "complete_attention_heads_preserved": True,
        "selected_source_heads_by_stage": tuple(
            selected_heads_by_block[sum(target.stage_depths[:stage_index])]
            for stage_index in range(4)
        ),
    }
    target.initialization_report = report
    return report


class BalancedSemanticFusion(nn.Module):
    """Learn a normalized P5/P4 semantic mixture, initialized at equal weight."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))
        self.last_weights: torch.Tensor | None = None

    def forward(
        self,
        high_level: torch.Tensor,
        lateral_semantic: torch.Tensor,
    ) -> torch.Tensor:
        high_level = _resize_like(high_level, lateral_semantic)
        weights = torch.softmax(self.logits, dim=0)
        self.last_weights = weights.detach()
        return weights[0] * high_level + weights[1] * lateral_semantic


class SemanticResidualGate(nn.Module):
    """Keep the top-down semantic tensor as the anchor and gate lateral detail."""

    def __init__(self, channels: int, *, gate_init_bias: float = -2.0) -> None:
        super().__init__()
        channels = int(channels)
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init_bias))
        self.last_gate_statistics: torch.Tensor | None = None

    def forward(
        self,
        semantic_anchor: torch.Tensor,
        lateral_residual: torch.Tensor,
    ) -> torch.Tensor:
        semantic_anchor = _resize_like(semantic_anchor, lateral_residual)
        gate = torch.sigmoid(
            self.gate(torch.cat((semantic_anchor, lateral_residual), dim=1))
        )
        detached = gate.detach()
        self.last_gate_statistics = torch.stack(
            (
                detached.mean(),
                detached.std(unbiased=False),
                detached.amin(),
                detached.amax(),
            )
        )
        return semantic_anchor + gate * lateral_residual


class GNObjectAttentionBlock(nn.Module):
    """Binary OCR object attention with GroupNorm-compatible output projection."""

    def __init__(self, channels: int, key_channels: int) -> None:
        super().__init__()
        channels = int(channels)
        key_channels = int(key_channels)
        self.query = nn.Conv2d(channels, key_channels, kernel_size=1, bias=False)
        self.key = nn.Conv2d(channels, key_channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.project = ConvGNAct(channels, channels, kernel_size=1)
        self.key_channels = key_channels

    def forward(
        self,
        features: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = features.shape
        query = self.query(features).flatten(2).transpose(1, 2)
        key = self.key(context).flatten(2)
        value = self.value(context).flatten(2).transpose(1, 2)
        attention = F.softmax(
            torch.bmm(query, key) / math.sqrt(self.key_channels),
            dim=-1,
        )
        pixels = (
            torch.bmm(attention, value)
            .transpose(1, 2)
            .reshape(batch, channels, height, width)
        )
        return self.project(pixels)


class SemanticAnchorGatedPyramidOCRHead(nn.Module):
    """P5-to-P3 semantic OCR followed by boundary-conditioned P2 refinement."""

    def __init__(
        self,
        input_channels: Sequence[int],
        *,
        decoder_channels: int = 128,
        ocr_channels: int = 256,
        boundary_channels: int | None = None,
        dropout: float = 0.05,
        p3_gate_init_bias: float = -1.0,
        detail_gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if len(input_channels) != 4:
            raise ValueError("semantic-anchor OCR head requires P2/P3/P4/P5 channels")
        decoder_channels = int(decoder_channels)
        ocr_channels = int(ocr_channels)
        boundary_channels = int(
            max(64, decoder_channels // 2)
            if boundary_channels is None
            else boundary_channels
        )
        self.output_strides = (4, 8, 16, 32)
        self.ocr_stride = 8
        self.auxiliary_stride = 16
        self.boundary_stride = 4
        self.lateral_projections = nn.ModuleList(
            ConvGNAct(int(channels), decoder_channels, kernel_size=1)
            for channels in input_channels
        )
        self.p5_refine = ConvGNAct(decoder_channels, decoder_channels)
        self.p4_semantic_fusion = BalancedSemanticFusion()
        self.p4_refine = ConvGNAct(decoder_channels, decoder_channels)
        self.p3_gate = SemanticResidualGate(
            decoder_channels, gate_init_bias=p3_gate_init_bias
        )
        self.p3_refine = ConvGNAct(decoder_channels, decoder_channels)

        self.auxiliary = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.ocr_shared = ConvGNAct(decoder_channels, ocr_channels)
        self.ocr_gather = OCRGather()
        self.ocr_attention = GNObjectAttentionBlock(
            ocr_channels, max(32, ocr_channels // 2)
        )
        self.ocr_bottleneck = nn.Sequential(
            ConvGNAct(ocr_channels * 2, ocr_channels),
            nn.Dropout2d(float(dropout)),
        )
        self.ocr_to_decoder = ConvGNAct(ocr_channels, decoder_channels, kernel_size=1)

        self.boundary_fusion = ConvGNAct(
            decoder_channels * 2,
            boundary_channels,
        )
        self.boundary_classifier = nn.Conv2d(boundary_channels, 1, kernel_size=1)
        self.detail_refine = ConvGNAct(decoder_channels, decoder_channels)
        self.detail_gate = nn.Conv2d(
            decoder_channels * 2 + boundary_channels,
            decoder_channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.detail_gate.weight)
        nn.init.constant_(self.detail_gate.bias, float(detail_gate_init_bias))
        self.classifier = nn.Sequential(
            ConvGNAct(decoder_channels, decoder_channels),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(decoder_channels, 1, kernel_size=1),
        )
        self.last_detail_gate_statistics: torch.Tensor | None = None

    def _forward_main_aux_boundary_features(
        self,
        features: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(features) != 4:
            raise ValueError(
                f"semantic-anchor OCR head requires four features, got {len(features)}"
            )
        p2, p3, p4, p5 = [
            projection(feature)
            for projection, feature in zip(
                self.lateral_projections, features, strict=True
            )
        ]
        semantic5 = self.p5_refine(p5)
        semantic4 = self.p4_refine(self.p4_semantic_fusion(semantic5, p4))
        semantic3 = self.p3_refine(self.p3_gate(semantic4, p3))

        auxiliary = self.auxiliary(semantic4)
        ocr_features = self.ocr_shared(semantic3)
        auxiliary_for_context = _resize_like(auxiliary, ocr_features)
        object_context = self.ocr_attention(
            ocr_features,
            self.ocr_gather(ocr_features, auxiliary_for_context),
        )
        ocr_refined = self.ocr_bottleneck(
            torch.cat((ocr_features, object_context), dim=1)
        )
        semantic2 = _resize_like(self.ocr_to_decoder(ocr_refined), p2)

        boundary_features = self.boundary_fusion(
            torch.cat((p2, _resize_like(semantic3, p2)), dim=1)
        )
        detail = self.detail_refine(p2)
        detail_gate = torch.sigmoid(
            self.detail_gate(torch.cat((semantic2, detail, boundary_features), dim=1))
        )
        detached = detail_gate.detach()
        self.last_detail_gate_statistics = torch.stack(
            (
                detached.mean(),
                detached.std(unbiased=False),
                detached.amin(),
                detached.amax(),
            )
        )
        main = self.classifier(semantic2 + detail_gate * detail)
        return main, auxiliary, boundary_features

    def forward_main(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        main, _auxiliary, _boundary_features = self._forward_main_aux_boundary_features(
            features
        )
        return main

    def forward(self, features: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        main, auxiliary, boundary_features = self._forward_main_aux_boundary_features(
            features
        )
        return {
            "out": main,
            "aux": auxiliary,
            "edge": self.boundary_classifier(boundary_features),
        }

    def gate_statistics(self) -> dict[str, tuple[float, ...]]:
        values: dict[str, tuple[float, ...]] = {}
        if self.p4_semantic_fusion.last_weights is not None:
            values["p4_p5_semantic_weights"] = tuple(
                float(value)
                for value in self.p4_semantic_fusion.last_weights.float().cpu()
            )
        if self.p3_gate.last_gate_statistics is not None:
            values["p3"] = tuple(
                float(value)
                for value in self.p3_gate.last_gate_statistics.float().cpu()
            )
        if self.last_detail_gate_statistics is not None:
            values["p2_detail"] = tuple(
                float(value) for value in self.last_detail_gate_statistics.float().cpu()
            )
        return values


class MultiscaleDINOv3SSepNet(nn.Module):
    active_backbones = ("dinov3",)

    def __init__(
        self,
        *,
        multiscale_backbone: nn.Module,
        decoder_channels: int = 128,
        ocr_channels: int = 256,
        num_classes: int = 1,
        freeze_backbone: bool = False,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("multiscale DINOv3 SSepNet requires num_classes=1")
        self.multiscale_backbone = multiscale_backbone
        self.freeze_backbone = bool(freeze_backbone)
        channels = tuple(int(value) for value in multiscale_backbone.out_channels)
        if len(channels) != 4:
            raise ValueError(
                "multiscale DINOv3 backbone must expose four feature widths"
            )
        self.head = SemanticAnchorGatedPyramidOCRHead(
            channels,
            decoder_channels=int(decoder_channels),
            ocr_channels=int(ocr_channels),
            dropout=float(dropout),
        )
        if self.freeze_backbone:
            self.multiscale_backbone.requires_grad_(False)

    @property
    def pretrained_initialization_report(self) -> Mapping[str, object]:
        return dict(getattr(self.multiscale_backbone, "initialization_report", {}))

    def routing_statistics(self) -> tuple[dict[str, tuple[float, ...]], ...]:
        callback = getattr(self.multiscale_backbone, "routing_statistics", None)
        return () if callback is None else callback()

    def head_gate_statistics(self) -> dict[str, tuple[float, ...]]:
        return self.head.gate_statistics()

    def train(self, mode: bool = True) -> "MultiscaleDINOv3SSepNet":
        super().train(mode)
        if self.freeze_backbone:
            self.multiscale_backbone.eval()
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        main_only: bool = False,
        disable_hrnet: bool = False,
        disable_dinov3: bool = False,
    ) -> dict[str, torch.Tensor]:
        del disable_hrnet
        features = list(self.multiscale_backbone(inputs))
        if len(features) != 4:
            raise ValueError(
                f"multiscale DINOv3 must return four features, got {len(features)}"
            )
        if disable_dinov3:
            features = [torch.zeros_like(value) for value in features]
        if main_only:
            outputs = {"out": self.head.forward_main(features)}
        else:
            outputs = self.head(features)
        return _resize_outputs(outputs, tuple(inputs.shape[-2:]))  # type: ignore[return-value]

    def forward_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs, main_only=True)["out"]


def _configure_flat_adaptation(
    backbone: nn.Module,
    *,
    mode: str,
    rank: int,
    alpha: float | None,
) -> object:
    from .lora import configure_dinov3_adaptation

    return configure_dinov3_adaptation(
        backbone,
        mode=mode,
        rank=int(rank),
        alpha=alpha,
    )


def build_ssepnet_multiscale_dinov3(
    name: str,
    *,
    in_chans: int = 4,
    num_classes: int = 1,
    pretrained: bool = True,
    dinov3_checkpoint: str | Path | None = None,
    freeze_backbone: bool = False,
    activation_checkpoint: bool = True,
    dinov3_adaptation: str = "full",
    dinov3_lora_rank: int = 0,
    dinov3_lora_alpha: float | None = None,
    decoder_channels: int = 128,
    ocr_channels: int = 256,
) -> MultiscaleDINOv3SSepNet:
    if int(in_chans) not in {3, 4}:
        raise ValueError("multiscale DINOv3 requires RGB or B/G/R/NIR input")
    spec = get_ssepnet_multiscale_dinov3(name)
    from .dinov3 import build_dinov3_vitl16_sat493m

    if spec.architecture in {"stage_adapter", "pool_hierarchy"}:
        flat = build_dinov3_vitl16_sat493m(
            in_chans=in_chans,
            pretrained=pretrained,
            checkpoint_path=dinov3_checkpoint,
        )
        adaptation_report = _configure_flat_adaptation(
            flat,
            mode=dinov3_adaptation,
            rank=dinov3_lora_rank,
            alpha=dinov3_lora_alpha,
        )
        if spec.architecture == "stage_adapter":
            backbone: nn.Module = StageAdapterDINOv3Backbone(
                dinov3_backbone=flat,
                stage_depths=spec.stage_depths,
                spatial_dims=spec.stage_dims,
                in_chans=in_chans,
                activation_checkpoint=activation_checkpoint,
            )
        else:
            backbone = PoolHierarchyDINOv3Backbone(
                dinov3_backbone=flat,
                stage_depths=spec.stage_depths,
                in_chans=in_chans,
                window_size=spec.window_size,
                activation_checkpoint=activation_checkpoint,
            )
        backbone.initialization_report = {
            **backbone.initialization_report,
            "dinov3_adaptation": getattr(adaptation_report, "mode", dinov3_adaptation),
        }
    else:
        if (
            str(dinov3_adaptation).strip().lower() != "full"
            or int(dinov3_lora_rank) != 0
        ):
            raise ValueError(
                "projected hierarchical DINOv3 variants currently support full adaptation only"
            )
        backbone = HierarchicalDINOv3Backbone(
            in_chans=in_chans,
            stage_depths=spec.stage_depths,
            stage_dims=spec.stage_dims,
            stage_heads=spec.stage_heads,
            window_size=spec.window_size,
            activation_checkpoint=activation_checkpoint,
            dynamic_base_depths=spec.dynamic_base_depths,
        )
        if pretrained:
            flat = build_dinov3_vitl16_sat493m(
                in_chans=in_chans,
                pretrained=True,
                checkpoint_path=dinov3_checkpoint,
            )
            initialize_hierarchical_from_flat_dinov3(backbone, flat)
            backbone.initialization_report["policy"] = spec.pretrained_policy
            del flat
        elif dinov3_checkpoint is not None:
            raise ValueError("dinov3_checkpoint requires pretrained=True")
    backbone.initialization_report.update(
        {
            "architecture": spec.architecture,
            "stage_depths": spec.stage_depths,
            "stage_output_layers": spec.output_layers,
            "output_strides": (4, 8, 16, 32),
        }
    )
    return MultiscaleDINOv3SSepNet(
        multiscale_backbone=backbone,
        decoder_channels=decoder_channels,
        ocr_channels=ocr_channels,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
    )


def build_ssepnet_multiscale_dinov3_loss(name: str) -> nn.Module:
    get_ssepnet_multiscale_dinov3(name)
    from geoseg.losses.ssepnet import SSepNetLoss

    return SSepNetLoss(boundary_weight=0.6, pbcl_weight=0.1)


__all__ = [
    "SSEPNET_MULTISCALE_DINOV3",
    "DinoStage",
    "DynamicDinoStage",
    "HierarchicalDINOv3Backbone",
    "MultiscaleDINOv3SSepNet",
    "MultiscaleDINOv3Spec",
    "PoolHierarchyDINOv3Backbone",
    "BalancedSemanticFusion",
    "SemanticAnchorGatedPyramidOCRHead",
    "SemanticResidualGate",
    "StageAdapterDINOv3Backbone",
    "build_ssepnet_multiscale_dinov3",
    "build_ssepnet_multiscale_dinov3_loss",
    "get_ssepnet_multiscale_dinov3",
    "initialize_hierarchical_from_flat_dinov3",
    "is_ssepnet_multiscale_dinov3",
]
