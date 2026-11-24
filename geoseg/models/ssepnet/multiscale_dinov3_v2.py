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

from .multiscale import _resize_like, _resize_outputs
from .multiscale_dinov3 import (
    ConvGNAct,
    MultiscaleDINOv3SSepNet,
    SemanticAnchorGatedPyramidOCRHead,
    SemanticResidualGate,
    SpatialPyramidStem,
    _group_count,
    _normalize_feature,
    _partition_windows,
    _reverse_windows,
    _run_token_block,
    _uniform_indices,
)
from .third_party.dinov3.vit import (
    RopePositionEmbedding,
    SelfAttentionBlock,
    _initialize_module,
)


@dataclass(frozen=True)
class MultiscaleDINOv3V2Spec:
    name: str
    architecture: str
    stage_depths: tuple[int, int, int, int]
    teacher_taps: tuple[int, int, int, int]
    feature_loss_weights: tuple[float, float, float, float]
    stage_dims: tuple[int, int, int, int]
    stage_heads: tuple[int, int, int, int]
    pretrained_policy: str
    parent: str
    dynamic_base_depths: tuple[int, int, int, int] | None = None
    pool_stage_depths: tuple[int, int, int] | None = None
    window_size: int = 8

    @property
    def active_backbones(self) -> tuple[str, ...]:
        return ("dinov3",)


_HIER_DIMS = (256, 384, 768, 1024)
_HIER_HEADS = (4, 6, 12, 16)

_SPECS = {
    "ssepnet_ms_dinov3_v2_hier_6_6_6_6": MultiscaleDINOv3V2Spec(
        "ssepnet_ms_dinov3_v2_hier_6_6_6_6",
        "hierarchy_v2",
        (6, 6, 6, 6),
        (5, 11, 17, 23),
        (0.1, 0.2, 0.4, 0.3),
        _HIER_DIMS,
        _HIER_HEADS,
        "nested_complete_head_projection_with_storage_tokens_v2",
        "ssepnet_ms_dinov3_stage_adapter_6_6_6_6",
    ),
    "ssepnet_ms_dinov3_v2_hier_6_6_8_4": MultiscaleDINOv3V2Spec(
        "ssepnet_ms_dinov3_v2_hier_6_6_8_4",
        "hierarchy_v2",
        (6, 6, 8, 4),
        (5, 11, 19, 23),
        (0.1, 0.2, 0.4, 0.3),
        _HIER_DIMS,
        _HIER_HEADS,
        "nested_complete_head_projection_with_storage_tokens_v2",
        "ssepnet_ms_dinov3_v2_hier_6_6_6_6",
    ),
    "ssepnet_ms_dinov3_v2_hier_8_6_6_4": MultiscaleDINOv3V2Spec(
        "ssepnet_ms_dinov3_v2_hier_8_6_6_4",
        "hierarchy_v2",
        (8, 6, 6, 4),
        (7, 13, 19, 23),
        (0.1, 0.2, 0.4, 0.3),
        _HIER_DIMS,
        _HIER_HEADS,
        "nested_complete_head_projection_with_storage_tokens_v2",
        "ssepnet_ms_dinov3_v2_hier_6_6_6_6",
    ),
    "ssepnet_ms_dinov3_v2_dynamic_depth": MultiscaleDINOv3V2Spec(
        "ssepnet_ms_dinov3_v2_dynamic_depth",
        "dynamic_hierarchy_v2",
        (8, 6, 6, 4),
        (7, 13, 19, 23),
        (0.1, 0.2, 0.4, 0.3),
        _HIER_DIMS,
        _HIER_HEADS,
        "nested_projection_with_storage_and_budgeted_routing_v2",
        "ssepnet_ms_dinov3_v2_hier_8_6_6_4",
        dynamic_base_depths=(4, 4, 6, 4),
    ),
    "ssepnet_ms_dinov3_v2_poolhier_6_6_12": MultiscaleDINOv3V2Spec(
        "ssepnet_ms_dinov3_v2_poolhier_6_6_12",
        "pool_hierarchy_v2",
        (0, 6, 6, 12),
        (5, 11, 17, 23),
        (0.2, 0.35, 0.2, 0.25),
        (128, 1024, 1024, 1024),
        (16, 16, 16, 16),
        "strict_flat_blocks_with_storage_and_anti_alias_transitions_v2",
        "ssepnet_ms_dinov3_poolhier_6_6_6_6",
        pool_stage_depths=(6, 6, 12),
    ),
}

SSEPNET_MULTISCALE_DINOV3_V2: Mapping[str, MultiscaleDINOv3V2Spec] = (
    MappingProxyType(_SPECS)
)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def is_ssepnet_multiscale_dinov3_v2(name: str) -> bool:
    return _normalize_name(name) in SSEPNET_MULTISCALE_DINOV3_V2


def get_ssepnet_multiscale_dinov3_v2(name: str) -> MultiscaleDINOv3V2Spec:
    key = _normalize_name(name)
    try:
        return SSEPNET_MULTISCALE_DINOV3_V2[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported multiscale DINOv3 V2 variant {name!r}; choose from "
            f"{', '.join(SSEPNET_MULTISCALE_DINOV3_V2)}"
        ) from exc


def nested_source_head_ids(
    target_heads: Sequence[int], *, source_heads: int = 16
) -> dict[int, tuple[int, ...]]:
    """Build deterministic nested source-head sets by farthest-point expansion."""

    requested = tuple(sorted({int(value) for value in target_heads}))
    if not requested or requested[-1] > int(source_heads):
        raise ValueError("target head counts must be non-empty and <= source_heads")
    smallest = requested[0]
    selected = list(
        dict.fromkeys(
            int(value)
            for value in torch.linspace(0, source_heads - 1, steps=smallest)
            .round()
            .long()
            .tolist()
        )
    )
    if len(selected) != smallest:
        raise ValueError("initial nested head selection produced duplicates")
    results: dict[int, tuple[int, ...]] = {smallest: tuple(sorted(selected))}
    for target in requested[1:]:
        while len(selected) < target:
            remaining = [value for value in range(source_heads) if value not in selected]
            choice = max(
                remaining,
                key=lambda value: (
                    min(abs(value - current) for current in selected),
                    -value,
                ),
            )
            selected.append(choice)
        results[target] = tuple(sorted(selected))
    return results


def _head_channels(
    head_ids: Sequence[int], *, head_dim: int, device: torch.device
) -> torch.Tensor:
    return torch.cat(
        tuple(
            torch.arange(
                int(head_id) * int(head_dim),
                (int(head_id) + 1) * int(head_dim),
                device=device,
            )
            for head_id in head_ids
        )
    )


class AntiAliasTransition(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.depthwise = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=self.in_channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(self.in_channels), self.in_channels)
        self.pointwise = nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False)
        self.storage_projection = nn.Linear(
            self.in_channels, self.out_channels, bias=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
        ) / 16.0
        with torch.no_grad():
            self.depthwise.weight.copy_(
                kernel.view(1, 1, 3, 3).expand(self.in_channels, 1, 3, 3)
            )
            nn.init.normal_(self.pointwise.weight, std=1e-3)
            nn.init.normal_(self.storage_projection.weight, std=1e-3)
            diagonal = min(self.in_channels, self.out_channels)
            indices = torch.arange(diagonal)
            self.pointwise.weight[indices, indices, 0, 0] = 1.0
            self.storage_projection.weight[indices, indices] = 1.0
        self.norm.reset_parameters()

    def forward(
        self, features: torch.Tensor, storage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.pointwise(F.gelu(self.norm(self.depthwise(features))))
        return features, self.storage_projection(storage)


def _run_window_block_with_storage(
    block: SelfAttentionBlock,
    rope_embed: RopePositionEmbedding,
    features: torch.Tensor,
    storage: torch.Tensor,
    *,
    window_size: int,
    shift: int,
    activation_checkpoint: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    windows, metadata = _partition_windows(
        features, window_size=int(window_size), shift=int(shift)
    )
    windows_per_image = int(windows.shape[0] // metadata.batch)
    expanded_storage = (
        storage[:, None]
        .expand(-1, windows_per_image, -1, -1)
        .reshape(windows.shape[0], storage.shape[1], storage.shape[2])
    )
    tokens = torch.cat((expanded_storage, windows), dim=1)
    tokens = _run_token_block(
        block,
        tokens,
        rope_embed(height=int(window_size), width=int(window_size)),
        activation_checkpoint=activation_checkpoint,
    )
    storage_count = int(storage.shape[1])
    updated_storage = tokens[:, :storage_count].reshape(
        metadata.batch, windows_per_image, storage_count, storage.shape[2]
    ).mean(dim=1)
    features = _reverse_windows(tokens[:, storage_count:], metadata)
    return features, updated_storage


def _run_global_block_with_storage(
    block: SelfAttentionBlock,
    rope_embed: RopePositionEmbedding,
    features: torch.Tensor,
    storage: torch.Tensor,
    *,
    activation_checkpoint: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels, height, width = features.shape
    patches = features.flatten(2).transpose(1, 2)
    storage_count = int(storage.shape[1])
    tokens = _run_token_block(
        block,
        torch.cat((storage, patches), dim=1),
        rope_embed(height=height, width=width),
        activation_checkpoint=activation_checkpoint,
    )
    storage = tokens[:, :storage_count]
    features = (
        tokens[:, storage_count:]
        .transpose(1, 2)
        .reshape(batch, channels, height, width)
        .contiguous()
    )
    return features, storage


class StorageDinoStageV2(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        depth: int,
        window_size: int | None,
        activation_checkpoint: bool,
        storage_count: int = 4,
    ) -> None:
        super().__init__()
        self.window_size = None if window_size is None else int(window_size)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.blocks = nn.ModuleList(
            SelfAttentionBlock(int(dim), int(heads), ffn_ratio=4.0)
            for _ in range(int(depth))
        )
        self.rope_embed = RopePositionEmbedding(int(dim), num_heads=int(heads))
        self.storage_tokens = nn.Parameter(torch.empty(1, int(storage_count), int(dim)))
        self.storage_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.output_norm = nn.LayerNorm(int(dim), eps=1e-5)
        self.blocks.apply(_initialize_module)
        nn.init.normal_(self.storage_tokens, std=0.02)

    def ordered_blocks(self) -> tuple[SelfAttentionBlock, ...]:
        return tuple(self.blocks)

    def forward(
        self, features: torch.Tensor, incoming_storage: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stage_storage = self.storage_tokens.expand(features.shape[0], -1, -1)
        storage = (
            stage_storage
            if incoming_storage is None
            else incoming_storage
            + torch.sigmoid(self.storage_gate_logit)
            * (stage_storage - incoming_storage)
        )
        for index, block in enumerate(self.blocks):
            if self.window_size is None:
                features, storage = _run_global_block_with_storage(
                    block,
                    self.rope_embed,
                    features,
                    storage,
                    activation_checkpoint=self.activation_checkpoint
                    and self.training,
                )
            else:
                features, storage = _run_window_block_with_storage(
                    block,
                    self.rope_embed,
                    features,
                    storage,
                    window_size=self.window_size,
                    shift=0 if index % 2 == 0 else self.window_size // 2,
                    activation_checkpoint=self.activation_checkpoint
                    and self.training,
                )
        return _normalize_feature(features, self.output_norm), storage


class DynamicStorageDinoStageV2(StorageDinoStageV2):
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
        super().__init__(
            dim=dim,
            heads=heads,
            depth=base_depth,
            window_size=window_size,
            activation_checkpoint=activation_checkpoint,
        )
        self.extra_blocks = nn.ModuleList(
            SelfAttentionBlock(int(dim), int(heads), ffn_ratio=4.0)
            for _ in range(int(extra_depth))
        )
        self.extra_blocks.apply(_initialize_module)
        self.router = nn.Linear(int(dim), 1)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.routing_temperature = float(routing_temperature)
        self.register_buffer(
            "routing_thresholds",
            torch.linspace(0.35, 0.65, steps=int(extra_depth)),
            persistent=True,
        )
        self.last_soft_gates: tuple[torch.Tensor, ...] = ()
        self.last_active_fractions: tuple[torch.Tensor, ...] = ()

    def ordered_blocks(self) -> tuple[SelfAttentionBlock, ...]:
        return tuple(self.blocks) + tuple(self.extra_blocks)

    def forward(
        self, features: torch.Tensor, incoming_storage: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stage_storage = self.storage_tokens.expand(features.shape[0], -1, -1)
        storage = (
            stage_storage
            if incoming_storage is None
            else incoming_storage
            + torch.sigmoid(self.storage_gate_logit)
            * (stage_storage - incoming_storage)
        )
        for index, block in enumerate(self.blocks):
            features, storage = _run_window_block_with_storage(
                block,
                self.rope_embed,
                features,
                storage,
                window_size=int(self.window_size),
                shift=0 if index % 2 == 0 else int(self.window_size) // 2,
                activation_checkpoint=self.activation_checkpoint and self.training,
            )
        soft_gates: list[torch.Tensor] = []
        active_fractions: list[torch.Tensor] = []
        for index, block in enumerate(self.extra_blocks):
            windows, metadata = _partition_windows(
                features, window_size=int(self.window_size), shift=0
            )
            difficulty = torch.sigmoid(self.router(windows.mean(dim=1))).squeeze(-1)
            threshold = self.routing_thresholds[index].to(difficulty.dtype)
            windows_per_image = int(windows.shape[0] // metadata.batch)
            expanded_storage = (
                storage[:, None]
                .expand(-1, windows_per_image, -1, -1)
                .reshape(windows.shape[0], storage.shape[1], storage.shape[2])
            )
            storage_count = int(storage.shape[1])
            tokens = torch.cat((expanded_storage, windows), dim=1)
            rope = self.rope_embed(
                height=int(self.window_size), width=int(self.window_size)
            )
            if self.training:
                gate = torch.sigmoid(
                    (difficulty - threshold) / self.routing_temperature
                )
                candidate = _run_token_block(
                    block,
                    tokens,
                    rope,
                    activation_checkpoint=self.activation_checkpoint,
                )
                tokens = tokens + gate[:, None, None] * (candidate - tokens)
            else:
                gate = (difficulty >= threshold).to(difficulty.dtype)
                active = gate.bool()
                if bool(active.any()):
                    routed = tokens.clone()
                    routed[active] = block(tokens[active], rope)
                    tokens = routed
            storage = tokens[:, :storage_count].reshape(
                metadata.batch,
                windows_per_image,
                storage_count,
                storage.shape[2],
            ).mean(dim=1)
            features = _reverse_windows(tokens[:, storage_count:], metadata)
            soft_gates.append(gate)
            active_fractions.append((gate >= 0.5).float().mean())
        self.last_soft_gates = tuple(soft_gates)
        self.last_active_fractions = tuple(active_fractions)
        return _normalize_feature(features, self.output_norm), storage

    def routing_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.last_soft_gates:
            anchor = self.router.weight.sum() * 0.0
            return anchor, anchor
        probabilities = torch.cat(tuple(gate.reshape(-1) for gate in self.last_soft_gates))
        budget = (probabilities.mean() - 0.5).square()
        eps = torch.finfo(probabilities.dtype).eps
        probabilities = probabilities.clamp(eps, 1.0 - eps)
        negative_entropy = (
            probabilities * probabilities.log()
            + (1.0 - probabilities) * (1.0 - probabilities).log()
        ).mean()
        return budget, negative_entropy


class HierarchicalDINOv3BackboneV2(nn.Module):
    def __init__(
        self,
        *,
        in_chans: int,
        stage_depths: Sequence[int],
        stage_dims: Sequence[int],
        stage_heads: Sequence[int],
        window_size: int,
        activation_checkpoint: bool,
        dynamic_base_depths: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.stage_depths = tuple(int(value) for value in stage_depths)
        self.stage_heads = tuple(int(value) for value in stage_heads)
        self.out_channels = tuple(int(value) for value in stage_dims)
        self.out_strides = (4, 8, 16, 32)
        self.stem = nn.Sequential(
            nn.Conv2d(int(in_chans), self.out_channels[0], 4, stride=4),
            nn.GroupNorm(_group_count(self.out_channels[0]), self.out_channels[0]),
        )
        self.transitions = nn.ModuleList(
            AntiAliasTransition(self.out_channels[index], self.out_channels[index + 1])
            for index in range(3)
        )
        base_depths = (
            None
            if dynamic_base_depths is None
            else tuple(int(value) for value in dynamic_base_depths)
        )
        stages: list[nn.Module] = []
        for index, (depth, dim, heads) in enumerate(
            zip(self.stage_depths, self.out_channels, self.stage_heads, strict=True)
        ):
            window = int(window_size) if index < 2 else None
            base = depth if base_depths is None else base_depths[index]
            extra = depth - base
            if extra > 0:
                stages.append(
                    DynamicStorageDinoStageV2(
                        dim=dim,
                        heads=heads,
                        base_depth=base,
                        extra_depth=extra,
                        window_size=int(window_size),
                        activation_checkpoint=activation_checkpoint,
                    )
                )
            else:
                stages.append(
                    StorageDinoStageV2(
                        dim=dim,
                        heads=heads,
                        depth=depth,
                        window_size=window,
                        activation_checkpoint=activation_checkpoint,
                    )
                )
        self.stages = nn.ModuleList(stages)
        self.distill_features: tuple[torch.Tensor, ...] = ()
        self.initialization_report: dict[str, object] = {"policy": "random"}

    def ordered_blocks(self) -> tuple[SelfAttentionBlock, ...]:
        return tuple(
            block for stage in self.stages for block in stage.ordered_blocks()
        )

    def student_transformer_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for stage in self.stages
            for block in stage.ordered_blocks()
            for parameter in block.parameters()
        )

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        features = self.stem(inputs)
        storage: torch.Tensor | None = None
        outputs: list[torch.Tensor] = []
        for index, stage in enumerate(self.stages):
            features, storage = stage(features, storage)
            outputs.append(features)
            if index < 3:
                features, storage = self.transitions[index](features, storage)
        self.distill_features = tuple(outputs)
        return outputs

    def routing_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        values = [
            stage.routing_regularization()
            for stage in self.stages
            if isinstance(stage, DynamicStorageDinoStageV2)
        ]
        if not values:
            anchor = next(self.parameters()).sum() * 0.0
            return anchor, anchor
        return (
            torch.stack([value[0] for value in values]).mean(),
            torch.stack([value[1] for value in values]).mean(),
        )

    def routing_statistics(self) -> tuple[dict[str, tuple[float, ...]], ...]:
        rows: list[dict[str, tuple[float, ...]]] = []
        for stage in self.stages:
            if not isinstance(stage, DynamicStorageDinoStageV2):
                continue
            rows.append(
                {
                    "active_fraction": tuple(
                        float(value.detach().cpu())
                        for value in stage.last_active_fractions
                    )
                }
            )
        return tuple(rows)

    def routing_compute_reduction(self, *, image_size: int = 512) -> float | None:
        base_depths: list[int] = []
        active_means: list[float] = []
        has_dynamic = False
        for stage, maximum_depth in zip(
            self.stages, self.stage_depths, strict=True
        ):
            if isinstance(stage, DynamicStorageDinoStageV2):
                has_dynamic = True
                base_depths.append(len(stage.blocks))
                active_means.append(
                    float(
                        sum(
                            float(value.detach().cpu())
                            for value in stage.last_active_fractions
                        )
                        / max(1, len(stage.last_active_fractions))
                    )
                )
            else:
                base_depths.append(int(maximum_depth))
                active_means.append(0.0)
        if not has_dynamic:
            return None
        costs = [
            (int(image_size) // stride) ** 2 * dim**2
            for stride, dim in zip(
                self.out_strides, self.out_channels, strict=True
            )
        ]
        full = sum(
            depth * cost
            for depth, cost in zip(self.stage_depths, costs, strict=True)
        )
        executed = 0.0
        for maximum, base, active, cost in zip(
            self.stage_depths,
            base_depths,
            active_means,
            costs,
            strict=True,
        ):
            executed += (base + (maximum - base) * active) * cost
        return 1.0 - executed / full


def _copy_projected_block_v2(
    target: SelfAttentionBlock,
    source: SelfAttentionBlock,
    channels: torch.Tensor,
) -> None:
    source_dim = int(source.norm1.weight.numel())
    source_hidden = int(source.mlp.fc1.out_features)
    target_hidden = int(target.mlp.fc1.out_features)
    hidden = _uniform_indices(source_hidden, target_hidden, device=channels.device)
    qkv_rows = torch.cat(
        tuple(channels + branch * source_dim for branch in range(3)), dim=0
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


def initialize_hierarchical_v2_from_flat(
    target: HierarchicalDINOv3BackboneV2, source_backbone: nn.Module
) -> dict[str, object]:
    source = source_backbone.model
    head_sets = nested_source_head_ids(target.stage_heads, source_heads=source.num_heads)
    head_dim = int(source.embed_dim) // int(source.num_heads)
    stage_channels = [
        _head_channels(
            head_sets[heads], head_dim=head_dim, device=source.norm.weight.device
        )
        for heads in target.stage_heads
    ]
    block_index = 0
    for stage, channels in zip(target.stages, stage_channels, strict=True):
        for block in stage.ordered_blocks():
            _copy_projected_block_v2(block, source.blocks[block_index], channels)
            block_index += 1
        with torch.no_grad():
            stage.storage_tokens.copy_(
                source.storage_tokens.index_select(2, channels)
            )
            stage.output_norm.weight.copy_(source.norm.weight.index_select(0, channels))
            stage.output_norm.bias.copy_(source.norm.bias.index_select(0, channels))
            if stage.rope_embed.periods.shape == source.rope_embed.periods.shape:
                stage.rope_embed.periods.copy_(source.rope_embed.periods)
    with torch.no_grad():
        source_weight = source.patch_embed.proj.weight.index_select(0, stage_channels[0])
        resized = F.adaptive_avg_pool2d(source_weight, (4, 4))
        target.stem[0].weight.copy_(resized * 16.0)
        if target.stem[0].bias is not None and source.patch_embed.proj.bias is not None:
            target.stem[0].bias.copy_(
                source.patch_embed.proj.bias.index_select(0, stage_channels[0])
            )
        for transition, left, right in zip(
            target.transitions, stage_channels[:-1], stage_channels[1:], strict=True
        ):
            left_lookup = {int(value): index for index, value in enumerate(left.tolist())}
            transition.pointwise.weight.normal_(std=1e-3)
            transition.storage_projection.weight.normal_(std=1e-3)
            for output_index, source_channel in enumerate(right.tolist()):
                input_index = left_lookup.get(int(source_channel))
                if input_index is None:
                    continue
                transition.pointwise.weight[output_index, input_index, 0, 0] = 1.0
                transition.storage_projection.weight[output_index, input_index] = 1.0
    report = {
        "policy": "nested_complete_head_projection_with_storage_tokens_v2",
        "source_blocks": 24,
        "projected_source_blocks": sum(
            depth
            for depth, dim in zip(
                target.stage_depths, target.out_channels, strict=True
            )
            if dim != int(source.embed_dim)
        ),
        "nested_source_heads_by_stage": tuple(
            head_sets[heads] for heads in target.stage_heads
        ),
        "storage_tokens_preserved": True,
        "anti_alias_transitions": True,
    }
    target.initialization_report = report
    target.register_buffer(
        "distillation_channel_indices_0", stage_channels[0].cpu(), persistent=False
    )
    target.register_buffer(
        "distillation_channel_indices_1", stage_channels[1].cpu(), persistent=False
    )
    target.register_buffer(
        "distillation_channel_indices_2", stage_channels[2].cpu(), persistent=False
    )
    target.register_buffer(
        "distillation_channel_indices_3", stage_channels[3].cpu(), persistent=False
    )
    return report


def register_random_distillation_indices(
    target: HierarchicalDINOv3BackboneV2, *, source_heads: int = 16, head_dim: int = 64
) -> None:
    head_sets = nested_source_head_ids(target.stage_heads, source_heads=source_heads)
    for index, heads in enumerate(target.stage_heads):
        channels = _head_channels(
            head_sets[heads], head_dim=head_dim, device=torch.device("cpu")
        )
        target.register_buffer(
            f"distillation_channel_indices_{index}", channels, persistent=False
        )


class PoolHierarchyDINOv3BackboneV2(nn.Module):
    def __init__(
        self,
        *,
        dinov3_backbone: nn.Module,
        in_chans: int,
        window_size: int,
        activation_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.dinov3_backbone = dinov3_backbone
        self.spatial_stem = SpatialPyramidStem(in_chans, (128, 256, 512, 512))
        dino_dim = int(dinov3_backbone.model.embed_dim)
        self.p3_projection = nn.Conv2d(256, dino_dim, 1, bias=False)
        self.p3_prior_gate = SemanticResidualGate(dino_dim, gate_init_bias=-2.0)
        self.p4_transition = AntiAliasTransition(dino_dim, dino_dim)
        self.p5_transition = AntiAliasTransition(dino_dim, dino_dim)
        self.p4_detail_gate = SemanticResidualGate(dino_dim, gate_init_bias=-2.0)
        self.window_size = int(window_size)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.out_channels = (128, dino_dim, dino_dim, dino_dim)
        self.out_strides = (4, 8, 16, 32)
        self.distill_features: tuple[torch.Tensor, ...] = ()
        self.initialization_report = {
            "policy": "strict_flat_blocks_with_storage_and_anti_alias_transitions_v2",
            "strict_source_blocks": 24,
            "projected_source_blocks": 0,
            "storage_tokens_preserved": True,
            "stage_mapping": "p2_stem+p3_6+p4_6+p5_12",
        }

    def student_transformer_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.dinov3_backbone.model.blocks.parameters())

    def _run_blocks(
        self,
        features: torch.Tensor,
        storage: torch.Tensor,
        *,
        start: int,
        end: int,
        windowed: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.dinov3_backbone.model
        for block_index in range(start, end):
            block = model.blocks[block_index]
            if windowed:
                features, storage = _run_window_block_with_storage(
                    block,
                    model.rope_embed,
                    features,
                    storage,
                    window_size=self.window_size,
                    shift=0
                    if (block_index - start) % 2 == 0
                    else self.window_size // 2,
                    activation_checkpoint=self.activation_checkpoint
                    and self.training,
                )
            else:
                features, storage = _run_global_block_with_storage(
                    block,
                    model.rope_embed,
                    features,
                    storage,
                    activation_checkpoint=self.activation_checkpoint
                    and self.training,
                )
        return _normalize_feature(features, model.norm), storage

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
        storage = model.storage_tokens.expand(inputs.shape[0], -1, -1)
        p3, storage = self._run_blocks(
            p3, storage, start=0, end=6, windowed=True
        )
        transitioned_p4, storage = self.p4_transition(p3, storage)
        patch4 = _resize_like(patch4, transitioned_p4)
        p4 = self.p4_detail_gate(patch4, transitioned_p4)
        p4, storage = self._run_blocks(
            p4, storage, start=6, end=12, windowed=False
        )
        p5, storage = self.p5_transition(p4, storage)
        p5_mid, storage = self._run_blocks(
            p5, storage, start=12, end=18, windowed=False
        )
        p5_final, storage = self._run_blocks(
            p5_mid, storage, start=18, end=24, windowed=False
        )
        del storage
        self.distill_features = (p3, p4, p5_mid, p5_final)
        return [p2, p3, p4, p5_final]


class P4AnchoredSemanticFusion(nn.Module):
    def __init__(self, channels: int, *, gate_bias: float = -1.0) -> None:
        super().__init__()
        self.gate = nn.Conv2d(int(channels) * 2, int(channels), 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias))
        self.last_gate_statistics: torch.Tensor | None = None
        self.last_weights: torch.Tensor | None = None

    def forward(
        self, high_level: torch.Tensor, p4_anchor: torch.Tensor
    ) -> torch.Tensor:
        high_level = _resize_like(high_level, p4_anchor)
        gate = torch.sigmoid(self.gate(torch.cat((p4_anchor, high_level), dim=1)))
        detached = gate.detach()
        self.last_gate_statistics = torch.stack(
            (
                detached.mean(),
                detached.std(unbiased=False),
                detached.amin(),
                detached.amax(),
            )
        )
        return p4_anchor + gate * high_level


class P4AnchoredPyramidOCRHead(SemanticAnchorGatedPyramidOCRHead):
    def __init__(self, input_channels: Sequence[int], **kwargs) -> None:
        super().__init__(input_channels, **kwargs)
        self.p4_semantic_fusion = P4AnchoredSemanticFusion(
            int(kwargs.get("decoder_channels", 128)), gate_bias=-1.0
        )

    def gate_statistics(self) -> dict[str, tuple[float, ...]]:
        values = super().gate_statistics()
        values.pop("p4_p5_semantic_weights", None)
        stats = self.p4_semantic_fusion.last_gate_statistics
        if stats is not None:
            values["p4_p5"] = tuple(float(value) for value in stats.float().cpu())
        return values


class DistillableMultiscaleDINOv3V2(MultiscaleDINOv3SSepNet):
    def __init__(
        self,
        *,
        multiscale_backbone: nn.Module,
        spec: MultiscaleDINOv3V2Spec,
        decoder_channels: int,
        ocr_channels: int,
        num_classes: int,
        freeze_backbone: bool,
    ) -> None:
        super().__init__(
            multiscale_backbone=multiscale_backbone,
            decoder_channels=decoder_channels,
            ocr_channels=ocr_channels,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
        )
        self.head = P4AnchoredPyramidOCRHead(
            tuple(int(value) for value in multiscale_backbone.out_channels),
            decoder_channels=decoder_channels,
            ocr_channels=ocr_channels,
        )
        self.v2_spec = spec

    @property
    def teacher_taps(self) -> tuple[int, int, int, int]:
        return self.v2_spec.teacher_taps

    @property
    def feature_loss_weights(self) -> tuple[float, float, float, float]:
        return self.v2_spec.feature_loss_weights

    def distillation_channel_indices(self) -> tuple[torch.Tensor, ...]:
        if isinstance(self.multiscale_backbone, PoolHierarchyDINOv3BackboneV2):
            channels = int(self.multiscale_backbone.out_channels[1])
            return tuple(torch.arange(channels) for _ in range(4))
        return tuple(
            getattr(
                self.multiscale_backbone,
                f"distillation_channel_indices_{index}",
            )
            for index in range(4)
        )

    def student_transformer_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.multiscale_backbone.student_transformer_parameters()

    def routing_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        callback = getattr(self.multiscale_backbone, "routing_regularization", None)
        if callback is None:
            anchor = next(self.parameters()).sum() * 0.0
            return anchor, anchor
        return callback()

    def routing_compute_reduction(self) -> float | None:
        callback = getattr(
            self.multiscale_backbone, "routing_compute_reduction", None
        )
        return None if callback is None else callback(image_size=512)

    def forward_distill_features(
        self, inputs: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
        features = list(self.multiscale_backbone(inputs))
        outputs = self.head(features)
        outputs = _resize_outputs(outputs, tuple(inputs.shape[-2:]))
        distill_features = tuple(self.multiscale_backbone.distill_features)
        if len(distill_features) != 4:
            raise RuntimeError("V2 backbone must expose four distillation features")
        return outputs, distill_features  # type: ignore[return-value]


def build_ssepnet_multiscale_dinov3_v2(
    name: str,
    *,
    in_chans: int = 4,
    num_classes: int = 1,
    pretrained: bool = True,
    dinov3_checkpoint: str | Path | None = None,
    freeze_backbone: bool = False,
    activation_checkpoint: bool = True,
    decoder_channels: int = 128,
    ocr_channels: int = 256,
    **_ignored,
) -> DistillableMultiscaleDINOv3V2:
    spec = get_ssepnet_multiscale_dinov3_v2(name)
    from .dinov3 import build_dinov3_vitl16_sat493m

    flat = build_dinov3_vitl16_sat493m(
        in_chans=in_chans,
        pretrained=pretrained,
        checkpoint_path=dinov3_checkpoint,
    )
    if spec.architecture == "pool_hierarchy_v2":
        backbone: nn.Module = PoolHierarchyDINOv3BackboneV2(
            dinov3_backbone=flat,
            in_chans=in_chans,
            window_size=spec.window_size,
            activation_checkpoint=activation_checkpoint,
        )
    else:
        backbone = HierarchicalDINOv3BackboneV2(
            in_chans=in_chans,
            stage_depths=spec.stage_depths,
            stage_dims=spec.stage_dims,
            stage_heads=spec.stage_heads,
            window_size=spec.window_size,
            activation_checkpoint=activation_checkpoint,
            dynamic_base_depths=spec.dynamic_base_depths,
        )
        if pretrained:
            initialize_hierarchical_v2_from_flat(backbone, flat)
        else:
            register_random_distillation_indices(backbone)
        del flat
    backbone.initialization_report.update(
        {
            "architecture": spec.architecture,
            "stage_depths": spec.stage_depths,
            "teacher_taps": spec.teacher_taps,
            "feature_loss_weights": spec.feature_loss_weights,
        }
    )
    return DistillableMultiscaleDINOv3V2(
        multiscale_backbone=backbone,
        spec=spec,
        decoder_channels=decoder_channels,
        ocr_channels=ocr_channels,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
    )


def build_ssepnet_multiscale_dinov3_v2_loss(name: str) -> nn.Module:
    get_ssepnet_multiscale_dinov3_v2(name)
    from geoseg.losses.ssepnet import SSepNetLoss

    return SSepNetLoss(boundary_weight=0.6, pbcl_weight=0.1)


__all__ = [
    "SSEPNET_MULTISCALE_DINOV3_V2",
    "AntiAliasTransition",
    "DistillableMultiscaleDINOv3V2",
    "DynamicStorageDinoStageV2",
    "HierarchicalDINOv3BackboneV2",
    "P4AnchoredSemanticFusion",
    "PoolHierarchyDINOv3BackboneV2",
    "build_ssepnet_multiscale_dinov3_v2",
    "build_ssepnet_multiscale_dinov3_v2_loss",
    "get_ssepnet_multiscale_dinov3_v2",
    "initialize_hierarchical_v2_from_flat",
    "is_ssepnet_multiscale_dinov3_v2",
    "nested_source_head_ids",
]
