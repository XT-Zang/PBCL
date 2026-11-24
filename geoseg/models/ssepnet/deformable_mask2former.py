# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# The deformable-attention and masked-decoder portions are standalone adaptations of the
# official DINOv3 segmentation implementation and are governed by the DINOv3 license in
# third_party/dinov3/LICENSE.md. GeoSeg-specific dual-backbone wiring is original here.

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiscale import (
    ConvGNReLU,
    DPTReassemble,
    DualBackboneCandidateBase,
    _resize_like,
    _resize_outputs,
)


def multi_scale_deformable_attention_pytorch(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Autograd-safe PyTorch implementation of multi-scale deformable attention."""

    output_dtype = value.dtype
    if value.dtype in {torch.float16, torch.bfloat16}:
        value = value.float()
        sampling_locations = sampling_locations.float()
        attention_weights = attention_weights.float()
    batch, _length, heads, head_channels = value.shape
    _batch, queries, _heads, levels, points, _xy = sampling_locations.shape
    splits = [int(height * width) for height, width in spatial_shapes.tolist()]
    values = value.split(splits, dim=1)
    grids = sampling_locations.mul(2.0).sub(1.0)
    sampled_values: list[torch.Tensor] = []
    for level, (height, width) in enumerate(spatial_shapes.tolist()):
        level_value = (
            values[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch * heads, head_channels, int(height), int(width))
        )
        level_grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled_values.append(
            F.grid_sample(
                level_value,
                level_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    weights = attention_weights.transpose(1, 2).reshape(
        batch * heads, 1, queries, levels * points
    )
    output = (
        torch.stack(sampled_values, dim=-2).flatten(-2).mul(weights).sum(-1)
    ).view(batch, heads * head_channels, queries)
    return output.transpose(1, 2).contiguous().to(dtype=output_dtype)


class MultiScaleDeformableAttention(nn.Module):
    """Deformable-DETR attention without an external CUDA extension requirement."""

    def __init__(
        self,
        channels: int,
        *,
        levels: int,
        heads: int,
        points: int,
    ) -> None:
        super().__init__()
        channels = int(channels)
        if channels % int(heads) != 0:
            raise ValueError("deformable attention heads must divide channels")
        self.channels = channels
        self.levels = int(levels)
        self.heads = int(heads)
        self.points = int(points)
        self.sampling_offsets = nn.Linear(
            channels, self.heads * self.levels * self.points * 2
        )
        self.attention_weights = nn.Linear(
            channels, self.heads * self.levels * self.points
        )
        self.value_projection = nn.Linear(channels, channels)
        self.output_projection = nn.Linear(channels, channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.sampling_offsets.weight)
        theta = torch.arange(self.heads, dtype=torch.float32) * (
            2.0 * math.pi / self.heads
        )
        directions = torch.stack((theta.cos(), theta.sin()), dim=-1)
        directions = directions / directions.abs().amax(dim=-1, keepdim=True)
        directions = directions.view(self.heads, 1, 1, 2).repeat(
            1, self.levels, self.points, 1
        )
        for index in range(self.points):
            directions[:, :, index] *= index + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(directions.flatten())
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.zeros_(self.value_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        values: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_count, _channels = query.shape
        projected = self.value_projection(values).view(
            batch,
            values.shape[1],
            self.heads,
            self.channels // self.heads,
        )
        offsets = self.sampling_offsets(query).view(
            batch,
            query_count,
            self.heads,
            self.levels,
            self.points,
            2,
        )
        weights = self.attention_weights(query).view(
            batch,
            query_count,
            self.heads,
            self.levels * self.points,
        )
        weights = torch.softmax(weights, dim=-1).view(
            batch,
            query_count,
            self.heads,
            self.levels,
            self.points,
        )
        normalizer = torch.stack((spatial_shapes[:, 1], spatial_shapes[:, 0]), dim=-1)
        locations = (
            reference_points[:, :, None, :, None]
            + offsets / normalizer[None, None, None, :, None]
        )
        output = multi_scale_deformable_attention_pytorch(
            projected, spatial_shapes, locations, weights
        )
        return self.output_projection(output)


def _reference_points(
    spatial_shapes: torch.Tensor,
    *,
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    references: list[torch.Tensor] = []
    for height, width in spatial_shapes.tolist():
        y, x = torch.meshgrid(
            torch.arange(int(height), device=device, dtype=torch.float32) + 0.5,
            torch.arange(int(width), device=device, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        references.append(
            torch.stack((x / float(width), y / float(height)), dim=-1).flatten(0, 1)
        )
    reference = torch.cat(references, dim=0)[None, :, None, :]
    return reference.repeat(int(batch), 1, int(spatial_shapes.shape[0]), 1)


class SinePositionEmbedding(nn.Module):
    def __init__(self, channels: int, temperature: float = 10000.0) -> None:
        super().__init__()
        if channels % 4 != 0:
            raise ValueError(
                "sine position embedding channels must be divisible by four"
            )
        self.channels = int(channels)
        self.temperature = float(temperature)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, _channels, height, width = feature.shape
        y = torch.linspace(0.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        x = torch.linspace(0.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        y, x = torch.meshgrid(y, x, indexing="ij")
        dim = torch.arange(
            self.channels // 4, device=feature.device, dtype=feature.dtype
        )
        dim = self.temperature ** (
            2.0 * torch.div(dim, 2, rounding_mode="floor") / max(1, self.channels // 4)
        )
        pos_x = x[..., None] * (2.0 * math.pi) / dim
        pos_y = y[..., None] * (2.0 * math.pi) / dim
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)
        position = torch.cat((pos_y, pos_x), dim=-1)[..., : self.channels]
        return position.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)


class DeformableEncoderLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        levels: int,
        heads: int,
        points: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = MultiScaleDeformableAttention(
            channels, levels=levels, heads=heads, points=points
        )
        self.dropout1 = nn.Dropout(float(dropout))
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(channels * 4, channels),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        source: torch.Tensor,
        position: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm1(source)
        attended = self.attention(
            normalized + position,
            reference_points,
            normalized,
            spatial_shapes,
        )
        source = source + self.dropout1(attended)
        return source + self.ffn(self.norm2(source))


class DinoHRPyramidAdapter(nn.Module):
    """Trainable downstream adapter aligning DINO layers with HRNet spatial priors."""

    def __init__(
        self,
        hr_channels: Sequence[int],
        dino_layer_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        if len(hr_channels) != 4:
            raise ValueError("DINO-HR pyramid adapter requires four HRNet levels")
        self.hr_projections = nn.ModuleList(
            ConvGNReLU(channels, hidden_channels, 1) for channels in hr_channels
        )
        self.dino_reassemble = DPTReassemble(dino_layer_channels, hidden_channels)
        self.fusions = nn.ModuleList(
            ConvGNReLU(hidden_channels * 2, hidden_channels, 3) for _ in range(4)
        )

    def forward(
        self,
        hr_features: Sequence[torch.Tensor],
        dino_layers: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        hr = [
            projection(feature)
            for projection, feature in zip(
                self.hr_projections, hr_features, strict=True
            )
        ]
        dino = self.dino_reassemble(dino_layers, hr)
        return [
            fusion(torch.cat((hr_feature, dino_feature), dim=1))
            for fusion, hr_feature, dino_feature in zip(
                self.fusions, hr, dino, strict=True
            )
        ]


class DeformablePixelDecoder(nn.Module):
    """MS-Deformable-Attention encoder plus a stride-4 top-down mask feature path."""

    def __init__(
        self,
        hidden_channels: int,
        *,
        encoder_layers: int,
        heads: int,
        points: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.position = SinePositionEmbedding(hidden_channels)
        self.level_embedding = nn.Parameter(torch.zeros(3, hidden_channels))
        nn.init.normal_(self.level_embedding, std=0.02)
        self.encoder = nn.ModuleList(
            DeformableEncoderLayer(
                hidden_channels,
                levels=3,
                heads=heads,
                points=points,
                dropout=dropout,
            )
            for _ in range(int(encoder_layers))
        )
        self.smooth = nn.ModuleList(
            ConvGNReLU(hidden_channels, hidden_channels, 3) for _ in range(4)
        )
        self.mask_projection = nn.Conv2d(hidden_channels, hidden_channels, 1)

    def forward(
        self, features: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if len(features) != 4:
            raise ValueError("deformable pixel decoder requires four feature levels")
        transformer_features = list(features[1:])
        spatial_shapes = torch.as_tensor(
            [feature.shape[-2:] for feature in transformer_features],
            dtype=torch.long,
            device=features[0].device,
        )
        source_parts: list[torch.Tensor] = []
        position_parts: list[torch.Tensor] = []
        for level, feature in enumerate(transformer_features):
            source_parts.append(feature.flatten(2).transpose(1, 2))
            position_parts.append(
                self.position(feature).flatten(2).transpose(1, 2)
                + self.level_embedding[level].view(1, 1, -1)
            )
        source = torch.cat(source_parts, dim=1)
        position = torch.cat(position_parts, dim=1)
        references = _reference_points(
            spatial_shapes,
            batch=int(source.shape[0]),
            device=source.device,
        )
        for layer in self.encoder:
            source = layer(source, position, references, spatial_shapes)
        lengths = [int(height * width) for height, width in spatial_shapes.tolist()]
        encoded_tokens = source.split(lengths, dim=1)
        encoded = [
            tokens.transpose(1, 2).reshape(
                source.shape[0], source.shape[2], int(shape[0]), int(shape[1])
            )
            for tokens, shape in zip(
                encoded_tokens, spatial_shapes.tolist(), strict=True
            )
        ]
        p3 = self.smooth[3](encoded[2])
        p2 = self.smooth[2](encoded[1] + _resize_like(p3, encoded[1]))
        p1 = self.smooth[1](encoded[0] + _resize_like(p2, encoded[0]))
        p0 = self.smooth[0](features[0] + _resize_like(p1, features[0]))
        return self.mask_projection(p0), [p1, p2, p3]


class MLP(nn.Module):
    def __init__(self, channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, output_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class MaskedDecoderLayer(nn.Module):
    def __init__(self, channels: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            channels, heads, dropout=float(dropout), batch_first=True
        )
        self.self_attention = nn.MultiheadAttention(
            channels, heads, dropout=float(dropout), batch_first=True
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(channels * 4, channels),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        query: torch.Tensor,
        query_position: torch.Tensor,
        memory: torch.Tensor,
        memory_position: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.norm1(query)
        cross, _ = self.cross_attention(
            normalized + query_position,
            memory + memory_position,
            memory,
            attn_mask=attention_mask,
            need_weights=False,
        )
        query = query + cross
        normalized = self.norm2(query)
        self_attended, _ = self.self_attention(
            normalized + query_position,
            normalized + query_position,
            normalized,
            need_weights=False,
        )
        query = query + self_attended
        return query + self.ffn(self.norm3(query))


class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        *,
        num_queries: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.num_queries = int(num_queries)
        self.heads = int(heads)
        self.position = SinePositionEmbedding(hidden_channels)
        self.level_embedding = nn.Parameter(torch.zeros(3, hidden_channels))
        nn.init.normal_(self.level_embedding, std=0.02)
        self.query_content = nn.Embedding(num_queries, hidden_channels)
        self.query_position = nn.Embedding(num_queries, hidden_channels)
        self.layers = nn.ModuleList(
            MaskedDecoderLayer(hidden_channels, heads, dropout)
            for _ in range(int(layers))
        )
        self.output_norm = nn.LayerNorm(hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 2)
        self.mask_embedding = MLP(hidden_channels, hidden_channels)

    def _prediction(
        self,
        query: torch.Tensor,
        mask_features: torch.Tensor,
        target_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.output_norm(query)
        class_logits = self.classifier(normalized)
        mask_logits = torch.einsum(
            "bqc,bchw->bqhw", self.mask_embedding(normalized), mask_features
        )
        attention_mask = F.interpolate(
            mask_logits, size=target_size, mode="bilinear", align_corners=False
        )
        attention_mask = attention_mask.sigmoid().flatten(2) < 0.5
        attention_mask = (
            attention_mask[:, None].repeat(1, self.heads, 1, 1).flatten(0, 1).detach()
        )
        fully_masked = attention_mask.all(dim=-1)
        attention_mask[fully_masked] = False
        return class_logits, mask_logits, attention_mask

    def forward(
        self,
        features: Sequence[torch.Tensor],
        mask_features: torch.Tensor,
    ) -> dict[str, object]:
        if len(features) != 3:
            raise ValueError("masked transformer decoder requires three feature levels")
        batch = int(features[0].shape[0])
        memory = [feature.flatten(2).transpose(1, 2) for feature in features]
        positions = [
            self.position(feature).flatten(2).transpose(1, 2)
            + self.level_embedding[index].view(1, 1, -1)
            for index, feature in enumerate(features)
        ]
        query = self.query_content.weight.unsqueeze(0).repeat(batch, 1, 1)
        query_position = self.query_position.weight.unsqueeze(0).repeat(batch, 1, 1)
        class_predictions: list[torch.Tensor] = []
        mask_predictions: list[torch.Tensor] = []
        class_logits, mask_logits, attention_mask = self._prediction(
            query, mask_features, tuple(features[0].shape[-2:])
        )
        class_predictions.append(class_logits)
        mask_predictions.append(mask_logits)
        for index, layer in enumerate(self.layers):
            level = index % 3
            query = layer(
                query,
                query_position,
                memory[level],
                positions[level],
                attention_mask,
            )
            next_level = (index + 1) % 3
            class_logits, mask_logits, attention_mask = self._prediction(
                query, mask_features, tuple(features[next_level].shape[-2:])
            )
            class_predictions.append(class_logits)
            mask_predictions.append(mask_logits)
        return {
            "pred_logits": class_predictions[-1],
            "pred_masks": mask_predictions[-1],
            "aux_outputs": [
                {"pred_logits": logits, "pred_masks": masks}
                for logits, masks in zip(
                    class_predictions[:-1], mask_predictions[:-1], strict=True
                )
            ],
        }


def semantic_union_logits(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
) -> torch.Tensor:
    foreground = torch.softmax(class_logits, dim=-1)[..., 0]
    contribution = foreground[:, :, None, None] * torch.sigmoid(mask_logits)
    probability = contribution.amax(dim=1, keepdim=True)
    return torch.logit(probability.clamp(1e-6, 1.0 - 1e-6))


class DeformableMask2FormerSSepNet(DualBackboneCandidateBase):
    """Candidate 5: DINO-HR adapter, deformable pixel decoder, and masked instance queries."""

    def __init__(
        self,
        *,
        hrnet_backbone: nn.Module,
        dinov3_backbone: nn.Module,
        hidden_channels: int = 128,
        num_queries: int = 64,
        pixel_encoder_layers: int = 2,
        decoder_layers: int = 3,
        num_heads: int = 8,
        deformable_points: int = 4,
        freeze_backbones: bool = False,
        activation_checkpoint: bool = True,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(
            hrnet_backbone=hrnet_backbone,
            dinov3_backbone=dinov3_backbone,
            freeze_backbones=freeze_backbones,
            activation_checkpoint=activation_checkpoint,
        )
        if int(hidden_channels) % int(num_heads) != 0:
            raise ValueError("Mask2Former num_heads must divide hidden_channels")
        self.num_queries = int(num_queries)
        self.adapter = DinoHRPyramidAdapter(
            self.stage_channels, self.dino_layer_channels, hidden_channels
        )
        self.pixel_decoder = DeformablePixelDecoder(
            hidden_channels,
            encoder_layers=pixel_encoder_layers,
            heads=num_heads,
            points=deformable_points,
            dropout=dropout,
        )
        self.mask_decoder = MultiScaleMaskedTransformerDecoder(
            hidden_channels,
            num_queries=num_queries,
            layers=decoder_layers,
            heads=num_heads,
            dropout=dropout,
        )
        self.auxiliary = nn.Conv2d(hidden_channels, 1, 1)
        self.boundary = nn.Sequential(
            ConvGNReLU(self.stage_channels[0], max(32, hidden_channels // 2), 3),
            nn.Conv2d(max(32, hidden_channels // 2), 1, 1),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        main_only: bool = False,
        disable_hrnet: bool = False,
        disable_dinov3: bool = False,
    ) -> dict[str, object]:
        hr_features, dino_layers = self._backbone_features(
            inputs, disable_hrnet=disable_hrnet, disable_dinov3=disable_dinov3
        )
        adapted = self.adapter(hr_features, dino_layers)
        mask_features, decoder_features = self.pixel_decoder(adapted)
        query_outputs = self.mask_decoder(decoder_features, mask_features)
        main = semantic_union_logits(
            query_outputs["pred_logits"], query_outputs["pred_masks"]
        )
        outputs: dict[str, object] = {
            "out": main,
            "aux": self.auxiliary(decoder_features[0]),
            "edge": self.boundary(hr_features[0]),
            **query_outputs,
        }
        if main_only:
            outputs = {"out": outputs["out"]}
        return _resize_outputs(outputs, tuple(inputs.shape[-2:]))


__all__ = [
    "DeformableMask2FormerSSepNet",
    "DeformablePixelDecoder",
    "DinoHRPyramidAdapter",
    "MultiScaleDeformableAttention",
    "MultiScaleMaskedTransformerDecoder",
    "multi_scale_deformable_attention_pytorch",
    "semantic_union_logits",
]
