from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint_forward

from .deformable_mask2former import (
    DeformablePixelDecoder,
    MultiScaleMaskedTransformerDecoder,
    semantic_union_logits,
)
from .multiscale import (
    BiFPNCell,
    ConvGNReLU,
    DPTReassemble,
    DirectPyramidAggregator,
    HistoricalDinoScaleMapper,
    PyramidBinaryHead,
    PyramidOCRHead,
    SoftLayerMixPyramid,
    StandardPyramidDecoder,
    _resize_like,
    _resize_outputs,
)


def _pyramid_references(inputs: torch.Tensor) -> list[torch.Tensor]:
    batch, _channels, height, width = inputs.shape
    return [
        inputs.new_zeros(
            (
                batch,
                1,
                (int(height) + stride - 1) // stride,
                (int(width) + stride - 1) // stride,
            )
        )
        for stride in (4, 8, 16, 32)
    ]


class DinoOnlyCandidateBase(nn.Module):
    """Physical DINOv3-only contract shared by the five matched ablations."""

    active_backbones = ("dinov3",)

    def __init__(
        self,
        *,
        dinov3_backbone: nn.Module,
        freeze_backbone: bool,
        activation_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.dinov3_backbone = dinov3_backbone
        self.freeze_backbone = bool(freeze_backbone)
        self.activation_checkpoint = bool(activation_checkpoint)
        out_channels = int(dinov3_backbone.out_channels)
        self.dino_layer_channels = int(
            getattr(dinov3_backbone, "embed_dim", out_channels // 4)
        )
        if self.dino_layer_channels * 4 != out_channels:
            raise ValueError(
                "DINO-only candidates require four equal-width intermediate layers, got "
                f"out_channels={out_channels}, layer_channels={self.dino_layer_channels}"
            )
        if self.freeze_backbone:
            self.dinov3_backbone.requires_grad_(False)

    def _forward_dinov3_layers(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        def forward_layers(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
            if hasattr(self.dinov3_backbone, "forward_layers"):
                values = self.dinov3_backbone.forward_layers(tensor)
            else:
                values = self.dinov3_backbone(tensor).chunk(4, dim=1)
            return tuple(values)

        trainable = self.training and any(
            parameter.requires_grad for parameter in self.dinov3_backbone.parameters()
        )
        if trainable and self.activation_checkpoint:
            layers = activation_checkpoint_forward(
                forward_layers, inputs, use_reentrant=False
            )
        else:
            layers = forward_layers(inputs)
        if len(layers) != 4:
            raise ValueError(f"DINOv3 must expose four layers, got {len(layers)}")
        return list(layers)

    def _features(
        self, inputs: torch.Tensor, *, disable_dinov3: bool
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        layers = self._forward_dinov3_layers(inputs)
        if disable_dinov3:
            layers = [torch.zeros_like(value) for value in layers]
        return layers, _pyramid_references(inputs)

    def train(self, mode: bool = True) -> "DinoOnlyCandidateBase":
        super().train(mode)
        if self.freeze_backbone:
            self.dinov3_backbone.eval()
        return self

    def forward_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.forward(inputs, main_only=True)["out"]
        if not isinstance(output, torch.Tensor):
            raise TypeError("DINO-only candidate inference output must be a tensor")
        return output


class DinoOnlyScaleProjector(nn.Module):
    def __init__(self, input_channels: Sequence[int], output_channels: int) -> None:
        super().__init__()
        if len(input_channels) != 4:
            raise ValueError("DINO scale projector requires four channel definitions")
        self.projections = nn.ModuleList(
            ConvGNReLU(channels, output_channels, 1) for channels in input_channels
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        references: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if len(features) != 4 or len(references) != 4:
            raise ValueError(
                "DINO scale projection requires four features and references"
            )
        return [
            _resize_like(projection(feature), reference)
            for projection, feature, reference in zip(
                self.projections, features, references, strict=True
            )
        ]


class AsymmetricFPNOCRDinoOnlySSepNet(DinoOnlyCandidateBase):
    """Matched ablation for candidate 1: layer mixing + FPN + OCR/Boundary."""

    def __init__(
        self,
        *,
        dinov3_backbone: nn.Module,
        decoder_channels: int = 128,
        ocr_channels: int = 256,
        freeze_backbone: bool = False,
        activation_checkpoint: bool = True,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(
            dinov3_backbone=dinov3_backbone,
            freeze_backbone=freeze_backbone,
            activation_checkpoint=activation_checkpoint,
        )
        self.dino_pyramid = SoftLayerMixPyramid(
            self.dino_layer_channels, decoder_channels
        )
        self.decoder = StandardPyramidDecoder(decoder_channels, dropout)
        self.head = PyramidOCRHead(decoder_channels, ocr_channels, dropout)

    def layer_mixing_weights(self) -> torch.Tensor:
        return self.dino_pyramid.mixing_weights()

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        main_only: bool = False,
        disable_hrnet: bool = False,
        disable_dinov3: bool = False,
    ) -> dict[str, object]:
        del disable_hrnet
        layers, references = self._features(inputs, disable_dinov3=disable_dinov3)
        outputs = self.head(self.decoder(self.dino_pyramid(layers, references)))
        if main_only:
            outputs = {"out": outputs["out"]}
        return _resize_outputs(outputs, tuple(inputs.shape[-2:]))


class LayerwiseFPNDinoOnlySSepNet(DinoOnlyCandidateBase):
    """Matched ablation for candidate 2: historical layer mapping + plain FPN."""

    def __init__(
        self,
        *,
        dinov3_backbone: nn.Module,
        decoder_channels: int = 128,
        freeze_backbone: bool = False,
        activation_checkpoint: bool = True,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(
            dinov3_backbone=dinov3_backbone,
            freeze_backbone=freeze_backbone,
            activation_checkpoint=activation_checkpoint,
        )
        mapped_channels = (
            self.dino_layer_channels,
            self.dino_layer_channels,
            self.dino_layer_channels * 4,
            self.dino_layer_channels * 2,
        )
        self.dino_mapper = HistoricalDinoScaleMapper()
        self.projector = DinoOnlyScaleProjector(mapped_channels, decoder_channels)
        self.decoder = StandardPyramidDecoder(decoder_channels, dropout)
        self.head = PyramidBinaryHead(decoder_channels, dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        main_only: bool = False,
        disable_hrnet: bool = False,
        disable_dinov3: bool = False,
    ) -> dict[str, object]:
        del disable_hrnet
        layers, references = self._features(inputs, disable_dinov3=disable_dinov3)
        pyramid = self.projector(self.dino_mapper(layers), references)
        outputs = self.head(self.decoder(pyramid))
        if main_only:
            outputs = {"out": outputs["out"]}
        return _resize_outputs(outputs, tuple(inputs.shape[-2:]))


class DPTBiFPNDinoOnlySSepNet(DinoOnlyCandidateBase):
    """Matched ablation for candidate 3: DPT reassembly + identical BiFPN/head."""

    def __init__(
        self,
        *,
        dinov3_backbone: nn.Module,
        decoder_channels: int = 128,
        freeze_backbone: bool = False,
        activation_checkpoint: bool = True,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(
            dinov3_backbone=dinov3_backbone,
            freeze_backbone=freeze_backbone,
            activation_checkpoint=activation_checkpoint,
        )
        self.dino_reassemble = DPTReassemble(self.dino_layer_channels, decoder_channels)
        self.bifpn = nn.Sequential(
            BiFPNCell(decoder_channels), BiFPNCell(decoder_channels)
        )
        self.aggregator = DirectPyramidAggregator(decoder_channels, dropout)
        self.head = PyramidBinaryHead(decoder_channels, dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        main_only: bool = False,
        disable_hrnet: bool = False,
        disable_dinov3: bool = False,
    ) -> dict[str, object]:
        del disable_hrnet
        layers, references = self._features(inputs, disable_dinov3=disable_dinov3)
        pyramid = self.bifpn(self.dino_reassemble(layers, references))
        outputs = self.head(self.aggregator(pyramid))
        if main_only:
            outputs = {"out": outputs["out"]}
        return _resize_outputs(outputs, tuple(inputs.shape[-2:]))


class BoundaryCascadeDinoOnlySSepNet(DinoOnlyCandidateBase):
    """Matched ablation for candidate 4: DINO features drive semantics and detail refinement."""

    def __init__(
        self,
        *,
        dinov3_backbone: nn.Module,
        decoder_channels: int = 128,
        freeze_backbone: bool = False,
        activation_checkpoint: bool = True,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(
            dinov3_backbone=dinov3_backbone,
            freeze_backbone=freeze_backbone,
            activation_checkpoint=activation_checkpoint,
        )
        self.dino_reassemble = DPTReassemble(self.dino_layer_channels, decoder_channels)
        self.detail_refine = ConvGNReLU(decoder_channels * 2, decoder_channels, 3)
        self.semantic_decoder = StandardPyramidDecoder(decoder_channels, dropout)
        self.coarse_classifier = nn.Conv2d(decoder_channels, 1, 1)
        self.auxiliary_classifier = nn.Conv2d(decoder_channels, 1, 1)
        structure_channels = max(32, decoder_channels // 2)
        self.boundary = nn.Sequential(
            ConvGNReLU(decoder_channels, structure_channels, 3),
            nn.Conv2d(structure_channels, 1, 1),
        )
        self.residual = nn.Sequential(
            ConvGNReLU(decoder_channels * 2 + 1, decoder_channels, 3),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(decoder_channels, 1, 1),
        )
        self.last_refinement_gate_mean: torch.Tensor | None = None

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        main_only: bool = False,
        disable_hrnet: bool = False,
        disable_dinov3: bool = False,
    ) -> dict[str, object]:
        del disable_hrnet
        layers, references = self._features(inputs, disable_dinov3=disable_dinov3)
        dino_pyramid = self.dino_reassemble(layers, references)
        detail = self.detail_refine(
            torch.cat(
                (dino_pyramid[0], _resize_like(dino_pyramid[1], dino_pyramid[0])), dim=1
            )
        )
        _p0, _p1, p2, _p3, semantic = self.semantic_decoder(dino_pyramid)
        semantic = _resize_like(semantic, detail)
        coarse = self.coarse_classifier(semantic)
        edge = self.boundary(detail)
        probability = torch.sigmoid(coarse)
        uncertainty = 4.0 * probability * (1.0 - probability)
        refinement_gate = (uncertainty + torch.sigmoid(edge)).clamp(max=1.0)
        self.last_refinement_gate_mean = refinement_gate.detach().mean()
        residual = self.residual(torch.cat((semantic, detail, coarse), dim=1))
        outputs: dict[str, object] = {
            "out": coarse + refinement_gate * residual,
            "aux": self.auxiliary_classifier(p2),
            "edge": edge,
        }
        if main_only:
            outputs = {"out": outputs["out"]}
        return _resize_outputs(outputs, tuple(inputs.shape[-2:]))


class DinoOnlyPyramidAdapter(nn.Module):
    def __init__(self, layer_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.reassemble = DPTReassemble(layer_channels, hidden_channels)
        self.refinements = nn.ModuleList(
            ConvGNReLU(hidden_channels, hidden_channels, 3) for _ in range(4)
        )

    def forward(
        self,
        layers: Sequence[torch.Tensor],
        references: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [
            refinement(feature)
            for refinement, feature in zip(
                self.refinements,
                self.reassemble(layers, references),
                strict=True,
            )
        ]


class DeformableMask2FormerDinoOnlySSepNet(DinoOnlyCandidateBase):
    """Matched ablation for candidate 5 with no HRNet modules or state-dict keys."""

    def __init__(
        self,
        *,
        dinov3_backbone: nn.Module,
        hidden_channels: int = 128,
        num_queries: int = 64,
        pixel_encoder_layers: int = 2,
        decoder_layers: int = 3,
        num_heads: int = 8,
        deformable_points: int = 4,
        freeze_backbone: bool = False,
        activation_checkpoint: bool = True,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(
            dinov3_backbone=dinov3_backbone,
            freeze_backbone=freeze_backbone,
            activation_checkpoint=activation_checkpoint,
        )
        self.num_queries = int(num_queries)
        self.adapter = DinoOnlyPyramidAdapter(self.dino_layer_channels, hidden_channels)
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
            ConvGNReLU(hidden_channels, max(32, hidden_channels // 2), 3),
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
        del disable_hrnet
        layers, references = self._features(inputs, disable_dinov3=disable_dinov3)
        adapted = self.adapter(layers, references)
        mask_features, decoder_features = self.pixel_decoder(adapted)
        query_outputs = self.mask_decoder(decoder_features, mask_features)
        outputs: dict[str, object] = {
            "out": semantic_union_logits(
                query_outputs["pred_logits"], query_outputs["pred_masks"]
            ),
            "aux": self.auxiliary(decoder_features[0]),
            "edge": self.boundary(adapted[0]),
            **query_outputs,
        }
        if main_only:
            outputs = {"out": outputs["out"]}
        return _resize_outputs(outputs, tuple(inputs.shape[-2:]))


__all__ = [
    "AsymmetricFPNOCRDinoOnlySSepNet",
    "BoundaryCascadeDinoOnlySSepNet",
    "DPTBiFPNDinoOnlySSepNet",
    "DeformableMask2FormerDinoOnlySSepNet",
    "DinoOnlyCandidateBase",
    "LayerwiseFPNDinoOnlySSepNet",
]
