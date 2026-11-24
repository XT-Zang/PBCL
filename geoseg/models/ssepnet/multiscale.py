from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint_forward

from .common import ConvBNReLU
from .ocr import OCRGather, ObjectAttentionBlock


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNReLU(nn.Sequential):
    """Projection used to align LayerNorm ViT and BatchNorm CNN feature domains."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__(
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                padding=int(kernel_size) // 2,
                bias=False,
            ),
            nn.GroupNorm(_group_count(int(out_channels)), int(out_channels)),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableGN(nn.Sequential):
    def __init__(self, channels: int) -> None:
        channels = int(channels)
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(_group_count(channels), channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.ReLU(inplace=True),
        )


def _resize_outputs(
    outputs: dict[str, object], input_size: tuple[int, int]
) -> dict[str, object]:
    resized: dict[str, object] = {}
    for key, value in outputs.items():
        if key in {"out", "aux", "edge"}:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{key} output must be a tensor")
            value = F.interpolate(
                value, size=input_size, mode="bilinear", align_corners=False
            )
        resized[key] = value
    return resized


def _resize_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape[-2:] == target.shape[-2:]:
        return source
    return F.interpolate(
        source, size=target.shape[-2:], mode="bilinear", align_corners=False
    )


class DualBackboneCandidateBase(nn.Module):
    """Shared standalone execution contract for the dual-backbone candidates."""

    def __init__(
        self,
        *,
        hrnet_backbone: nn.Module,
        dinov3_backbone: nn.Module,
        freeze_backbones: bool,
        activation_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.hrnet_backbone = hrnet_backbone
        self.dinov3_backbone = dinov3_backbone
        self.freeze_backbones = bool(freeze_backbones)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.stage_channels = tuple(
            int(value) for value in hrnet_backbone.stage_channels
        )
        if len(self.stage_channels) != 4:
            raise ValueError(
                "dual-backbone candidates require four HRNet feature levels"
            )
        dino_out_channels = int(dinov3_backbone.out_channels)
        self.dino_layer_channels = int(
            getattr(dinov3_backbone, "embed_dim", dino_out_channels // 4)
        )
        if self.dino_layer_channels * 4 != dino_out_channels:
            raise ValueError(
                "dual-backbone candidates require four equal-width DINOv3 layers, got "
                f"out_channels={dino_out_channels}, layer_channels={self.dino_layer_channels}"
            )
        if self.freeze_backbones:
            self.hrnet_backbone.requires_grad_(False)
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
            raise ValueError(
                f"DINOv3 must expose four intermediate layers, got {len(layers)}"
            )
        return list(layers)

    def _backbone_features(
        self,
        inputs: torch.Tensor,
        *,
        disable_hrnet: bool,
        disable_dinov3: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        hr_features = list(self.hrnet_backbone(inputs))
        dino_layers = self._forward_dinov3_layers(inputs)
        if disable_hrnet:
            hr_features = [torch.zeros_like(value) for value in hr_features]
        if disable_dinov3:
            dino_layers = [torch.zeros_like(value) for value in dino_layers]
        return hr_features, dino_layers

    def train(self, mode: bool = True) -> "DualBackboneCandidateBase":
        super().train(mode)
        if self.freeze_backbones:
            self.hrnet_backbone.eval()
            self.dinov3_backbone.eval()
        return self

    def fusion_gate_means(self) -> tuple[float, ...]:
        blocks = getattr(self, "fusion_blocks", ())
        return tuple(
            float(block.last_gate_mean.float().cpu())
            for block in blocks
            if getattr(block, "last_gate_mean", None) is not None
        )

    def fusion_gate_statistics(self) -> tuple[dict[str, float], ...]:
        blocks = getattr(self, "fusion_blocks", ())
        names = ("mean", "std", "min", "max")
        return tuple(
            {
                name: float(value.float().cpu())
                for name, value in zip(names, block.last_gate_statistics, strict=True)
            }
            for block in blocks
            if getattr(block, "last_gate_statistics", None) is not None
        )

    def forward_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.forward(inputs, main_only=True)["out"]
        if not isinstance(output, torch.Tensor):
            raise TypeError("candidate inference output must be a tensor")
        return output


class ResidualGateFusion(nn.Module):
    def __init__(
        self,
        first_channels: int,
        second_channels: int,
        out_channels: int,
        *,
        anchor: str = "first",
        gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if anchor not in {"first", "second"}:
            raise ValueError("anchor must be 'first' or 'second'")
        self.anchor = anchor
        self.first_projection = ConvGNReLU(first_channels, out_channels, 1)
        self.second_projection = ConvGNReLU(second_channels, out_channels, 1)
        self.gate = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init_bias))
        self.last_gate_mean: torch.Tensor | None = None
        self.last_gate_statistics: torch.Tensor | None = None

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first = self.first_projection(first)
        second = _resize_like(self.second_projection(second), first)
        gate = torch.sigmoid(self.gate(torch.cat((first, second), dim=1)))
        self.last_gate_mean = gate.detach().mean()
        self.last_gate_statistics = torch.stack(
            (
                gate.detach().mean(),
                gate.detach().std(unbiased=False),
                gate.detach().amin(),
                gate.detach().amax(),
            )
        )
        if self.anchor == "first":
            return first + gate * second
        return second + gate * first


class StandardPyramidDecoder(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.smooth = nn.ModuleList(ConvGNReLU(channels, channels, 3) for _ in range(4))
        self.aggregate = nn.Sequential(
            ConvGNReLU(channels * 4, channels, 3),
            nn.Dropout2d(float(dropout)),
        )

    def forward(
        self, features: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(features) != 4:
            raise ValueError(
                f"pyramid decoder requires four features, got {len(features)}"
            )
        p3 = self.smooth[3](features[3])
        p2 = self.smooth[2](features[2] + _resize_like(p3, features[2]))
        p1 = self.smooth[1](features[1] + _resize_like(p2, features[1]))
        p0 = self.smooth[0](features[0] + _resize_like(p1, features[0]))
        fused = self.aggregate(
            torch.cat(
                (p0, _resize_like(p1, p0), _resize_like(p2, p0), _resize_like(p3, p0)),
                dim=1,
            )
        )
        return p0, p1, p2, p3, fused


class PyramidBinaryHead(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        structure_channels = max(32, int(channels) // 2)
        self.main = nn.Sequential(
            nn.Dropout2d(float(dropout)), nn.Conv2d(channels, 1, 1)
        )
        self.auxiliary = nn.Conv2d(channels, 1, 1)
        self.boundary = nn.Sequential(
            ConvGNReLU(channels, structure_channels, 3),
            nn.Conv2d(structure_channels, 1, 1),
        )

    def forward(
        self,
        pyramid: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> dict[str, torch.Tensor]:
        p0, _p1, p2, _p3, fused = pyramid
        return {
            "out": self.main(fused),
            "aux": self.auxiliary(p2),
            "edge": self.boundary(p0),
        }


class PyramidOCRHead(nn.Module):
    """OCR refinement after scale fusion; Boundary remains tied to the P2 detail map."""

    def __init__(self, channels: int, ocr_channels: int, dropout: float) -> None:
        super().__init__()
        ocr_channels = int(ocr_channels)
        structure_channels = max(32, ocr_channels // 2)
        self.shared = ConvBNReLU(channels, ocr_channels, 3)
        self.auxiliary = nn.Conv2d(channels, 1, 1)
        self.gather = OCRGather()
        self.object_attention = ObjectAttentionBlock(
            ocr_channels, max(32, ocr_channels // 2)
        )
        self.bottleneck = nn.Sequential(
            ConvBNReLU(ocr_channels * 2, ocr_channels, 3),
            nn.Dropout2d(float(dropout)),
        )
        self.classifier = nn.Conv2d(ocr_channels, 1, 1)
        self.boundary = nn.Sequential(
            ConvGNReLU(channels, structure_channels, 3),
            nn.Conv2d(structure_channels, 1, 1),
        )

    def _forward_main_and_auxiliary(
        self,
        pyramid: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _p0, _p1, p2, _p3, fused = pyramid
        features = self.shared(fused)
        auxiliary = self.auxiliary(p2)
        auxiliary_for_context = _resize_like(auxiliary, features)
        context = self.object_attention(
            features, self.gather(features, auxiliary_for_context)
        )
        refined = self.bottleneck(torch.cat((features, context), dim=1))
        return self.classifier(refined), auxiliary

    def forward_main_and_auxiliary(
        self,
        pyramid: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the shared OCR path for heads with an external Boundary source."""

        return self._forward_main_and_auxiliary(pyramid)

    def forward_main(
        self,
        pyramid: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        main, _auxiliary = self.forward_main_and_auxiliary(pyramid)
        return main

    def forward(
        self,
        pyramid: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> dict[str, torch.Tensor]:
        p0, _p1, _p2, _p3, _fused = pyramid
        main, auxiliary = self.forward_main_and_auxiliary(pyramid)
        return {
            "out": main,
            "aux": auxiliary,
            "edge": self.boundary(p0),
        }


class HistoricalDinoScaleMapper(nn.Module):
    """The layer-to-scale mapping used by the prior multiscale FPN experiment."""

    def forward(self, layers: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(layers) != 4:
            raise ValueError(
                f"historical mapper requires four layers, got {len(layers)}"
            )
        d0, d1, d2, d3 = layers
        return [d0, d1, torch.cat((d0, d1, d2, d3), dim=1), torch.cat((d2, d3), dim=1)]


class LayerwiseGatedFPNSSepNet(DualBackboneCandidateBase):
    """Candidate 2: four-scale residual gated fusion followed by a plain FPN head."""

    def __init__(
        self,
        *,
        hrnet_backbone: nn.Module,
        dinov3_backbone: nn.Module,
        decoder_channels: int = 128,
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
        mapped_channels = (
            self.dino_layer_channels,
            self.dino_layer_channels,
            self.dino_layer_channels * 4,
            self.dino_layer_channels * 2,
        )
        self.dino_mapper = HistoricalDinoScaleMapper()
        self.fusion_blocks = nn.ModuleList(
            ResidualGateFusion(hr, dino, decoder_channels, anchor="first")
            for hr, dino in zip(self.stage_channels, mapped_channels, strict=True)
        )
        self.decoder = StandardPyramidDecoder(decoder_channels, dropout)
        self.head = PyramidBinaryHead(decoder_channels, dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        main_only: bool = False,
        disable_hrnet: bool = False,
        disable_dinov3: bool = False,
    ) -> dict[str, torch.Tensor]:
        hr_features, dino_layers = self._backbone_features(
            inputs, disable_hrnet=disable_hrnet, disable_dinov3=disable_dinov3
        )
        dino_features = self.dino_mapper(dino_layers)
        fused = [
            block(hr, dino)
            for block, hr, dino in zip(
                self.fusion_blocks, hr_features, dino_features, strict=True
            )
        ]
        outputs = self.head(self.decoder(fused))
        if main_only:
            outputs = {"out": outputs["out"]}
        return _resize_outputs(outputs, tuple(inputs.shape[-2:]))


class SoftLayerMixPyramid(nn.Module):
    def __init__(self, layer_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            ConvGNReLU(layer_channels, out_channels, 1) for _ in range(4)
        )
        initial = torch.full((4, 4), -2.0)
        initial.fill_diagonal_(2.0)
        self.layer_mix_logits = nn.Parameter(initial)

    def forward(
        self, layers: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]
    ) -> list[torch.Tensor]:
        if len(layers) != 4 or len(targets) != 4:
            raise ValueError(
                "soft layer mixing requires four DINO layers and four target scales"
            )
        projected = torch.stack(
            [
                projection(layer)
                for projection, layer in zip(self.projections, layers, strict=True)
            ],
            dim=1,
        )
        weights = torch.softmax(self.layer_mix_logits, dim=1)
        mixed = torch.einsum("sl,blchw->bschw", weights, projected)
        return [
            _resize_like(mixed[:, index], target)
            for index, target in enumerate(targets)
        ]

    def mixing_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_mix_logits.detach(), dim=1)


class AsymmetricGatedFPNOCRSSepNet(DualBackboneCandidateBase):
    """Candidate 1: role-aware asymmetric fusion, FPN, then OCR/Boundary refinement."""

    def __init__(
        self,
        *,
        hrnet_backbone: nn.Module,
        dinov3_backbone: nn.Module,
        decoder_channels: int = 128,
        ocr_channels: int = 256,
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
        self.dino_pyramid = SoftLayerMixPyramid(
            self.dino_layer_channels, decoder_channels
        )
        anchors = ("first", "first", "second", "second")
        self.fusion_blocks = nn.ModuleList(
            ResidualGateFusion(hr, decoder_channels, decoder_channels, anchor=anchor)
            for hr, anchor in zip(self.stage_channels, anchors, strict=True)
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
    ) -> dict[str, torch.Tensor]:
        hr_features, dino_layers = self._backbone_features(
            inputs, disable_hrnet=disable_hrnet, disable_dinov3=disable_dinov3
        )
        dino_pyramid = self.dino_pyramid(dino_layers, hr_features)
        fused = [
            block(hr, dino)
            for block, hr, dino in zip(
                self.fusion_blocks, hr_features, dino_pyramid, strict=True
            )
        ]
        outputs = self.head(self.decoder(fused))
        if main_only:
            outputs = {"out": outputs["out"]}
        return _resize_outputs(outputs, tuple(inputs.shape[-2:]))


class DPTReassemble(nn.Module):
    def __init__(self, layer_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            ConvGNReLU(layer_channels, out_channels, 1) for _ in range(4)
        )

    def forward(
        self, layers: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]
    ) -> list[torch.Tensor]:
        if len(layers) != 4 or len(targets) != 4:
            raise ValueError(
                "DPT reassembly requires four layers and four target scales"
            )
        return [
            _resize_like(projection(layer), target)
            for projection, layer, target in zip(
                self.projections, layers, targets, strict=True
            )
        ]


class NormalizedWeightedFusion(nn.Module):
    def __init__(self, input_count: int, channels: int) -> None:
        super().__init__()
        if input_count < 2:
            raise ValueError("weighted fusion requires at least two inputs")
        self.weights = nn.Parameter(torch.ones(int(input_count)))
        self.refine = DepthwiseSeparableGN(channels)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != int(self.weights.numel()):
            raise ValueError(
                f"expected {self.weights.numel()} fusion inputs, got {len(inputs)}"
            )
        target = inputs[0]
        values = [_resize_like(value, target) for value in inputs]
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + 1e-4)
        combined = sum(
            weight * value for weight, value in zip(weights, values, strict=True)
        )
        return self.refine(combined)


class BiFPNCell(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.top_down = nn.ModuleList(
            NormalizedWeightedFusion(2, channels) for _ in range(3)
        )
        self.bottom_up = nn.ModuleList(
            [
                NormalizedWeightedFusion(3, channels),
                NormalizedWeightedFusion(3, channels),
                NormalizedWeightedFusion(2, channels),
            ]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != 4:
            raise ValueError("BiFPN requires four feature levels")
        x0, x1, x2, x3 = features
        td3 = x3
        td2 = self.top_down[2]((x2, td3))
        td1 = self.top_down[1]((x1, td2))
        td0 = self.top_down[0]((x0, td1))
        out0 = td0
        out1 = self.bottom_up[0]((x1, td1, out0))
        out2 = self.bottom_up[1]((x2, td2, out1))
        out3 = self.bottom_up[2]((x3, out2))
        return [out0, out1, out2, out3]


class DirectPyramidAggregator(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.aggregate = nn.Sequential(
            ConvGNReLU(channels * 4, channels, 3),
            nn.Dropout2d(float(dropout)),
        )

    def forward(
        self, features: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(features) != 4:
            raise ValueError("pyramid aggregation requires four feature levels")
        p0, p1, p2, p3 = features
        fused = self.aggregate(
            torch.cat(
                (p0, _resize_like(p1, p0), _resize_like(p2, p0), _resize_like(p3, p0)),
                dim=1,
            )
        )
        return p0, p1, p2, p3, fused


class DPTBiFPNSSepNet(DualBackboneCandidateBase):
    """Candidate 3: DPT layer reassembly and two lightweight bidirectional FPN cells."""

    def __init__(
        self,
        *,
        hrnet_backbone: nn.Module,
        dinov3_backbone: nn.Module,
        decoder_channels: int = 128,
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
        self.hr_projections = nn.ModuleList(
            ConvGNReLU(channels, decoder_channels, 1)
            for channels in self.stage_channels
        )
        self.dino_reassemble = DPTReassemble(self.dino_layer_channels, decoder_channels)
        self.input_fusions = nn.ModuleList(
            NormalizedWeightedFusion(2, decoder_channels) for _ in range(4)
        )
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
    ) -> dict[str, torch.Tensor]:
        hr_features, dino_layers = self._backbone_features(
            inputs, disable_hrnet=disable_hrnet, disable_dinov3=disable_dinov3
        )
        hr_projected = [
            projection(feature)
            for projection, feature in zip(
                self.hr_projections, hr_features, strict=True
            )
        ]
        dino_pyramid = self.dino_reassemble(dino_layers, hr_projected)
        fused = [
            block((hr, dino))
            for block, hr, dino in zip(
                self.input_fusions, hr_projected, dino_pyramid, strict=True
            )
        ]
        fused = self.bifpn(fused)
        outputs = self.head(self.aggregator(fused))
        if main_only:
            outputs = {"out": outputs["out"]}
        return _resize_outputs(outputs, tuple(inputs.shape[-2:]))


class DinoSemanticBoundaryCascadeSSepNet(DualBackboneCandidateBase):
    """Candidate 4: preserve DINO semantics and restrict HRNet to boundary residual refinement."""

    def __init__(
        self,
        *,
        hrnet_backbone: nn.Module,
        dinov3_backbone: nn.Module,
        decoder_channels: int = 128,
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
        self.hr_detail = nn.ModuleList(
            [
                ConvGNReLU(self.stage_channels[0], decoder_channels, 1),
                ConvGNReLU(self.stage_channels[1], decoder_channels, 1),
            ]
        )
        self.detail_refine = ConvGNReLU(decoder_channels * 2, decoder_channels, 3)
        self.dino_reassemble = DPTReassemble(self.dino_layer_channels, decoder_channels)
        self.semantic_decoder = StandardPyramidDecoder(decoder_channels, dropout)
        self.coarse_classifier = nn.Conv2d(decoder_channels, 1, 1)
        self.auxiliary_classifier = nn.Conv2d(decoder_channels, 1, 1)
        self.boundary = nn.Sequential(
            ConvGNReLU(decoder_channels, max(32, decoder_channels // 2), 3),
            nn.Conv2d(max(32, decoder_channels // 2), 1, 1),
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
    ) -> dict[str, torch.Tensor]:
        hr_features, dino_layers = self._backbone_features(
            inputs, disable_hrnet=disable_hrnet, disable_dinov3=disable_dinov3
        )
        detail0 = self.hr_detail[0](hr_features[0])
        detail1 = _resize_like(self.hr_detail[1](hr_features[1]), detail0)
        detail = self.detail_refine(torch.cat((detail0, detail1), dim=1))
        dino_pyramid = self.dino_reassemble(dino_layers, hr_features)
        p0, _p1, p2, _p3, semantic = self.semantic_decoder(dino_pyramid)
        semantic = _resize_like(semantic, detail)
        coarse = self.coarse_classifier(semantic)
        edge = self.boundary(detail)
        uncertainty = 4.0 * torch.sigmoid(coarse) * (1.0 - torch.sigmoid(coarse))
        refinement_gate = (uncertainty + torch.sigmoid(edge)).clamp(max=1.0)
        self.last_refinement_gate_mean = refinement_gate.detach().mean()
        residual = self.residual(torch.cat((semantic, detail, coarse), dim=1))
        outputs = {
            "out": coarse + refinement_gate * residual,
            "aux": self.auxiliary_classifier(p2),
            "edge": edge,
        }
        if main_only:
            outputs = {"out": outputs["out"]}
        return _resize_outputs(outputs, tuple(inputs.shape[-2:]))


__all__ = [
    "AsymmetricGatedFPNOCRSSepNet",
    "BiFPNCell",
    "DPTBiFPNSSepNet",
    "DinoSemanticBoundaryCascadeSSepNet",
    "DualBackboneCandidateBase",
    "LayerwiseGatedFPNSSepNet",
    "NormalizedWeightedFusion",
    "ResidualGateFusion",
    "SoftLayerMixPyramid",
    "StandardPyramidDecoder",
]
