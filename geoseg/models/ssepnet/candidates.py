from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch.nn as nn


@dataclass(frozen=True)
class SSepNetCandidateSpec:
    name: str
    architecture: str
    loss_name: str
    decoder_channels: int = 128
    num_queries: int = 0
    branch_ablation: str = "none"
    base_candidate: str = ""

    @property
    def active_backbones(self) -> tuple[str, ...]:
        return (
            ("dinov3",)
            if self.branch_ablation == "dinov3_only"
            else ("hrnet", "dinov3")
        )


_CANDIDATES = {
    "ssepnet_asymmetric_fpn_ocr": SSepNetCandidateSpec(
        "ssepnet_asymmetric_fpn_ocr", "asymmetric_fpn_ocr", "SSepNetLoss"
    ),
    "ssepnet_layerwise_fpn": SSepNetCandidateSpec(
        "ssepnet_layerwise_fpn", "layerwise_fpn", "SSepNetLoss"
    ),
    "ssepnet_dpt_bifpn": SSepNetCandidateSpec(
        "ssepnet_dpt_bifpn", "dpt_bifpn", "SSepNetLoss"
    ),
    "ssepnet_boundary_cascade": SSepNetCandidateSpec(
        "ssepnet_boundary_cascade", "boundary_cascade", "SSepNetLoss"
    ),
    "ssepnet_deformable_mask2former": SSepNetCandidateSpec(
        "ssepnet_deformable_mask2former",
        "deformable_mask2former",
        "DeformableMask2FormerLoss",
        num_queries=64,
    ),
    "ssepnet_asymmetric_fpn_ocr_dinov3_only": SSepNetCandidateSpec(
        "ssepnet_asymmetric_fpn_ocr_dinov3_only",
        "asymmetric_fpn_ocr",
        "SSepNetLoss",
        branch_ablation="dinov3_only",
        base_candidate="ssepnet_asymmetric_fpn_ocr",
    ),
    "ssepnet_layerwise_fpn_dinov3_only": SSepNetCandidateSpec(
        "ssepnet_layerwise_fpn_dinov3_only",
        "layerwise_fpn",
        "SSepNetLoss",
        branch_ablation="dinov3_only",
        base_candidate="ssepnet_layerwise_fpn",
    ),
    "ssepnet_dpt_bifpn_dinov3_only": SSepNetCandidateSpec(
        "ssepnet_dpt_bifpn_dinov3_only",
        "dpt_bifpn",
        "SSepNetLoss",
        branch_ablation="dinov3_only",
        base_candidate="ssepnet_dpt_bifpn",
    ),
    "ssepnet_boundary_cascade_dinov3_only": SSepNetCandidateSpec(
        "ssepnet_boundary_cascade_dinov3_only",
        "boundary_cascade",
        "SSepNetLoss",
        branch_ablation="dinov3_only",
        base_candidate="ssepnet_boundary_cascade",
    ),
    "ssepnet_deformable_mask2former_dinov3_only": SSepNetCandidateSpec(
        "ssepnet_deformable_mask2former_dinov3_only",
        "deformable_mask2former",
        "DeformableMask2FormerLoss",
        num_queries=64,
        branch_ablation="dinov3_only",
        base_candidate="ssepnet_deformable_mask2former",
    ),
}

SSEPNET_CANDIDATES: Mapping[str, SSepNetCandidateSpec] = MappingProxyType(_CANDIDATES)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def is_ssepnet_candidate(name: str) -> bool:
    return _normalize_name(name) in SSEPNET_CANDIDATES


def get_ssepnet_candidate(name: str) -> SSepNetCandidateSpec:
    key = _normalize_name(name)
    try:
        return SSEPNET_CANDIDATES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported SSepNet candidate {name!r}; choose from "
            f"{', '.join(SSEPNET_CANDIDATES)}"
        ) from exc


def build_ssepnet_candidate_loss(name: str) -> nn.Module:
    spec = get_ssepnet_candidate(name)
    if spec.loss_name == "DeformableMask2FormerLoss":
        from geoseg.losses.mask2former import DeformableMask2FormerLoss

        return DeformableMask2FormerLoss(
            num_queries=spec.num_queries,
            boundary_weight=0.6,
            pbcl_weight=0.1,
        )
    from geoseg.losses.ssepnet import SSepNetLoss

    return SSepNetLoss(boundary_weight=0.6, pbcl_weight=0.1)


def build_ssepnet_candidate(
    name: str,
    *,
    in_chans: int = 4,
    num_classes: int = 1,
    pretrained: bool = True,
    hrnet_checkpoint: str | Path | None = None,
    dinov3_checkpoint: str | Path | None = None,
    freeze_backbones: bool = False,
    activation_checkpoint: bool = True,
) -> nn.Module:
    if int(num_classes) != 1:
        raise ValueError("SSepNet candidates require num_classes=1")
    spec = get_ssepnet_candidate(name)
    from .dinov3 import build_dinov3_vitl16_sat493m

    dinov3_backbone = build_dinov3_vitl16_sat493m(
        in_chans=in_chans,
        pretrained=pretrained,
        checkpoint_path=dinov3_checkpoint,
    )
    if spec.branch_ablation == "dinov3_only":
        if hrnet_checkpoint is not None:
            raise ValueError("DINO-only candidates cannot load an HRNet checkpoint")
        from .dinov3_only_candidates import (
            AsymmetricFPNOCRDinoOnlySSepNet,
            BoundaryCascadeDinoOnlySSepNet,
            DPTBiFPNDinoOnlySSepNet,
            DeformableMask2FormerDinoOnlySSepNet,
            LayerwiseFPNDinoOnlySSepNet,
        )

        dino_builders: dict[str, type[nn.Module]] = {
            "asymmetric_fpn_ocr": AsymmetricFPNOCRDinoOnlySSepNet,
            "layerwise_fpn": LayerwiseFPNDinoOnlySSepNet,
            "dpt_bifpn": DPTBiFPNDinoOnlySSepNet,
            "boundary_cascade": BoundaryCascadeDinoOnlySSepNet,
            "deformable_mask2former": DeformableMask2FormerDinoOnlySSepNet,
        }
        keyword = (
            {"hidden_channels": spec.decoder_channels, "num_queries": spec.num_queries}
            if spec.architecture == "deformable_mask2former"
            else {"decoder_channels": spec.decoder_channels}
        )
        return dino_builders[spec.architecture](
            dinov3_backbone=dinov3_backbone,
            freeze_backbone=freeze_backbones,
            activation_checkpoint=activation_checkpoint,
            **keyword,
        )
    from .hrnet import build_hrnetv2_w32

    common = {
        "hrnet_backbone": build_hrnetv2_w32(
            in_chans=in_chans,
            pretrained=pretrained,
            checkpoint_path=hrnet_checkpoint,
        ),
        "dinov3_backbone": dinov3_backbone,
        "freeze_backbones": freeze_backbones,
        "activation_checkpoint": activation_checkpoint,
    }
    if spec.architecture == "deformable_mask2former":
        from .deformable_mask2former import DeformableMask2FormerSSepNet

        return DeformableMask2FormerSSepNet(
            **common,
            hidden_channels=spec.decoder_channels,
            num_queries=spec.num_queries,
        )
    from .multiscale import (
        AsymmetricGatedFPNOCRSSepNet,
        DPTBiFPNSSepNet,
        DinoSemanticBoundaryCascadeSSepNet,
        LayerwiseGatedFPNSSepNet,
    )

    builders: dict[str, type[nn.Module]] = {
        "asymmetric_fpn_ocr": AsymmetricGatedFPNOCRSSepNet,
        "layerwise_fpn": LayerwiseGatedFPNSSepNet,
        "dpt_bifpn": DPTBiFPNSSepNet,
        "boundary_cascade": DinoSemanticBoundaryCascadeSSepNet,
    }
    return builders[spec.architecture](
        **common,
        decoder_channels=spec.decoder_channels,
    )


__all__ = [
    "SSEPNET_CANDIDATES",
    "SSepNetCandidateSpec",
    "build_ssepnet_candidate",
    "build_ssepnet_candidate_loss",
    "get_ssepnet_candidate",
    "is_ssepnet_candidate",
]
