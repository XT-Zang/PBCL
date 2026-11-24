from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from geoseg.losses.ssepnet import SSepNetLoss

    from .model import SSepNet


@dataclass(frozen=True)
class SSepNetVariantSpec:
    name: str
    branch_ablation: str
    fusion_mode: str
    boundary_enabled: bool
    pbcl_weight: float


_VARIANTS = {
    "ssepnet": SSepNetVariantSpec("ssepnet", "none", "gate_concat", True, 0.1),
    "ssepnet_hrnet_only": SSepNetVariantSpec(
        "ssepnet_hrnet_only", "hrnet_only", "gate_concat", True, 0.1
    ),
    "ssepnet_dinov3_only": SSepNetVariantSpec(
        "ssepnet_dinov3_only", "dinov3_only", "gate_concat", True, 0.1
    ),
    "ssepnet_base": SSepNetVariantSpec("ssepnet_base", "none", "gate_concat", False, 0.0),
    "ssepnet_boundary": SSepNetVariantSpec(
        "ssepnet_boundary", "none", "gate_concat", True, 0.0
    ),
    "ssepnet_pbcl": SSepNetVariantSpec("ssepnet_pbcl", "none", "gate_concat", False, 0.1),
    "ssepnet_fusion_add": SSepNetVariantSpec(
        "ssepnet_fusion_add", "none", "add", True, 0.1
    ),
    "ssepnet_fusion_concat": SSepNetVariantSpec(
        "ssepnet_fusion_concat", "none", "concat", True, 0.1
    ),
    "ssepnet_fusion_gate": SSepNetVariantSpec(
        "ssepnet_fusion_gate", "none", "gate", True, 0.1
    ),
}

SSEPNET_VARIANTS: Mapping[str, SSepNetVariantSpec] = MappingProxyType(_VARIANTS)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def get_ssepnet_variant(name: str) -> SSepNetVariantSpec:
    key = _normalize_name(name)
    try:
        return SSEPNET_VARIANTS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported SSepNet variant {name!r}; choose from {', '.join(SSEPNET_VARIANTS)}"
        ) from exc


def build_ssepnet_loss(name: str) -> "SSepNetLoss":
    from .stage_adapter_fpn_ocr import (
        build_ssepnet_stage_adapter_fpn_ocr_loss,
        is_ssepnet_stage_adapter_fpn_ocr,
    )

    if is_ssepnet_stage_adapter_fpn_ocr(name):
        return build_ssepnet_stage_adapter_fpn_ocr_loss(name)
    from .multiscale_dinov3_v2 import (
        build_ssepnet_multiscale_dinov3_v2_loss,
        is_ssepnet_multiscale_dinov3_v2,
    )

    if is_ssepnet_multiscale_dinov3_v2(name):
        return build_ssepnet_multiscale_dinov3_v2_loss(name)
    from .multiscale_dinov3 import (
        build_ssepnet_multiscale_dinov3_loss,
        is_ssepnet_multiscale_dinov3,
    )

    if is_ssepnet_multiscale_dinov3(name):
        return build_ssepnet_multiscale_dinov3_loss(name)
    from .dual_fpn_ocr import (
        build_ssepnet_dual_fpn_ocr_loss,
        is_ssepnet_dual_fpn_ocr,
    )

    if is_ssepnet_dual_fpn_ocr(name):
        return build_ssepnet_dual_fpn_ocr_loss(name)
    from .ocr_optimization import (
        build_ssepnet_ocr_optimization_loss,
        is_ssepnet_ocr_optimization,
    )

    if is_ssepnet_ocr_optimization(name):
        return build_ssepnet_ocr_optimization_loss(name)
    from .candidates import build_ssepnet_candidate_loss, is_ssepnet_candidate

    if is_ssepnet_candidate(name):
        return build_ssepnet_candidate_loss(name)
    from geoseg.losses.ssepnet import SSepNetLoss

    spec = get_ssepnet_variant(name)
    return SSepNetLoss(
        boundary_weight=0.6 if spec.boundary_enabled else 0.0,
        pbcl_weight=spec.pbcl_weight,
    )


def build_ssepnet_variant(
    name: str,
    *,
    in_chans: int = 4,
    num_classes: int = 1,
    pretrained: bool = True,
    hrnet_checkpoint: str | Path | None = None,
    dinov3_checkpoint: str | Path | None = None,
    freeze_backbones: bool = True,
    activation_checkpoint: bool = False,
    dinov3_adaptation: str = "full",
    dinov3_lora_rank: int = 0,
    dinov3_lora_alpha: float | None = None,
    physical_pruning: bool = False,
) -> "SSepNet":
    from .stage_adapter_fpn_ocr import (
        build_ssepnet_stage_adapter_fpn_ocr,
        is_ssepnet_stage_adapter_fpn_ocr,
    )

    if is_ssepnet_stage_adapter_fpn_ocr(name):
        if physical_pruning:
            raise ValueError(
                "Stage-Adapter FPN/OCR is already an isolated physical model"
            )
        if hrnet_checkpoint is not None:
            raise ValueError("Stage-Adapter FPN/OCR cannot load an HRNet checkpoint")
        return build_ssepnet_stage_adapter_fpn_ocr(
            name,
            in_chans=in_chans,
            num_classes=num_classes,
            pretrained=pretrained,
            dinov3_checkpoint=dinov3_checkpoint,
            freeze_backbone=freeze_backbones,
            activation_checkpoint=activation_checkpoint,
            dinov3_adaptation=dinov3_adaptation,
            dinov3_lora_rank=dinov3_lora_rank,
            dinov3_lora_alpha=dinov3_lora_alpha,
        )
    from .multiscale_dinov3_v2 import (
        build_ssepnet_multiscale_dinov3_v2,
        is_ssepnet_multiscale_dinov3_v2,
    )

    if is_ssepnet_multiscale_dinov3_v2(name):
        if physical_pruning:
            raise ValueError(
                "multiscale DINOv3 V2 variants are already isolated physical backbones"
            )
        if hrnet_checkpoint is not None:
            raise ValueError(
                "multiscale DINOv3 V2 variants cannot load an HRNet checkpoint"
            )
        return build_ssepnet_multiscale_dinov3_v2(
            name,
            in_chans=in_chans,
            num_classes=num_classes,
            pretrained=pretrained,
            dinov3_checkpoint=dinov3_checkpoint,
            freeze_backbone=freeze_backbones,
            activation_checkpoint=activation_checkpoint,
            dinov3_adaptation=dinov3_adaptation,
            dinov3_lora_rank=dinov3_lora_rank,
            dinov3_lora_alpha=dinov3_lora_alpha,
        )
    from .multiscale_dinov3 import (
        build_ssepnet_multiscale_dinov3,
        is_ssepnet_multiscale_dinov3,
    )

    if is_ssepnet_multiscale_dinov3(name):
        if physical_pruning:
            raise ValueError(
                "multiscale DINOv3 variants are already isolated physical backbones"
            )
        if hrnet_checkpoint is not None:
            raise ValueError("multiscale DINOv3 variants cannot load an HRNet checkpoint")
        return build_ssepnet_multiscale_dinov3(
            name,
            in_chans=in_chans,
            num_classes=num_classes,
            pretrained=pretrained,
            dinov3_checkpoint=dinov3_checkpoint,
            freeze_backbone=freeze_backbones,
            activation_checkpoint=activation_checkpoint,
            dinov3_adaptation=dinov3_adaptation,
            dinov3_lora_rank=dinov3_lora_rank,
            dinov3_lora_alpha=dinov3_lora_alpha,
        )
    from .dual_fpn_ocr import (
        build_ssepnet_dual_fpn_ocr,
        is_ssepnet_dual_fpn_ocr,
    )

    if is_ssepnet_dual_fpn_ocr(name):
        if physical_pruning:
            raise ValueError("dual FPN+OCR models already use physical detail pruning")
        if hrnet_checkpoint is not None:
            raise ValueError(
                "dual FPN+OCR models use detail_checkpoint rather than hrnet_checkpoint"
            )
        return build_ssepnet_dual_fpn_ocr(
            name,
            in_chans=in_chans,
            num_classes=num_classes,
            pretrained=pretrained,
            dinov3_checkpoint=dinov3_checkpoint,
            activation_checkpoint=activation_checkpoint,
            dinov3_adaptation=dinov3_adaptation,
            dinov3_lora_rank=dinov3_lora_rank,
            dinov3_lora_alpha=dinov3_lora_alpha,
        )
    from .ocr_optimization import (
        build_ssepnet_ocr_optimization,
        is_ssepnet_ocr_optimization,
    )

    if is_ssepnet_ocr_optimization(name):
        if physical_pruning:
            raise ValueError(
                "OCR optimization variants are already physically DINOv3-only"
            )
        if hrnet_checkpoint is not None:
            raise ValueError("DINOv3-only OCR optimizations cannot load an HRNet checkpoint")
        return build_ssepnet_ocr_optimization(
            name,
            in_chans=in_chans,
            num_classes=num_classes,
            pretrained=pretrained,
            dinov3_checkpoint=dinov3_checkpoint,
            freeze_backbone=freeze_backbones,
            activation_checkpoint=activation_checkpoint,
            dinov3_adaptation=dinov3_adaptation,
            dinov3_lora_rank=dinov3_lora_rank,
            dinov3_lora_alpha=dinov3_lora_alpha,
        )
    from .candidates import build_ssepnet_candidate, is_ssepnet_candidate

    if is_ssepnet_candidate(name):
        if physical_pruning:
            raise ValueError("physical_pruning is only defined for paper single-branch variants")
        return build_ssepnet_candidate(
            name,
            in_chans=in_chans,
            num_classes=num_classes,
            pretrained=pretrained,
            hrnet_checkpoint=hrnet_checkpoint,
            dinov3_checkpoint=dinov3_checkpoint,
            freeze_backbones=freeze_backbones,
            activation_checkpoint=activation_checkpoint,
        )
    from .model import DINOv3OnlySSepNet, HRNetOnlySSepNet, SSepNet

    spec = get_ssepnet_variant(name)
    if physical_pruning and spec.branch_ablation != "none" and dinov3_adaptation != "full":
        raise ValueError("DINOv3 adaptation controls require the full dual-backbone variant")
    if physical_pruning and spec.branch_ablation == "hrnet_only":
        from .hrnet import build_hrnetv2_w32

        return HRNetOnlySSepNet(
            hrnet_backbone=build_hrnetv2_w32(
                in_chans=in_chans,
                pretrained=pretrained,
                checkpoint_path=hrnet_checkpoint,
            ),
            boundary_enabled=spec.boundary_enabled,
            num_classes=num_classes,
            freeze_backbone=freeze_backbones,
        )
    if physical_pruning and spec.branch_ablation == "dinov3_only":
        from .dinov3 import build_dinov3_vitl16_sat493m

        return DINOv3OnlySSepNet(
            dinov3_backbone=build_dinov3_vitl16_sat493m(
                in_chans=in_chans,
                pretrained=pretrained,
                checkpoint_path=dinov3_checkpoint,
            ),
            boundary_enabled=spec.boundary_enabled,
            num_classes=num_classes,
            freeze_backbone=freeze_backbones,
            activation_checkpoint=activation_checkpoint,
        )
    return SSepNet.from_variant(
        spec,
        in_chans=in_chans,
        num_classes=num_classes,
        pretrained=pretrained,
        hrnet_checkpoint=hrnet_checkpoint,
        dinov3_checkpoint=dinov3_checkpoint,
        freeze_backbones=freeze_backbones,
        activation_checkpoint=activation_checkpoint,
        dinov3_adaptation=dinov3_adaptation,
        dinov3_lora_rank=dinov3_lora_rank,
        dinov3_lora_alpha=dinov3_lora_alpha,
        prune_disabled_boundary=physical_pruning,
    )
