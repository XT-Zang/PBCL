from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from .mask2former import instance_maps_to_mask_labels


def _masked_instance_targets(
    instance_maps: torch.Tensor,
    *,
    valid_masks: torch.Tensor | None,
    max_instances: int,
    crop_to_valid: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    maps = instance_maps
    if maps.ndim == 4 and maps.shape[1] == 1:
        maps = maps[:, 0]
    if maps.ndim != 3:
        raise ValueError(
            f"instance_maps must have shape [B,H,W] or [B,1,H,W], got {tuple(instance_maps.shape)}"
        )
    if valid_masks is None:
        return instance_maps_to_mask_labels(maps, max_instances=max_instances)

    valid = torch.as_tensor(valid_masks, device=maps.device, dtype=torch.bool)
    if valid.ndim == 4 and valid.shape[1] == 1:
        valid = valid[:, 0]
    if valid.shape != maps.shape:
        raise ValueError(
            f"valid_masks shape {tuple(valid.shape)} does not match instance maps {tuple(maps.shape)}"
        )

    all_masks: list[torch.Tensor] = []
    all_classes: list[torch.Tensor] = []
    for sample_map, sample_valid in zip(maps, valid, strict=True):
        masked = torch.where(sample_valid, sample_map, torch.zeros_like(sample_map))
        if crop_to_valid:
            rows = torch.nonzero(sample_valid.any(dim=1), as_tuple=False).flatten()
            cols = torch.nonzero(sample_valid.any(dim=0), as_tuple=False).flatten()
            if rows.numel() and cols.numel():
                masked = masked[
                    int(rows[0]) : int(rows[-1]) + 1,
                    int(cols[0]) : int(cols[-1]) + 1,
                ]
            else:
                masked = masked[:1, :1]
        masks, classes = instance_maps_to_mask_labels(
            masked.unsqueeze(0),
            max_instances=max_instances,
        )
        all_masks.append(masks[0])
        all_classes.append(classes[0])
    return all_masks, all_classes


class Mask2FormerInstanceLoss(nn.Module):
    """Adapter for a model-owned official Mask2Former set-prediction loss."""

    requires_instance_supervision = True
    model_computes_loss = True

    def __init__(self, *, num_queries: int = 100) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.last_components: dict[str, float] = {}

    def model_forward_kwargs(
        self,
        *,
        instance_maps: torch.Tensor,
        instance_pairs: Sequence[torch.Tensor] | None = None,
        valid_masks: torch.Tensor | None = None,
    ) -> dict[str, object]:
        del instance_pairs
        masks, classes = _masked_instance_targets(
            instance_maps,
            valid_masks=valid_masks,
            max_instances=self.num_queries,
            # Keep targets in the padded image coordinate system. The official
            # Mask2Former matcher/loss samples prediction and target masks with
            # the same normalized coordinates, so cropping only the target
            # would geometrically stretch it against the padded prediction.
            crop_to_valid=False,
        )
        result: dict[str, object] = {
            "mask_labels": [mask.float() for mask in masks],
            "class_labels": classes,
        }
        if valid_masks is not None:
            pixel_mask = valid_masks
            if pixel_mask.ndim == 4 and pixel_mask.shape[1] == 1:
                pixel_mask = pixel_mask[:, 0]
            if pixel_mask.ndim != 3:
                raise ValueError(
                    "valid_masks must have shape [B,H,W] or [B,1,H,W], got "
                    f"{tuple(valid_masks.shape)}"
                )
            result["pixel_mask"] = pixel_mask.to(dtype=torch.bool)
        return result

    def forward(
        self,
        outputs: Mapping[str, Any],
        target: torch.Tensor,
        *,
        instance_maps: torch.Tensor | None = None,
        instance_pairs: Sequence[torch.Tensor] | None = None,
        valid_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del target, instance_maps, instance_pairs, valid_masks
        loss = outputs.get("loss")
        if not isinstance(loss, torch.Tensor):
            raise KeyError("Mask2Former instance output does not contain the official loss")
        loss_dict = outputs.get("loss_dict", {})
        if not isinstance(loss_dict, Mapping):
            raise TypeError("Mask2Former loss_dict must be a mapping")
        self.last_components = {
            str(name): float(value.detach().float().cpu())
            for name, value in loss_dict.items()
            if isinstance(value, torch.Tensor)
        }
        return loss


def instance_metric_targets(
    instance_maps: torch.Tensor,
    *,
    max_instances: int = 100,
    valid_masks: torch.Tensor | None = None,
) -> list[dict[str, torch.Tensor]]:
    masks, classes = _masked_instance_targets(
        instance_maps,
        valid_masks=valid_masks,
        max_instances=max_instances,
        crop_to_valid=False,
    )
    return [
        {"masks": sample_masks, "labels": sample_classes}
        for sample_masks, sample_classes in zip(masks, classes, strict=True)
    ]


__all__ = ["Mask2FormerInstanceLoss", "instance_metric_targets"]
