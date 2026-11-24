from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssepnet import SSepNetLoss


def instance_maps_to_mask_labels(
    instance_maps: torch.Tensor,
    *,
    max_instances: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Convert dense instance IDs into Mask2Former mask_labels/class_labels."""

    if instance_maps.ndim == 4 and instance_maps.shape[1] == 1:
        instance_maps = instance_maps[:, 0]
    if instance_maps.ndim != 3:
        raise ValueError(
            "instance_maps must have shape [B,H,W] or [B,1,H,W], got "
            f"{tuple(instance_maps.shape)}"
        )
    mask_labels: list[torch.Tensor] = []
    class_labels: list[torch.Tensor] = []
    for batch_index, instance_map in enumerate(instance_maps.to(dtype=torch.long)):
        instance_ids = torch.unique(instance_map)
        instance_ids = instance_ids[instance_ids > 0]
        if max_instances is not None and int(instance_ids.numel()) > int(max_instances):
            raise ValueError(
                f"sample {batch_index} contains {instance_ids.numel()} instances but only "
                f"{max_instances} queries are configured"
            )
        if int(instance_ids.numel()) == 0:
            masks = torch.empty(
                (0, *instance_map.shape), dtype=torch.bool, device=instance_map.device
            )
        else:
            masks = instance_map.unsqueeze(0) == instance_ids[:, None, None]
        mask_labels.append(masks)
        class_labels.append(
            torch.zeros((masks.shape[0],), dtype=torch.long, device=instance_map.device)
        )
    return mask_labels, class_labels


def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).flatten(1)
    targets = targets.float().flatten(1)
    numerator = 2.0 * (probabilities * targets).sum(dim=1) + 1.0
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1) + 1.0
    return (1.0 - numerator / denominator).mean()


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        *,
        class_cost: float = 2.0,
        mask_cost: float = 5.0,
        dice_cost: float = 5.0,
        matching_size: int = 32,
    ) -> None:
        super().__init__()
        self.class_cost = float(class_cost)
        self.mask_cost = float(mask_cost)
        self.dice_cost = float(dice_cost)
        self.matching_size = int(matching_size)
        if min(self.class_cost, self.mask_cost, self.dice_cost) < 0.0:
            raise ValueError("Hungarian matching costs must be non-negative")
        if self.class_cost + self.mask_cost + self.dice_cost == 0.0:
            raise ValueError("at least one Hungarian matching cost must be positive")

    def _cost_matrix(
        self,
        class_logits: torch.Tensor,
        mask_logits: torch.Tensor,
        target_masks: torch.Tensor,
        target_classes: torch.Tensor,
    ) -> torch.Tensor:
        queries = int(class_logits.shape[0])
        targets = int(target_masks.shape[0])
        if targets == 0:
            return class_logits.new_empty((queries, 0))
        size = (self.matching_size, self.matching_size)
        predicted = F.interpolate(
            mask_logits.unsqueeze(1), size=size, mode="bilinear", align_corners=False
        )[:, 0]
        truth = F.interpolate(target_masks[:, None].float(), size=size, mode="nearest")[
            :, 0
        ]
        probabilities = torch.sigmoid(predicted).flatten(1).clamp(1e-6, 1.0 - 1e-6)
        truth = truth.flatten(1)
        pixel_count = float(truth.shape[1])
        bce = (
            -(
                torch.log(probabilities) @ truth.transpose(0, 1)
                + torch.log1p(-probabilities) @ (1.0 - truth).transpose(0, 1)
            )
            / pixel_count
        )
        intersection = probabilities @ truth.transpose(0, 1)
        dice = 1.0 - (2.0 * intersection + 1.0) / (
            probabilities.sum(dim=1, keepdim=True) + truth.sum(dim=1).unsqueeze(0) + 1.0
        )
        foreground_cost = -torch.softmax(class_logits, dim=-1)[:, target_classes]
        return (
            self.class_cost * foreground_cost
            + self.mask_cost * bce
            + self.dice_cost * dice
        )

    @torch.no_grad()
    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        target_masks: Sequence[torch.Tensor],
        target_classes: Sequence[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise RuntimeError(
                "DeformableMask2FormerLoss requires scipy for Hungarian matching"
            ) from exc
        class_logits = outputs["pred_logits"]
        mask_logits = outputs["pred_masks"]
        assignments: list[tuple[torch.Tensor, torch.Tensor]] = []
        for batch_index, masks in enumerate(target_masks):
            if int(masks.shape[0]) == 0:
                empty = torch.empty(0, dtype=torch.long, device=class_logits.device)
                assignments.append((empty, empty))
                continue
            cost = self._cost_matrix(
                class_logits[batch_index],
                mask_logits[batch_index],
                masks,
                target_classes[batch_index],
            )
            query_indices, target_indices = linear_sum_assignment(
                cost.detach().float().cpu().numpy()
            )
            assignments.append(
                (
                    torch.as_tensor(
                        query_indices, dtype=torch.long, device=class_logits.device
                    ),
                    torch.as_tensor(
                        target_indices, dtype=torch.long, device=class_logits.device
                    ),
                )
            )
        return assignments


class DeformableMask2FormerLoss(nn.Module):
    """True set-prediction loss plus the locked semantic/Boundary/PBCL objective."""

    def __init__(
        self,
        *,
        num_queries: int = 64,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        no_object_weight: float = 0.1,
        auxiliary_weight: float = 0.5,
        boundary_weight: float = 0.6,
        pbcl_weight: float = 0.1,
        boundary_kernel: int = 5,
        pbcl_band_radius: int = 3,
        pbcl_iterations: int = 9,
        pbcl_threshold: float = 0.5,
        pbcl_margin: float = 0.1,
        pbcl_neighbors: int = 4,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.class_weight = float(class_weight)
        self.mask_weight = float(mask_weight)
        self.dice_weight = float(dice_weight)
        self.no_object_weight = float(no_object_weight)
        self.auxiliary_weight = float(auxiliary_weight)
        self.requires_instance_supervision = True
        self.matcher = HungarianMatcher(
            class_cost=class_weight,
            mask_cost=mask_weight,
            dice_cost=dice_weight,
        )
        self.semantic_loss = SSepNetLoss(
            boundary_weight=boundary_weight,
            pbcl_weight=pbcl_weight,
            boundary_kernel=boundary_kernel,
            pbcl_band_radius=pbcl_band_radius,
            pbcl_iterations=pbcl_iterations,
            pbcl_threshold=pbcl_threshold,
            pbcl_margin=pbcl_margin,
            pbcl_neighbors=pbcl_neighbors,
        )
        self.last_components: dict[str, float | int] = {}

    @staticmethod
    def _require_query_outputs(outputs: Mapping[str, object]) -> None:
        missing = [
            name
            for name in ("pred_logits", "pred_masks", "out", "aux", "edge")
            if name not in outputs
        ]
        if missing:
            raise KeyError(
                f"Deformable Mask2Former loss requires output tensors {missing}"
            )

    def _set_loss(
        self,
        outputs: Mapping[str, torch.Tensor],
        mask_labels: Sequence[torch.Tensor],
        class_labels: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        assignments = self.matcher(outputs, mask_labels, class_labels)
        class_logits = outputs["pred_logits"]
        mask_logits = outputs["pred_masks"]
        target_classes = torch.full(
            class_logits.shape[:2],
            fill_value=class_logits.shape[-1] - 1,
            dtype=torch.long,
            device=class_logits.device,
        )
        matched_predictions: list[torch.Tensor] = []
        matched_targets: list[torch.Tensor] = []
        matched_count = 0
        for batch_index, (query_indices, target_indices) in enumerate(assignments):
            if int(query_indices.numel()) == 0:
                continue
            target_classes[batch_index, query_indices] = class_labels[batch_index][
                target_indices
            ]
            predicted = mask_logits[batch_index, query_indices]
            targets = mask_labels[batch_index][target_indices].float()
            predicted = F.interpolate(
                predicted[:, None],
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            matched_predictions.append(predicted)
            matched_targets.append(targets)
            matched_count += int(query_indices.numel())
        class_weights = class_logits.new_tensor([1.0, self.no_object_weight])
        class_loss = F.cross_entropy(
            class_logits.flatten(0, 1), target_classes.flatten(), weight=class_weights
        )
        anchor = class_logits.float().sum() * 0.0
        mask_loss = anchor
        dice_loss = anchor
        if matched_predictions:
            predictions = torch.cat(matched_predictions, dim=0)
            targets = torch.cat(matched_targets, dim=0)
            mask_loss = F.binary_cross_entropy_with_logits(predictions, targets)
            dice_loss = _dice_loss(predictions, targets)
        total = (
            self.class_weight * class_loss
            + self.mask_weight * mask_loss
            + self.dice_weight * dice_loss
        )
        return total, class_loss, mask_loss + dice_loss, matched_count

    def forward(
        self,
        outputs: Mapping[str, object],
        target: torch.Tensor,
        *,
        instance_maps: torch.Tensor | None = None,
        instance_pairs: Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not isinstance(outputs, Mapping):
            raise TypeError("DeformableMask2FormerLoss requires named model outputs")
        self._require_query_outputs(outputs)
        if instance_maps is None or instance_pairs is None:
            raise KeyError(
                "DeformableMask2FormerLoss requires instance_maps and instance_pairs"
            )
        mask_labels, class_labels = instance_maps_to_mask_labels(
            instance_maps, max_instances=self.num_queries
        )
        semantic = self.semantic_loss(
            outputs,
            target,
            instance_maps=instance_maps,
            instance_pairs=instance_pairs,
        )
        query_outputs = {
            "pred_logits": outputs["pred_logits"],
            "pred_masks": outputs["pred_masks"],
        }
        set_loss, class_loss, mask_and_dice, matched_count = self._set_loss(
            query_outputs, mask_labels, class_labels
        )
        auxiliary_total = set_loss.new_zeros(())
        auxiliary_outputs = outputs.get("aux_outputs", [])
        if not isinstance(auxiliary_outputs, Sequence):
            raise TypeError("Mask2Former aux_outputs must be a sequence")
        for auxiliary in auxiliary_outputs:
            auxiliary_loss, _class, _mask, _count = self._set_loss(
                auxiliary, mask_labels, class_labels
            )
            auxiliary_total = auxiliary_total + self.auxiliary_weight * auxiliary_loss
        total = semantic + set_loss + auxiliary_total
        self.last_components = {
            "semantic": float(semantic.detach().float().cpu()),
            "set": float(set_loss.detach().float().cpu()),
            "class": float(class_loss.detach().float().cpu()),
            "mask_and_dice": float(mask_and_dice.detach().float().cpu()),
            "auxiliary_set": float(auxiliary_total.detach().float().cpu()),
            "matched_instances": int(matched_count),
        }
        return total


__all__ = [
    "DeformableMask2FormerLoss",
    "HungarianMatcher",
    "instance_maps_to_mask_labels",
]
