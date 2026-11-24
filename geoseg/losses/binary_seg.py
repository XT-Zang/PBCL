from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice import DiceLoss
from .pairwise_connectivity import pair_bottleneck_connectivity_loss
from .soft_bce import SoftBCEWithLogitsLoss


class BinarySegmentationLoss(nn.Module):
    def __init__(self, main_bce: float = 1.0, main_dice: float = 1.0, aux_bce: float = 0.3, aux_dice: float = 0.3):
        super().__init__()
        self.main_bce_weight = float(main_bce)
        self.main_dice_weight = float(main_dice)
        self.aux_bce_weight = float(aux_bce)
        self.aux_dice_weight = float(aux_dice)
        self.bce = SoftBCEWithLogitsLoss(ignore_index=None)
        self.dice = DiceLoss(mode="binary", from_logits=True)
        self.requires_instance_supervision = False

    @staticmethod
    def _valid_mask(
        valid_masks: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        valid = valid_masks.to(device=target.device, dtype=target.dtype)
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.shape[-2:] != target.shape[-2:]:
            valid = F.interpolate(valid, size=target.shape[-2:], mode="nearest")
        if valid.shape[1] == 1 and target.shape[1] != 1:
            valid = valid.expand(-1, target.shape[1], -1, -1)
        if valid.shape != target.shape:
            raise ValueError(
                f"valid mask shape {tuple(valid.shape)} does not match target {tuple(target.shape)}"
            )
        return valid

    def _single_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        bce_weight: float,
        dice_weight: float,
        valid_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = target.float()
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        if valid_masks is not None:
            valid = self._valid_mask(valid_masks, target)
            denominator = valid.sum().clamp_min(1.0)
            bce = (
                F.binary_cross_entropy_with_logits(logits, target, reduction="none")
                * valid
            ).sum() / denominator
            probabilities = torch.sigmoid(logits) * valid
            masked_target = target * valid
            intersection = (probabilities * masked_target).flatten(1).sum(dim=1)
            dice_denominator = (
                probabilities.flatten(1).sum(dim=1)
                + masked_target.flatten(1).sum(dim=1)
            )
            dice = 1.0 - (
                (2.0 * intersection + 1e-6) / (dice_denominator + 1e-6)
            ).mean()
            return bce_weight * bce + dice_weight * dice
        return bce_weight * self.bce(logits, target) + dice_weight * self.dice(logits, target)

    def _boundary_supervision_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weight: float,
        kernel: int,
        valid_masks: torch.Tensor | None,
    ) -> torch.Tensor:
        target = target.float()
        boundary_target = _soft_boundary(target, kernel)
        boundary_valid = None
        if valid_masks is not None:
            valid = self._valid_mask(valid_masks, target)
            boundary_valid = 1.0 - F.max_pool2d(
                1.0 - valid, kernel_size=kernel, stride=1, padding=kernel // 2,
            )
        return self._single_loss(logits, boundary_target, weight, weight, boundary_valid)

    def forward(
        self,
        outputs,
        target: torch.Tensor,
        *,
        valid_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(outputs, (tuple, list)):
            main = outputs[0]
            aux_outputs = outputs[1:]
        else:
            main = outputs
            aux_outputs = ()

        loss = self._single_loss(
            main,
            target,
            self.main_bce_weight,
            self.main_dice_weight,
            valid_masks,
        )
        for aux in aux_outputs:
            loss = loss + self._single_loss(
                aux,
                target,
                self.aux_bce_weight,
                self.aux_dice_weight,
                valid_masks,
            )
        return loss


def loss_batch_kwargs(batch: Mapping, device: torch.device) -> dict[str, object]:
    """Move optional PBCL supervision to the loss device without changing legacy batches."""
    kwargs: dict[str, object] = {}
    valid_mask = batch.get("valid_mask")
    if valid_mask is not None:
        kwargs["valid_masks"] = torch.as_tensor(valid_mask).to(
            device=device, non_blocking=True
        )
    ewc_weight = batch.get("ewc_weight")
    if ewc_weight is not None:
        kwargs["ewc_weights"] = torch.as_tensor(ewc_weight).to(
            device=device, non_blocking=True
        )
    wprf_support = batch.get("wprf_support")
    wprf_dense_components = batch.get("wprf_dense_components")
    wprf_pairs = batch.get("wprf_pairs")
    if any(value is not None for value in (wprf_support, wprf_dense_components, wprf_pairs)):
        if any(value is None for value in (wprf_support, wprf_dense_components, wprf_pairs)):
            raise KeyError("WPRF support, dense components, and pairs must be present together")
        kwargs.update(
            wprf_support=torch.as_tensor(wprf_support).to(
                device=device, non_blocking=True
            ),
            wprf_dense_components=torch.as_tensor(wprf_dense_components).to(
                device=device, non_blocking=True
            ),
            wprf_pairs=wprf_pairs,
        )
    instance_map = batch.get("instance_map")
    instance_pairs = batch.get("instance_pairs")
    if instance_map is None and instance_pairs is None:
        return kwargs
    if instance_map is None or instance_pairs is None:
        raise KeyError("instance_map and instance_pairs must be present together")
    kwargs.update(
        instance_maps=instance_map.to(device=device, non_blocking=True),
        instance_pairs=[
            torch.as_tensor(pair).to(device=device, non_blocking=True)
            for pair in instance_pairs
        ],
    )
    return kwargs


class PBCLBinarySegmentationLoss(BinarySegmentationLoss):
    """Binary BCE+Dice loss with an optional pair bottleneck connectivity term."""

    def __init__(
        self,
        main_bce: float = 1.0,
        main_dice: float = 1.0,
        aux_bce: float = 0.3,
        aux_dice: float = 0.3,
        pbcl_weight: float = 0.0,
        pbcl_band_radius: int = 3,
        pbcl_iterations: int = 9,
        pbcl_threshold: float = 0.5,
        pbcl_margin: float = 0.1,
        pbcl_neighbors: int = 4,
        pbcl_max_pairs_per_tile: int | None = None,
        pbcl_pair_sampling_salt: str = "pbcl-pair-sampling-v1",
        boundary_weight: float = 0.0,
        boundary_kernel: int = 5,
    ):
        super().__init__(
            main_bce=main_bce,
            main_dice=main_dice,
            aux_bce=aux_bce,
            aux_dice=aux_dice,
        )
        self.pbcl_weight = float(pbcl_weight)
        self.pbcl_band_radius = int(pbcl_band_radius)
        self.pbcl_iterations = int(pbcl_iterations)
        self.pbcl_threshold = float(pbcl_threshold)
        self.pbcl_margin = float(pbcl_margin)
        self.pbcl_neighbors = int(pbcl_neighbors)
        self.pbcl_max_pairs_per_tile = (
            None if pbcl_max_pairs_per_tile is None else int(pbcl_max_pairs_per_tile)
        )
        self.pbcl_pair_sampling_salt = str(pbcl_pair_sampling_salt)
        self.boundary_weight = float(boundary_weight)
        self.boundary_kernel = int(boundary_kernel)
        self.requires_instance_supervision = self.pbcl_weight > 0.0
        if self.pbcl_weight < 0:
            raise ValueError("pbcl_weight must be non-negative")
        if self.boundary_weight < 0.0:
            raise ValueError("boundary_weight must be non-negative")
        if self.boundary_kernel < 3 or self.boundary_kernel % 2 == 0:
            raise ValueError("boundary_kernel must be an odd integer >= 3")
        self.last_components: dict[str, float | int] = {}

    @staticmethod
    def _main_logits(outputs) -> torch.Tensor:
        if isinstance(outputs, Mapping):
            if "out" not in outputs:
                raise KeyError("PBCL loss requires model output dictionary to contain 'out'")
            return outputs["out"]
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs

    def forward(
        self,
        outputs,
        target: torch.Tensor,
        *,
        instance_maps: torch.Tensor | None = None,
        instance_pairs: list[torch.Tensor] | None = None,
        valid_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        semantic_outputs = outputs
        if isinstance(outputs, Mapping):
            semantic_outputs = (outputs["out"],)
            if "aux" in outputs:
                semantic_outputs += (outputs["aux"],)
        segmentation_loss = super().forward(
            semantic_outputs,
            target,
            valid_masks=valid_masks,
        )
        total = segmentation_loss
        boundary_components = {}
        if self.boundary_weight > 0.0:
            if not isinstance(outputs, Mapping) or "edge" not in outputs:
                raise KeyError("Boundary supervision requires an 'edge' output tensor")
            boundary = self._boundary_supervision_loss(
                outputs["edge"], target, self.boundary_weight, self.boundary_kernel, valid_masks,
            )
            total = total + boundary
            boundary_components["boundary"] = float(boundary.detach().float().cpu())
        if self.pbcl_weight == 0.0:
            self.last_components = {
                **boundary_components,
                "segmentation": float(segmentation_loss.detach().float().cpu()),
                "pbcl": 0.0,
                "pbcl_weighted": 0.0,
                "pbcl_score": 0.0,
                "pbcl_pair_count": 0,
            }
            return total
        if instance_maps is None or instance_pairs is None:
            raise KeyError("PBCL loss requires instance_maps and instance_pairs")
        main = self._main_logits(outputs)
        if main.shape[-2:] != instance_maps.shape[-2:]:
            main = F.interpolate(main, size=instance_maps.shape[-2:], mode="bilinear", align_corners=False)
        pbcl_loss, pbcl_score, pair_count = pair_bottleneck_connectivity_loss(
            main,
            instance_maps,
            instance_pairs,
            band_radius=self.pbcl_band_radius,
            iterations=self.pbcl_iterations,
            threshold=self.pbcl_threshold,
            margin=self.pbcl_margin,
            neighbors=self.pbcl_neighbors,
            valid_masks=valid_masks,
            max_pairs_per_tile=(self.pbcl_max_pairs_per_tile if self.training else None),
            pair_sampling_salt=self.pbcl_pair_sampling_salt,
        )
        weighted = self.pbcl_weight * pbcl_loss
        total = total + weighted
        self.last_components = {
            **boundary_components,
            "segmentation": float(segmentation_loss.detach().float().cpu()),
            "pbcl": float(pbcl_loss.detach().cpu()),
            "pbcl_weighted": float(weighted.detach().cpu()),
            "pbcl_score": float(pbcl_score.cpu()),
            "pbcl_pair_count": pair_count,
            "pbcl_pair_total_count": int(sum(len(pair) for pair in instance_pairs)),
        }
        return total


def _soft_dice_from_probs(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = probs.flatten(1)
    target = target.flatten(1)
    intersection = (probs * target).sum(dim=1)
    denominator = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _soft_boundary(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def _erode_mask(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)


def _distance_from_mask(mask: torch.Tensor, iterations: int = 32, kernel_size: int = 3) -> torch.Tensor:
    current = mask.float()
    distance = torch.zeros_like(current)
    for _ in range(iterations):
        distance = distance + current
        current = _erode_mask(current, kernel_size=kernel_size)
    denom = distance.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1.0)
    return (distance / denom).clamp(0.0, 1.0)


def _skeleton_from_distance(distance: torch.Tensor, mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    local_max = F.max_pool2d(distance, kernel_size=kernel_size, stride=1, padding=pad)
    skeleton = ((distance >= local_max) & (distance > 0)).to(distance.dtype) * mask
    return F.max_pool2d(skeleton, kernel_size=kernel_size, stride=1, padding=pad).clamp(0.0, 1.0)


class StructureAwareBinarySegmentationLoss(BinarySegmentationLoss):
    def __init__(
        self,
        main_bce: float = 1.0,
        main_dice: float = 1.0,
        aux_bce: float = 0.3,
        aux_dice: float = 0.3,
        boundary_bce: float = 0.1,
        boundary_dice: float = 0.2,
        boundary_kernel: int = 5,
        edge_bce: float | None = None,
        edge_dice: float | None = None,
        skeleton_bce: float = 0.0,
        skeleton_dice: float = 0.0,
        distance_smooth_l1: float = 0.0,
        distance_iterations: int = 32,
        distance_kernel: int = 3,
        skeleton_kernel: int = 3,
    ):
        super().__init__(main_bce=main_bce, main_dice=main_dice, aux_bce=aux_bce, aux_dice=aux_dice)
        self.boundary_bce_weight = float(boundary_bce if edge_bce is None else edge_bce)
        self.boundary_dice_weight = float(boundary_dice if edge_dice is None else edge_dice)
        self.edge_bce_weight = self.boundary_bce_weight
        self.edge_dice_weight = self.boundary_dice_weight
        self.skeleton_bce_weight = float(skeleton_bce)
        self.skeleton_dice_weight = float(skeleton_dice)
        self.distance_smooth_l1_weight = float(distance_smooth_l1)
        self.boundary_kernel = int(boundary_kernel)
        if self.boundary_kernel < 3 or self.boundary_kernel % 2 == 0:
            raise ValueError("boundary_kernel must be an odd integer >= 3")
        self.distance_iterations = int(distance_iterations)
        self.distance_kernel = int(distance_kernel)
        self.skeleton_kernel = int(skeleton_kernel)
        for name, value in (
            ("distance_kernel", self.distance_kernel),
            ("skeleton_kernel", self.skeleton_kernel),
        ):
            if value < 3 or value % 2 == 0:
                raise ValueError(f"{name} must be an odd integer >= 3")
        if self.distance_iterations <= 0:
            raise ValueError("distance_iterations must be positive")

    def _boundary_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        pred_boundary = _soft_boundary(torch.sigmoid(logits).float(), self.boundary_kernel)
        target_boundary = _soft_boundary(target.float(), self.boundary_kernel)
        pred_boundary = pred_boundary.clamp(1e-4, 1.0 - 1e-4)
        with torch.autocast(device_type=pred_boundary.device.type, enabled=False):
            bce = F.binary_cross_entropy(pred_boundary.float(), target_boundary.float())
            dice = _soft_dice_from_probs(pred_boundary.float(), target_boundary.float())
        return self.boundary_bce_weight * bce + self.boundary_dice_weight * dice

    def _required_output(self, outputs: Mapping[str, torch.Tensor], key: str, weight_names: tuple[str, ...]) -> torch.Tensor:
        if key not in outputs:
            weights = ", ".join(weight_names)
            raise KeyError(f"loss config enables {weights}, but model output has no '{key}' tensor")
        return outputs[key]

    def _dict_loss(self, outputs: Mapping[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if "out" not in outputs:
            raise KeyError("loss config requires model output dictionary to contain an 'out' tensor")
        loss = self._single_loss(outputs["out"], target, self.main_bce_weight, self.main_dice_weight)
        if self.aux_bce_weight > 0.0 or self.aux_dice_weight > 0.0:
            aux = self._required_output(outputs, "aux", ("aux_bce", "aux_dice"))
            loss = loss + self._single_loss(aux, target, self.aux_bce_weight, self.aux_dice_weight)

        target_fg = target.float()
        distance_target = None
        if self.edge_bce_weight > 0.0 or self.edge_dice_weight > 0.0:
            edge = self._required_output(outputs, "edge", ("edge_bce", "edge_dice"))
            edge_target = _soft_boundary(target_fg, self.boundary_kernel)
            loss = loss + self._single_loss(edge, edge_target, self.edge_bce_weight, self.edge_dice_weight)

        if self.skeleton_bce_weight > 0.0 or self.skeleton_dice_weight > 0.0:
            skeleton = self._required_output(outputs, "skeleton", ("skeleton_bce", "skeleton_dice"))
            distance_target = _distance_from_mask(target_fg, self.distance_iterations, self.distance_kernel)
            skeleton_target = _skeleton_from_distance(distance_target, target_fg, self.skeleton_kernel)
            loss = loss + self._single_loss(
                skeleton,
                skeleton_target,
                self.skeleton_bce_weight,
                self.skeleton_dice_weight,
            )

        if self.distance_smooth_l1_weight > 0.0:
            distance = self._required_output(outputs, "distance", ("distance_smooth_l1",))
            if distance_target is None:
                distance_target = _distance_from_mask(target_fg, self.distance_iterations, self.distance_kernel)
            if distance.shape[-2:] != distance_target.shape[-2:]:
                distance = F.interpolate(distance, size=distance_target.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss + self.distance_smooth_l1_weight * F.smooth_l1_loss(
                torch.sigmoid(distance),
                distance_target,
            )
        return loss

    def forward(self, outputs, target: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, Mapping):
            return self._dict_loss(outputs, target)
        loss = super().forward(outputs, target)
        main = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if self.boundary_bce_weight > 0.0 or self.boundary_dice_weight > 0.0:
            loss = loss + self._boundary_loss(main, target)
        return loss
