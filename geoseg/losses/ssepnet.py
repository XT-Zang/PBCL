from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from .binary_seg import BinarySegmentationLoss
from .pairwise_connectivity import pair_bottleneck_connectivity_loss


class SSepNetLoss(BinarySegmentationLoss):
    """Paper SSepNet objective for main, OCR auxiliary, Boundary, and PBCL outputs."""

    def __init__(
        self,
        *,
        boundary_weight: float,
        pbcl_weight: float,
        boundary_kernel: int = 5,
        pbcl_band_radius: int = 3,
        pbcl_iterations: int = 9,
        pbcl_threshold: float = 0.5,
        pbcl_margin: float = 0.1,
        pbcl_neighbors: int = 4,
        pbcl_max_pairs_per_tile: int | None = None,
        pbcl_pair_sampling_salt: str = "pbcl-pair-sampling-v1",
    ) -> None:
        super().__init__(main_bce=1.0, main_dice=1.0, aux_bce=0.3, aux_dice=0.3)
        self.boundary_weight = float(boundary_weight)
        self.pbcl_weight = float(pbcl_weight)
        self.boundary_kernel = int(boundary_kernel)
        self.pbcl_band_radius = int(pbcl_band_radius)
        self.pbcl_iterations = int(pbcl_iterations)
        self.pbcl_threshold = float(pbcl_threshold)
        self.pbcl_margin = float(pbcl_margin)
        self.pbcl_neighbors = int(pbcl_neighbors)
        self.pbcl_max_pairs_per_tile = (
            None if pbcl_max_pairs_per_tile is None else int(pbcl_max_pairs_per_tile)
        )
        self.pbcl_pair_sampling_salt = str(pbcl_pair_sampling_salt)
        self.requires_instance_supervision = self.pbcl_weight > 0.0
        self.requires_ewc_supervision = False
        if self.boundary_weight < 0.0 or self.pbcl_weight < 0.0:
            raise ValueError("boundary_weight and pbcl_weight must be non-negative")
        if self.boundary_kernel < 3 or self.boundary_kernel % 2 == 0:
            raise ValueError("boundary_kernel must be an odd integer >= 3")
        self.last_components: dict[str, float | int] = {}

    @staticmethod
    def _paper_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits).flatten(1)
        flattened_target = target.flatten(1)
        intersection = (probabilities * flattened_target).sum(dim=1)
        denominator = probabilities.sum(dim=1) + flattened_target.sum(dim=1)
        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        return 1.0 - dice.mean()

    def _single_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        bce_weight: float,
        dice_weight: float,
        valid_masks: torch.Tensor | None = None,
        bce_pixel_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = target.to(dtype=logits.dtype)
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        if valid_masks is not None or bce_pixel_weights is not None:
            valid = (
                self._valid_mask(valid_masks, target)
                if valid_masks is not None
                else torch.ones_like(target)
            )
            pixel_weights = torch.ones_like(target)
            if bce_pixel_weights is not None:
                pixel_weights = bce_pixel_weights.to(
                    device=target.device,
                    dtype=target.dtype,
                )
                if pixel_weights.ndim == 3:
                    pixel_weights = pixel_weights.unsqueeze(1)
                if pixel_weights.shape[-2:] != target.shape[-2:]:
                    pixel_weights = F.interpolate(
                        pixel_weights,
                        size=target.shape[-2:],
                        mode="nearest",
                    )
                if pixel_weights.shape[1] == 1 and target.shape[1] != 1:
                    pixel_weights = pixel_weights.expand(
                        -1, target.shape[1], -1, -1
                    )
                if pixel_weights.shape != target.shape:
                    raise ValueError(
                        "EWC pixel weight shape "
                        f"{tuple(pixel_weights.shape)} does not match target "
                        f"{tuple(target.shape)}"
                    )
                if not torch.isfinite(pixel_weights).all() or torch.any(
                    pixel_weights <= 0
                ):
                    raise ValueError("EWC pixel weights must be finite and positive")
            weighted_valid = pixel_weights * valid
            weighted_bce = (
                F.binary_cross_entropy_with_logits(logits, target, reduction="none")
                * weighted_valid
            )
            if bce_pixel_weights is not None:
                numerator = weighted_bce.flatten(1).sum(dim=1)
                denominator = weighted_valid.flatten(1).sum(dim=1).clamp_min(1.0)
                bce = (numerator / denominator).mean()
            else:
                denominator = valid.sum().clamp_min(1.0)
                bce = weighted_bce.sum() / denominator
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
        return bce_weight * F.binary_cross_entropy_with_logits(
            logits,
            target,
        ) + dice_weight * self._paper_dice_loss(logits, target)

    @staticmethod
    def _require_outputs(outputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        missing = [name for name in ("out", "aux") if name not in outputs]
        if missing:
            raise KeyError(f"SSepNet loss requires output tensors {missing}")
        return outputs["out"], outputs["aux"]

    def _main_semantic_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_masks: torch.Tensor | None,
        semantic_bce_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._single_loss(
            logits,
            target,
            1.0,
            1.0,
            valid_masks,
            semantic_bce_weights,
        )

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        instance_maps: torch.Tensor | None = None,
        instance_pairs: Sequence[torch.Tensor] | None = None,
        ewc_weights: torch.Tensor | None = None,
        valid_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(outputs, Mapping):
            raise TypeError("SSepNet loss requires a mapping of named model outputs")
        if self.requires_ewc_supervision and ewc_weights is None:
            raise KeyError("SSepNet EWC requires ewc_weights")
        semantic_bce_weights = ewc_weights if self.requires_ewc_supervision else None
        main, aux = self._require_outputs(outputs)
        main_loss = self._main_semantic_loss(
            main,
            target,
            valid_masks,
            semantic_bce_weights,
        )
        aux_loss = self._single_loss(
            aux,
            target,
            0.3,
            0.3,
            valid_masks,
            semantic_bce_weights,
        )
        total = main_loss + aux_loss

        anchor = main.float().sum() * 0.0
        boundary_loss = anchor
        if self.boundary_weight > 0.0:
            if "edge" not in outputs:
                raise KeyError("SSepNet Boundary loss requires an 'edge' output tensor")
            boundary_loss = self._boundary_supervision_loss(
                outputs["edge"],
                target,
                self.boundary_weight,
                self.boundary_kernel,
                valid_masks,
            )
            total = total + boundary_loss

        pbcl_loss = anchor
        pbcl_score = anchor.detach()
        pbcl_pair_count = 0
        if self.pbcl_weight > 0.0:
            if instance_maps is None or instance_pairs is None:
                raise KeyError("SSepNet PBCL requires instance_maps and instance_pairs")
            pbcl_logits = main
            if pbcl_logits.shape[-2:] != instance_maps.shape[-2:]:
                pbcl_logits = F.interpolate(
                    pbcl_logits,
                    size=instance_maps.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            pbcl_loss, pbcl_score, pbcl_pair_count = pair_bottleneck_connectivity_loss(
                pbcl_logits,
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
            total = total + self.pbcl_weight * pbcl_loss

        self.last_components = {
            "main": float(main_loss.detach().float().cpu()),
            "aux": float(aux_loss.detach().float().cpu()),
            "boundary": float(boundary_loss.detach().float().cpu()),
            "pbcl": float(pbcl_loss.detach().float().cpu()),
            "pbcl_weighted": float((self.pbcl_weight * pbcl_loss).detach().float().cpu()),
            "pbcl_score": float(pbcl_score.detach().float().cpu()),
            "pbcl_pair_count": int(pbcl_pair_count),
            "pbcl_pair_total_count": int(
                sum(len(pair) for pair in instance_pairs or ())
            ),
            "ewc_weight_mean": float(
                semantic_bce_weights.detach().float().mean().cpu()
            )
            if semantic_bce_weights is not None
            else 1.0,
            "ewc_weight_max": float(
                semantic_bce_weights.detach().float().amax().cpu()
            )
            if semantic_bce_weights is not None
            else 1.0,
        }
        return total


class SSepNetEWCLoss(SSepNetLoss):
    """SSepNet loss with loss-only exponentially weighted semantic BCE."""

    def __init__(
        self,
        *,
        boundary_weight: float = 0.6,
        boundary_kernel: int = 5,
        ewc_w0: float = 10.0,
        ewc_sigma: float = 7.5,
        ewc_power: float = 2.0,
    ) -> None:
        super().__init__(
            boundary_weight=boundary_weight,
            pbcl_weight=0.0,
            boundary_kernel=boundary_kernel,
        )
        self.ewc_w0 = float(ewc_w0)
        self.ewc_sigma = float(ewc_sigma)
        self.ewc_power = float(ewc_power)
        if self.ewc_w0 <= 0.0 or self.ewc_sigma <= 0.0 or self.ewc_power <= 0.0:
            raise ValueError("EWC w0, sigma, and power must be positive")
        self.requires_ewc_supervision = True


class SSepNetClDiceLoss(SSepNetLoss):
    """Matched Boundary carrier with the official 0.5 Dice/soft-clDice mixture."""

    def __init__(
        self,
        *,
        boundary_weight: float = 0.6,
        boundary_kernel: int = 5,
        cldice_alpha: float = 0.5,
        cldice_iterations: int = 10,
        cldice_smooth: float = 1.0,
    ) -> None:
        super().__init__(
            boundary_weight=boundary_weight,
            pbcl_weight=0.0,
            boundary_kernel=boundary_kernel,
        )
        self.cldice_alpha = float(cldice_alpha)
        self.cldice_iterations = int(cldice_iterations)
        self.cldice_smooth = float(cldice_smooth)
        if not 0.0 <= self.cldice_alpha <= 1.0:
            raise ValueError("clDice alpha must be in [0,1]")
        if self.cldice_iterations < 0 or self.cldice_smooth <= 0.0:
            raise ValueError("clDice iterations/smooth are invalid")
        self._last_cldice = torch.tensor(0.0)

    def _main_semantic_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_masks: torch.Tensor | None,
        semantic_bce_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if semantic_bce_weights is not None:
            raise ValueError("soft-clDice comparison does not accept EWC weights")
        from .soft_cldice import soft_cldice_loss

        bce = self._single_loss(logits, target, 1.0, 0.0, valid_masks)
        dice = self._single_loss(logits, target, 0.0, 1.0, valid_masks)
        cldice = soft_cldice_loss(
            logits,
            target,
            iterations=self.cldice_iterations,
            smooth=self.cldice_smooth,
            valid_masks=valid_masks,
        )
        self._last_cldice = cldice
        return bce + (1.0 - self.cldice_alpha) * dice + self.cldice_alpha * cldice

    def forward(self, *args, **kwargs) -> torch.Tensor:
        total = super().forward(*args, **kwargs)
        self.last_components.update(
            cldice=float(self._last_cldice.detach().float().cpu()),
            cldice_alpha=self.cldice_alpha,
        )
        return total


class SSepNetConstrainedMALISLoss(SSepNetLoss):
    """Matched Boundary objective plus a training-only constrained MALIS head."""

    def __init__(
        self,
        *,
        boundary_weight: float = 0.6,
        boundary_kernel: int = 5,
        malis_weight: float = 1.0,
        malis_cpu_workers: int = 6,
    ) -> None:
        super().__init__(
            boundary_weight=boundary_weight,
            pbcl_weight=0.0,
            boundary_kernel=boundary_kernel,
        )
        self.malis_weight = float(malis_weight)
        self.malis_cpu_workers = int(malis_cpu_workers)
        if self.malis_weight < 0.0:
            raise ValueError("MALIS weight must be non-negative")
        if self.malis_cpu_workers <= 0:
            raise ValueError("MALIS CPU workers must be positive")
        self.requires_instance_supervision = True
        self.training_only_affinity_head = True

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        instance_maps: torch.Tensor | None = None,
        instance_pairs: Sequence[torch.Tensor] | None = None,
        valid_masks: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if instance_maps is None:
            raise KeyError("Constrained MALIS requires instance_maps")
        if "affinity" not in outputs:
            raise KeyError("Constrained MALIS requires the training-only affinity output")
        base = super().forward(
            outputs,
            target,
            instance_maps=instance_maps,
            instance_pairs=instance_pairs,
            valid_masks=valid_masks,
        )
        from .constrained_malis import constrained_malis_loss

        malis, details = constrained_malis_loss(
            outputs["affinity"],
            instance_maps,
            valid_masks=valid_masks,
            cpu_workers=self.malis_cpu_workers,
        )
        weighted = self.malis_weight * malis
        self.last_components.update(
            malis=float(malis.detach().float().cpu()),
            malis_weighted=float(weighted.detach().float().cpu()),
            **details,
        )
        return base + weighted


class SSepNetWPRFLoss(SSepNetLoss):
    """Matched Boundary objective plus the WPRF affinity/reachability module."""

    def __init__(
        self,
        *,
        boundary_weight: float = 0.6,
        boundary_kernel: int = 5,
        wprf_edge_weight: float = 1.0,
        wprf_reach_weight: float = 1.0,
        wprf_warmup_epochs: int = 4,
        wprf_source_batch_size: int = 32,
        wprf_activation_checkpoint: bool = True,
        wprf_negative_support: str = "gt",
        wprf_upstream: bool = False,
        wprf_local_windows: bool = False,
    ) -> None:
        super().__init__(
            boundary_weight=boundary_weight,
            pbcl_weight=0.0,
            boundary_kernel=boundary_kernel,
        )
        self.wprf_edge_weight = float(wprf_edge_weight)
        self.wprf_reach_weight = float(wprf_reach_weight)
        self.wprf_warmup_epochs = int(wprf_warmup_epochs)
        self.wprf_source_batch_size = int(wprf_source_batch_size)
        self.wprf_activation_checkpoint = bool(wprf_activation_checkpoint)
        self.wprf_negative_support = str(wprf_negative_support)
        self.wprf_upstream = bool(wprf_upstream)
        self.wprf_local_windows = wprf_local_windows
        if self.wprf_negative_support not in {"gt", "valid"}:
            raise ValueError("WPRF negative support must be gt or valid")
        if min(self.wprf_edge_weight, self.wprf_reach_weight) < 0.0:
            raise ValueError("WPRF loss weights must be non-negative")
        if self.wprf_warmup_epochs < 0 or self.wprf_source_batch_size <= 0:
            raise ValueError("WPRF warmup/source batch parameters are invalid")
        self.requires_wprf_supervision = True
        self.training_only_affinity_head = True
        self._epoch = 0
        self.negative_gradient_audit: dict[str, object] = {}

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def wprf_active(self) -> bool:
        return self._epoch >= self.wprf_warmup_epochs

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        wprf_support: torch.Tensor | None = None,
        wprf_dense_components: torch.Tensor | None = None,
        wprf_pairs: Sequence[Mapping[str, torch.Tensor]] | None = None,
        valid_masks: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if "affinity" not in outputs:
            raise KeyError("WPRF requires the training-only affinity output")
        base = super().forward(outputs, target, valid_masks=valid_masks)
        edge, reach = self.structural_term(
            outputs["affinity"], wprf_support, wprf_dense_components, wprf_pairs, valid_masks,
        )
        return base + edge + reach

    def structural_term(self, affinity, wprf_support, wprf_dense_components, wprf_pairs, valid_masks):
        anchor = affinity.float().sum() * 0.0
        edge = reach = anchor
        pair_count = 0
        if self.wprf_active:
            if wprf_support is None or wprf_dense_components is None or wprf_pairs is None:
                raise KeyError("WPRF requires support, dense components, and pair caches")
            from .wprf import wprf_structural_loss

            grid_valid = None
            if valid_masks is not None:
                invalid = 1.0 - valid_masks.float()
                grid_valid = F.adaptive_max_pool2d(invalid, affinity.shape[-2:]) == 0
                wprf_support = wprf_support & grid_valid
            negative_support = None
            if self.wprf_negative_support == "valid":
                negative_support = grid_valid
                if negative_support is None:
                    negative_support = torch.ones_like(wprf_support, dtype=torch.bool)
            if self.training and self.wprf_upstream and self._epoch in {4, 19, 39} and self.negative_gradient_audit.get("epoch") != self._epoch + 1:
                from .wprf import audit_negative_reachability

                audited = audit_negative_reachability(
                    affinity, negative_support if negative_support is not None else wprf_support, wprf_pairs,
                    local_windows=self.wprf_local_windows,
                )
                if audited:
                    self.negative_gradient_audit = {"epoch": self._epoch + 1, "support": self.wprf_negative_support,
                                                    "denominator": "all cached negatives in the first batch with negatives at this epoch",
                                                    "scales": audited}

            edge, reach, pair_count = wprf_structural_loss(
                affinity,
                wprf_support,
                wprf_dense_components,
                wprf_pairs,
                source_batch_size=self.wprf_source_batch_size,
                activation_checkpoint=self.wprf_activation_checkpoint,
                valid_masks=grid_valid,
                negative_support_masks=negative_support,
                upstream=self.wprf_upstream,
                local_windows=self.wprf_local_windows,
            )
        weighted_edge = self.wprf_edge_weight * edge
        weighted_reach = self.wprf_reach_weight * reach
        self.last_components.update(
            wprf_active=int(self.wprf_active),
            wprf_epoch=self._epoch,
            wprf_edge=float(edge.detach().float().cpu()),
            wprf_reach=float(reach.detach().float().cpu()),
            wprf_edge_weighted=float(weighted_edge.detach().float().cpu()),
            wprf_reach_weighted=float(weighted_reach.detach().float().cpu()),
            wprf_pair_count=int(pair_count),
        )
        return weighted_edge, weighted_reach


class SSepNetSplitLoss(SSepNetLoss):
    """DINOv3 Boundary baseline augmented with the SPLIT mask separation penalty."""

    def __init__(
        self,
        *,
        image_size: int = 512,
        boundary_weight: float = 0.6,
        boundary_kernel: int = 5,
        split_weight: float = 0.1,
        split_kernel: int = 3,
    ) -> None:
        super().__init__(
            boundary_weight=boundary_weight,
            boundary_kernel=boundary_kernel,
            pbcl_weight=0.0,
        )
        from .split import SplitSegmentationLoss

        self.split_weight = float(split_weight)
        self.split = SplitSegmentationLoss(
            image_size=int(image_size),
            split_weight=float(split_weight),
            boundary_kernel=int(split_kernel),
        )

    def forward(self, outputs, target, *, valid_masks=None, **kwargs):
        total = super().forward(outputs, target, valid_masks=valid_masks, **kwargs)
        logits = outputs["out"].float()
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits, target.shape[-2:], mode="bilinear", align_corners=False
            )
        target_f = target.float()
        valid = (
            self._valid_mask(valid_masks, target_f).bool()
            if valid_masks is not None
            else None
        )
        split_term = self.split.penalty(logits, target_f, valid)
        self.last_components["split"] = float(split_term.detach().float().cpu())
        return total + self.split_weight * split_term
