"""Loss adaptations that share the archived SegFormer-B2 Boundary objective."""
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from .binary_seg import BinarySegmentationLoss
from .ssepnet import SSepNetWPRFLoss


class SegFormerComparisonLoss(BinarySegmentationLoss):
    def __init__(self, spec: Mapping, image_size: int):
        super().__init__(main_bce=spec["main_bce"], main_dice=spec["main_dice"])
        self.spec = dict(spec)
        self.method = spec["method"]
        self.requires_ewc_supervision = self.method == "ewc"
        self.requires_instance_supervision = self.method == "cmalis"
        self.requires_wprf_supervision = self.method in {"wprf_gt", "wprf_valid"}
        self.training_only_affinity_head = self.requires_wprf_supervision or self.requires_instance_supervision
        self.last_components = {}
        if self.method == "split":
            from .split import SplitSegmentationLoss

            self.split = SplitSegmentationLoss(
                image_size=image_size, split_weight=spec["split_weight"],
                boundary_kernel=spec["split_kernel"],
            )
        if self.requires_wprf_supervision:
            self.wprf = SSepNetWPRFLoss(wprf_local_windows=True, **{
                key: spec[key] for key in (
                    "wprf_edge_weight", "wprf_reach_weight", "wprf_warmup_epochs",
                    "wprf_source_batch_size", "wprf_activation_checkpoint",
                    "wprf_negative_support", "wprf_upstream",
                )
            })

    @property
    def negative_gradient_audit(self):
        return self.wprf.negative_gradient_audit if self.requires_wprf_supervision else {}

    def set_epoch(self, epoch):
        if self.requires_wprf_supervision:
            self.wprf.set_epoch(epoch)

    def model_forward_kwargs(self, **kwargs):
        # Skip the unused affinity branch during WPRF warmup.
        if self.requires_wprf_supervision:
            return {"return_affinity": self.wprf.wprf_active}
        if self.requires_instance_supervision:
            return {"return_affinity": True}
        return {}

    def forward(
        self, outputs, target, *, valid_masks=None, ewc_weights=None, instance_maps=None,
        instance_pairs=None, wprf_support=None, wprf_dense_components=None, wprf_pairs=None,
    ):
        logits = outputs["out"]
        if self.method == "ewc":
            if ewc_weights is None:
                raise KeyError("EW-BCE requires pixel weights")
            aligned = F.interpolate(logits, target.shape[-2:], mode="bilinear", align_corners=False)
            weights = ewc_weights.float()
            if valid_masks is not None:
                weights = weights * self._valid_mask(valid_masks, target.float())
            pixel_bce = F.binary_cross_entropy_with_logits(aligned, target.float(), reduction="none")
            weighted_bce = ((pixel_bce * weights).flatten(1).sum(1)
                            / weights.flatten(1).sum(1).clamp_min(1)).mean()
            semantic = (self.main_bce_weight * weighted_bce
                        + self._single_loss(logits, target, 0, self.main_dice_weight, valid_masks))
        elif self.method == "cldice":
            from .soft_cldice import soft_cldice_loss

            alpha = self.spec["cldice_alpha"]
            semantic = self._single_loss(
                logits, target, self.main_bce_weight, self.main_dice_weight * (1 - alpha), valid_masks,
            )
            semantic = semantic + self.main_dice_weight * alpha * soft_cldice_loss(
                logits, target, valid_masks=valid_masks, iterations=self.spec["cldice_iterations"],
                smooth=self.spec["cldice_smooth"],
            )
        else:
            semantic = self._single_loss(logits, target, self.main_bce_weight, self.main_dice_weight, valid_masks)

        boundary = self._boundary_supervision_loss(
            outputs["edge"], target, self.spec["boundary_weight"], self.spec["boundary_kernel"], valid_masks,
        )
        total = semantic + boundary
        if self.method == "split":
            aligned = F.interpolate(logits.float(), target.shape[-2:], mode="bilinear", align_corners=False)
            valid = None if valid_masks is None else self._valid_mask(valid_masks, target.float()).bool()
            total = total + self.spec["split_weight"] * self.split.penalty(aligned, target.float(), valid)
        elif self.requires_instance_supervision:
            from .constrained_malis import constrained_malis_loss

            malis, _ = constrained_malis_loss(
                outputs["affinity"], instance_maps, valid_masks=valid_masks,
                cpu_workers=self.spec["malis_cpu_workers"],
            )
            total = total + self.spec["malis_weight"] * malis
        elif self.requires_wprf_supervision and self.wprf.wprf_active:
            edge, reach = self.wprf.structural_term(
                outputs["affinity"], wprf_support, wprf_dense_components, wprf_pairs, valid_masks,
            )
            total = total + edge + reach
            self.last_components = self.wprf.last_components
        return total
