from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from geoseg.metrics.binary import binary_valid_mask
from geoseg.models.aquav3_outputs import get_main_logits


class SplitSegmentationLoss(nn.Module):
    """Mask BCE and the optional separation penalty in Shojaei et al., Eq. (1)."""

    def __init__(self, *, image_size: int, split_weight: float, boundary_kernel: int) -> None:
        super().__init__()
        self.image_size = (image_size, image_size)
        self.split_weight = split_weight
        self.boundary_kernel = boundary_kernel
        if boundary_kernel < 1 or boundary_kernel % 2 == 0:
            raise ValueError("SPLIT boundary kernel must be positive and odd")
        self.register_buffer(
            "laplacian",
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
            .reshape(1, 1, 3, 3),
            persistent=False,
        )

    def _boundary_band(self, mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # Each invalid neighbor contributes the center value, suppressing artificial edges.
        valid_float = valid.float()
        kernel = self.laplacian.to(device=mask.device)
        response = F.conv2d(mask * valid_float, kernel, padding=1)
        response = response - mask * F.conv2d(valid_float, kernel, padding=1)
        thin = (response.abs() > 0) & valid
        band = F.max_pool2d(
            thin.float(), self.boundary_kernel, stride=1, padding=self.boundary_kernel // 2
        )
        return band.bool() & valid

    def forward(
        self,
        outputs,
        target: torch.Tensor,
        *,
        valid_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = get_main_logits(outputs).float()
        target = target.float()
        valid = binary_valid_mask(valid_masks, target) if valid_masks is not None else None
        if target.shape[-2:] != self.image_size:
            target = F.interpolate(target, size=self.image_size, mode="nearest")
            if valid is not None:
                valid = F.interpolate(valid.float(), size=self.image_size, mode="nearest").bool()
        if logits.shape[-2:] != self.image_size:
            logits = F.interpolate(logits, size=self.image_size, mode="bilinear", align_corners=False)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if valid is None:
            mask_loss = bce.mean()
        else:
            mask_loss = bce.masked_fill(~valid, 0.0).sum() / valid.sum().clamp_min(1)
        if self.split_weight == 0:
            return mask_loss

        return (1.0 - self.split_weight) * mask_loss + self.split_weight * self.penalty(logits, target, valid)

    def penalty(
        self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None,
    ) -> torch.Tensor:
        with torch.no_grad():
            if valid is None:
                valid = torch.ones_like(target, dtype=torch.bool)
            band = self._boundary_band(target, valid)
        penalty = F.softplus(logits).masked_fill(~band, 0.0)
        return (penalty.flatten(1).sum(1) / (band.flatten(1).sum(1) + 1e-6)).mean()
