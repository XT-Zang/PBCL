from __future__ import annotations

import torch
import torch.nn.functional as F


CLDICE_SOURCE_COMMIT = "47d31a6cc4a8101b1ffe8052994821961e57af9f"


def _soft_erode(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"soft-clDice expects BCHW tensors, got {tuple(image.shape)}")
    vertical = -F.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_open(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(_soft_erode(image), kernel_size=3, stride=1, padding=1)


def soft_skeletonize(image: torch.Tensor, *, iterations: int = 10) -> torch.Tensor:
    """Official clDice min/max-pooling skeletonization for 2-D tensors."""
    iterations = int(iterations)
    if iterations < 0:
        raise ValueError("soft skeleton iterations must be non-negative")
    opened = _soft_open(image)
    skeleton = F.relu(image - opened)
    current = image
    for _ in range(iterations):
        current = _soft_erode(current)
        opened = _soft_open(current)
        delta = F.relu(current - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    iterations: int = 10,
    smooth: float = 1.0,
    valid_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the foreground soft-clDice loss without adding another Dice term."""
    if logits.ndim != 4 or target.ndim != 4:
        raise ValueError("soft-clDice requires BCHW logits and targets")
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    probabilities = torch.sigmoid(logits.float())
    target = target.to(device=logits.device, dtype=torch.float32)
    if target.shape[1] != probabilities.shape[1]:
        raise ValueError(
            f"soft-clDice channel mismatch: {probabilities.shape[1]} vs {target.shape[1]}"
        )
    if valid_masks is not None:
        valid = valid_masks.to(device=logits.device, dtype=torch.float32)
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.shape[-2:] != target.shape[-2:]:
            valid = F.interpolate(valid, size=target.shape[-2:], mode="nearest")
        if valid.shape[1] == 1 and target.shape[1] != 1:
            valid = valid.expand(-1, target.shape[1], -1, -1)
        if valid.shape != target.shape:
            raise ValueError(
                f"soft-clDice valid mask mismatch: {tuple(valid.shape)} vs {tuple(target.shape)}"
            )
        probabilities = probabilities * valid
        target = target * valid

    skeleton_prediction = soft_skeletonize(probabilities, iterations=iterations)
    skeleton_target = soft_skeletonize(target, iterations=iterations)
    smooth_value = float(smooth)
    if smooth_value <= 0.0:
        raise ValueError("soft-clDice smooth must be positive")

    reduce_dims = tuple(range(1, probabilities.ndim))
    topology_precision = (
        (skeleton_prediction * target).sum(dim=reduce_dims) + smooth_value
    ) / (skeleton_prediction.sum(dim=reduce_dims) + smooth_value)
    topology_sensitivity = (
        (skeleton_target * probabilities).sum(dim=reduce_dims) + smooth_value
    ) / (skeleton_target.sum(dim=reduce_dims) + smooth_value)
    score = (
        2.0
        * topology_precision
        * topology_sensitivity
        / (topology_precision + topology_sensitivity).clamp_min(1.0e-12)
    )
    return 1.0 - score.mean()


__all__ = [
    "CLDICE_SOURCE_COMMIT",
    "soft_cldice_loss",
    "soft_skeletonize",
]
