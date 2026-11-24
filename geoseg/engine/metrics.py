from __future__ import annotations

import numpy as np
import torch

from geoseg.metrics.binary import binary_confusion_tensor, binary_global_stats
from tools.metric import Evaluator


class BinaryMetricAccumulator:
    """Accumulate binary confusion counts on-device and synchronize once per epoch."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)
        self.counts = torch.zeros(4, dtype=torch.int64)

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        probabilities: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        batch_counts = binary_confusion_tensor(
            logits,
            targets,
            threshold=self.threshold,
            probabilities=probabilities,
            valid_mask=valid_mask,
        )
        if self.counts.device != batch_counts.device:
            self.counts = self.counts.to(batch_counts.device)
        self.counts.add_(batch_counts)

    def compute(self) -> dict[str, float]:
        tp, fp, fn, tn = self.counts.detach().cpu().tolist()
        return binary_global_stats({"tp": tp, "fp": fp, "fn": fn, "tn": tn})

    def reset(self) -> None:
        self.counts.zero_()


class MulticlassMetricAccumulator:
    """Batch-oriented compatibility wrapper around the existing Evaluator."""

    def __init__(self, num_classes: int) -> None:
        self._evaluator = Evaluator(num_class=int(num_classes))

    @property
    def confusion_matrix(self) -> np.ndarray:
        return self._evaluator.confusion_matrix

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = logits.argmax(dim=1)
        if targets.ndim == predictions.ndim + 1 and targets.shape[1] == 1:
            targets = targets[:, 0]
        packed = torch.stack((targets.long(), predictions.long())).detach().cpu().numpy()
        self._evaluator.add_batch(packed[0], packed[1])

    def compute(self) -> dict[str, object]:
        return {
            "iou_per_class": self._evaluator.Intersection_over_Union(),
            "f1_per_class": self._evaluator.F1(),
            "oa": float(self._evaluator.OA()),
        }

    def reset(self) -> None:
        self._evaluator.reset()
