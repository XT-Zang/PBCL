"""Standalone Pairwise Bottleneck Connectivity Loss (PBCL) module.

PBCL directly penalizes high-probability bridges connecting distinct neighboring
instances in a single binary probability map using bounded morphological
max-min propagation (four-neighbor bottleneck connectivity) inside each pair's
local padded crop.

Reference:
    Pairwise Bottleneck Connectivity Loss for Structurally Separable
    Raft-Aquaculture Mapping (IEEE J-STARS).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from geoseg.losses.pairwise_connectivity import (
    PAIR_BUCKET_CODE,
    PAIR_BUCKET_NAME,
    pair_bottleneck_connectivity_loss,
    stratified_pair_sample,
)


def extract_neighbor_pairs(
    instance_map: np.ndarray | torch.Tensor,
    *,
    max_gap: int = 5,
    policy: str = "all",
) -> np.ndarray:
    """Extract neighboring instance pairs ``(id_a, id_b, bucket_code)`` from a 2D label map.

    Bucket codes follow the paper's gap stratification:
      * ``0`` (``"touching"``): Euclidean pixel gap ``g_ij == 0`` (four-neighbor or diagonal contact)
      * ``1`` (``"1_2"``): ``1 <= g_ij <= 2`` pixels
      * ``2`` (``"3_5"``): ``3 <= g_ij <= max_gap`` pixels

    Supported contact policies:
      * ``"all"``: retain all contact and noncontact pairs within ``max_gap`` (paper default).
      * ``"no_four_contact"``: exclude direct four-neighbor touching pairs, keep diagonal-only contacts and noncontact pairs.
      * ``"noncontact"``: retain only noncontact pairs (``g_ij >= 1``).
      * ``"four_contact"``: retain only direct four-neighbor touching pairs.
    """
    from scipy.ndimage import distance_transform_edt

    if policy not in {"all", "no_four_contact", "noncontact", "four_contact"}:
        raise ValueError(f"Unsupported contact policy: {policy!r}")

    if isinstance(instance_map, torch.Tensor):
        labels = instance_map.detach().cpu().numpy()
    else:
        labels = np.asarray(instance_map)
    if labels.ndim != 2:
        raise ValueError(f"instance_map must be 2D (H, W), got shape {labels.shape}")

    ids = [int(v) for v in np.unique(labels) if int(v) > 0]
    if len(ids) < 2:
        return np.zeros((0, 3), dtype=np.int64)

    # Precompute direct four-neighbor touching pairs via shifted equality checks.
    four_touching: set[tuple[int, int]] = set()
    for a, b in (
        (labels[:-1, :], labels[1:, :]),
        (labels[:, :-1], labels[:, 1:]),
    ):
        diff = (a > 0) & (b > 0) & (a != b)
        if np.any(diff):
            p1 = a[diff]
            p2 = b[diff]
            lo = np.minimum(p1, p2)
            hi = np.maximum(p1, p2)
            for u, v in zip(lo.tolist(), hi.tolist()):
                four_touching.add((int(u), int(v)))

    # Precompute bounding boxes for quick pruning.
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for inst_id in ids:
        ys, xs = np.nonzero(labels == inst_id)
        bboxes[inst_id] = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))

    h, w = labels.shape
    pad = int(max_gap) + 2
    rows: list[tuple[int, int, int]] = []
    for idx_i, id_i in enumerate(ids):
        y0_i, y1_i, x0_i, x1_i = bboxes[id_i]
        for id_j in ids[idx_i + 1 :]:
            y0_j, y1_j, x0_j, x1_j = bboxes[id_j]
            if (
                y1_i + max_gap + 1 < y0_j
                or y1_j + max_gap + 1 < y0_i
                or x1_i + max_gap + 1 < x0_j
                or x1_j + max_gap + 1 < x0_i
            ):
                continue

            is_four = (id_i, id_j) in four_touching
            if policy == "four_contact":
                if is_four:
                    rows.append((id_i, id_j, PAIR_BUCKET_CODE["touching"]))
                continue

            ry0 = max(0, min(y0_i, y0_j) - pad)
            ry1 = min(h, max(y1_i, y1_j) + pad + 1)
            rx0 = max(0, min(x0_i, x0_j) - pad)
            rx1 = min(w, max(x1_i, x1_j) + pad + 1)
            crop = labels[ry0:ry1, rx0:rx1]
            mask_i = crop == id_i
            mask_j = crop == id_j
            dist = distance_transform_edt(~mask_i)
            min_dist = float(dist[mask_j].min())
            # Exact pixel gap g_ij = max(0, floor(min_dist + 1e-9) - 1) (Eq. (1)):
            # 4-neighbor (d=1) and diagonal (d=sqrt(2)) both yield g_ij = 0;
            # worst-case g_ij <= 5 displacements (3,6) and (4,5) satisfy d < 7 -> g_ij = 5.
            gap = max(0, int(np.floor(min_dist + 1e-9)) - 1)

            if gap > max_gap:
                continue
            if gap == 0:
                if policy == "noncontact":
                    continue
                if policy == "no_four_contact" and is_four:
                    continue
                bucket = PAIR_BUCKET_CODE["touching"]
            elif gap <= 2:
                bucket = PAIR_BUCKET_CODE["1_2"]
            else:
                bucket = PAIR_BUCKET_CODE["3_5"]
            rows.append((id_i, id_j, bucket))

    if not rows:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


class PairwiseBottleneckConnectivityLoss(nn.Module):
    """Differentiable Pairwise Bottleneck Connectivity Loss (PBCL).

    Drop-in training-only structural loss for binary foreground segmentation.
    Requires no extra decoder head at inference time.

    Args:
        band_radius: Eight-neighbor dilation radius ``r`` defining the local
            interaction band (default: ``3``).
        iterations: Number of four-neighbor max-min propagation steps ``K``
            (default: ``9``).
        threshold: Binary decision threshold ``tau`` (default: ``0.5``).
        margin: Separation safety margin ``m``; bottleneck scores above
            ``tau - m`` receive a squared hinge penalty (default: ``0.1``).
        neighbors: Connectivity neighborhood (currently ``4``).
        max_pairs_per_tile: Optional deterministic stratified pair cap per image.
        pair_sampling_salt: Deterministic hash salt for stratified sampling.
    """

    def __init__(
        self,
        *,
        band_radius: int = 3,
        iterations: int = 9,
        threshold: float = 0.5,
        margin: float = 0.1,
        neighbors: int = 4,
        max_pairs_per_tile: int | None = None,
        pair_sampling_salt: str = "pbcl-pair-sampling-v1",
    ) -> None:
        super().__init__()
        self.band_radius = int(band_radius)
        self.iterations = int(iterations)
        self.threshold = float(threshold)
        self.margin = float(margin)
        self.neighbors = int(neighbors)
        self.max_pairs_per_tile = (
            None if max_pairs_per_tile is None else int(max_pairs_per_tile)
        )
        self.pair_sampling_salt = str(pair_sampling_salt)

    def forward(
        self,
        logits: torch.Tensor,
        instance_maps: torch.Tensor,
        instance_pairs: Sequence[torch.Tensor] | None = None,
        *,
        valid_masks: torch.Tensor | None = None,
        max_gap: int = 5,
        policy: str = "all",
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, int]:
        """Compute PBCL from binary logits ``(B, 1, H, W)`` and instance maps ``(B, H, W)``.

        If ``instance_pairs`` is omitted, neighboring pairs are extracted on the fly
        via :func:`extract_neighbor_pairs` (convenient for prototyping; for training
        at scale, precompute ``instance_pairs`` once per tile).
        """
        if instance_pairs is None:
            maps_3d = instance_maps[:, 0] if instance_maps.ndim == 4 else instance_maps
            instance_pairs = [
                torch.from_numpy(
                    extract_neighbor_pairs(maps_3d[i], max_gap=max_gap, policy=policy)
                )
                for i in range(maps_3d.shape[0])
            ]

        loss, mean_bottleneck, pair_count = pair_bottleneck_connectivity_loss(
            logits,
            instance_maps,
            instance_pairs,
            band_radius=self.band_radius,
            iterations=self.iterations,
            threshold=self.threshold,
            margin=self.margin,
            neighbors=self.neighbors,
            valid_masks=valid_masks,
            max_pairs_per_tile=self.max_pairs_per_tile,
            pair_sampling_salt=self.pair_sampling_salt,
        )
        if return_stats:
            return loss, mean_bottleneck, pair_count
        return loss


__all__ = [
    "PAIR_BUCKET_CODE",
    "PAIR_BUCKET_NAME",
    "PairwiseBottleneckConnectivityLoss",
    "extract_neighbor_pairs",
    "pair_bottleneck_connectivity_loss",
    "stratified_pair_sample",
]
