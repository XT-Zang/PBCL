from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


WPRF_SOURCE_COMMIT = "418e362b440df08e72ae6da54abeec6559a8f442"
WPRF_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
WPRF_SCALES: tuple[int, ...] = (1, 2, 4, 8, 16)


def _shift_to_target(
    values: torch.Tensor,
    dy: int,
    dx: int,
) -> torch.Tensor:
    """Move a value stored at source u to target u+(dy,dx), without wrapping."""
    output = torch.zeros_like(values)
    height, width = values.shape[-2:]
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    target_y0, target_y1 = source_y0 + dy, source_y1 + dy
    target_x0, target_x1 = source_x0 + dx, source_x1 + dx
    if source_y1 > source_y0 and source_x1 > source_x0:
        output[..., target_y0:target_y1, target_x0:target_x1] = values[
            ..., source_y0:source_y1, source_x0:source_x1
        ]
    return output


def symmetrize_affinities(
    affinity_logits: torch.Tensor,
    *,
    offsets: Sequence[tuple[int, int]] = WPRF_OFFSETS,
) -> torch.Tensor:
    if affinity_logits.ndim != 4:
        raise ValueError("WPRF affinity logits must be BCHW")
    if affinity_logits.shape[1] != len(offsets):
        raise ValueError(
            f"WPRF affinity channels {affinity_logits.shape[1]} do not match offsets {len(offsets)}"
        )
    probabilities = torch.sigmoid(affinity_logits.float())
    offset_index = {tuple(offset): index for index, offset in enumerate(offsets)}
    symmetric: list[torch.Tensor] = []
    for index, (dy, dx) in enumerate(offsets):
        reverse_index = offset_index.get((-dy, -dx))
        if reverse_index is None:
            raise ValueError("WPRF offsets must be origin-symmetric")
        reverse_at_source = _shift_to_target(
            probabilities[:, reverse_index], -dy, -dx
        )
        symmetric.append(0.5 * (probabilities[:, index] + reverse_at_source))
    return torch.stack(symmetric, dim=1)


def edge_affinity_loss(
    symmetric_affinities: torch.Tensor,
    dense_component_ids: torch.Tensor,
    *,
    offsets: Sequence[tuple[int, int]] = WPRF_OFFSETS,
    valid_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    if dense_component_ids.ndim == 4:
        dense_component_ids = dense_component_ids[:, 0]
    if dense_component_ids.ndim != 3:
        raise ValueError("WPRF dense component ids must be BHW or B1HW")
    if symmetric_affinities.shape[-2:] != dense_component_ids.shape[-2:]:
        raise ValueError("WPRF affinity and component grids do not match")
    component = dense_component_ids.to(
        device=symmetric_affinities.device, dtype=torch.long
    )
    positive_values: list[torch.Tensor] = []
    negative_values: list[torch.Tensor] = []
    for index, (dy, dx) in enumerate(offsets):
        neighbor = _shift_to_target(component, -dy, -dx)
        valid_neighbor = _shift_to_target(torch.ones_like(component), -dy, -dx) > 0
        source_foreground = component > 0
        neighbor_foreground = neighbor > 0
        valid = valid_neighbor & (source_foreground | neighbor_foreground)
        if valid_masks is not None:
            grid_valid = valid_masks.to(device=component.device, dtype=torch.bool)
            if grid_valid.ndim == 4:
                grid_valid = grid_valid[:, 0]
            valid &= grid_valid & _shift_to_target(grid_valid, -dy, -dx)
        positive = valid & source_foreground & (component == neighbor)
        negative = valid & ~positive
        values = symmetric_affinities[:, index].clamp(1.0e-6, 1.0 - 1.0e-6)
        if positive.any():
            positive_values.append(-torch.log(values[positive]))
        if negative.any():
            negative_values.append(-torch.log1p(-values[negative]))
    anchor = symmetric_affinities.sum() * 0.0
    positive_loss = (
        torch.cat(positive_values).mean() if positive_values else anchor
    )
    negative_loss = (
        torch.cat(negative_values).mean() if negative_values else anchor
    )
    if positive_values and negative_values:
        return 0.5 * (positive_loss + negative_loss)
    return positive_loss + negative_loss


def _max_min_step(
    reachability: torch.Tensor,
    edge_weights: torch.Tensor,
    support: torch.Tensor,
    offsets: Sequence[tuple[int, int]],
    retain_previous: bool = True,
) -> torch.Tensor:
    candidates = [reachability] if retain_previous else []
    for index, (dy, dx) in enumerate(offsets):
        source_strength = torch.minimum(reachability, edge_weights[index].unsqueeze(0))
        candidates.append(_shift_to_target(source_strength, dy, dx))
    updated = torch.stack(candidates, dim=0).amax(dim=0)
    return updated * support.unsqueeze(0).to(dtype=updated.dtype)


def _local_widest_paths(edge_weights, support, pairs, sources, inverse, steps, offsets,
                        chunk_size, activation_checkpoint):
    """Vectorized source-centered windows bound every path of at most k steps."""
    radius = steps * max(max(abs(dy), abs(dx)) for dy, dx in offsets)
    axis = torch.arange(-radius, radius + 1, device=edge_weights.device)
    side = len(axis)
    height, width = support.shape
    result = edge_weights.new_zeros(len(pairs))
    for start in range(0, len(sources), chunk_size):
        centers = sources[start:start + chunk_size]
        y = centers[:, 0, None, None] + axis[None, :, None]
        x = centers[:, 1, None, None] + axis[None, None, :]
        inside = (y >= 0) & (y < height) & (x >= 0) & (x < width)
        y, x = y.clamp(0, height - 1), x.clamp(0, width - 1)
        local_support = support[y, x] & inside
        local_edges = edge_weights[:, y, x]
        reach = edge_weights.new_zeros((len(centers), side, side))
        reach[:, radius, radius] = 1
        reach = reach * local_support
        accumulated = reach

        def step(previous, weights, valid):
            candidates = [_shift_to_target(torch.minimum(previous, weights[index]), dy, dx)
                          for index, (dy, dx) in enumerate(offsets)]
            return torch.stack(candidates).amax(0) * valid

        for _ in range(steps):
            if activation_checkpoint and edge_weights.requires_grad:
                reach = checkpoint(step, reach, local_edges, local_support, use_reentrant=False)
            else:
                reach = step(reach, local_edges, local_support)
            accumulated = torch.maximum(accumulated, reach)
        selected = (inverse >= start) & (inverse < start + len(centers))
        indices = torch.nonzero(selected, as_tuple=False).flatten()
        source_index = inverse[indices] - start
        targets = pairs[indices, 2:] - centers[source_index] + radius
        in_window = ((targets >= 0) & (targets < side)).all(dim=1)
        targets = targets.clamp(0, side - 1)
        result[indices] = accumulated[source_index, targets[:, 0], targets[:, 1]] * in_window
    return result


def widest_path_for_pairs(
    edge_weights: torch.Tensor,
    support: torch.Tensor,
    pairs_yxyx: torch.Tensor,
    *,
    num_steps: int,
    offsets: Sequence[tuple[int, int]] = WPRF_OFFSETS,
    source_batch_size: int = 32,
    activation_checkpoint: bool = True,
    upstream: bool = False,
    local_windows: bool = False,
) -> torch.Tensor:
    """Compute differentiable k-step max-min reachability for one image."""
    if pairs_yxyx.numel() == 0:
        return edge_weights.new_zeros((0,))
    pairs = pairs_yxyx.to(device=edge_weights.device, dtype=torch.long)
    if pairs.ndim != 2 or pairs.shape[1] != 4:
        raise ValueError("WPRF pairs must have shape (N,4)=[yu,xu,yv,xv]")
    support = support.to(device=edge_weights.device, dtype=torch.bool)
    height, width = support.shape
    if torch.any(pairs[:, (0, 2)] < 0) or torch.any(pairs[:, (0, 2)] >= height):
        raise ValueError("WPRF pair y-coordinate is outside the support grid")
    if torch.any(pairs[:, (1, 3)] < 0) or torch.any(pairs[:, (1, 3)] >= width):
        raise ValueError("WPRF pair x-coordinate is outside the support grid")
    sources, inverse = torch.unique(pairs[:, :2], dim=0, return_inverse=True)
    if local_windows:
        if not upstream:
            raise ValueError("Local windows are enabled for upstream max-over-steps reachability")
        return _local_widest_paths(
            edge_weights, support, pairs, sources, inverse, int(num_steps), offsets,
            max(1, int(source_batch_size)), activation_checkpoint,
        )
    results = edge_weights.new_empty((pairs.shape[0],), dtype=torch.float32)
    chunk_size = max(1, int(source_batch_size))
    for start in range(0, int(sources.shape[0]), chunk_size):
        chunk = sources[start : start + chunk_size]
        reachability = edge_weights.new_zeros(
            (chunk.shape[0], height, width), dtype=torch.float32
        )
        row = torch.arange(chunk.shape[0], device=edge_weights.device)
        reachability[row, chunk[:, 0], chunk[:, 1]] = 1.0
        reachability = reachability * support.unsqueeze(0)
        accumulated = reachability
        for _ in range(int(num_steps)):
            if activation_checkpoint and edge_weights.requires_grad:
                reachability = checkpoint(
                    lambda value: _max_min_step(
                        value, edge_weights, support, offsets, retain_previous=not upstream
                    ),
                    reachability,
                    use_reentrant=False,
                )
            else:
                reachability = _max_min_step(
                    reachability, edge_weights, support, offsets, retain_previous=not upstream
                )
            if upstream:
                accumulated = torch.maximum(accumulated, reachability)
        if upstream:
            reachability = accumulated
        selected_pairs = (inverse >= start) & (inverse < start + chunk.shape[0])
        selected_indices = torch.nonzero(selected_pairs, as_tuple=False).flatten()
        local_source = inverse[selected_indices] - start
        targets = pairs[selected_indices, 2:]
        results[selected_indices] = reachability[
            local_source, targets[:, 0], targets[:, 1]
        ]
    return results


def reachability_loss(
    symmetric_affinities: torch.Tensor,
    support_masks: torch.Tensor,
    pair_batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    scales: Sequence[int] = WPRF_SCALES,
    offsets: Sequence[tuple[int, int]] = WPRF_OFFSETS,
    gamma: Sequence[float] | None = None,
    source_batch_size: int = 32,
    activation_checkpoint: bool = True,
    negative_support_masks: torch.Tensor | None = None,
    upstream: bool = False,
    local_windows: bool = False,
) -> tuple[torch.Tensor, int]:
    if support_masks.ndim == 4:
        support_masks = support_masks[:, 0]
    if len(pair_batches) != int(symmetric_affinities.shape[0]):
        raise ValueError("WPRF pair batches must align with the model batch")
    weights = tuple(float(value) for value in (gamma or [1.0 / len(scales)] * len(scales)))
    if len(weights) != len(scales):
        raise ValueError("WPRF gamma and scale lengths differ")
    image_losses: list[torch.Tensor] = []
    total_pairs = 0
    for batch_index, pair_mapping in enumerate(pair_batches):
        scale_losses: list[torch.Tensor] = []
        for scale, scale_weight in zip(scales, weights, strict=True):
            positive = pair_mapping.get(f"pos_{int(scale)}")
            negative = pair_mapping.get(f"neg_{int(scale)}")
            if positive is None or negative is None:
                raise KeyError(f"WPRF cache is missing scale {scale} pairs")
            positive = torch.as_tensor(positive, dtype=torch.long)
            negative = torch.as_tensor(negative, dtype=torch.long)
            all_pairs = torch.cat((positive, negative), dim=0)
            if all_pairs.numel() == 0:
                continue
            scores = widest_path_for_pairs(
                symmetric_affinities[batch_index],
                support_masks[batch_index],
                positive if negative_support_masks is not None else all_pairs,
                num_steps=int(scale),
                offsets=offsets,
                source_batch_size=source_batch_size,
                activation_checkpoint=activation_checkpoint,
                upstream=upstream,
                local_windows=local_windows,
            )
            count_positive = int(positive.shape[0])
            if negative_support_masks is not None and negative.shape[0]:
                negative_support = negative_support_masks[batch_index].squeeze(0)
                negative_scores = widest_path_for_pairs(
                    symmetric_affinities[batch_index], negative_support, negative,
                    num_steps=int(scale), offsets=offsets,
                    source_batch_size=source_batch_size,
                    activation_checkpoint=activation_checkpoint,
                    upstream=upstream,
                    local_windows=local_windows,
                )
                scores = torch.cat((scores[:count_positive], negative_scores))
            if not upstream:
                scores = scores.clamp(1.0e-6, 1.0 - 1.0e-6)
            losses: list[torch.Tensor] = []
            if count_positive:
                positive_scores = scores[:count_positive] + (1.0e-6 if upstream else 0.0)
                losses.append(-torch.log(positive_scores))
            if int(negative.shape[0]):
                negative_scores = scores[count_positive:].clamp(0.0, 1.0 - 1.0e-6)
                losses.append(-torch.log(1.0 - negative_scores + 1.0e-6) if upstream else -torch.log1p(-negative_scores))
            vector = torch.cat(losses)
            scale_losses.append(float(scale_weight) * vector.mean())
            total_pairs += int(vector.numel())
        if scale_losses:
            image_losses.append(torch.stack(scale_losses).sum())
        else:
            image_losses.append(symmetric_affinities[batch_index].sum() * 0.0)
    return torch.stack(image_losses).mean(), total_pairs


def upstream_edge_affinity_loss(
    symmetric: torch.Tensor,
    component_ids: torch.Tensor,
    valid_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pinned upstream half-space, source-foreground, per-image balanced BCE."""
    components = component_ids.squeeze(1) if component_ids.ndim == 4 else component_ids
    batch_size = symmetric.shape[0]
    positive_sum = symmetric.new_zeros(batch_size)
    negative_sum = symmetric.new_zeros(batch_size)
    positive_count = symmetric.new_zeros(batch_size)
    negative_count = symmetric.new_zeros(batch_size)
    for index, (dy, dx) in enumerate(WPRF_OFFSETS):
        if not (dy > 0 or (dy == 0 and dx > 0)):
            continue
        neighbor = _shift_to_target(components, -dy, -dx)
        selected = (components > 0) & _shift_to_target(torch.ones_like(components), -dy, -dx).bool()
        if valid_masks is not None:
            valid = valid_masks.squeeze(1) if valid_masks.ndim == 4 else valid_masks
            selected &= valid & _shift_to_target(valid, -dy, -dx)
        positive = selected & (components == neighbor) & (neighbor > 0)
        negative = selected & ~positive
        # CUDA autocast rejects probability-space BCE even for FP32 inputs.
        with torch.autocast(device_type=symmetric.device.type, enabled=False):
            probabilities = symmetric[:, index].float().clamp(1e-6, 1 - 1e-6)
            values = F.binary_cross_entropy(probabilities, positive.float(), reduction="none")
        positive_sum = positive_sum + (values * positive).sum(dim=(1, 2))
        negative_sum = negative_sum + (values * negative).sum(dim=(1, 2))
        positive_count = positive_count + positive.sum(dim=(1, 2))
        negative_count = negative_count + negative.sum(dim=(1, 2))
    positive_mean = positive_sum / (positive_count + 1e-6)
    negative_mean = negative_sum / (negative_count + 1e-6)
    both = (positive_count > 0) & (negative_count > 0)
    return torch.where(both, 0.5 * (positive_mean + negative_mean), positive_mean + negative_mean).mean()


def audit_negative_reachability(
    affinity_logits: torch.Tensor,
    support_masks: torch.Tensor,
    pair_batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    local_windows: bool = False,
) -> dict[str, object]:
    """Measure negative-query scores and isolated gradients on actual affinities."""
    with torch.enable_grad():
        logits = affinity_logits.detach().float().requires_grad_()
        symmetric = symmetrize_affinities(logits)
        result: dict[str, object] = {}
        for scale in WPRF_SCALES:
            scores = []
            legal_records = 0
            for index, mapping in enumerate(pair_batches):
                negative = mapping[f"neg_{scale}"]
                if len(negative):
                    support = support_masks[index].squeeze(0)
                    if local_windows:
                        with torch.no_grad():
                            legal = widest_path_for_pairs(
                                torch.ones_like(symmetric[index]), support, negative,
                                num_steps=scale, activation_checkpoint=False, upstream=True,
                                local_windows=True,
                            )
                        legal_records += int((legal > 0).sum())
                    scores.append(widest_path_for_pairs(
                        symmetric[index], support, negative,
                        num_steps=scale, activation_checkpoint=True, upstream=True,
                        local_windows=local_windows,
                    ))
            if not scores:
                continue
            combined = torch.cat(scores)
            loss = -torch.log(1 - combined.clamp(0, 1 - 1e-6) + 1e-6).mean()
            gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
            result[str(scale)] = {
                "negative_records": len(combined),
                "positive_scores": int((combined > 0).sum()),
                "derivative_eligible_scores": int(((combined > 0) & (combined < 1 - 1e-6)).sum()),
                "legal_path_records": legal_records if local_windows else int((combined > 0).sum()),
                "nonzero_query_gradient_fraction": float(((combined > 0) & (combined < 1 - 1e-6)).float().mean()),
                "isolated_negative_gradient_norm": float(gradient.norm()),
                "isolated_negative_gradient_max": float(gradient.abs().max()),
            }
    return result


def wprf_structural_loss(
    affinity_logits: torch.Tensor,
    support_masks: torch.Tensor,
    dense_component_ids: torch.Tensor,
    pair_batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    source_batch_size: int = 32,
    activation_checkpoint: bool = True,
    valid_masks: torch.Tensor | None = None,
    negative_support_masks: torch.Tensor | None = None,
    upstream: bool = False,
    local_windows: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    symmetric = symmetrize_affinities(affinity_logits)
    edge = (
        upstream_edge_affinity_loss(symmetric, dense_component_ids, valid_masks)
        if upstream else edge_affinity_loss(symmetric, dense_component_ids, valid_masks=valid_masks)
    )
    reach, count = reachability_loss(
        symmetric,
        support_masks,
        pair_batches,
        source_batch_size=source_batch_size,
        activation_checkpoint=activation_checkpoint,
        negative_support_masks=negative_support_masks,
        upstream=upstream,
        local_windows=local_windows,
    )
    return edge, reach, count


__all__ = [
    "WPRF_OFFSETS",
    "WPRF_SCALES",
    "WPRF_SOURCE_COMMIT",
    "edge_affinity_loss",
    "reachability_loss",
    "symmetrize_affinities",
    "widest_path_for_pairs",
    "wprf_structural_loss",
]
