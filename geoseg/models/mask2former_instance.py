from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aquav3_external import replace_first_conv
from geoseg.pretrained import default_pretrained_root


MASK2FORMER_INSTANCE_REPO_ID = "facebook/mask2former-swin-tiny-coco-instance"
MASK2FORMER_INSTANCE_REVISION = "22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6"
MASK2FORMER_INSTANCE_WEIGHTS_SHA256 = (
    "3550e8b21381d082c4d12b06a4b7c5f0f4a530125eee4c5f264b1e5f43dfab80"
)
MASK2FORMER_INSTANCE_NUM_QUERIES = 100


def mask2former_instance_snapshot_path() -> str:
    path = (
        default_pretrained_root()
        / "huggingface/hub/models--facebook--mask2former-swin-tiny-coco-instance"
        / "snapshots"
        / MASK2FORMER_INSTANCE_REVISION
    )
    required = ("config.json", "model.safetensors", "preprocessor_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Pinned Mask2Former instance snapshot is incomplete at {path}: {missing}"
        )
    return str(path.resolve())


@torch.no_grad()
def semantic_union_logits(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    *,
    size: tuple[int, int],
) -> torch.Tensor:
    """Return a stable one-channel union without summing over queries."""

    resized_masks = F.interpolate(
        mask_logits,
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    foreground = torch.softmax(class_logits, dim=-1)[..., 0]
    contribution = foreground[:, :, None, None] * torch.sigmoid(resized_masks)
    probability = contribution.amax(dim=1, keepdim=True)
    return torch.logit(probability.clamp(1e-6, 1.0 - 1e-6))


@torch.no_grad()
def postprocess_mask2former_instances(
    outputs: Mapping[str, Any],
    *,
    size: tuple[int, int],
    score_threshold: float | None,
    mask_probability_threshold: float = 0.5,
    max_instances: int = MASK2FORMER_INSTANCE_NUM_QUERIES,
    valid_masks: torch.Tensor | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Convert query outputs into independent, possibly overlapping instances."""

    class_logits = outputs.get("pred_logits")
    mask_logits = outputs.get("pred_masks")
    if not isinstance(class_logits, torch.Tensor) or not isinstance(mask_logits, torch.Tensor):
        raise KeyError("Mask2Former instance outputs require pred_logits and pred_masks tensors")
    if class_logits.ndim != 3 or mask_logits.ndim != 4:
        raise ValueError(
            "Expected pred_logits [B,Q,C+1] and pred_masks [B,Q,H,W], got "
            f"{tuple(class_logits.shape)} and {tuple(mask_logits.shape)}"
        )
    if class_logits.shape[:2] != mask_logits.shape[:2]:
        raise ValueError("Mask2Former class and mask query dimensions do not match")

    resized_logits = F.interpolate(
        mask_logits,
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    mask_probabilities = torch.sigmoid(resized_logits)
    if valid_masks is not None:
        valid = torch.as_tensor(
            valid_masks,
            device=mask_probabilities.device,
            dtype=torch.bool,
        )
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.ndim != 4 or valid.shape[1] != 1:
            raise ValueError(
                "valid_masks must have shape [B,H,W] or [B,1,H,W], got "
                f"{tuple(valid_masks.shape)}"
            )
        if valid.shape[0] != mask_probabilities.shape[0]:
            raise ValueError("valid_masks batch dimension does not match predictions")
        if valid.shape[-2:] != size:
            valid = F.interpolate(valid.float(), size=size, mode="nearest").bool()
        mask_probabilities = mask_probabilities * valid
    binary_masks = mask_probabilities >= float(mask_probability_threshold)
    foreground_scores = torch.softmax(class_logits, dim=-1)[..., 0]
    mask_areas = binary_masks.flatten(2).sum(dim=-1)
    mask_quality = (
        (mask_probabilities * binary_masks).flatten(2).sum(dim=-1)
        / mask_areas.clamp_min(1)
    )
    scores = foreground_scores * mask_quality

    results: list[dict[str, torch.Tensor]] = []
    for batch_index in range(int(class_logits.shape[0])):
        order = torch.argsort(scores[batch_index], descending=True)
        order = order[: min(int(max_instances), int(order.numel()))]
        keep = mask_areas[batch_index, order] > 0
        if score_threshold is not None:
            keep = keep & (scores[batch_index, order] >= float(score_threshold))
        selected = order[keep]
        selected_masks = binary_masks[batch_index, selected]
        selected_scores = scores[batch_index, selected]
        results.append(
            {
                "masks": selected_masks,
                "scores": selected_scores,
                "labels": torch.zeros(
                    (int(selected.numel()),),
                    dtype=torch.long,
                    device=class_logits.device,
                ),
            }
        )
    return results


@torch.no_grad()
def union_instance_masks(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    *,
    size: tuple[int, int],
    device: torch.device | None = None,
) -> torch.Tensor:
    unions: list[torch.Tensor] = []
    for prediction in predictions:
        masks = prediction["masks"]
        if masks.ndim != 3 or tuple(masks.shape[-2:]) != tuple(size):
            raise ValueError(
                f"Instance masks must have shape [N,{size[0]},{size[1]}], got {tuple(masks.shape)}"
            )
        if int(masks.shape[0]) == 0:
            union = torch.zeros(size, dtype=torch.bool, device=masks.device)
        else:
            union = masks.any(dim=0)
        unions.append(union)
    if not unions:
        resolved_device = device if device is not None else torch.device("cpu")
        return torch.empty((0, 1, *size), dtype=torch.bool, device=resolved_device)
    return torch.stack(unions, dim=0).unsqueeze(1)


class Mask2FormerInstanceWrapper(nn.Module):
    """AquaDateset wrapper around the official COCO-instance Mask2Former loss."""

    def __init__(
        self,
        model: nn.Module,
        *,
        loading_info: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_queries = int(model.config.num_queries)
        self.pretrained_revision = MASK2FORMER_INSTANCE_REVISION
        self.pretrained_sha256 = MASK2FORMER_INSTANCE_WEIGHTS_SHA256
        self.loading_info = dict(loading_info or {})
        self.instance_prediction = True
        self.train()

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mask_labels: list[torch.Tensor] | None = None,
        class_labels: list[torch.Tensor] | None = None,
        pixel_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        has_targets = mask_labels is not None or class_labels is not None
        if has_targets and (mask_labels is None or class_labels is None):
            raise ValueError("mask_labels and class_labels must be provided together")
        model_kwargs: dict[str, Any] = {
            "pixel_values": inputs,
            "output_auxiliary_logits": bool(has_targets),
        }
        if pixel_mask is not None:
            model_kwargs["pixel_mask"] = pixel_mask
        raw = self.model(**model_kwargs)
        if raw.class_queries_logits is None or raw.masks_queries_logits is None:
            raise RuntimeError("Mask2Former did not return query predictions")
        result: dict[str, Any] = {
            "out": semantic_union_logits(
                raw.class_queries_logits,
                raw.masks_queries_logits,
                size=tuple(inputs.shape[-2:]),
            ),
            "pred_logits": raw.class_queries_logits,
            "pred_masks": raw.masks_queries_logits,
        }
        if has_targets:
            loss_dict = self.model.get_loss_dict(
                masks_queries_logits=raw.masks_queries_logits,
                class_queries_logits=raw.class_queries_logits,
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=raw.auxiliary_logits,
            )
            result["loss_dict"] = loss_dict
            result["loss"] = self.model.get_loss(loss_dict)
        return result


def expand_mask2former_queries(
    model: nn.Module,
    *,
    num_queries: int,
    seed: int,
) -> None:
    """Expand official query tables without changing the first 100 pretrained rows."""

    requested = int(num_queries)
    module = model.model.transformer_module
    current = int(module.queries_embedder.num_embeddings)
    if requested < current:
        raise ValueError(
            f"Refusing to truncate pretrained Mask2Former queries: {current} -> {requested}"
        )
    if requested == current:
        model.config.num_queries = requested
        return

    init_std = float(getattr(model.config, "init_std", 0.02))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for name in ("queries_embedder", "queries_features"):
        old = getattr(module, name)
        replacement = nn.Embedding(
            requested,
            int(old.embedding_dim),
            device=old.weight.device,
            dtype=old.weight.dtype,
        )
        with torch.no_grad():
            extra = torch.randn(
                (requested - current, int(old.embedding_dim)),
                generator=generator,
                dtype=torch.float32,
            ).mul_(init_std)
            replacement.weight[:current].copy_(old.weight)
            replacement.weight[current:].copy_(
                extra.to(device=old.weight.device, dtype=old.weight.dtype)
            )
        setattr(module, name, replacement)
    model.config.num_queries = requested


def _validate_loading_info(loading_info: Mapping[str, Any]) -> None:
    errors = list(loading_info.get("error_msgs", ()))
    missing = list(loading_info.get("missing_keys", ()))
    unexpected = list(loading_info.get("unexpected_keys", ()))
    mismatched = list(loading_info.get("mismatched_keys", ()))
    allowed_missing = {
        "model.pixel_level_module.encoder.swin.layernorm.weight",
        "model.pixel_level_module.encoder.swin.layernorm.bias",
    }
    invalid_missing = [key for key in missing if key not in allowed_missing]
    invalid_unexpected = [
        key for key in unexpected if not key.endswith("attention.self.relative_position_index")
    ]
    if errors or invalid_missing or invalid_unexpected:
        raise RuntimeError(
            "COCO-instance Mask2Former loading was not complete: "
            f"errors={errors}, missing={invalid_missing}, unexpected={invalid_unexpected}"
        )
    allowed_prefixes = ("class_predictor.", "criterion.empty_weight")
    invalid_mismatches: list[str] = []
    for entry in mismatched:
        key = entry[0] if isinstance(entry, (tuple, list)) else str(entry)
        if not str(key).startswith(allowed_prefixes):
            invalid_mismatches.append(str(key))
    if invalid_mismatches:
        raise RuntimeError(
            "Only the one-class Mask2Former class predictor may be reinitialized; "
            f"unexpected mismatches: {invalid_mismatches}"
        )


def build_mask2former_swin_t_instance(
    *,
    in_chans: int = 4,
    num_classes: int = 1,
    pretrained: bool = True,
    bgrnir_pretrained_stem: bool = True,
    num_queries: int = MASK2FORMER_INSTANCE_NUM_QUERIES,
    query_seed: int = 0,
) -> Mask2FormerInstanceWrapper:
    if not pretrained:
        raise RuntimeError("The formal instance baseline requires official COCO-instance pretraining")
    if int(num_classes) != 1:
        raise ValueError("AquaDateset Mask2Former instance baseline is one-class only")

    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    snapshot = mask2former_instance_snapshot_path()
    config = Mask2FormerConfig.from_pretrained(
        snapshot,
        local_files_only=True,
    )
    config.num_labels = 1
    config.id2label = {0: "foreground"}
    config.label2id = {"foreground": 0}

    model, loading_info = Mask2FormerForUniversalSegmentation.from_pretrained(
        snapshot,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=True,
        use_safetensors=True,
        output_loading_info=True,
    )
    _validate_loading_info(loading_info)
    # Transformers 5.x added a trainable final Swin LayerNorm that is absent from
    # this official 2023 checkpoint. Identity restores the checkpoint architecture
    # and avoids silently introducing randomly initialized parameters.
    model.model.pixel_level_module.encoder.swin.layernorm = nn.Identity()
    source_indices = (2, 1, 0, None) if bgrnir_pretrained_stem and in_chans == 4 else None
    if not replace_first_conv(model, in_chans=in_chans, source_indices=source_indices):
        raise RuntimeError("Could not locate the Mask2Former-Swin-T input stem")
    model.config.backbone_config.num_channels = int(in_chans)
    expand_mask2former_queries(
        model,
        num_queries=int(num_queries),
        seed=int(query_seed),
    )
    model.config.class_weight = 2.0
    model.config.mask_weight = 5.0
    model.config.dice_weight = 5.0
    model.config.no_object_weight = 0.1
    return Mask2FormerInstanceWrapper(model, loading_info=loading_info)


__all__ = [
    "MASK2FORMER_INSTANCE_NUM_QUERIES",
    "MASK2FORMER_INSTANCE_REPO_ID",
    "MASK2FORMER_INSTANCE_REVISION",
    "MASK2FORMER_INSTANCE_WEIGHTS_SHA256",
    "Mask2FormerInstanceWrapper",
    "build_mask2former_swin_t_instance",
    "expand_mask2former_queries",
    "mask2former_instance_snapshot_path",
    "postprocess_mask2former_instances",
    "semantic_union_logits",
    "union_instance_masks",
]
