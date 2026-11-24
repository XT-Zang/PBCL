"""Swin-Unet adapted for AquaV3 four-band semantic segmentation.

The network follows the symmetric pure-Transformer architecture described in
"Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation" while
reusing this GPL project's existing dynamic-resolution Swin building blocks.
The decoder and checkpoint adapter are implemented locally because the
authors' reference repository does not publish a code license.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

from .DCSwin import BasicLayer, PatchEmbed, PatchMerging


SWIN_TINY_CHECKPOINT_NAME = "swin_tiny_patch4_window7_224.pth"
SWIN_TINY_CHECKPOINT_URL = (
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/"
    "swin_tiny_patch4_window7_224.pth"
)


class SwinUnetPretrainedError(RuntimeError):
    """Raised when the required official Swin-T checkpoint is unavailable."""


@dataclass(frozen=True)
class SwinUnetLoadReport:
    loaded_keys: int
    required_keys: int
    covered_required_keys: int
    coverage: float
    group_coverage: tuple[tuple[str, int, int], ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes"}


def expand_swin_patch_embed_weight(
    weight: torch.Tensor, in_chans: int = 4
) -> torch.Tensor:
    """Map an RGB patch projection to AquaV3 ``[B,G,R,NIR]`` without response-scale drift."""

    if weight.ndim != 4:
        raise ValueError(
            "patch embedding weight must have shape [out_channels, in_channels, kh, kw]"
        )
    if in_chans <= 0:
        raise ValueError("in_chans must be positive")
    source_channels = int(weight.shape[1])
    if source_channels == in_chans:
        return weight.clone()
    if source_channels != 3:
        raise ValueError(
            f"expected an RGB patch embedding with 3 channels, got {source_channels}"
        )

    mean_channel = weight.mean(dim=1, keepdim=True)
    if in_chans > 3:
        visible_bgr = weight[:, [2, 1, 0], :, :]
        extra = mean_channel.repeat(1, in_chans - 3, 1, 1)
        expanded = torch.cat([visible_bgr, extra], dim=1)
    else:
        expanded = mean_channel.repeat(1, in_chans, 1, 1)
    return expanded * (3.0 / float(in_chans))


class PatchExpand(nn.Module):
    """Double token-map resolution and halve its channel width."""

    def __init__(self, dim: int, norm_layer: type[nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("PatchExpand requires an even channel dimension")
        self.dim = int(dim)
        self.expand = nn.Linear(self.dim, 2 * self.dim, bias=False)
        self.norm = norm_layer(self.dim // 2)

    def forward(
        self, x: torch.Tensor, height: int, width: int
    ) -> tuple[torch.Tensor, int, int]:
        batch, length, channels = x.shape
        if length != height * width or channels != self.dim:
            raise ValueError(
                "PatchExpand input tokens do not match the supplied spatial shape"
            )
        x = self.expand(x).view(batch, height, width, 2, 2, channels // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch, height * 2, width * 2, channels // 2)
        x = self.norm(x).view(batch, height * width * 4, channels // 2)
        return x, height * 2, width * 2


class FinalPatchExpand(nn.Module):
    """Restore patch tokens to pixel resolution while retaining channel width."""

    def __init__(
        self, dim: int, scale: int = 4, norm_layer: type[nn.Module] = nn.LayerNorm
    ) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.dim = int(dim)
        self.scale = int(scale)
        self.expand = nn.Linear(
            self.dim, self.scale * self.scale * self.dim, bias=False
        )
        self.norm = norm_layer(self.dim)

    def forward(
        self, x: torch.Tensor, height: int, width: int
    ) -> tuple[torch.Tensor, int, int]:
        batch, length, channels = x.shape
        if length != height * width or channels != self.dim:
            raise ValueError(
                "FinalPatchExpand input tokens do not match the supplied spatial shape"
            )
        scale = self.scale
        x = self.expand(x).view(batch, height, width, scale, scale, channels)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch, height * scale, width * scale, channels)
        x = self.norm(x).view(batch, height * width * scale * scale, channels)
        return x, height * scale, width * scale


class SwinDecoderLayer(BasicLayer):
    """A Swin stage followed by the patch expansion used by the symmetric decoder."""

    def __init__(self, *, dim: int, upsample: bool, **kwargs) -> None:
        super().__init__(dim=dim, downsample=None, **kwargs)
        self.upsample = PatchExpand(dim) if upsample else None

    def forward(
        self, x: torch.Tensor, height: int, width: int
    ) -> tuple[torch.Tensor, int, int]:
        x, height, width, _, _, _ = super().forward(x, height, width)
        if self.upsample is not None:
            x, height, width = self.upsample(x, height, width)
        return x, height, width


class SwinUnet(nn.Module):
    """Symmetric four-stage Swin encoder-decoder with concat skip connections."""

    def __init__(
        self,
        in_chans: int = 4,
        num_classes: int = 1,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(num_heads) != 4:
            raise ValueError("SwinUnet requires exactly four encoder stages")
        if patch_size != 4:
            raise ValueError(
                "the AquaV3 Swin-Unet implementation requires patch_size=4"
            )
        if in_chans <= 0 or num_classes <= 0:
            raise ValueError("in_chans and num_classes must be positive")

        self.in_chans = int(in_chans)
        self.num_classes = int(num_classes)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.depths = tuple(int(depth) for depth in depths)
        self.num_heads = tuple(int(heads) for heads in num_heads)
        self.window_size = int(window_size)
        self.num_layers = 4
        self.total_stride = self.patch_size * 2 ** (self.num_layers - 1)

        norm_layer = nn.LayerNorm
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, sum(self.depths)).tolist()
        self.layers = nn.ModuleList()
        for stage in range(self.num_layers):
            start = sum(self.depths[:stage])
            stop = start + self.depths[stage]
            self.layers.append(
                BasicLayer(
                    dim=self.embed_dim * 2**stage,
                    depth=self.depths[stage],
                    num_heads=self.num_heads[stage],
                    window_size=self.window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[start:stop],
                    norm_layer=norm_layer,
                    downsample=PatchMerging if stage < self.num_layers - 1 else None,
                    use_checkpoint=use_checkpoint,
                )
            )

        self.norm = norm_layer(self.embed_dim * 8)
        self.layers_up = nn.ModuleList([PatchExpand(self.embed_dim * 8)])
        self.concat_back_dim = nn.ModuleList([nn.Identity()])
        for decoder_index, encoder_stage in enumerate((2, 1, 0), start=1):
            dim = self.embed_dim * 2**encoder_stage
            start = sum(self.depths[:encoder_stage])
            stop = start + self.depths[encoder_stage]
            self.concat_back_dim.append(nn.Linear(2 * dim, dim))
            self.layers_up.append(
                SwinDecoderLayer(
                    dim=dim,
                    depth=self.depths[encoder_stage],
                    num_heads=self.num_heads[encoder_stage],
                    window_size=self.window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=list(reversed(dpr[start:stop])),
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    upsample=decoder_index < 3,
                )
            )

        self.norm_up = norm_layer(self.embed_dim)
        self.final_up = FinalPatchExpand(
            self.embed_dim, scale=self.patch_size, norm_layer=norm_layer
        )
        self.output = nn.Conv2d(
            self.embed_dim, self.num_classes, kernel_size=1, bias=True
        )
        self.pretrained_report: SwinUnetLoadReport | None = None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, int, int, list[tuple[torch.Tensor, int, int]]]:
        x = self.patch_embed(x)
        height, width = x.shape[-2:]
        x = self.pos_drop(x.flatten(2).transpose(1, 2))
        skips: list[tuple[torch.Tensor, int, int]] = []
        for layer in self.layers:
            skips.append((x, height, width))
            _, _, _, x, height, width = layer(x, height, width)
        return self.norm(x), height, width, skips

    def _decode(
        self,
        x: torch.Tensor,
        height: int,
        width: int,
        skips: list[tuple[torch.Tensor, int, int]],
    ) -> tuple[torch.Tensor, int, int]:
        x, height, width = self.layers_up[0](x, height, width)
        for decoder_index in range(1, self.num_layers):
            skip, skip_height, skip_width = skips[self.num_layers - 1 - decoder_index]
            if (height, width) != (skip_height, skip_width):
                raise RuntimeError(
                    "Swin-Unet decoder and encoder skip resolutions do not match"
                )
            x = self.concat_back_dim[decoder_index](torch.cat([x, skip], dim=-1))
            x, height, width = self.layers_up[decoder_index](x, height, width)
        return self.norm_up(x), height, width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("SwinUnet expects an NCHW tensor")
        if x.shape[1] != self.in_chans:
            raise ValueError(
                f"SwinUnet expects {self.in_chans} input channels, got {x.shape[1]}"
            )
        original_height, original_width = x.shape[-2:]
        pad_height = (
            self.total_stride - original_height % self.total_stride
        ) % self.total_stride
        pad_width = (
            self.total_stride - original_width % self.total_stride
        ) % self.total_stride
        if pad_height or pad_width:
            x = F.pad(x, (0, pad_width, 0, pad_height))

        x, height, width, skips = self._encode(x)
        x, height, width = self._decode(x, height, width, skips)
        x, height, width = self.final_up(x, height, width)
        batch = x.shape[0]
        x = (
            x.view(batch, height, width, self.embed_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        logits = self.output(x)
        return logits[:, :, :original_height, :original_width].contiguous()


def _checkpoint_state_dict(checkpoint: Mapping[str, object]) -> dict[str, torch.Tensor]:
    candidate = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(candidate, Mapping):
        raise ValueError(
            "Swin-T checkpoint must contain a mapping under 'model' or 'state_dict'"
        )
    state: dict[str, torch.Tensor] = {}
    for raw_key, value in candidate.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(raw_key)
        for prefix in ("module.", "swin_unet."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        state[key] = value
    return state


def adapt_swinunet_checkpoint(
    checkpoint: Mapping[str, object], model: SwinUnet
) -> dict[str, torch.Tensor]:
    """Adapt official Swin-T encoder weights to four bands and the symmetric decoder."""

    source = _checkpoint_state_dict(checkpoint)
    patch_key = "patch_embed.proj.weight"
    if patch_key in source and source[patch_key].shape[1] != model.in_chans:
        source[patch_key] = expand_swin_patch_embed_weight(
            source[patch_key], model.in_chans
        )

    model_state = model.state_dict()
    candidates = dict(source)
    for key, value in source.items():
        timm_downsample = re.match(r"^layers\.(\d+)\.downsample(\..+)$", key)
        if timm_downsample is not None and int(timm_downsample.group(1)) > 0:
            previous_stage = int(timm_downsample.group(1)) - 1
            previous_key = (
                f"layers.{previous_stage}.downsample{timm_downsample.group(2)}"
            )
            if (
                previous_key in model_state
                and model_state[previous_key].shape == value.shape
            ):
                candidates[previous_key] = value
        match = re.match(r"^layers\.(\d+)(\..+)$", key)
        if match is None:
            continue
        encoder_stage = int(match.group(1))
        decoder_stage = model.num_layers - 1 - encoder_stage
        candidates[f"layers_up.{decoder_stage}{match.group(2)}"] = value

    return {
        key: value
        for key, value in candidates.items()
        if key in model_state and model_state[key].shape == value.shape
    }


def _pretrained_parameter_groups(model: SwinUnet) -> dict[str, set[str]]:
    parameter_keys = set(dict(model.named_parameters()))
    return {
        "patch_embed": {
            key for key in parameter_keys if key.startswith("patch_embed.")
        },
        "encoder_blocks": {
            key
            for key in parameter_keys
            if key.startswith("layers.") and ".blocks." in key
        },
        "encoder_downsample": {
            key
            for key in parameter_keys
            if key.startswith("layers.") and ".downsample." in key
        },
        "encoder_norm": {key for key in parameter_keys if key.startswith("norm.")},
        "decoder_blocks": {
            key
            for key in parameter_keys
            if key.startswith("layers_up.") and ".blocks." in key
        },
    }


def _validate_pretrained_coverage(
    adapted: Mapping[str, torch.Tensor], model: SwinUnet
) -> tuple[int, int, float, tuple[tuple[str, int, int], ...]]:
    groups = _pretrained_parameter_groups(model)
    adapted_keys = set(adapted)
    group_coverage = tuple(
        (name, len(required & adapted_keys), len(required))
        for name, required in groups.items()
    )
    missing = sorted(
        key
        for required in groups.values()
        for key in required
        if key not in adapted_keys
    )
    required_count = sum(len(required) for required in groups.values())
    covered_count = required_count - len(missing)
    coverage = covered_count / required_count if required_count else 0.0
    if missing:
        group_summary = ", ".join(
            f"{name}={covered}/{required}" for name, covered, required in group_coverage
        )
        raise SwinUnetPretrainedError(
            "official Swin-T checkpoint is incomplete for Swin-Unet "
            f"({group_summary}); missing keys include: {missing[:8]}"
        )
    return required_count, covered_count, coverage, group_coverage


def default_swinunet_checkpoint_path() -> Path:
    configured = os.environ.get("AQUAV3_SWINUNET_CKPT")
    if configured:
        return Path(configured).expanduser()
    return Path(torch.hub.get_dir()) / "checkpoints" / SWIN_TINY_CHECKPOINT_NAME


def ensure_swinunet_checkpoint(allow_download: bool | None = None) -> Path:
    checkpoint_path = default_swinunet_checkpoint_path()
    if checkpoint_path.is_file():
        return checkpoint_path
    if allow_download is None:
        allow_download = _env_flag("AQUAV3_ALLOW_PRETRAIN_DOWNLOAD")
    if allow_download:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.hub.download_url_to_file(
                SWIN_TINY_CHECKPOINT_URL, str(checkpoint_path), progress=True
            )
        except Exception as exc:
            raise SwinUnetPretrainedError(
                f"could not download official Swin-T checkpoint from "
                f"{SWIN_TINY_CHECKPOINT_URL}: {exc}"
            ) from None
        return checkpoint_path
    raise SwinUnetPretrainedError(
        f"Swin-Unet requires the official Swin-T checkpoint at '{checkpoint_path}'. "
        "Set AQUAV3_SWINUNET_CKPT to a cached file or AQUAV3_ALLOW_PRETRAIN_DOWNLOAD=1 "
        f"to download {SWIN_TINY_CHECKPOINT_URL}."
    )


def load_swinunet_pretrained(
    model: SwinUnet, checkpoint_path: str | Path
) -> SwinUnetLoadReport:
    try:
        try:
            checkpoint = torch.load(
                Path(checkpoint_path), map_location="cpu", weights_only=True
            )
        except TypeError:  # PyTorch < 2.0 compatibility
            checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    except Exception as exc:
        raise SwinUnetPretrainedError(
            f"could not read Swin-T checkpoint '{checkpoint_path}': {exc}"
        ) from None
    if not isinstance(checkpoint, Mapping):
        raise SwinUnetPretrainedError(
            "official Swin-T checkpoint did not contain a state dictionary"
        )
    try:
        adapted = adapt_swinunet_checkpoint(checkpoint, model)
    except (ValueError, KeyError, IndexError, RuntimeError) as exc:
        raise SwinUnetPretrainedError(
            f"invalid Swin-T checkpoint '{checkpoint_path}': {exc}"
        ) from None
    if not adapted:
        raise SwinUnetPretrainedError(
            "official Swin-T checkpoint contained no compatible Swin-Unet weights"
        )
    required_count, covered_count, coverage, group_coverage = (
        _validate_pretrained_coverage(adapted, model)
    )
    incompatible = model.load_state_dict(adapted, strict=False)
    report = SwinUnetLoadReport(
        loaded_keys=len(adapted),
        required_keys=required_count,
        covered_required_keys=covered_count,
        coverage=coverage,
        group_coverage=group_coverage,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )
    model.pretrained_report = report
    return report


def build_swinunet(
    in_chans: int = 4,
    num_classes: int = 1,
    pretrained: bool = True,
    **kwargs,
) -> SwinUnet:
    model = SwinUnet(in_chans=in_chans, num_classes=num_classes, **kwargs)
    if pretrained:
        checkpoint_path = ensure_swinunet_checkpoint()
        load_swinunet_pretrained(model, checkpoint_path)
    return model
