from __future__ import annotations

import ctypes
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoseg.models.DCSwin import SwinTransformer, load_dcswin_base_pretrained
from geoseg.models.aquav3_torchgeo import adapt_first_conv_to_channels
from geoseg.pretrained import resolve_pretrained_model_path


@dataclass(frozen=True)
class ContextHybridSpec:
    name: str
    kind: str
    pretrained_key: str
    source_indices: tuple[int, ...]
    sha256: str


_SPECS = {
    "hrnet_swin_b_ocr_boundary_pbcl": ContextHybridSpec(
        "hrnet_swin_b_ocr_boundary_pbcl",
        "swin_b",
        "geoseg-stseg-base",
        (2, 1, 0, -1),
        "00585aef2972df877f53db302eebbb9b9162821a328f8200bf10f758a9d8ed78",
    ),
    "hrnet_swinv2_b_ocr_boundary_pbcl": ContextHybridSpec(
        "hrnet_swinv2_b_ocr_boundary_pbcl",
        "swin_v2_b",
        "torchgeo-swin-v2-b",
        (2, 1, 0, 6),
        "39c86721744331ca21654869fb268f763e23df6a59bc0b79137868a01af4e448",
    ),
    "hrnet_vit_b_ocr_boundary_pbcl": ContextHybridSpec(
        "hrnet_vit_b_ocr_boundary_pbcl",
        "vit_b",
        "torchgeo-vit-base-sentinel2-mae",
        (1, 2, 3, 7),
        "a3f1cd4f9bbff53a289298518ae8e0ade5214e9f80799c0d852aa1dca7e9bc60",
    ),
}

CONTEXT_HYBRID_SPECS: Mapping[str, ContextHybridSpec] = MappingProxyType(_SPECS)


def _preload_conda_libjpeg() -> None:
    path = Path(sys.prefix) / "lib" / "libjpeg.so.8"
    if path.exists():
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


class SwinBContext(nn.Module):
    out_channels = 1024

    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        self.backbone = SwinTransformer(
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            frozen_stages=-1,
            in_chans=4,
        )
        self.pretrained_report = None
        if pretrained:
            self.pretrained_report = load_dcswin_base_pretrained(
                self,
                resolve_pretrained_model_path("geoseg-stseg-base"),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)[-1]


class SwinV2BContext(nn.Module):
    out_channels = 1024

    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        _preload_conda_libjpeg()
        from torchgeo.models import swin_v2_b

        self.encoder = swin_v2_b(weights=None)
        if pretrained:
            self.encoder.features[0][0] = adapt_first_conv_to_channels(
                self.encoder.features[0][0],
                in_chans=9,
            )
            state = torch.load(
                resolve_pretrained_model_path("torchgeo-swin-v2-b"),
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            self.encoder.load_state_dict(state, strict=True)
        self.encoder.features[0][0] = adapt_first_conv_to_channels(
            self.encoder.features[0][0],
            in_chans=4,
            source_indices=(2, 1, 0, 6) if pretrained else None,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.encoder.features(inputs)
        return self.encoder.norm(features).permute(0, 3, 1, 2).contiguous()


def _interpolate_vit_position(pos_embed: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    prefix = pos_embed[:, :1]
    patches = pos_embed[:, 1:]
    source_size = int(math.sqrt(patches.shape[1]))
    if source_size * source_size != patches.shape[1]:
        raise ValueError(f"Unexpected ViT-B position embedding shape: {tuple(pos_embed.shape)}")
    patches = patches.transpose(1, 2).reshape(1, patches.shape[-1], source_size, source_size)
    patches = F.interpolate(patches, size=grid_size, mode="bicubic", align_corners=False)
    return torch.cat((prefix, patches.flatten(2).transpose(1, 2)), dim=1)


class ViTBContext(nn.Module):
    out_channels = 768

    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        _preload_conda_libjpeg()
        from torchgeo.models import vit_base_patch16_224

        self.encoder = vit_base_patch16_224(
            weights=None,
            num_classes=0,
            in_chans=13 if pretrained else 3,
        )
        if pretrained:
            state = torch.load(
                resolve_pretrained_model_path("torchgeo-vit-base-sentinel2-mae"),
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            self.encoder.load_state_dict(state, strict=True)
        self.encoder.patch_embed.proj = adapt_first_conv_to_channels(
            self.encoder.patch_embed.proj,
            in_chans=4,
            source_indices=(1, 2, 3, 7) if pretrained else None,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        patches = self.encoder.patch_embed.proj(inputs)
        grid_size = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)
        norm = getattr(self.encoder.patch_embed, "norm", None)
        if norm is not None:
            tokens = norm(tokens)
        cls_token = self.encoder.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = tokens + _interpolate_vit_position(
            self.encoder.pos_embed.to(dtype=tokens.dtype, device=tokens.device),
            grid_size,
        )
        tokens = self.encoder.pos_drop(tokens)
        tokens = self.encoder.patch_drop(tokens)
        tokens = self.encoder.norm_pre(tokens)
        tokens = self.encoder.blocks(tokens)
        tokens = self.encoder.norm(tokens)[:, 1:]
        return tokens.transpose(1, 2).reshape(tokens.shape[0], self.out_channels, *grid_size).contiguous()


def build_context_backbone(name: str, *, pretrained: bool = True) -> nn.Module:
    try:
        spec = CONTEXT_HYBRID_SPECS[str(name).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported HRNet context hybrid: {name!r}") from exc
    if spec.kind == "swin_b":
        return SwinBContext(pretrained=pretrained)
    if spec.kind == "swin_v2_b":
        return SwinV2BContext(pretrained=pretrained)
    if spec.kind == "vit_b":
        return ViTBContext(pretrained=pretrained)
    raise AssertionError(spec.kind)
