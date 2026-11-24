"""VMamba-T V2 + FPN adapted from RoadSeg's plain vmamba_fpn baseline.

VMamba source revision: MzeroMiko/VMamba 2ed52ead062a51a64521ed3871d52914bf532876.
"""

from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from geoseg.pretrained import default_pretrained_root

from .aquav3_external import expand_first_conv_to_channels
from .ssepnet.multiscale import ConvGNReLU, StandardPyramidDecoder


class VMambaTBackbone(nn.Module):
    stage_channels = (96, 192, 384, 768)

    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        root = default_pretrained_root() / "roadscale"
        source = (root / "vmamba_code").resolve()
        expected = source / "vmamba_official/vmamba.py"
        if not expected.is_file():
            raise FileNotFoundError(expected)
        try:
            import selective_scan_cuda_oflex  # noqa: F401
        except ImportError as error:
            raise RuntimeError("VMamba-T requires the SM120 selective-scan CUDA extension") from error
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from vmamba_official import vmamba

        if Path(vmamba.__file__).resolve() != expected:
            raise RuntimeError(f"Expected VMamba source at {expected}")
        model = vmamba.VSSM(
            depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2,
            ssm_d_state=1, ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False,
            forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3",
            patchembed_version="v2", norm_layer="ln2d",
        )
        if pretrained:
            weights = torch.load(
                root / "vmamba_t_imagenet_official.pth", map_location="cpu", weights_only=True,
            )
            model.load_state_dict(weights["model"], strict=True)
        del model.classifier
        model.patch_embed[0] = expand_first_conv_to_channels(
            model.patch_embed[0], in_chans=4, source_indices=(2, 1, 0, None),
        )
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.model.patch_embed(image)
        features = []
        for stage in self.model.layers:
            if self.training and torch.is_grad_enabled():
                x = checkpoint(stage.blocks, x, use_reentrant=False)
            else:
                x = stage.blocks(x)
            features.append(x if self.model.channel_first else x.permute(0, 3, 1, 2).contiguous())
            x = stage.downsample(x)
        return tuple(features)


class VMambaFPN(nn.Module):
    fpn_channels = 128

    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        self.backbone = VMambaTBackbone(pretrained=pretrained)
        self.projections = nn.ModuleList(
            ConvGNReLU(channels, self.fpn_channels, 1) for channels in self.backbone.stage_channels
        )
        self.decoder = StandardPyramidDecoder(self.fpn_channels, dropout=0.05)
        self.main_head = nn.Conv2d(self.fpn_channels, 1, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        features = tuple(
            project(feature) for project, feature in zip(self.projections, self.backbone(image), strict=True)
        )
        fused = self.decoder(features)[-1]
        logits = F.interpolate(self.main_head(fused), size=image.shape[-2:], mode="bilinear", align_corners=False)
        if hasattr(self, "boundary_head"):
            return {"out": logits, "edge": self.boundary_head(fused)}
        return logits
