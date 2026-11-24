from __future__ import annotations

import os
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class HFPretrainedCacheError(RuntimeError):
    pass


def _hf_cache_error_message(model_label: str, repo_id: str) -> str:
    return (
        f"{model_label} requires Hugging Face checkpoint '{repo_id}', but it is not available in the local cache. "
        "Pre-cache the checkpoint, or rerun with AQUAV3_ALLOW_HF_DOWNLOAD=1."
    )


def _load_hf_pretrained(loader, model_label: str, repo_id: str, **kwargs):
    try:
        return loader.from_pretrained(repo_id, **kwargs)
    except OSError:
        raise HFPretrainedCacheError(_hf_cache_error_message(model_label, repo_id)) from None


def expand_first_conv_to_channels(
    conv: nn.Conv2d,
    in_chans: int = 4,
    source_indices: tuple[int | None, ...] | list[int | None] | None = None,
) -> nn.Conv2d:
    if in_chans <= 0:
        raise ValueError("in_chans must be positive")
    if conv.in_channels == in_chans and source_indices is None:
        return conv
    if source_indices is not None and len(source_indices) != in_chans:
        raise ValueError("source_indices length must match in_chans")
    expanded = nn.Conv2d(
        in_channels=in_chans,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups if conv.groups == 1 else 1,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    with torch.no_grad():
        expanded.weight.zero_()
        mean_channel = conv.weight.mean(dim=1, keepdim=True)
        if source_indices is None:
            copied = min(conv.in_channels, in_chans)
            expanded.weight[:, :copied] = conv.weight[:, :copied]
            if in_chans > conv.in_channels:
                expanded.weight[:, conv.in_channels :] = mean_channel.repeat(
                    1, in_chans - conv.in_channels, 1, 1
                )
        else:
            for destination, source in enumerate(source_indices):
                if source is None:
                    expanded.weight[:, destination : destination + 1] = mean_channel
                elif source < 0 or source >= conv.in_channels:
                    raise ValueError(
                        f"source_indices contains {source}, but convolution has {conv.in_channels} channels"
                    )
                else:
                    expanded.weight[:, destination : destination + 1] = conv.weight[
                        :, source : source + 1
                    ]
        if expanded.bias is not None and conv.bias is not None:
            expanded.bias.copy_(conv.bias)
    return expanded


def replace_first_conv(
    module: nn.Module,
    in_chans: int = 4,
    source_indices: tuple[int | None, ...] | list[int | None] | None = None,
) -> bool:
    for _, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            adapted = expand_first_conv_to_channels(
                child,
                in_chans=in_chans,
                source_indices=source_indices,
            )
            child.in_channels = adapted.in_channels
            child.groups = adapted.groups
            child.weight = nn.Parameter(
                adapted.weight.detach().clone(),
                requires_grad=child.weight.requires_grad,
            )
            if child.bias is not None and adapted.bias is not None:
                child.bias.data.copy_(adapted.bias.data)
            return True
        if replace_first_conv(child, in_chans=in_chans, source_indices=source_indices):
            return True
    return False


class HuggingFaceSegmentationWrapper(nn.Module):
    def __init__(self, model: nn.Module, output_kind: str = "semantic"):
        super().__init__()
        self.model = model
        self.output_kind = output_kind
        self.train()

    def add_comparison_affinity(self, method: str) -> None:
        self.comparison_affinity = method
        self.training_only_affinity_head = True
        channels = self.model.decode_head.classifier.in_channels
        self.affinity_head = nn.Conv2d(
            channels, 8 if method == "wprf" else 2, kernel_size=3,
            padding=1, stride=4 if method == "wprf" else 1,
        )

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        result = self.forward(x, return_affinity=False)
        return result["out"] if isinstance(result, dict) else result

    def _wprf_affinity(self, features: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        stride = 4
        height, width = features.shape[-2:]
        if size != (height * stride, width * stride):
            raise ValueError("WPRF requires the SegFormer stride-4 grid")
        stripes = []
        for start in range(0, height, 12):
            stop = min(start + 12, height)
            halo_start = max(start - 1, 0)
            # Every output is centered at 4*y; its three sampled rows use
            # native rows y-1 and y with align_corners=False.
            def project(stripe, trim=start - halo_start):
                upsampled = F.interpolate(stripe, scale_factor=4, mode="bilinear", align_corners=False)
                return self.affinity_head(upsampled)[:, :, trim:]
            stripe = features[:, :, halo_start:stop]
            value = checkpoint(project, stripe, use_reentrant=False) if torch.is_grad_enabled() else project(stripe)
            stripes.append(value)
        return torch.cat(stripes, dim=2)

    def forward(self, x: torch.Tensor, *, return_affinity: bool | None = None) -> torch.Tensor | dict[str, torch.Tensor]:
        if return_affinity is None:
            return_affinity = self.training
        features_before_classifier = []
        hook = None
        if return_affinity and hasattr(self, "affinity_head"):
            hook = self.model.decode_head.classifier.register_forward_pre_hook(
                lambda _module, args: features_before_classifier.append(args[0])
            )
        needs_encoder_feature = hasattr(self, "boundary_head") and self.output_kind != "mask2former"
        try:
            outputs = self.model(pixel_values=x, **({"output_hidden_states": True} if needs_encoder_feature else {}))
        finally:
            if hook is not None:
                hook.remove()
        if self.output_kind == "mask2former":
            class_logits = outputs.class_queries_logits
            mask_logits = outputs.masks_queries_logits
            foreground_logits = class_logits[..., 0]
            if class_logits.shape[-1] > 1:
                foreground_logits = foreground_logits - class_logits[..., -1]
            logits = (mask_logits + foreground_logits[:, :, None, None]).amax(dim=1, keepdim=True)
        else:
            logits = outputs.logits
            if logits.ndim == 4 and logits.shape[1] > 1:
                logits = logits[:, :1]
        if hasattr(self, "boundary_head"):
            features = (
                outputs.pixel_decoder_last_hidden_state
                if self.output_kind == "mask2former"
                else outputs.hidden_states[0]
            )
            result = {"out": logits, "edge": self.boundary_head(features)}
            if features_before_classifier:
                fused = features_before_classifier[0]
                if self.comparison_affinity == "wprf":
                    result["affinity"] = self._wprf_affinity(fused, x.shape[-2:])
                else:
                    result["affinity"] = self.affinity_head(fused)
            return result
        return logits


@dataclass(frozen=True)
class ExternalBuildSpec:
    key: str
    family: str
    architecture: str
    backbone: str
    pretrained_name: str = ""


def _hf_local_files_only() -> bool:
    return os.environ.get("AQUAV3_ALLOW_HF_DOWNLOAD", "0").strip().lower() not in {"1", "true", "yes"}


def _pretrained_enabled(default: bool) -> bool:
    if os.environ.get("AQUAV3_COMPARE_RANDOM_INIT", "0").strip().lower() in {"1", "true", "yes"}:
        return False
    return default


def build_smp_binary_model(
    architecture: str,
    encoder_name: str,
    in_chans: int = 4,
    num_classes: int = 1,
    pretrained: bool = True,
    bgrnir_pretrained_stem: bool = False,
) -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except Exception as exc:  # pragma: no cover - dependency is present in the target env
        raise RuntimeError("segmentation_models_pytorch is required for SMP baselines") from exc
    cls = getattr(smp, architecture)
    encoder_weights = "imagenet" if _pretrained_enabled(pretrained) else None
    build_in_chans = 3 if _pretrained_enabled(pretrained) and in_chans != 3 else in_chans
    model = cls(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=build_in_chans,
        classes=num_classes,
        activation=None,
    )
    if build_in_chans != in_chans and not replace_first_conv(
        model.encoder,
        in_chans=in_chans,
        source_indices=(2, 1, 0, None) if bgrnir_pretrained_stem and in_chans == 4 else None,
    ):
        raise RuntimeError(f"Could not adapt {encoder_name} input stem to {in_chans} channels")
    return model


def build_segformer_b2(
    in_chans: int = 4,
    num_classes: int = 1,
    pretrained: bool = True,
    bgrnir_pretrained_stem: bool = False,
) -> nn.Module:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
    if _pretrained_enabled(pretrained):
        model = _load_hf_pretrained(
            SegformerForSemanticSegmentation,
            "SegFormer-B2",
            model_name,
            num_labels=num_classes,
            num_channels=3,
            ignore_mismatched_sizes=True,
            local_files_only=_hf_local_files_only(),
        )
        if in_chans != 3 and not replace_first_conv(
            model.segformer,
            in_chans=in_chans,
            source_indices=(2, 1, 0, None) if bgrnir_pretrained_stem and in_chans == 4 else None,
        ):
            raise RuntimeError("Could not locate SegFormer input stem for channel adaptation")
        model.config.num_channels = in_chans
    else:
        config = SegformerConfig(
            num_channels=in_chans,
            num_labels=num_classes,
            depths=[3, 4, 6, 3],
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=768,
        )
        model = SegformerForSemanticSegmentation(config)
    return HuggingFaceSegmentationWrapper(model)


def build_upernet_swin_t(
    in_chans: int = 4,
    num_classes: int = 1,
    pretrained: bool = True,
    bgrnir_pretrained_stem: bool = False,
) -> nn.Module:
    from transformers import UperNetForSemanticSegmentation

    if not _pretrained_enabled(pretrained):
        raise RuntimeError("UPerNet-Swin-T random-init construction is not supported; pre-cache a HF checkpoint or set another baseline.")
    model = _load_hf_pretrained(
        UperNetForSemanticSegmentation,
        "UPerNet-Swin-T",
        "openmmlab/upernet-swin-tiny",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        local_files_only=_hf_local_files_only(),
    )
    if not replace_first_conv(
        model,
        in_chans=in_chans,
        source_indices=(2, 1, 0, None) if bgrnir_pretrained_stem and in_chans == 4 else None,
    ):
        raise RuntimeError("Could not locate the first convolution in UPerNet-Swin-T for 4-channel adaptation")
    return HuggingFaceSegmentationWrapper(model)


def build_mask2former_swin_t(
    in_chans: int = 4,
    num_classes: int = 1,
    pretrained: bool = True,
    bgrnir_pretrained_stem: bool = False,
) -> nn.Module:
    from transformers import Mask2FormerForUniversalSegmentation

    if not _pretrained_enabled(pretrained):
        raise RuntimeError("Mask2Former-Swin-T random-init construction is not supported; pre-cache a HF checkpoint or set another baseline.")
    model = _load_hf_pretrained(
        Mask2FormerForUniversalSegmentation,
        "Mask2Former-Swin-T",
        "facebook/mask2former-swin-tiny-ade-semantic",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        local_files_only=_hf_local_files_only(),
    )
    if not replace_first_conv(
        model,
        in_chans=in_chans,
        source_indices=(2, 1, 0, None) if bgrnir_pretrained_stem and in_chans == 4 else None,
    ):
        raise RuntimeError("Could not locate the first convolution in Mask2Former-Swin-T for 4-channel adaptation")
    return HuggingFaceSegmentationWrapper(model, output_kind="mask2former")


def build_swinunet(in_chans: int = 4, num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    from .SwinUnet import build_swinunet as build_model

    skip_base_pretrain = os.environ.get("AQUAV3_SWINUNET_SKIP_PRETRAIN", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    return build_model(
        in_chans=in_chans,
        num_classes=num_classes,
        pretrained=_pretrained_enabled(pretrained) and not skip_base_pretrain,
    )


def build_compare_model(
    key: str,
    in_chans: int = 4,
    num_classes: int = 1,
    pretrained: bool = True,
    *,
    bgrnir_pretrained_stem: bool = False,
) -> nn.Module:
    normalized = key.strip().lower().replace("-", "_")
    if normalized == "vmamba_fpn":
        from .vmamba_fpn import VMambaFPN

        if (in_chans, num_classes) != (4, 1):
            raise ValueError("VMamba-FPN expects four input bands and one foreground logit")
        return VMambaFPN(pretrained=pretrained)
    if normalized == "dcswin_base":
        from .DCSwin import build_dcswin_base_aquadataset

        return build_dcswin_base_aquadataset(
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
        )
    if normalized == "unetplusplus":
        return build_smp_binary_model(
            "UnetPlusPlus",
            "efficientnet-b4",
            in_chans,
            num_classes,
            pretrained=pretrained,
            bgrnir_pretrained_stem=bgrnir_pretrained_stem,
        )
    if normalized == "deeplabv3plus":
        return build_smp_binary_model(
            "DeepLabV3Plus",
            "resnet50",
            in_chans,
            num_classes,
            pretrained=pretrained,
            bgrnir_pretrained_stem=bgrnir_pretrained_stem,
        )
    if normalized == "swinunet":
        return build_swinunet(in_chans=in_chans, num_classes=num_classes, pretrained=pretrained)
    if normalized == "segformer_b2":
        return build_segformer_b2(
            in_chans=in_chans,
            num_classes=num_classes,
            pretrained=pretrained,
            bgrnir_pretrained_stem=bgrnir_pretrained_stem,
        )
    if normalized == "upernet_swin_t":
        return build_upernet_swin_t(
            in_chans=in_chans,
            num_classes=num_classes,
            pretrained=pretrained,
            bgrnir_pretrained_stem=bgrnir_pretrained_stem,
        )
    if normalized == "mask2former_swin_t":
        return build_mask2former_swin_t(
            in_chans=in_chans,
            num_classes=num_classes,
            pretrained=pretrained,
            bgrnir_pretrained_stem=bgrnir_pretrained_stem,
        )
    if normalized == "torchgeo_dofa_base":
        from .aquav3_torchgeo import build_torchgeo_dofa_base

        return build_torchgeo_dofa_base(in_chans=in_chans, num_classes=num_classes, pretrained=pretrained)
    if normalized == "torchgeo_swin_b":
        from .aquav3_torchgeo import build_torchgeo_swin_b

        return build_torchgeo_swin_b(in_chans=in_chans, num_classes=num_classes, pretrained=pretrained)
    if normalized == "torchgeo_swin_t":
        from .aquav3_torchgeo import build_torchgeo_swin_t

        return build_torchgeo_swin_t(in_chans=in_chans, num_classes=num_classes, pretrained=pretrained)
    if normalized == "geoseg_a2fpn":
        from .aquav3_registry import build_aquav3_model

        return build_aquav3_model("a2fpn", in_chans=in_chans, num_classes=num_classes, pretrained=False)
    if normalized == "geoseg_dcswin":
        from .aquav3_registry import build_aquav3_model

        return build_aquav3_model("dcswin", in_chans=in_chans, num_classes=num_classes, pretrained=False)
    if normalized in {"unetformer", "abcnet"}:
        if not pretrained:
            raise ValueError(f"{normalized} formal retraining requires official SWSL ResNet-18 pretraining")
        from .aquav3_registry import build_aquav3_model

        return build_aquav3_model(
            normalized,
            in_chans=in_chans,
            num_classes=num_classes,
            pretrained=True,
        )
    raise ValueError(f"Unsupported AquaV3 comparison model: {key}")
