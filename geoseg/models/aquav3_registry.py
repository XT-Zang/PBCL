from __future__ import annotations

import torch.nn as nn


AQUAV3_MODEL_NAMES = (
    "a2fpn",
    "abcnet",
    "dcswin",
    "unetformer",
)


def build_aquav3_model(name: str, in_chans: int = 4, num_classes: int = 1, pretrained: bool = False) -> nn.Module:
    key = name.strip().lower().replace("-", "").replace("_", "")
    if key == "a2fpn":
        from .A2FPN import A2FPN

        return A2FPN(band=in_chans, class_num=num_classes, pretrained=pretrained)
    if key == "abcnet":
        from .ABCNet import ABCNet

        return ABCNet(band=in_chans, n_classes=num_classes, pretrained=pretrained)
    if key == "dcswin":
        from .DCSwin import dcswin_small

        return dcswin_small(pretrained=pretrained, num_classes=num_classes, weight_path=None if not pretrained else "pretrain_weights/stseg_small.pth", in_chans=in_chans)
    if key == "unetformer":
        from .UNetFormer import UNetFormer

        return UNetFormer(num_classes=num_classes, in_chans=in_chans, pretrained=pretrained)
    raise ValueError(f"Unsupported AquaV3 model: {name}")
