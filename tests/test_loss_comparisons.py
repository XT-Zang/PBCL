from __future__ import annotations

import numpy as np
import torch

from geoseg.losses.binary_seg import PBCLBinarySegmentationLoss
from geoseg.losses.segformer_comparison import SegFormerComparisonLoss
from geoseg.losses.soft_cldice import soft_cldice_loss
from geoseg.losses.split import SplitSegmentationLoss
from geoseg.losses.ssepnet import (
    SSepNetClDiceLoss,
    SSepNetEWCLoss,
    SSepNetLoss,
    SSepNetSplitLoss,
)
from scripts.evaluate_full_scene_corridor import (
    edge_anchored_starts,
    filter_interior_small_components,
    make_triangular_weight,
)


def _synthetic_batch():
    inst = torch.zeros((2, 32, 32), dtype=torch.long)
    inst[:, 6:14, 6:12] = 1
    inst[:, 6:14, 14:20] = 2
    target = (inst > 0).float().unsqueeze(1)
    pairs = [torch.tensor([[1, 2, 1]], dtype=torch.long) for _ in range(2)]
    valid = torch.ones_like(target, dtype=torch.bool)
    ewc = torch.ones_like(target)
    return target, inst, pairs, valid, ewc


def test_all_primary_losses_forward_backward():
    target, inst, pairs, valid, ewc = _synthetic_batch()

    # 1. SSepNetLoss (Full: Boundary + PBCL)
    out = {
        "out": torch.randn(2, 1, 32, 32, requires_grad=True),
        "aux": torch.randn(2, 1, 32, 32, requires_grad=True),
        "edge": torch.randn(2, 1, 32, 32, requires_grad=True),
    }
    loss_full = SSepNetLoss(boundary_weight=0.6, pbcl_weight=0.1)(
        out, target, instance_maps=inst, instance_pairs=pairs, valid_masks=valid
    )
    assert torch.isfinite(loss_full)
    loss_full.backward()
    assert out["out"].grad is not None and torch.isfinite(out["out"].grad).all()

    # 2. PBCLBinarySegmentationLoss (external architectures)
    out_ext = {
        "out": torch.randn(2, 1, 32, 32, requires_grad=True),
        "edge": torch.randn(2, 1, 32, 32, requires_grad=True),
    }
    loss_ext = PBCLBinarySegmentationLoss(boundary_weight=0.6, pbcl_weight=0.1)(
        out_ext, target, instance_maps=inst, instance_pairs=pairs, valid_masks=valid
    )
    assert torch.isfinite(loss_ext)
    loss_ext.backward()
    assert out_ext["out"].grad is not None

    # 3. EW-BCE
    out_ewc = {
        "out": torch.randn(2, 1, 32, 32, requires_grad=True),
        "aux": torch.randn(2, 1, 32, 32, requires_grad=True),
        "edge": torch.randn(2, 1, 32, 32, requires_grad=True),
    }
    loss_ewc = SSepNetEWCLoss()(out_ewc, target, ewc_weights=ewc, valid_masks=valid)
    assert torch.isfinite(loss_ewc)
    loss_ewc.backward()
    assert out_ewc["out"].grad is not None

    # 4. soft-clDice
    logits_cd = torch.randn(2, 1, 32, 32, requires_grad=True)
    loss_cd_fn = soft_cldice_loss(logits_cd, target, valid_masks=valid, iterations=3)
    assert torch.isfinite(loss_cd_fn)
    out_cd = {
        "out": torch.randn(2, 1, 32, 32, requires_grad=True),
        "aux": torch.randn(2, 1, 32, 32, requires_grad=True),
        "edge": torch.randn(2, 1, 32, 32, requires_grad=True),
    }
    loss_cd = SSepNetClDiceLoss(cldice_iterations=3)(out_cd, target, valid_masks=valid)
    assert torch.isfinite(loss_cd)
    loss_cd.backward()

    # 5. BCE + SPLIT (Eq. 15) & Boundary + SPLIT
    logits_sp = torch.randn(2, 1, 32, 32, requires_grad=True)
    loss_sp = SplitSegmentationLoss(image_size=32, split_weight=0.1, boundary_kernel=3)(
        {"out": logits_sp}, target, valid_masks=valid
    )
    assert torch.isfinite(loss_sp)
    loss_sp.backward()

    out_bsp = {
        "out": torch.randn(2, 1, 32, 32, requires_grad=True),
        "aux": torch.randn(2, 1, 32, 32, requires_grad=True),
        "edge": torch.randn(2, 1, 32, 32, requires_grad=True),
    }
    loss_bsp = SSepNetSplitLoss(image_size=32, split_weight=0.1, split_kernel=3)(
        out_bsp, target, valid_masks=valid
    )
    assert torch.isfinite(loss_bsp)
    loss_bsp.backward()

    # 6. SegFormerComparisonLoss (ewc, cldice, split)
    for method, extra in (
        ("ewc", {"ewc_weights": ewc}),
        ("cldice", {}),
        ("split", {}),
    ):
        spec = {
            "main_bce": 1.0,
            "main_dice": 1.0,
            "boundary_weight": 0.6,
            "boundary_kernel": 5,
            "method": method,
            "cldice_alpha": 0.5,
            "cldice_iterations": 3,
            "cldice_smooth": 1.0,
            "split_weight": 0.1,
            "split_kernel": 3,
        }
        out_sf = {
            "out": torch.randn(2, 1, 32, 32, requires_grad=True),
            "edge": torch.randn(2, 1, 32, 32, requires_grad=True),
        }
        loss_sf = SegFormerComparisonLoss(spec, image_size=32)(
            out_sf, target, valid_masks=valid, **extra
        )
        assert torch.isfinite(loss_sf)
        loss_sf.backward()


def test_corridor_post_filter_and_mosaicking_helpers():
    weight = make_triangular_weight(512, floor=0.15)
    assert weight.shape == (512, 512)
    assert abs(float(weight.min()) - 0.0225) < 1e-4
    assert edge_anchored_starts(1000, 512, 384) == [0, 384, 488]

    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:20] = True  # 100 px interior -> keep
    mask[30:34, 30:34] = True  # 16 px interior speck -> drop
    mask[0:4, 40:44] = True    # 16 px border-touching -> keep
    filtered = filter_interior_small_components(mask, min_area_px=64)
    assert filtered[15, 15]
    assert not filtered[31, 31]
    assert filtered[1, 41]
