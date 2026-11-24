from __future__ import annotations

import numpy as np
import torch

from pbcl import (
    PAIR_BUCKET_CODE,
    PairwiseBottleneckConnectivityLoss,
    extract_neighbor_pairs,
)


def _two_instance_scene(gap: int = 2) -> np.ndarray:
    """Create a 32x32 instance map with two 6x6 squares separated by `gap` pixels."""
    inst = np.zeros((32, 32), dtype=np.int64)
    inst[10:16, 6:12] = 1
    x2 = 12 + gap
    inst[10:16, x2 : x2 + 6] = 2
    return inst


def test_extract_neighbor_pairs_gap_and_policies():
    # Gap = 2 pixels -> bucket "1_2" (code 1)
    pairs = extract_neighbor_pairs(_two_instance_scene(gap=2), max_gap=5, policy="all")
    assert pairs.shape == (1, 3)
    assert tuple(pairs[0].tolist()) == (1, 2, PAIR_BUCKET_CODE["1_2"])

    # Direct 4-neighbor contact (gap = 0)
    touching = _two_instance_scene(gap=0)
    assert len(extract_neighbor_pairs(touching, max_gap=5, policy="all")) == 1
    assert len(extract_neighbor_pairs(touching, max_gap=5, policy="four_contact")) == 1
    assert len(extract_neighbor_pairs(touching, max_gap=5, policy="no_four_contact")) == 0
    assert len(extract_neighbor_pairs(touching, max_gap=5, policy="noncontact")) == 0

    # Diagonal-only contact (corner touching, no 4-neighbor edge)
    diag = np.zeros((20, 20), dtype=np.int64)
    diag[4:8, 4:8] = 1
    diag[8:12, 8:12] = 2
    assert len(extract_neighbor_pairs(diag, max_gap=5, policy="all")) == 1
    assert len(extract_neighbor_pairs(diag, max_gap=5, policy="no_four_contact")) == 1
    assert len(extract_neighbor_pairs(diag, max_gap=5, policy="four_contact")) == 0
    assert len(extract_neighbor_pairs(diag, max_gap=5, policy="noncontact")) == 0


def test_extract_neighbor_pairs_eq1_and_prop3_tightness():
    """Verify g_ij = max(0, floor(d_ij + 1e-9) - 1) including Proposition 3 boundary cases."""
    cases = [
        ((1, 0), PAIR_BUCKET_CODE["touching"]),   # d = 1.0 -> g = 0
        ((1, 1), PAIR_BUCKET_CODE["touching"]),   # d = sqrt(2) -> g = 0
        ((0, 2), PAIR_BUCKET_CODE["1_2"]),        # d = 2.0 -> g = 1
        ((1, 2), PAIR_BUCKET_CODE["1_2"]),        # d = sqrt(5) ~ 2.236 -> g = 1
        ((3, 6), PAIR_BUCKET_CODE["3_5"]),        # d = sqrt(45) ~ 6.708 -> g = 5 (Prop. 3 tight)
        ((4, 5), PAIR_BUCKET_CODE["3_5"]),        # d = sqrt(41) ~ 6.403 -> g = 5 (Prop. 3 tight)
    ]
    for (dy, dx), expected_bucket in cases:
        inst = np.zeros((20, 20), dtype=np.int64)
        inst[2, 2] = 1
        inst[2 + dy, 2 + dx] = 2
        pairs = extract_neighbor_pairs(inst, max_gap=5, policy="all")
        assert pairs.shape == (1, 3), f"Displacement {(dy, dx)} should be retained with g <= 5"
        assert tuple(pairs[0].tolist()) == (1, 2, expected_bucket)

    # Displacement (0, 7) has d = 7.0 -> g = 6 > 5 and must be excluded
    far = np.zeros((20, 20), dtype=np.int64)
    far[2, 2] = 1
    far[2, 9] = 2
    assert len(extract_neighbor_pairs(far, max_gap=5, policy="all")) == 0


def test_pbcl_penalizes_bridge_and_vanishes_when_separated():
    inst = _two_instance_scene(gap=2)
    inst_t = torch.from_numpy(inst).unsqueeze(0)
    criterion = PairwiseBottleneckConnectivityLoss(
        band_radius=3, iterations=9, threshold=0.5, margin=0.1
    )

    # Case 1: Clean separation (low probability in gap -> bottleneck < 0.4 -> zero loss)
    logits_clean = torch.full((1, 1, 32, 32), -6.0)
    logits_clean[0, 0, inst > 0] = 6.0
    loss_clean, score_clean, n_pairs = criterion(logits_clean, inst_t, return_stats=True)
    assert n_pairs == 1
    assert float(loss_clean) == 0.0
    assert float(score_clean) < 0.05

    # Case 2: False-positive bridge across the 2-pixel gap -> positive loss & gradient on bridge
    logits_bridge = logits_clean.clone().detach().requires_grad_(True)
    with torch.no_grad():
        logits_bridge[0, 0, 12, 12:14] = 3.0  # bridge pixels at row 12, cols 12..13
    loss_bridge, score_bridge, n_pairs = criterion(logits_bridge, inst_t, return_stats=True)
    assert n_pairs == 1
    assert float(score_bridge) > 0.9
    assert float(loss_bridge.detach()) > 0.25

    loss_bridge.backward()
    grad = logits_bridge.grad[0, 0]
    assert torch.isfinite(grad).all()
    # Gradient must push down the bottleneck bridge pixels
    assert float(grad[12, 12:14].sum()) > 0.0
    # Background pixels far from the pair receive zero gradient
    assert float(grad[0:4, 0:4].abs().sum()) == 0.0
