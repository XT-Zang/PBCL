"""Numerical and efficiency gates for the SegFormer-B2 loss comparison."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from geoseg.datasets.segformer_wprf_targets import build_semantic_target, row_major_dense_components
from geoseg.losses.wprf import WPRF_SCALES, symmetrize_affinities, widest_path_for_pairs
from geoseg.models.aquav3_external import HuggingFaceSegmentationWrapper


def verify():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    fixture = HuggingFaceSegmentationWrapper(nn.Identity())
    device = torch.device("cuda")
    fixture.affinity_head = nn.Conv2d(8, 8, 3, stride=4, padding=1).to(device)
    features = torch.randn(2, 8, 31, 37, device=device, requires_grad=True)
    size = (124, 148)
    results = {}
    for mixed in (False, True):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed):
            direct = fixture.affinity_head(F.interpolate(features, size=size, mode="bilinear", align_corners=False))
            striped = fixture._wprf_affinity(features, size)
        tensors = (features, *fixture.affinity_head.parameters())
        signal = torch.rand_like(direct)
        expected = torch.autograd.grad(direct, tensors, signal)
        actual = torch.autograd.grad(striped, tensors, signal)
        error = (striped - direct).abs().max().item()
        tolerance = 0.008 if mixed else 1e-5
        torch.testing.assert_close(direct, striped, atol=tolerance, rtol=tolerance)
        for reference, candidate in zip(expected, actual, strict=True):
            torch.testing.assert_close(reference, candidate, atol=0.12 if mixed else 2e-4,
                                       rtol=0.012 if mixed else 2e-5)
        results["affinity_bf16" if mixed else "affinity_fp32"] = {
            "max_output_error": error,
            "max_gradient_errors": [(ref - got).abs().max().item()
                                    for ref, got in zip(expected, actual, strict=True)],
        }

    for placement in (torch.device("cpu"), device):
        logits = torch.randn(1, 8, 37, 47, device=placement, requires_grad=True)
        symmetric = symmetrize_affinities(logits)
        support = torch.rand((37, 47), device=placement) > 0.15
        support[30:] = False
        support[:, 41:] = False
        pairs = torch.tensor([[0, 0, 0, 1], [0, 0, 5, 5], [1, 2, 36, 44],
                              [12, 10, 6, 4], [29, 39, 27, 35], [28, 40, 36, 44],
                              [12, 10, 20, 22], [12, 10, 12, 10]])
        for scale in WPRF_SCALES:
            for recompute in (True, False):
                reference = widest_path_for_pairs(symmetric[0], support, pairs, num_steps=scale,
                                                  activation_checkpoint=recompute, upstream=True)
                candidate = widest_path_for_pairs(symmetric[0], support, pairs, num_steps=scale,
                                                  activation_checkpoint=recompute, upstream=True, local_windows=True)
                torch.testing.assert_close(reference, candidate, atol=0, rtol=0)
                gradients = [torch.autograd.grad(x.sum(), logits, retain_graph=True)[0]
                             for x in (reference, candidate)]
                torch.testing.assert_close(*gradients, atol=1e-6, rtol=1e-6)
    support = np.zeros((3, 3), dtype=bool)
    support[0, 2] = support[2, 0] = True
    components = np.zeros_like(support, dtype=np.int32)
    components[0, 2], components[2, 0] = 1, 2
    assert row_major_dense_components(support, components, np.ones_like(support))[1, 1] == 1
    mask = np.zeros((64, 64), dtype=bool)
    mask[5:55, 6:10] = mask[3:59, 14:20] = True
    valid = np.ones_like(mask)
    valid[:, -6:] = False
    target = build_semantic_target(mask, valid, 42)
    assert any(len(target[f"neg_{scale}"]) for scale in WPRF_SCALES)
    assert not target["support"][:, -2:].any()
    assert not build_semantic_target(mask & False, valid, 42)["support"].any()
    return {**results, "scales": list(WPRF_SCALES), "local_window_scores_and_gradients_match": True,
            "row_major_vna_and_padding": True}


def efficiency_review(directory):
    from geoseg.experiments.segformer_loss_comparison import DATASETS, METHODS

    records = {}
    for dataset in DATASETS:
        base = json.loads((directory / f"{dataset}__boundary.json").read_text())
        for method in METHODS:
            key = f"{dataset}__{method}"
            candidate = json.loads((directory / f"{key}.json").read_text())
            ratio = candidate["step_median"] / base["step_median"]
            if method.startswith("wprf"):
                reason = "Active loss propagates reachability at every scale. Source-local windows preserve values and gradients."
            elif method == "cmalis":
                reason = "CPU maximum-spanning-tree affinity weights. Existing verified empty/single-instance fast paths and bounded workers."
            elif method == "cldice":
                reason = "Ten differentiable skeletonization iterations on the valid semantic grid."
            else:
                reason = "Additional pixel-wise supervision. Loader and GPU step times reported separately."
            records[key] = {"gpu_step_ratio_to_boundary": ratio, "diagnosis": reason,
                            "profile": str(directory / f"{key}.json")}
    return records
