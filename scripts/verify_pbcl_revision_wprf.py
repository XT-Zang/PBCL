#!/usr/bin/env python3
"""Verify the revision WPRF port against pinned upstream values and gradients."""
from __future__ import annotations

from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from geoseg.datasets.aquadataset_wprf import WPRF_SOURCE_COMMIT, WPRF_SCALES, _adjacency, load_wprf_target
from geoseg.datasets.geographic_holdout import sha256_file
from geoseg.experiments.pbcl_revision import DATA_ROOT, OUTPUT_ROOT
from geoseg.losses.wprf import (
    WPRF_OFFSETS, reachability_loss, symmetrize_affinities,
    upstream_edge_affinity_loss, widest_path_for_pairs,
)
from scripts.run_pbcl_revision import write_json


def upstream_modules():
    source = OUTPUT_ROOT / "references/WPRF_open/src/wprf"
    package = types.ModuleType("revision_upstream_wprf")
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    loaded = {}
    for name in ("config", "reachability", "losses"):
        spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", source / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def verify_upstream() -> dict:
    import subprocess

    checkout = OUTPUT_ROOT / "references/WPRF_open"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if commit != WPRF_SOURCE_COMMIT:
        raise RuntimeError(f"Upstream commit mismatch: {commit}")
    upstream = upstream_modules()
    constants = upstream["config"].WPRFConstants()
    torch.manual_seed(42)
    logits = torch.randn(2, 8, 5, 6, requires_grad=True)
    symmetric = symmetrize_affinities(logits)
    components = torch.tensor([
        [[1, 1, 0, 0, 2, 2]] * 5,
        [[0, 0, 0, 0, 0, 0]] * 5,
    ])
    local_edge = upstream_edge_affinity_loss(symmetric, components)
    original_edge = upstream["losses"].edge_affinity_bce_loss(
        logits.permute(0, 2, 3, 1), components, constants=constants,
        offsets=WPRF_OFFSETS,
    )
    local_gradient = torch.autograd.grad(local_edge, logits, retain_graph=True)[0]
    original_gradient = torch.autograd.grad(original_edge, logits, retain_graph=True)[0]
    torch.testing.assert_close(local_edge, original_edge, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(local_gradient, original_gradient, atol=1e-6, rtol=1e-6)
    cases = []
    for steps in WPRF_SCALES:
        support = torch.ones((5, 6), dtype=torch.bool)
        support[2, 1:5] = False
        pairs = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 2], [0, 0, 4, 4], [1, 1, 3, 1]])
        for checkpoint in (False, True):
            local = widest_path_for_pairs(symmetric[0], support, pairs, num_steps=steps,
                                          activation_checkpoint=checkpoint, upstream=True)
            original = upstream["reachability"].max_min_for_pairs(
                symmetric[0].permute(1, 2, 0), offsets=WPRF_OFFSETS, pairs_yxyx=pairs,
                num_steps=steps, source_batch_size=32, support_mask=support,
            )
            torch.testing.assert_close(local, original, atol=1e-6, rtol=1e-6)
            local_grad = torch.autograd.grad(local.sum(), logits, retain_graph=True)[0]
            original_grad = torch.autograd.grad(original.sum(), logits, retain_graph=True)[0]
            torch.testing.assert_close(local_grad, original_grad, atol=1e-5, rtol=1e-5)
            mapping = {f"pos_{steps}": pairs[:2], f"neg_{steps}": pairs[2:]}
            local_loss, _ = reachability_loss(symmetric[:1], support[None], [mapping], scales=(steps,),
                                               activation_checkpoint=checkpoint, upstream=True)
            original_loss = upstream["losses"].reachability_loss_k(
                symmetric[0].permute(1, 2, 0), constants=constants, offsets=WPRF_OFFSETS,
                pos_pairs_yxyx=pairs[:2], neg_pairs_yxyx=pairs[2:], num_steps=steps,
                source_batch_size=32, support_mask=support,
            )
            torch.testing.assert_close(local_loss, original_loss, atol=1e-6, rtol=1e-6)
            cases.append({"steps": steps, "checkpoint": checkpoint, "score_error": float((local - original).abs().max().detach()),
                          "gradient_error": float((local_grad - original_grad).abs().max()), "loss_error": float((local_loss - original_loss).abs().detach())})
    return {"commit": commit, "cases": cases, "edge_error": float((local_edge - original_edge).abs().detach()),
            "source_files": {str(path.relative_to(checkout)): sha256_file(path) for path in (checkout / "src/wprf").glob("*.py")}}


def audit_cache() -> dict:
    rows = [json.loads(line) for line in (DATA_ROOT / "splits/train.jsonl").read_text().splitlines()]
    totals = defaultdict(lambda: defaultdict(int))
    probes = []
    for row in rows:
        target = load_wprf_target(DATA_ROOT / "instance_targets/wprf_matched_v1" / f"{row['tile_id']}.npz")
        support = target["support"].astype(bool)
        coordinates, neighbors = _adjacency(support)
        edges = [(source, destination) for source, group in enumerate(neighbors) for destination in group]
        graph = csr_matrix((np.ones(len(edges)), tuple(zip(*edges))), shape=(len(coordinates), len(coordinates))) if edges else csr_matrix((len(coordinates), len(coordinates)))
        index = np.full(support.shape, -1, dtype=np.int32)
        if len(coordinates):
            index[coordinates[:, 0], coordinates[:, 1]] = np.arange(len(coordinates))
        pairs_by_scale = {scale: target[f"neg_{scale}"] for scale in WPRF_SCALES}
        nonempty = [value for value in pairs_by_scale.values() if len(value)]
        distances = None
        if nonempty:
            sources = np.unique(np.concatenate(nonempty)[:, :2], axis=0)
            source_nodes = index[sources[:, 0], sources[:, 1]]
            distances = dijkstra(graph, directed=False, indices=source_nodes, unweighted=True, limit=16)
            source_index = {tuple(point): location for location, point in enumerate(sources.tolist())}
        for scale, pairs in pairs_by_scale.items():
            if not len(pairs):
                continue
            legal_gt = np.array([distances[source_index[tuple(pair[:2])], index[pair[2], pair[3]]] <= scale for pair in pairs])
            # Positive Train tiles have complete valid support. The real-batch
            # autograd probe below also checks the transformed runtime mask.
            if int(row["valid_height"]) != 512 or int(row["valid_width"]) != 512:
                raise RuntimeError("WPRF cache legality requires auditing partial positive tiles")
            legal_valid = np.max(np.abs(pairs[:, :2] - pairs[:, 2:]), axis=1) <= scale
            totals[scale]["negative_records"] += len(pairs)
            totals[scale]["legal_gt"] += int(legal_gt.sum())
            totals[scale]["legal_valid"] += int(legal_valid.sum())
            if legal_valid.any() and not any(item["scale"] == scale for item in probes):
                pair = pairs[np.flatnonzero(legal_valid)[0]:np.flatnonzero(legal_valid)[0] + 1]
                logits = torch.zeros((1, 8, *support.shape), requires_grad=True)
                scores = widest_path_for_pairs(symmetrize_affinities(logits)[0], torch.ones_like(torch.from_numpy(support)),
                                                torch.from_numpy(pair.astype(np.int64)), num_steps=scale,
                                                activation_checkpoint=True, upstream=True)
                gradient = torch.autograd.grad(-torch.log(1 - scores + 1e-6).mean(), logits)[0]
                probes.append({"tile_id": row["tile_id"], "scale": scale, "pair": pair.tolist(),
                               "score": float(scores.detach()[0]), "gradient_max": float(gradient.abs().max())})
    result = {str(scale): {**dict(values), "legal_gt_ratio": values["legal_gt"] / values["negative_records"],
                          "legal_valid_ratio": values["legal_valid"] / values["negative_records"]} for scale, values in totals.items()}
    return {"train_negative_by_scale": result, "neutral_affinity_real_cache_probes": probes}


def real_batch_probe() -> dict:
    import rasterio
    from geoseg.runtime import load_experiment
    from geoseg.losses.binary_seg import loss_batch_kwargs

    experiment = load_experiment(ROOT / "configs/aquadataset/wprf_valid.py", build=True)
    device = torch.device("cuda")
    model = experiment.components.model.to(device).train()
    selected = None
    for number, batch in enumerate(experiment.components.datamodule.train_dataloader()):
        if any(len(item[f"neg_{scale}"]) for item in batch["wprf_pairs"] for scale in WPRF_SCALES):
            selected = batch
            break
    if selected is None:
        raise RuntimeError("No real Train batch contains WPRF negatives")
    image = selected["img"].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            outputs = model(image)
    logits = outputs["affinity"].float().detach().requires_grad_()
    symmetric = symmetrize_affinities(logits)
    valid = torch.nn.functional.adaptive_max_pool2d(1 - selected["valid_mask"].float().to(device), logits.shape[-2:]) == 0
    counts = {}
    maximum_gradient = 0.0
    for scale in WPRF_SCALES:
        scores = []
        for index, pairs in enumerate(selected["wprf_pairs"]):
            negative = pairs[f"neg_{scale}"].to(device)
            if len(negative):
                scores.append(widest_path_for_pairs(symmetric[index], valid[index, 0], negative,
                                                    num_steps=scale, activation_checkpoint=True, upstream=True))
        if not scores:
            continue
        scores = torch.cat(scores)
        loss = -torch.log(1 - scores.clamp(0, 1 - 1e-6) + 1e-6).mean()
        gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        gradient_max = float(gradient.abs().max())
        maximum_gradient = max(maximum_gradient, gradient_max)
        counts[str(scale)] = {"negative_records": len(scores), "positive_scores": int((scores > 0).sum()),
                              "interior_scores": int(((scores > 0) & (scores < 1 - 1e-6)).sum()), "gradient_max": gradient_max}
    return {"batch_index": number, "tile_ids": selected["tile_id"], "affinity_state": "initial_pretrained_encoder_and_seed42_random_head",
            "scales": counts, "gradient_max": maximum_gradient}


def main() -> None:
    torch.set_num_threads(4)
    verification = verify_upstream()
    cache = audit_cache()
    batch = real_batch_probe()
    gate = batch["gradient_max"] > 0 and any(probe["gradient_max"] > 0 for probe in cache["neutral_affinity_real_cache_probes"])
    write_json(OUTPUT_ROOT / "analysis/wprf_verification.json",
               {"upstream_verified": True, "upstream": verification, **cache, "real_batch_probe": batch,
                "valid_support_gate_passed": gate,
                "adaptation": "WPRF-valid changes only the negative-query support from GT skeleton to the valid grid",
                "legacy_port_differences": ["half-space/source-foreground local edge sampling", "per-image edge normalization", "additive epsilon in reachability BCE", "exact-step propagation with accumulated maxima"]})


if __name__ == "__main__":
    main()
