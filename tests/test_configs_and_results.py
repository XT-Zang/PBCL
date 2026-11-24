from __future__ import annotations

from pathlib import Path
import re

from geoseg.runtime import load_experiment
from run import discover_configs, resolve_config
from scripts.verify_results import verify_all


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_configs_resolve_cleanly():
    configs = discover_configs()
    assert len(configs) >= 100
    for name, path in configs:
        exp = load_experiment(path, build=False, seed=42)
        assert exp.spec.optimization["loss"]["name"], f"Missing loss name for {name}"
        assert exp.spec.runtime["seed"] == 42


def test_cli_aliases_and_loss_coverage():
    expected_aliases = (
        "pbcl",
        "dinov3-full",
        "dinov3-boundary",
        "segformer-full",
        "vegas-segformer-full",
    )
    for alias in expected_aliases:
        path = resolve_config(alias)
        assert path.is_file()

    # Verify all 9 same-carrier loss objectives resolve on both DINOv3 and SegFormer-B2
    for name in (
        "aquadataset/base",
        "aquadataset/boundary",
        "aquadataset/pbcl_only",
        "aquadataset/full",
        "aquadataset/ewc",
        "aquadataset/cldice",
        "aquadataset/cmalis",
        "aquadataset/wprf_gt",
        "aquadataset/wprf_valid",
        "aquadataset/bce_split",
        "aquadataset/boundary_split",
        "aquadataset/segformer_b2_ewc",
        "aquadataset/segformer_b2_cldice",
        "aquadataset/segformer_b2_cmalis",
        "aquadataset/segformer_b2_wprf_gt",
        "aquadataset/segformer_b2_wprf_valid",
        "aquadataset/segformer_b2_bce_split",
        "aquadataset/segformer_b2_boundary_split",
        "vegas/segformer_b2__full",
        "vegas/segformer_b2__ewc",
        "vegas/segformer_b2__cmalis",
        "vegas/segformer_b2__wprf_valid",
        "vegas/segformer_b2__bce_split",
    ):
        exp = load_experiment(resolve_config(name), build=False, seed=42)
        assert exp.spec.optimization["loss"]["name"]


def test_paper_results_verification():
    summary = verify_all(verbose=False)
    assert summary["status"] == "PASSED"
    assert summary["aqua_8arch_paired_wins"] == "48/48"


def test_no_hardcoded_machine_paths():
    pattern = re.compile(r"/(?:mnt|home/zang|Users/zang|run/media)/")
    checked_suffixes = {".py", ".json", ".csv", ".txt", ".md"}
    offenders: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.name == "test_configs_and_results.py":
            continue
        if path.suffix not in checked_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"Found hardcoded machine paths in: {offenders}"
