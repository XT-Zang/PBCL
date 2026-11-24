from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PretrainedModelSpec:
    key: str
    framework: str
    relative_path: str
    source: str
    consumers: tuple[str, ...]
    revision: str = ""
    fallback_root_paths: tuple[str, ...] = ()
    fallback_workspace_paths: tuple[str, ...] = ()


DEFAULT_PRETRAINED_MODELS = (
    PretrainedModelSpec(
        "segformer-b2-ade",
        "huggingface",
        "huggingface/hub/models--nvidia--segformer-b2-finetuned-ade-512-512",
        "nvidia/segformer-b2-finetuned-ade-512-512",
        ("AquaV3 SegFormer-B2",),
    ),
    PretrainedModelSpec(
        "upernet-swin-tiny",
        "huggingface",
        "huggingface/hub/models--openmmlab--upernet-swin-tiny",
        "openmmlab/upernet-swin-tiny",
        ("AquaV3 UPerNet-Swin-T",),
    ),
    PretrainedModelSpec(
        "mask2former-swin-tiny",
        "huggingface",
        "huggingface/hub/models--facebook--mask2former-swin-tiny-ade-semantic",
        "facebook/mask2former-swin-tiny-ade-semantic",
        ("AquaV3 Mask2Former-Swin-T",),
    ),
    PretrainedModelSpec(
        "mask2former-swin-tiny-coco-instance",
        "huggingface",
        "huggingface/hub/models--facebook--mask2former-swin-tiny-coco-instance",
        "facebook/mask2former-swin-tiny-coco-instance",
        ("AquaDateset Mask2Former-Swin-T Instance",),
        revision="22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6",
    ),
    PretrainedModelSpec(
        "smp-efficientnet-b4-imagenet",
        "huggingface",
        "huggingface/hub/models--smp-hub--efficientnet-b4.imagenet",
        "smp-hub/efficientnet-b4.imagenet",
        ("AquaV3 UNet++",),
    ),
    PretrainedModelSpec(
        "smp-resnet50-imagenet",
        "huggingface",
        "huggingface/hub/models--smp-hub--resnet50.imagenet",
        "smp-hub/resnet50.imagenet",
        ("AquaV3 DeepLabV3+",),
    ),
    PretrainedModelSpec(
        "swsl-resnet18",
        "huggingface",
        "huggingface/hub/models--timm--resnet18.fb_swsl_ig1b_ft_in1k",
        "timm/resnet18.fb_swsl_ig1b_ft_in1k",
        ("AquaV3 UNetFormer", "AquaV3 ABCNet"),
    ),
    PretrainedModelSpec(
        "swin-tiny-imagenet",
        "torch-hub",
        "torch/hub/checkpoints/swin_tiny_patch4_window7_224.pth",
        "SwinTransformer/swin_tiny_patch4_window7_224",
        ("AquaV3 Swin-Unet",),
    ),
    PretrainedModelSpec(
        "torchgeo-dofa-base",
        "torch-hub",
        "torch/hub/checkpoints/B13_vitb16_mae_ep99_enc.pth",
        "torchgeo DOFABase16_Weights.DOFA_MAE",
        ("AquaV3 TorchGeo DOFA-Base",),
        fallback_root_paths=("torchgeo/B13_vitb16_mae_ep99_enc.pth",),
    ),
    PretrainedModelSpec(
        "torchgeo-vit-base-sentinel2-mae",
        "torchgeo",
        "torchgeo/B13_vitb16_mae_ep99_enc.pth",
        "torchgeo ViTBase16_Weights.SENTINEL2_ALL_MAE",
        ("AquaDateset HRNet-ViT-B context",),
        fallback_root_paths=("torchgeo/B13_vitb16_mae_ep99_enc.pth",),
    ),
    PretrainedModelSpec(
        "torchgeo-swin-v2-b",
        "torch-hub",
        "torch/hub/checkpoints/sentinel2_swinb_mi_ms-39c86721.pth",
        "torchgeo Swin_V2_B_Weights.SENTINEL2_MI_MS_SATLAS",
        ("AquaV3 TorchGeo Swin-V2-B",),
        fallback_root_paths=("torchgeo/sentinel2_swinb_mi_ms-39c86721.pth",),
    ),
    PretrainedModelSpec(
        "torchgeo-swin-v2-t",
        "torch-hub",
        "torch/hub/checkpoints/sentinel2_swint_mi_ms-d8c659e3.pth",
        "torchgeo Swin_V2_T_Weights.SENTINEL2_MI_MS_SATLAS",
        ("AquaV3 TorchGeo Swin-V2-T",),
        fallback_root_paths=("torchgeo/sentinel2_swint_mi_ms-d8c659e3.pth",),
    ),
    PretrainedModelSpec(
        "torchgeo-vit-small-sentinel2-dino",
        "torchgeo",
        "torchgeo/vit_small_patch16_224_sentinel2_all_dino-36bcc127.pth",
        "torchgeo Sentinel-2 DINO ViT-S/16",
        ("TorchGeo remote-sensing backbones",),
    ),
    PretrainedModelSpec(
        "dinov3-vitl16-sat493m",
        "torch-hub",
        "torch/hub/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        "DINOv3 ViT-L/16 SAT-493M",
        ("DINOv3 ViT-L/16 Stage-Adapter FPN",),
    ),
    PretrainedModelSpec(
        "swinv2-large-imagenet22k",
        "torch-hub",
        "torch/hub/checkpoints/swinv2_large_patch4_window12_192_22k.pth",
        "Swin-V2 Large ImageNet-22K",
        ("Swin-V2 experiments",),
    ),
    PretrainedModelSpec(
        "hrnetv2-w32-imagenet",
        "pytorch",
        "hrnetv2_w32_imagenet_pretrained.pth",
        "HRNetV2-W32 ImageNet",
        ("GeoSeg SSepNet",),
    ),
    PretrainedModelSpec(
        "hrformer-s-imagenet",
        "pytorch",
        "hrformer/hrt_small_imagenet.pth",
        "OpenMMLab MMPose HRFormer-S ImageNet-1K backbone 09516375",
        ("GeoSeg DINOv3-HRFormer-S FPN+OCR",),
    ),
    PretrainedModelSpec(
        "pidnet-s-imagenet",
        "pytorch",
        "pidnet/PIDNet_S_ImageNet.pth",
        "XuJiacong/PIDNet PIDNet-S ImageNet",
        ("GeoSeg DINOv3-PIDNet-S-PD FPN+OCR",),
    ),
    PretrainedModelSpec(
        "ddrnet23-slim-imagenet",
        "pytorch",
        "ddrnet/DDRNet23s_imagenet.pth",
        "ydhongHIT/DDRNet DDRNet-23-slim ImageNet",
        ("GeoSeg DINOv3-DDRNet-23-slim FPN+OCR",),
    ),
    PretrainedModelSpec(
        "hrnetv2-w32-distilled",
        "pytorch",
        "dinov3_distill/hrnetv2_w32_distilled.pth",
        "HRNetV2-W32 distilled",
        ("GeoSeg SSepNet",),
    ),
    PretrainedModelSpec(
        "sam-vit-b",
        "pytorch",
        "sam/sam_vit_b_01ec64.pth",
        "Segment Anything ViT-B",
        ("SAMRS",),
    ),
    PretrainedModelSpec(
        "geoseg-rest-lite",
        "legacy-geoseg",
        "geoseg/rest_lite.pth",
        "WangLibo1995/GeoSeg backbone weights",
        ("BANet",),
    ),
    PretrainedModelSpec(
        "geoseg-stseg-tiny",
        "legacy-geoseg",
        "geoseg/stseg_tiny.pth",
        "WangLibo1995/GeoSeg backbone weights",
        ("DC-Swin-Tiny",),
    ),
    PretrainedModelSpec(
        "geoseg-stseg-small",
        "legacy-geoseg",
        "geoseg/stseg_small.pth",
        "WangLibo1995/GeoSeg backbone weights",
        ("DC-Swin-Small",),
    ),
    PretrainedModelSpec(
        "geoseg-stseg-base",
        "legacy-geoseg",
        "geoseg/stseg_base.pth",
        "WangLibo1995/GeoSeg backbone weights",
        ("DC-Swin-Base", "FT-UNetFormer"),
    ),
)


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_pretrained_root() -> Path:
    configured = os.environ.get("PBCL_PRETRAINED_ROOT") or os.environ.get("GEOSEG_PRETRAINED_ROOT")
    root = Path(configured).expanduser() if configured else Path.home() / ".cache" / "pbcl"
    return root.resolve()


def configure_pretrained_environment(root: str | Path | None = None) -> dict[str, str]:
    resolved_root = Path(root).expanduser().resolve() if root is not None else default_pretrained_root()
    defaults = {
        "GEOSEG_PRETRAINED_ROOT": str(resolved_root),
        "HF_HOME": str((resolved_root / "huggingface").resolve()),
        "TORCH_HOME": str((resolved_root / "torch").resolve()),
        "GEOSEG_PRETRAINED_WEIGHTS": str((resolved_root / "geoseg").resolve()),
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    swin_checkpoint = resolved_root / "torch" / "hub" / "checkpoints" / "swin_tiny_patch4_window7_224.pth"
    if swin_checkpoint.is_file():
        os.environ.setdefault("AQUAV3_SWINUNET_CKPT", str(swin_checkpoint.resolve()))
    return {
        "root": str(resolved_root),
        "hf_home": os.environ["HF_HOME"],
        "torch_home": os.environ["TORCH_HOME"],
        "legacy_weights": os.environ["GEOSEG_PRETRAINED_WEIGHTS"],
        "swinunet_checkpoint": os.environ.get("AQUAV3_SWINUNET_CKPT", ""),
    }


def pretrained_weight_path(filename: str) -> Path:
    root = Path(os.environ.get("GEOSEG_PRETRAINED_WEIGHTS", default_pretrained_root() / "geoseg"))
    return root.expanduser().resolve() / filename


def resolve_legacy_pretrained_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if len(candidate.parts) == 2 and candidate.parts[0] == "pretrain_weights":
        return pretrained_weight_path(candidate.name)
    return candidate


def _has_payload(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(item.is_file() and item.stat().st_size > 0 for item in path.rglob("*"))
    return False


def resolve_pretrained_model_path(key: str, *, root: str | Path | None = None) -> Path:
    """Resolve one indexed pretrained artifact without downloading or mutating the index."""
    normalized = str(key).strip().lower()
    try:
        spec = next(model for model in DEFAULT_PRETRAINED_MODELS if model.key == normalized)
    except StopIteration as exc:
        raise KeyError(f"Unknown pretrained model key: {key!r}") from exc
    resolved_root = Path(root).expanduser().resolve() if root is not None else default_pretrained_root()
    path = (resolved_root / spec.relative_path).resolve()
    if not _has_payload(path):
        raise FileNotFoundError(
            f"Pretrained model {spec.key!r} is missing at {path}; "
            "populate GEOSEG_PRETRAINED_ROOT before requesting pretrained=True"
        )
    return path


def _fallback_candidates(spec: PretrainedModelSpec, root: Path, workspace: Path) -> Iterable[Path]:
    for value in spec.fallback_root_paths:
        yield root / value
    for value in spec.fallback_workspace_paths:
        yield workspace / value


def _link_first_fallback(spec: PretrainedModelSpec, root: Path, workspace: Path) -> Path | None:
    target = root / spec.relative_path
    if target.exists() or target.is_symlink():
        return None
    for candidate in _fallback_candidates(spec, root, workspace):
        if not _has_payload(candidate):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        relative = os.path.relpath(candidate.resolve(), target.parent.resolve())
        target.symlink_to(relative, target_is_directory=candidate.is_dir())
        return candidate.resolve()
    return None


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def _render_markdown(manifest: dict[str, object]) -> str:
    rows = [
        "# Workspace Pretrained Model Index",
        "",
        f"Root: `{manifest['root']}`",
        "",
        "| Key | Framework | Revision | Status | Size (bytes) | Path | Consumers |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for entry in manifest["models"]:
        consumers = ", ".join(entry["consumers"])
        rows.append(
            f"| {entry['key']} | {entry['framework']} | {entry['revision']} | {entry['status']} | "
            f"{entry['size_bytes']} | `{entry['path']}` | {consumers} |"
        )
    rows.extend(
        [
            "",
            "GeoSeg automatically uses this root unless `GEOSEG_PRETRAINED_ROOT`, `HF_HOME`, or `TORCH_HOME` is explicitly set.",
            "Missing entries are catalogued but are never downloaded or replaced automatically.",
            "",
        ]
    )
    return "\n".join(rows)


def index_pretrained_models(
    *,
    root: str | Path | None = None,
    workspace_root: str | Path | None = None,
    models: Iterable[PretrainedModelSpec] = DEFAULT_PRETRAINED_MODELS,
) -> dict[str, object]:
    resolved_root = Path(root).expanduser().resolve() if root is not None else default_pretrained_root()
    resolved_workspace = (
        Path(workspace_root).expanduser().resolve() if workspace_root is not None else globals()["workspace_root"]()
    )
    resolved_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for spec in models:
        linked_from = _link_first_fallback(spec, resolved_root, resolved_workspace)
        path = resolved_root / spec.relative_path
        available = _has_payload(path)
        entry = asdict(spec)
        entry.update(
            {
                "path": str(path),
                "status": "available" if available else "missing",
                "size_bytes": _path_size(path),
                "is_symlink": path.is_symlink(),
                "linked_from": str(linked_from) if linked_from is not None else None,
            }
        )
        entries.append(entry)
    available_count = sum(entry["status"] == "available" for entry in entries)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(resolved_root),
        "environment": {
            "GEOSEG_PRETRAINED_ROOT": str(resolved_root),
            "HF_HOME": str(resolved_root / "huggingface"),
            "TORCH_HOME": str(resolved_root / "torch"),
        },
        "summary": {
            "available": available_count,
            "missing": len(entries) - available_count,
            "total": len(entries),
        },
        "models": entries,
    }
    (resolved_root / "index.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (resolved_root / "INDEX.md").write_text(_render_markdown(manifest), encoding="utf-8")
    return manifest
