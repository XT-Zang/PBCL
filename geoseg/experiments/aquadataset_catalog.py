from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from geoseg.datasets.paths import DATASETS_ROOT
from geoseg.pretrained import default_pretrained_root
from geoseg.metrics.aquadataset import AQUADATASET_OVERALL_SCORE_WEIGHTS
from geoseg.runtime.specs import ExperimentSpec


AQUADATASET_ROOT = str(DATASETS_ROOT / "AquaDateset")
AQUADATASET_VERSION = "aquav3-sceneholdout-v1"
AQUADATASET_MEAN = (0.2259130624, 0.2878868692, 0.3214092823, 0.1363257618)
AQUADATASET_STD = (0.2089709094, 0.2542853528, 0.2718809957, 0.1339930134)
AQUADATASET_SPLIT_COUNTS = MappingProxyType({"train": 1734, "val": 432, "test": 1382})
AQUADATASET_SPLIT_HASHES = MappingProxyType(
    {
        "train": "2a039416e8ee73ae31296d1ac505006239f54b07ff5b947b801950d25d0c823c",
        "val": "b409dc3ca697a998f3a814a1c5525b066cb6103e66d50bb2437951c645377783",
        "test": "72edfa9ec561137b1aed94cc96ea803b537cd418ce7e3929ae803d7e6df2f3a7",
    }
)
AQUADATASET_SEEDS = (42, 43, 44)
PBCL_BASE_IDS = frozenset({"unetplusplus", "segformer_b2", "mask2former_swin_t"})
_PRETRAINED_ARTIFACT_PAYLOADS = {
    "vmamba-t-imagenet": (
        "roadscale/vmamba_t_imagenet_official.pth",
        "d855b57224a652bc692804932f88dc3a39b9028e64be99f745a7fd6792ca672a",
    ),
    "vmamba-modeling": (
        "roadscale/vmamba_code/vmamba_official/vmamba.py",
        "841cf8d244890ff74e08557ccb4fb75232b0bb6eed1adca632515b6b02c9137f",
    ),
    "vmamba-scan": (
        "roadscale/vmamba_code/vmamba_official/csms6s.py",
        "b709d303a0b0917d3d1e46acf4b0a5ad01a91ca508d289557b8c87a4432902c5",
    ),
    "vmamba-cross-scan": (
        "roadscale/vmamba_code/vmamba_official/csm_triton.py",
        "6835a3794e41ffbdbfd57d6d5c6c5c2dc8e420e357b5daa22c4ffbc69bb16568",
    ),
    "vmamba-scan-wheel": (
        "roadscale/vmamba_selective_scan_cuda13_sm120.whl",
        "c23c12118af2e323784304949a1e32c997ef0a015c7cb3e52304e57964fe057e",
    ),
    "smp-efficientnet-b4-imagenet": (
        "huggingface/hub/models--smp-hub--efficientnet-b4.imagenet/blobs/"
        "c465ce05723692769f0d069312aaf8fcdf9a6f5153e48127f5b7eae68e4ec1dd",
        "c465ce05723692769f0d069312aaf8fcdf9a6f5153e48127f5b7eae68e4ec1dd",
    ),
    "smp-resnet50-imagenet": (
        "huggingface/hub/models--smp-hub--resnet50.imagenet/blobs/"
        "df1aad85e18536504a4c8597118364e291ff3a9c4b56dd9b3a4900642e4c3a7c",
        "df1aad85e18536504a4c8597118364e291ff3a9c4b56dd9b3a4900642e4c3a7c",
    ),
    "swin-tiny-imagenet": (
        "torch/hub/checkpoints/swin_tiny_patch4_window7_224.pth",
        "9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3",
    ),
    "segformer-b2-ade": (
        "huggingface/hub/models--nvidia--segformer-b2-finetuned-ade-512-512/blobs/"
        "4a7ab8f05afe62dfdd75338b7fc2eb10ad1347bf5ade78ca2109951f5c717b86",
        "4a7ab8f05afe62dfdd75338b7fc2eb10ad1347bf5ade78ca2109951f5c717b86",
    ),
    "upernet-swin-tiny": (
        "huggingface/hub/models--openmmlab--upernet-swin-tiny/blobs/"
        "1970b8c114a49db5a58180de85e323c4d8e4c02729aee9d1398616d79769fa6e",
        "1970b8c114a49db5a58180de85e323c4d8e4c02729aee9d1398616d79769fa6e",
    ),
    "mask2former-swin-tiny": (
        "huggingface/hub/models--facebook--mask2former-swin-tiny-ade-semantic/blobs/"
        "d615cdba267d6647662bec15dd9bb4c366b91a4e354d865d5fbee7ad2f7e61f9",
        "d615cdba267d6647662bec15dd9bb4c366b91a4e354d865d5fbee7ad2f7e61f9",
    ),
    "mask2former-swin-tiny-coco-instance": (
        "huggingface/hub/models--facebook--mask2former-swin-tiny-coco-instance/blobs/"
        "3550e8b21381d082c4d12b06a4b7c5f0f4a530125eee4c5f264b1e5f43dfab80",
        "3550e8b21381d082c4d12b06a4b7c5f0f4a530125eee4c5f264b1e5f43dfab80",
    ),
    "swsl-resnet18": (
        "huggingface/hub/models--timm--resnet18.fb_swsl_ig1b_ft_in1k/blobs/"
        "69834a6388a8f9a929176775935920a8b1925c1a9b834ea4208b6e749b9bc20f",
        "69834a6388a8f9a929176775935920a8b1925c1a9b834ea4208b6e749b9bc20f",
    ),
    "geoseg-stseg-base": (
        "geoseg/stseg_base.pth",
        "00585aef2972df877f53db302eebbb9b9162821a328f8200bf10f758a9d8ed78",
    ),
    "dinov3-vitl16-sat493m": (
        "torch/hub/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        "eadcf0ffc02418b6c22a885ea1a7aaeeef84fbf0f5bb4d0b7d1d36e68a964f48",
    ),
    "hrnetv2-w32-imagenet": (
        "hrnetv2_w32_imagenet_pretrained.pth",
        "dc9eeb4f307e3f9bd6e7a13e546962bd8c578db1ca9caed6f7fff71770beb7bb",
    ),
    "hrformer-s-imagenet": (
        "hrformer/hrt_small_imagenet.pth",
        "095163755e66ae54d73c988bdb9e29e9ffa9d30ecd018cfcf492ac7e4fe76966",
    ),
    "pidnet-s-imagenet": (
        "pidnet/PIDNet_S_ImageNet.pth",
        "f96e2c96b1aca1400a6f54ac41093d98b4817f6008a5d869abe6248551a5f359",
    ),
    "ddrnet23-slim-imagenet": (
        "ddrnet/DDRNet23s_imagenet.pth",
        "f6d03fa29b80e2e648f89a7639eb0c6d6c088e8f70f2cba1b2872b314a5c54b6",
    ),
    "torchgeo-swin-v2-b": (
        "torch/hub/checkpoints/sentinel2_swinb_mi_ms-39c86721.pth",
        "39c86721744331ca21654869fb268f763e23df6a59bc0b79137868a01af4e448",
    ),
    "torchgeo-vit-base-sentinel2-mae": (
        "torchgeo/B13_vitb16_mae_ep99_enc.pth",
        "a3f1cd4f9bbff53a289298518ae8e0ade5214e9f80799c0d852aa1dca7e9bc60",
    ),
}
_PRETRAINED_KEYS_BY_MODEL = {
    "vmamba_fpn": (
        "vmamba-t-imagenet", "vmamba-modeling", "vmamba-scan",
        "vmamba-cross-scan", "vmamba-scan-wheel",
    ),
    "unetplusplus": ("smp-efficientnet-b4-imagenet",),
    "deeplabv3plus": ("smp-resnet50-imagenet",),
    "swinunet": ("swin-tiny-imagenet",),
    "segformer_b2": ("segformer-b2-ade",),
    "upernet_swin_t": ("upernet-swin-tiny",),
    "mask2former_swin_t": ("mask2former-swin-tiny",),
    "mask2former_swin_t_instance": ("mask2former-swin-tiny-coco-instance",),
    "unetformer": ("swsl-resnet18",),
    "abcnet": ("swsl-resnet18",),
    "dcswin_base": ("geoseg-stseg-base",),
}


@dataclass(frozen=True)
class AquaDatasetRecipe:
    experiment_id: str
    family: str
    model_key: str
    loss_kind: str
    pretrained: bool
    encoder_prefixes: tuple[str, ...]
    train_batch_size: int
    accumulate_grad_batches: int
    base_experiment_id: str = ""
    dinov3_adaptation: str = "full"
    dinov3_lora_rank: int = 0


def _recipe(
    experiment_id: str,
    family: str,
    model_key: str,
    loss_kind: str,
    pretrained: bool,
    encoder_prefixes: tuple[str, ...],
    batch: int,
    accumulate: int,
    base: str = "",
    dinov3_adaptation: str = "full",
    dinov3_lora_rank: int = 0,
) -> AquaDatasetRecipe:
    return AquaDatasetRecipe(
        experiment_id,
        family,
        model_key,
        loss_kind,
        pretrained,
        encoder_prefixes,
        batch,
        accumulate,
        base,
        dinov3_adaptation,
        dinov3_lora_rank,
    )


_RECIPES = {
    "unetplusplus": _recipe(
        "unetplusplus",
        "main_table",
        "unetplusplus",
        "binary",
        True,
        ("encoder.",),
        12,
        1,
    ),
    "deeplabv3plus": _recipe(
        "deeplabv3plus",
        "main_table",
        "deeplabv3plus",
        "binary",
        True,
        ("encoder.",),
        12,
        1,
    ),
    "swinunet": _recipe(
        "swinunet",
        "main_table",
        "swinunet",
        "binary",
        True,
        ("patch_embed.", "layers.", "norm."),
        12,
        1,
    ),
    "segformer_b2": _recipe(
        "segformer_b2",
        "main_table",
        "segformer_b2",
        "binary",
        True,
        ("model.segformer.",),
        12,
        1,
    ),
    "upernet_swin_t": _recipe(
        "upernet_swin_t",
        "main_table",
        "upernet_swin_t",
        "binary",
        True,
        ("model.backbone.",),
        12,
        1,
    ),
    "mask2former_swin_t": _recipe(
        "mask2former_swin_t",
        "main_table",
        "mask2former_swin_t",
        "binary",
        True,
        ("model.model.pixel_level_module.encoder.",),
        6,
        2,
    ),
    "a2fpn": _recipe("a2fpn", "main_table", "geoseg_a2fpn", "binary", False, (), 12, 1),
    "unetformer": _recipe(
        "unetformer", "main_table", "unetformer", "binary", True, ("backbone.",), 12, 1
    ),
    "abcnet": _recipe(
        "abcnet", "main_table", "abcnet", "binary", True, ("cp.resnet.",), 12, 1
    ),
    "dcswin_base": _recipe(
        "dcswin_base", "main_table", "dcswin_base", "binary", True, ("backbone.",), 6, 2
    ),
    "unetplusplus_pbcl": _recipe(
        "unetplusplus_pbcl",
        "pbcl_crossarch",
        "unetplusplus",
        "pbcl",
        True,
        ("encoder.",),
        12,
        1,
        "unetplusplus",
    ),
    "segformer_b2_pbcl": _recipe(
        "segformer_b2_pbcl",
        "pbcl_crossarch",
        "segformer_b2",
        "pbcl",
        True,
        ("model.segformer.",),
        12,
        1,
        "segformer_b2",
    ),
    "mask2former_swin_t_pbcl": _recipe(
        "mask2former_swin_t_pbcl",
        "pbcl_crossarch",
        "mask2former_swin_t",
        "pbcl",
        True,
        ("model.model.pixel_level_module.encoder.",),
        6,
        2,
        "mask2former_swin_t",
    ),
    "ssepnet": _recipe(
        "ssepnet",
        "ssepnet",
        "ssepnet",
        "ssepnet",
        True,
        ("hrnet_backbone.", "dinov3_backbone."),
        6,
        2,
    ),
    "ssepnet_hrnet_only": _recipe(
        "ssepnet_hrnet_only",
        "ssepnet",
        "ssepnet_hrnet_only",
        "ssepnet",
        True,
        ("hrnet_backbone.",),
        6,
        2,
    ),
    "ssepnet_dinov3_only": _recipe(
        "ssepnet_dinov3_only",
        "ssepnet",
        "ssepnet_dinov3_only",
        "ssepnet",
        True,
        ("dinov3_backbone.",),
        6,
        2,
    ),
    "ssepnet_base": _recipe(
        "ssepnet_base",
        "ssepnet",
        "ssepnet_base",
        "ssepnet",
        True,
        ("hrnet_backbone.", "dinov3_backbone."),
        6,
        2,
    ),
    "ssepnet_boundary": _recipe(
        "ssepnet_boundary",
        "ssepnet",
        "ssepnet_boundary",
        "ssepnet",
        True,
        ("hrnet_backbone.", "dinov3_backbone."),
        6,
        2,
    ),
    "ssepnet_pbcl": _recipe(
        "ssepnet_pbcl",
        "ssepnet",
        "ssepnet_pbcl",
        "ssepnet",
        True,
        ("hrnet_backbone.", "dinov3_backbone."),
        6,
        2,
    ),
    "ssepnet_fusion_add": _recipe(
        "ssepnet_fusion_add",
        "ssepnet",
        "ssepnet_fusion_add",
        "ssepnet",
        True,
        ("hrnet_backbone.", "dinov3_backbone."),
        6,
        2,
    ),
    "ssepnet_fusion_concat": _recipe(
        "ssepnet_fusion_concat",
        "ssepnet",
        "ssepnet_fusion_concat",
        "ssepnet",
        True,
        ("hrnet_backbone.", "dinov3_backbone."),
        6,
        2,
    ),
    "ssepnet_fusion_gate": _recipe(
        "ssepnet_fusion_gate",
        "ssepnet",
        "ssepnet_fusion_gate",
        "ssepnet",
        True,
        ("hrnet_backbone.", "dinov3_backbone."),
        6,
        2,
    ),
    "hrnet_swin_b_ocr_boundary_pbcl": _recipe(
        "hrnet_swin_b_ocr_boundary_pbcl",
        "context_hybrid",
        "hrnet_swin_b_ocr_boundary_pbcl",
        "hybrid",
        True,
        ("hrnet_backbone.", "context_backbone."),
        4,
        3,
    ),
    "hrnet_swinv2_b_ocr_boundary_pbcl": _recipe(
        "hrnet_swinv2_b_ocr_boundary_pbcl",
        "context_hybrid",
        "hrnet_swinv2_b_ocr_boundary_pbcl",
        "hybrid",
        True,
        ("hrnet_backbone.", "context_backbone."),
        4,
        3,
    ),
    "hrnet_vit_b_ocr_boundary_pbcl": _recipe(
        "hrnet_vit_b_ocr_boundary_pbcl",
        "context_hybrid",
        "hrnet_vit_b_ocr_boundary_pbcl",
        "hybrid",
        True,
        ("hrnet_backbone.", "context_backbone."),
        4,
        3,
    ),
}

AQUADATASET_EXPERIMENTS: Mapping[str, AquaDatasetRecipe] = MappingProxyType(_RECIPES)

_GEOGRAPHIC_HOLDOUT_RECIPES = {
    "vmamba_fpn": _recipe(
        "vmamba_fpn", "geographic_holdout", "vmamba_fpn", "binary", True, ("backbone.",), 6, 2,
    ),
}

_INSTANCE_BASELINE_RECIPES = {
    "mask2former_swin_t_instance": _recipe(
        "mask2former_swin_t_instance",
        "instance_baselines",
        "mask2former_swin_t_instance",
        "mask2former_instance",
        True,
        ("model.model.pixel_level_module.encoder.",),
        6,
        2,
    ),
}

AQUADATASET_INSTANCE_BASELINE_EXPERIMENTS: Mapping[str, AquaDatasetRecipe] = (
    MappingProxyType(_INSTANCE_BASELINE_RECIPES)
)

_CANDIDATE_RECIPES = {
    "ssepnet_layerwise_fpn": _recipe(
        "ssepnet_layerwise_fpn",
        "ssepnet_candidates",
        "ssepnet_layerwise_fpn",
        "ssepnet_candidate",
        True,
        ("hrnet_backbone.", "dinov3_backbone."),
        6,
        2,
    ),
    "ssepnet_layerwise_fpn_dinov3_only": _recipe(
        "ssepnet_layerwise_fpn_dinov3_only",
        "ssepnet_candidates",
        "ssepnet_layerwise_fpn_dinov3_only",
        "ssepnet_candidate",
        True,
        ("dinov3_backbone.",),
        6,
        2,
        "ssepnet_layerwise_fpn",
    ),
    "ssepnet_asymmetric_fpn_ocr": _recipe(
        "ssepnet_asymmetric_fpn_ocr",
        "ssepnet_candidates",
        "ssepnet_asymmetric_fpn_ocr",
        "ssepnet_candidate",
        True,
        ("hrnet_backbone.", "dinov3_backbone."),
        6,
        2,
    ),
    "ssepnet_asymmetric_fpn_ocr_dinov3_only": _recipe(
        "ssepnet_asymmetric_fpn_ocr_dinov3_only",
        "ssepnet_candidates",
        "ssepnet_asymmetric_fpn_ocr_dinov3_only",
        "ssepnet_candidate",
        True,
        ("dinov3_backbone.",),
        6,
        2,
        "ssepnet_asymmetric_fpn_ocr",
    ),
    "ssepnet_dpt_bifpn": _recipe(
        "ssepnet_dpt_bifpn",
        "ssepnet_candidates",
        "ssepnet_dpt_bifpn",
        "ssepnet_candidate",
        True,
        ("hrnet_backbone.", "dinov3_backbone."),
        6,
        2,
    ),
    "ssepnet_dpt_bifpn_dinov3_only": _recipe(
        "ssepnet_dpt_bifpn_dinov3_only",
        "ssepnet_candidates",
        "ssepnet_dpt_bifpn_dinov3_only",
        "ssepnet_candidate",
        True,
        ("dinov3_backbone.",),
        6,
        2,
        "ssepnet_dpt_bifpn",
    ),
    "ssepnet_boundary_cascade": _recipe(
        "ssepnet_boundary_cascade",
        "ssepnet_candidates",
        "ssepnet_boundary_cascade",
        "ssepnet_candidate",
        True,
        ("hrnet_backbone.", "dinov3_backbone."),
        6,
        2,
    ),
    "ssepnet_boundary_cascade_dinov3_only": _recipe(
        "ssepnet_boundary_cascade_dinov3_only",
        "ssepnet_candidates",
        "ssepnet_boundary_cascade_dinov3_only",
        "ssepnet_candidate",
        True,
        ("dinov3_backbone.",),
        6,
        2,
        "ssepnet_boundary_cascade",
    ),
    "ssepnet_deformable_mask2former": _recipe(
        "ssepnet_deformable_mask2former",
        "ssepnet_candidates",
        "ssepnet_deformable_mask2former",
        "deformable_mask2former",
        True,
        ("hrnet_backbone.", "dinov3_backbone."),
        2,
        6,
    ),
    "ssepnet_deformable_mask2former_dinov3_only": _recipe(
        "ssepnet_deformable_mask2former_dinov3_only",
        "ssepnet_candidates",
        "ssepnet_deformable_mask2former_dinov3_only",
        "deformable_mask2former",
        True,
        ("dinov3_backbone.",),
        2,
        6,
        "ssepnet_deformable_mask2former",
    ),
}

AQUADATASET_CANDIDATE_EXPERIMENTS: Mapping[str, AquaDatasetRecipe] = MappingProxyType(
    _CANDIDATE_RECIPES
)

_MATCHED_CONTROL_RECIPES = {
    "ssepnet_asymmetric_fpn_ocr_dinov3_only_bs12": _recipe(
        "ssepnet_asymmetric_fpn_ocr_dinov3_only_bs12",
        "ssepnet_matched_controls",
        "ssepnet_asymmetric_fpn_ocr_dinov3_only",
        "ssepnet_candidate",
        True,
        ("dinov3_backbone.",),
        12,
        1,
        "ssepnet_asymmetric_fpn_ocr_dinov3_only",
    ),
    "ssepnet_layerwise_fpn_dinov3_only_bs12": _recipe(
        "ssepnet_layerwise_fpn_dinov3_only_bs12",
        "ssepnet_matched_controls",
        "ssepnet_layerwise_fpn_dinov3_only",
        "ssepnet_candidate",
        True,
        ("dinov3_backbone.",),
        12,
        1,
        "ssepnet_layerwise_fpn_dinov3_only",
    ),
}

AQUADATASET_MATCHED_CONTROL_EXPERIMENTS: Mapping[str, AquaDatasetRecipe] = (
    MappingProxyType(_MATCHED_CONTROL_RECIPES)
)

_LORA_EFFICIENCY_RECIPES = {
    "ssepnet_dinov3_frozen": _recipe(
        "ssepnet_dinov3_frozen",
        "ssepnet_lora_efficiency",
        "ssepnet",
        "ssepnet",
        True,
        ("hrnet_backbone.",),
        6,
        2,
        "ssepnet",
        dinov3_adaptation="frozen",
    ),
    "ssepnet_dinov3_lora_r3": _recipe(
        "ssepnet_dinov3_lora_r3",
        "ssepnet_lora_efficiency",
        "ssepnet",
        "ssepnet",
        True,
        ("hrnet_backbone.",),
        6,
        2,
        "ssepnet",
        dinov3_adaptation="lora",
        dinov3_lora_rank=3,
    ),
    "ssepnet_dinov3_full": _recipe(
        "ssepnet_dinov3_full",
        "ssepnet_lora_efficiency",
        "ssepnet",
        "ssepnet",
        True,
        ("hrnet_backbone.", "dinov3_backbone."),
        6,
        2,
        "ssepnet",
        dinov3_adaptation="full",
    ),
}

AQUADATASET_LORA_EFFICIENCY_EXPERIMENTS: Mapping[str, AquaDatasetRecipe] = (
    MappingProxyType(_LORA_EFFICIENCY_RECIPES)
)

_OCR_OPTIMIZATION_RECIPES = {
    "ssepnet_dinov3_unified_ocr_full": _recipe(
        "ssepnet_dinov3_unified_ocr_full",
        "ssepnet_ocr_optimization",
        "ssepnet_dinov3_unified_ocr",
        "ssepnet_ocr_optimization",
        True,
        ("dinov3_backbone.",),
        6,
        2,
        dinov3_adaptation="full",
    ),
    "ssepnet_dinov3_unified_ocr_lora_r3": _recipe(
        "ssepnet_dinov3_unified_ocr_lora_r3",
        "ssepnet_ocr_optimization",
        "ssepnet_dinov3_unified_ocr",
        "ssepnet_ocr_optimization",
        True,
        (),
        6,
        2,
        "ssepnet_dinov3_unified_ocr_full",
        dinov3_adaptation="lora",
        dinov3_lora_rank=3,
    ),
    "ssepnet_dinov3_layeraware_ocr_full": _recipe(
        "ssepnet_dinov3_layeraware_ocr_full",
        "ssepnet_ocr_optimization",
        "ssepnet_dinov3_layeraware_ocr",
        "ssepnet_ocr_optimization",
        True,
        ("dinov3_backbone.",),
        6,
        2,
        dinov3_adaptation="full",
    ),
    "ssepnet_dinov3_layeraware_ocr_lora_r3": _recipe(
        "ssepnet_dinov3_layeraware_ocr_lora_r3",
        "ssepnet_ocr_optimization",
        "ssepnet_dinov3_layeraware_ocr",
        "ssepnet_ocr_optimization",
        True,
        (),
        6,
        2,
        "ssepnet_dinov3_layeraware_ocr_full",
        dinov3_adaptation="lora",
        dinov3_lora_rank=3,
    ),
}

AQUADATASET_OCR_OPTIMIZATION_EXPERIMENTS: Mapping[str, AquaDatasetRecipe] = (
    MappingProxyType(_OCR_OPTIMIZATION_RECIPES)
)

_DUAL_FPN_OCR_RECIPES = {
    "ssepnet_dinov3_fpn_ocr_full": _recipe(
        "ssepnet_dinov3_fpn_ocr_full",
        "ssepnet_dual_fpn_ocr",
        "ssepnet_dinov3_fpn_ocr",
        "ssepnet_dual_fpn_ocr",
        True,
        ("dinov3_backbone.",),
        6,
        2,
        dinov3_adaptation="full",
    ),
    "ssepnet_dinov3_fpn_ocr_lora_r3": _recipe(
        "ssepnet_dinov3_fpn_ocr_lora_r3",
        "ssepnet_dual_fpn_ocr",
        "ssepnet_dinov3_fpn_ocr",
        "ssepnet_dual_fpn_ocr",
        True,
        (),
        6,
        2,
        "ssepnet_dinov3_fpn_ocr_full",
        dinov3_adaptation="lora",
        dinov3_lora_rank=3,
    ),
    "ssepnet_dinov3_hrformer_s_fpn_ocr_full": _recipe(
        "ssepnet_dinov3_hrformer_s_fpn_ocr_full",
        "ssepnet_dual_fpn_ocr",
        "ssepnet_dinov3_hrformer_s_fpn_ocr",
        "ssepnet_dual_fpn_ocr",
        True,
        ("dinov3_backbone.", "detail_backbone.encoder."),
        6,
        2,
        "ssepnet_dinov3_fpn_ocr_full",
        dinov3_adaptation="full",
    ),
    "ssepnet_dinov3_hrformer_s_fpn_ocr_lora_r3": _recipe(
        "ssepnet_dinov3_hrformer_s_fpn_ocr_lora_r3",
        "ssepnet_dual_fpn_ocr",
        "ssepnet_dinov3_hrformer_s_fpn_ocr",
        "ssepnet_dual_fpn_ocr",
        True,
        ("detail_backbone.encoder.",),
        6,
        2,
        "ssepnet_dinov3_hrformer_s_fpn_ocr_full",
        dinov3_adaptation="lora",
        dinov3_lora_rank=3,
    ),
    "ssepnet_dinov3_pidnet_s_pd_fpn_ocr_full": _recipe(
        "ssepnet_dinov3_pidnet_s_pd_fpn_ocr_full",
        "ssepnet_dual_fpn_ocr",
        "ssepnet_dinov3_pidnet_s_pd_fpn_ocr",
        "ssepnet_dual_fpn_ocr",
        True,
        ("dinov3_backbone.", "detail_backbone.encoder."),
        6,
        2,
        "ssepnet_dinov3_fpn_ocr_full",
        dinov3_adaptation="full",
    ),
    "ssepnet_dinov3_pidnet_s_pd_fpn_ocr_lora_r3": _recipe(
        "ssepnet_dinov3_pidnet_s_pd_fpn_ocr_lora_r3",
        "ssepnet_dual_fpn_ocr",
        "ssepnet_dinov3_pidnet_s_pd_fpn_ocr",
        "ssepnet_dual_fpn_ocr",
        True,
        ("detail_backbone.encoder.",),
        6,
        2,
        "ssepnet_dinov3_pidnet_s_pd_fpn_ocr_full",
        dinov3_adaptation="lora",
        dinov3_lora_rank=3,
    ),
    "ssepnet_dinov3_ddrnet23_slim_fpn_ocr_full": _recipe(
        "ssepnet_dinov3_ddrnet23_slim_fpn_ocr_full",
        "ssepnet_dual_fpn_ocr",
        "ssepnet_dinov3_ddrnet23_slim_fpn_ocr",
        "ssepnet_dual_fpn_ocr",
        True,
        ("dinov3_backbone.", "detail_backbone.encoder."),
        6,
        2,
        "ssepnet_dinov3_fpn_ocr_full",
        dinov3_adaptation="full",
    ),
    "ssepnet_dinov3_ddrnet23_slim_fpn_ocr_lora_r3": _recipe(
        "ssepnet_dinov3_ddrnet23_slim_fpn_ocr_lora_r3",
        "ssepnet_dual_fpn_ocr",
        "ssepnet_dinov3_ddrnet23_slim_fpn_ocr",
        "ssepnet_dual_fpn_ocr",
        True,
        ("detail_backbone.encoder.",),
        6,
        2,
        "ssepnet_dinov3_ddrnet23_slim_fpn_ocr_full",
        dinov3_adaptation="lora",
        dinov3_lora_rank=3,
    ),
}

AQUADATASET_DUAL_FPN_OCR_EXPERIMENTS: Mapping[str, AquaDatasetRecipe] = (
    MappingProxyType(_DUAL_FPN_OCR_RECIPES)
)

_MULTISCALE_DINOV3_RECIPES = {
    "ssepnet_ms_dinov3_stage_adapter_6_6_6_6": _recipe(
        "ssepnet_ms_dinov3_stage_adapter_6_6_6_6",
        "ssepnet_multiscale_dinov3",
        "ssepnet_ms_dinov3_stage_adapter_6_6_6_6",
        "ssepnet_multiscale_dinov3",
        True,
        ("multiscale_backbone.dinov3_backbone.",),
        12,
        1,
    ),
    "ssepnet_ms_dinov3_poolhier_6_6_6_6": _recipe(
        "ssepnet_ms_dinov3_poolhier_6_6_6_6",
        "ssepnet_multiscale_dinov3",
        "ssepnet_ms_dinov3_poolhier_6_6_6_6",
        "ssepnet_multiscale_dinov3",
        True,
        ("multiscale_backbone.dinov3_backbone.",),
        12,
        1,
        "ssepnet_ms_dinov3_stage_adapter_6_6_6_6",
    ),
    "ssepnet_ms_dinov3_hier_6_6_6_6": _recipe(
        "ssepnet_ms_dinov3_hier_6_6_6_6",
        "ssepnet_multiscale_dinov3",
        "ssepnet_ms_dinov3_hier_6_6_6_6",
        "ssepnet_multiscale_dinov3",
        True,
        ("multiscale_backbone.stages.",),
        12,
        1,
        "ssepnet_ms_dinov3_stage_adapter_6_6_6_6",
    ),
    "ssepnet_ms_dinov3_hier_6_6_8_4": _recipe(
        "ssepnet_ms_dinov3_hier_6_6_8_4",
        "ssepnet_multiscale_dinov3",
        "ssepnet_ms_dinov3_hier_6_6_8_4",
        "ssepnet_multiscale_dinov3",
        True,
        ("multiscale_backbone.stages.",),
        12,
        1,
        "ssepnet_ms_dinov3_hier_6_6_6_6",
    ),
    "ssepnet_ms_dinov3_hier_8_6_6_4": _recipe(
        "ssepnet_ms_dinov3_hier_8_6_6_4",
        "ssepnet_multiscale_dinov3",
        "ssepnet_ms_dinov3_hier_8_6_6_4",
        "ssepnet_multiscale_dinov3",
        True,
        ("multiscale_backbone.stages.",),
        12,
        1,
        "ssepnet_ms_dinov3_hier_6_6_6_6",
    ),
    "ssepnet_ms_dinov3_dynamic_depth": _recipe(
        "ssepnet_ms_dinov3_dynamic_depth",
        "ssepnet_multiscale_dinov3",
        "ssepnet_ms_dinov3_dynamic_depth",
        "ssepnet_multiscale_dinov3",
        True,
        ("multiscale_backbone.stages.",),
        12,
        1,
        "ssepnet_ms_dinov3_hier_8_6_6_4",
    ),
}

AQUADATASET_MULTISCALE_DINOV3_EXPERIMENTS: Mapping[str, AquaDatasetRecipe] = (
    MappingProxyType(_MULTISCALE_DINOV3_RECIPES)
)

_STAGE_ADAPTER_FPN_OCR_MAIN = (
    "ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6"
)
_STAGE_ADAPTER_FPN_OCR_RECIPES = {
    name: _recipe(
        name,
        "ssepnet_stage_adapter_fpn_ocr",
        name,
        "ssepnet_stage_adapter_fpn_ocr",
        True,
        ("multiscale_backbone.dinov3_backbone.",),
        12,
        1,
        (
            "ssepnet_ms_dinov3_stage_adapter_6_6_6_6"
            if name == _STAGE_ADAPTER_FPN_OCR_MAIN
            else _STAGE_ADAPTER_FPN_OCR_MAIN
        ),
    )
    for name in (
        _STAGE_ADAPTER_FPN_OCR_MAIN,
        f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_base",
        f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_boundary",
        f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_pbcl",
        f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_binary_head",
        f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_decoded_p2_boundary",
    )
}

AQUADATASET_STAGE_ADAPTER_FPN_OCR_EXPERIMENTS: Mapping[
    str, AquaDatasetRecipe
] = MappingProxyType(_STAGE_ADAPTER_FPN_OCR_RECIPES)

_STAGE_ADAPTER_FPN_BINARY_FULL = f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_binary_head"
_STAGE_ADAPTER_FPN_BINARY_RECIPES = {
    name: _recipe(
        name,
        "ssepnet_stage_adapter_fpn_ocr",
        name,
        "ssepnet_stage_adapter_fpn_ocr",
        True,
        ("multiscale_backbone.dinov3_backbone.",),
        12,
        1,
        _STAGE_ADAPTER_FPN_BINARY_FULL,
    )
    for name in (
        f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_binary_base",
        f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_binary_boundary",
        f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_binary_pbcl",
        f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_binary_decoded_p2_boundary",
    )
}

AQUADATASET_STAGE_ADAPTER_FPN_BINARY_EXPERIMENTS: Mapping[
    str, AquaDatasetRecipe
] = MappingProxyType(_STAGE_ADAPTER_FPN_BINARY_RECIPES)


_EWC_EXPERIMENT = f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_binary_boundary_ewc"
_EWC_RECIPES = {
    _EWC_EXPERIMENT: _recipe(
        _EWC_EXPERIMENT,
        "ssepnet_loss_baselines",
        f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_binary_boundary",
        "ssepnet_ewc",
        True,
        ("multiscale_backbone.dinov3_backbone.",),
        12,
        1,
        _STAGE_ADAPTER_FPN_BINARY_FULL,
    )
}

AQUADATASET_EWC_EXPERIMENTS: Mapping[str, AquaDatasetRecipe] = MappingProxyType(
    _EWC_RECIPES
)


_STRUCTURAL_LOSS_KINDS = MappingProxyType(
    {
        "cldice": "ssepnet_cldice",
        "cmalis": "ssepnet_cmalis",
        "wprf": "ssepnet_wprf",
    }
)
_STRUCTURAL_LOSS_RECIPES = {
    f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_binary_boundary_{suffix}": _recipe(
        f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_binary_boundary_{suffix}",
        "ssepnet_loss_baselines",
        f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_binary_boundary_{suffix}",
        loss_kind,
        True,
        ("multiscale_backbone.dinov3_backbone.",),
        12,
        1,
        _STAGE_ADAPTER_FPN_BINARY_FULL,
    )
    for suffix, loss_kind in _STRUCTURAL_LOSS_KINDS.items()
}

AQUADATASET_STRUCTURAL_LOSS_EXPERIMENTS: Mapping[str, AquaDatasetRecipe] = (
    MappingProxyType(_STRUCTURAL_LOSS_RECIPES)
)


_BINARY_STAGE_ADAPTER_REFERENCE = _STAGE_ADAPTER_FPN_BINARY_FULL
_BINARY_PROJECTION_RESIZE_CONTROL = (
    "ssepnet_ms_dinov3_projection_resize_fpn_ocr_6_6_6_6_binary_head"
)
_BINARY_STAGE_ADAPTER_ABLATION_RECIPES = {
    _BINARY_PROJECTION_RESIZE_CONTROL: _recipe(
        _BINARY_PROJECTION_RESIZE_CONTROL,
        "ssepnet_stage_adapter_fpn_ocr",
        _BINARY_PROJECTION_RESIZE_CONTROL,
        "ssepnet_stage_adapter_fpn_ocr",
        True,
        ("multiscale_backbone.dinov3_backbone.",),
        12,
        1,
        _BINARY_STAGE_ADAPTER_REFERENCE,
    )
}

AQUADATASET_BINARY_STAGE_ADAPTER_ABLATION_EXPERIMENTS: Mapping[
    str, AquaDatasetRecipe
] = MappingProxyType(_BINARY_STAGE_ADAPTER_ABLATION_RECIPES)


_STAGE_ADAPTER_FPN_BINARY_SENSITIVITY_OVERRIDES = MappingProxyType(
    {
        f"{_STAGE_ADAPTER_FPN_BINARY_FULL}_pbcl_iterations_5": MappingProxyType(
            {"pbcl_iterations": 5}
        ),
        f"{_STAGE_ADAPTER_FPN_BINARY_FULL}_pbcl_iterations_13": MappingProxyType(
            {"pbcl_iterations": 13}
        ),
        f"{_STAGE_ADAPTER_FPN_BINARY_FULL}_pbcl_radius_1": MappingProxyType(
            {"pbcl_band_radius": 1}
        ),
        f"{_STAGE_ADAPTER_FPN_BINARY_FULL}_pbcl_radius_5": MappingProxyType(
            {"pbcl_band_radius": 5}
        ),
        f"{_STAGE_ADAPTER_FPN_BINARY_FULL}_pbcl_margin_0p00": MappingProxyType(
            {"pbcl_margin": 0.0}
        ),
        f"{_STAGE_ADAPTER_FPN_BINARY_FULL}_pbcl_margin_0p05": MappingProxyType(
            {"pbcl_margin": 0.05}
        ),
        f"{_STAGE_ADAPTER_FPN_BINARY_FULL}_pbcl_margin_0p15": MappingProxyType(
            {"pbcl_margin": 0.15}
        ),
        f"{_STAGE_ADAPTER_FPN_BINARY_FULL}_pbcl_weight_0p025": MappingProxyType(
            {"pbcl_weight": 0.025}
        ),
        f"{_STAGE_ADAPTER_FPN_BINARY_FULL}_pbcl_weight_0p05": MappingProxyType(
            {"pbcl_weight": 0.05}
        ),
        f"{_STAGE_ADAPTER_FPN_BINARY_FULL}_pbcl_weight_0p20": MappingProxyType(
            {"pbcl_weight": 0.20}
        ),
    }
)

_STAGE_ADAPTER_FPN_BINARY_SENSITIVITY_RECIPES = {
    name: _recipe(
        name,
        "ssepnet_stage_adapter_fpn_ocr",
        _STAGE_ADAPTER_FPN_BINARY_FULL,
        "ssepnet_stage_adapter_fpn_ocr",
        True,
        ("multiscale_backbone.dinov3_backbone.",),
        12,
        1,
        _STAGE_ADAPTER_FPN_BINARY_FULL,
    )
    for name in _STAGE_ADAPTER_FPN_BINARY_SENSITIVITY_OVERRIDES
}

AQUADATASET_STAGE_ADAPTER_FPN_BINARY_SENSITIVITY_EXPERIMENTS: Mapping[
    str, AquaDatasetRecipe
] = MappingProxyType(_STAGE_ADAPTER_FPN_BINARY_SENSITIVITY_RECIPES)


def _stage_adapter_fpn_ablation_axis(
    experiment_id: str, spec: object
) -> str:
    if experiment_id in _EWC_RECIPES:
        return "loss"
    if getattr(spec, "feature_adapter", "stage_adapter") == "projection_resize":
        return "stage_adapter"
    if experiment_id in _STAGE_ADAPTER_FPN_BINARY_SENSITIVITY_OVERRIDES:
        return "pbcl_hyperparameter"
    if getattr(spec, "boundary_source") == "decoded_p2":
        return "boundary_source"
    if experiment_id == _STAGE_ADAPTER_FPN_BINARY_FULL:
        return "segmentation_head"
    return "supervision"


def _multiscale_v2_encoder_prefixes(name: str) -> tuple[str, ...]:
    if "poolhier" in name:
        return ("multiscale_backbone.dinov3_backbone.",)
    prefixes: list[str] = []
    for stage in range(4):
        base = f"multiscale_backbone.stages.{stage}."
        prefixes.extend(
            (
                f"{base}blocks.",
                f"{base}storage_tokens",
                f"{base}output_norm.",
            )
        )
        if "dynamic_depth" in name and stage < 2:
            prefixes.append(f"{base}extra_blocks.")
    return tuple(prefixes)


_MULTISCALE_DINOV3_V2_RECIPES = {
    name: _recipe(
        name,
        "ssepnet_multiscale_dinov3_v2",
        name,
        "ssepnet_multiscale_dinov3_v2",
        True,
        _multiscale_v2_encoder_prefixes(name),
        12,
        1,
        base,
    )
    for name, base in {
        "ssepnet_ms_dinov3_v2_hier_6_6_6_6": "ssepnet_ms_dinov3_stage_adapter_6_6_6_6",
        "ssepnet_ms_dinov3_v2_hier_6_6_8_4": "ssepnet_ms_dinov3_v2_hier_6_6_6_6",
        "ssepnet_ms_dinov3_v2_hier_8_6_6_4": "ssepnet_ms_dinov3_v2_hier_6_6_6_6",
        "ssepnet_ms_dinov3_v2_dynamic_depth": "ssepnet_ms_dinov3_v2_hier_8_6_6_4",
        "ssepnet_ms_dinov3_v2_poolhier_6_6_12": "ssepnet_ms_dinov3_poolhier_6_6_6_6",
    }.items()
}

AQUADATASET_MULTISCALE_DINOV3_V2_EXPERIMENTS: Mapping[str, AquaDatasetRecipe] = (
    MappingProxyType(_MULTISCALE_DINOV3_V2_RECIPES)
)


def get_aquadataset_recipe(experiment_id: str) -> AquaDatasetRecipe:
    key = str(experiment_id).strip().lower()
    for recipes in (
        AQUADATASET_EXPERIMENTS,
        _GEOGRAPHIC_HOLDOUT_RECIPES,
        AQUADATASET_INSTANCE_BASELINE_EXPERIMENTS,
        AQUADATASET_CANDIDATE_EXPERIMENTS,
        AQUADATASET_MATCHED_CONTROL_EXPERIMENTS,
        AQUADATASET_LORA_EFFICIENCY_EXPERIMENTS,
        AQUADATASET_OCR_OPTIMIZATION_EXPERIMENTS,
        AQUADATASET_DUAL_FPN_OCR_EXPERIMENTS,
        AQUADATASET_MULTISCALE_DINOV3_EXPERIMENTS,
        AQUADATASET_STAGE_ADAPTER_FPN_OCR_EXPERIMENTS,
        AQUADATASET_STAGE_ADAPTER_FPN_BINARY_EXPERIMENTS,
        AQUADATASET_EWC_EXPERIMENTS,
        AQUADATASET_STRUCTURAL_LOSS_EXPERIMENTS,
        AQUADATASET_BINARY_STAGE_ADAPTER_ABLATION_EXPERIMENTS,
        AQUADATASET_STAGE_ADAPTER_FPN_BINARY_SENSITIVITY_EXPERIMENTS,
        AQUADATASET_MULTISCALE_DINOV3_V2_EXPERIMENTS,
    ):
        if key in recipes:
            return recipes[key]
    raise ValueError(f"Unsupported AquaDateset experiment: {experiment_id!r}")


def _loss_spec(recipe: AquaDatasetRecipe) -> dict[str, object]:
    common: dict[str, object] = {
        "name": "BinarySegmentationLoss",
        "main_bce": 1.0,
        "main_dice": 1.0,
        "aux_bce": 0.3,
        "aux_dice": 0.3,
    }
    if recipe.loss_kind == "pbcl":
        common.update(
            name="PBCLBinarySegmentationLoss",
            pbcl_weight=0.1,
            pbcl_band_radius=3,
            pbcl_iterations=9,
            pbcl_threshold=0.5,
            pbcl_margin=0.1,
            pbcl_neighbors=4,
        )
    elif recipe.loss_kind == "ssepnet_ewc":
        common.update(
            name="SSepNetEWCLoss",
            boundary_weight=0.6,
            boundary_kernel=5,
            pbcl_weight=0.0,
            pbcl_band_radius=3,
            pbcl_iterations=9,
            pbcl_threshold=0.5,
            pbcl_margin=0.1,
            pbcl_neighbors=4,
            ewc_w0=10.0,
            ewc_sigma=7.5,
            ewc_power=2.0,
            ewc_class_weight="per_tile_inverse_frequency",
            ewc_single_instance="d1_only",
            fbeta_enabled=False,
            cross_dataset_training=False,
        )
    elif recipe.loss_kind == "ssepnet_cldice":
        common.update(
            name="SSepNetClDiceLoss",
            boundary_weight=0.6,
            boundary_kernel=5,
            pbcl_weight=0.0,
            pbcl_band_radius=3,
            pbcl_iterations=9,
            pbcl_threshold=0.5,
            pbcl_margin=0.1,
            pbcl_neighbors=4,
            cldice_alpha=0.5,
            cldice_iterations=10,
            cldice_smooth=1.0,
            source_commit="47d31a6cc4a8101b1ffe8052994821961e57af9f",
        )
    elif recipe.loss_kind == "ssepnet_cmalis":
        common.update(
            name="SSepNetConstrainedMALISLoss",
            boundary_weight=0.6,
            boundary_kernel=5,
            pbcl_weight=0.0,
            pbcl_band_radius=3,
            pbcl_iterations=9,
            pbcl_threshold=0.5,
            pbcl_margin=0.1,
            pbcl_neighbors=4,
            malis_weight=1.0,
            malis_cpu_workers=6,
            affinity_channels=2,
            source_commit="8386d68c87c0a07691c4c0a11b835e0bb33a93db",
        )
    elif recipe.loss_kind == "ssepnet_wprf":
        common.update(
            name="SSepNetWPRFLoss",
            boundary_weight=0.6,
            boundary_kernel=5,
            pbcl_weight=0.0,
            pbcl_band_radius=3,
            pbcl_iterations=9,
            pbcl_threshold=0.5,
            pbcl_margin=0.1,
            pbcl_neighbors=4,
            wprf_edge_weight=1.0,
            wprf_reach_weight=1.0,
            wprf_warmup_epochs=4,
            wprf_source_batch_size=32,
            wprf_activation_checkpoint=True,
            wprf_grid_stride=4,
            wprf_neighbors=8,
            wprf_scales=[1, 2, 4, 8, 16],
            wprf_gamma=[0.2] * 5,
            wprf_sources_per_scale=32,
            source_commit="418e362b440df08e72ae6da54abeec6559a8f442",
        )
    elif recipe.loss_kind == "mask2former_instance":
        common = {
            "name": "Mask2FormerInstanceLoss",
            "num_queries": 100,
            "class_weight": 2.0,
            "mask_weight": 5.0,
            "dice_weight": 5.0,
            "no_object_weight": 0.1,
            "official_point_sampling": True,
            "use_auxiliary_loss": True,
        }
    elif recipe.loss_kind in {
        "ssepnet",
        "ssepnet_candidate",
        "ssepnet_ocr_optimization",
        "ssepnet_dual_fpn_ocr",
        "ssepnet_multiscale_dinov3",
        "ssepnet_stage_adapter_fpn_ocr",
        "ssepnet_multiscale_dinov3_v2",
        "hybrid",
    }:
        boundary_weight, pbcl_weight = 0.6, 0.1
        if recipe.loss_kind == "ssepnet":
            from geoseg.models.ssepnet import get_ssepnet_variant

            variant = get_ssepnet_variant(recipe.model_key)
            boundary_weight = 0.6 if variant.boundary_enabled else 0.0
            pbcl_weight = variant.pbcl_weight
        elif recipe.loss_kind == "ssepnet_stage_adapter_fpn_ocr":
            from geoseg.models.ssepnet import get_ssepnet_stage_adapter_fpn_ocr

            variant = get_ssepnet_stage_adapter_fpn_ocr(recipe.model_key)
            boundary_weight = 0.6 if variant.boundary_enabled else 0.0
            pbcl_weight = float(variant.pbcl_weight)
        common.update(
            name="SSepNetLoss",
            boundary_weight=boundary_weight,
            boundary_kernel=5,
            pbcl_weight=pbcl_weight,
            pbcl_band_radius=3,
            pbcl_iterations=9,
            pbcl_threshold=0.5,
            pbcl_margin=0.1,
            pbcl_neighbors=4,
        )
        sensitivity_overrides = _STAGE_ADAPTER_FPN_BINARY_SENSITIVITY_OVERRIDES.get(
            recipe.experiment_id
        )
        if sensitivity_overrides is not None:
            common.update(sensitivity_overrides)
    elif recipe.loss_kind == "deformable_mask2former":
        common.update(
            name="DeformableMask2FormerLoss",
            num_queries=64,
            class_weight=2.0,
            mask_weight=5.0,
            dice_weight=5.0,
            no_object_weight=0.1,
            auxiliary_weight=0.5,
            boundary_weight=0.6,
            boundary_kernel=5,
            pbcl_weight=0.1,
            pbcl_band_radius=3,
            pbcl_iterations=9,
            pbcl_threshold=0.5,
            pbcl_margin=0.1,
            pbcl_neighbors=4,
        )
    return common


def build_aquadataset_spec(
    experiment_id: str,
    seed: int,
    *,
    source_config: str | Path | None = None,
) -> ExperimentSpec:
    recipe = get_aquadataset_recipe(experiment_id)
    seed = int(seed)
    if seed not in AQUADATASET_SEEDS:
        raise ValueError(
            f"AquaDateset seed must be one of {AQUADATASET_SEEDS}, got {seed}"
        )
    loss_spec = _loss_spec(recipe)
    is_structural_loss = recipe.loss_kind in {
        "ssepnet_cldice",
        "ssepnet_cmalis",
        "ssepnet_wprf",
    }
    ssepnet_variant = None
    active_backbones: tuple[str, ...] = ()
    head_input_channels: int | None = None
    boundary_head_enabled: bool | None = None
    requires_instance_supervision = recipe.loss_kind in {
        "pbcl",
        "hybrid",
        "mask2former_instance",
        "ssepnet_cmalis",
    }
    candidate_spec = None
    ocr_optimization_spec = None
    dual_fpn_ocr_spec = None
    multiscale_dinov3_spec = None
    stage_adapter_fpn_ocr_spec = None
    multiscale_dinov3_v2_spec = None
    if recipe.family in {
        "ssepnet_stage_adapter_fpn_ocr",
        "ssepnet_loss_baselines",
    }:
        from geoseg.models.ssepnet import get_ssepnet_stage_adapter_fpn_ocr

        stage_adapter_fpn_ocr_spec = get_ssepnet_stage_adapter_fpn_ocr(
            recipe.model_key
        )
        active_backbones = stage_adapter_fpn_ocr_spec.active_backbones
        head_input_channels = int(stage_adapter_fpn_ocr_spec.decoder_channels)
        boundary_head_enabled = bool(stage_adapter_fpn_ocr_spec.boundary_enabled)
        requires_instance_supervision = recipe.loss_kind == "ssepnet_cmalis" or (
            float(loss_spec.get("pbcl_weight", 0.0)) > 0.0
        )
    elif recipe.family == "ssepnet_multiscale_dinov3":
        from geoseg.models.ssepnet import get_ssepnet_multiscale_dinov3

        multiscale_dinov3_spec = get_ssepnet_multiscale_dinov3(recipe.model_key)
        active_backbones = multiscale_dinov3_spec.active_backbones
        head_input_channels = 128
        boundary_head_enabled = True
        requires_instance_supervision = True
    elif recipe.family == "ssepnet_multiscale_dinov3_v2":
        from geoseg.models.ssepnet import get_ssepnet_multiscale_dinov3_v2

        multiscale_dinov3_v2_spec = get_ssepnet_multiscale_dinov3_v2(
            recipe.model_key
        )
        active_backbones = multiscale_dinov3_v2_spec.active_backbones
        head_input_channels = 128
        boundary_head_enabled = True
        requires_instance_supervision = True
    elif recipe.family == "ssepnet_dual_fpn_ocr":
        from geoseg.models.ssepnet import get_ssepnet_dual_fpn_ocr

        dual_fpn_ocr_spec = get_ssepnet_dual_fpn_ocr(recipe.model_key)
        active_backbones = dual_fpn_ocr_spec.active_backbones
        head_input_channels = int(dual_fpn_ocr_spec.decoder_channels)
        boundary_head_enabled = True
        requires_instance_supervision = True
    elif recipe.family == "ssepnet_ocr_optimization":
        from geoseg.models.ssepnet import get_ssepnet_ocr_optimization

        ocr_optimization_spec = get_ssepnet_ocr_optimization(recipe.model_key)
        active_backbones = ("dinov3",)
        head_input_channels = int(ocr_optimization_spec.head_input_channels)
        boundary_head_enabled = True
        requires_instance_supervision = True
    elif recipe.family in {"ssepnet", "ssepnet_lora_efficiency"}:
        from geoseg.models.ssepnet import get_ssepnet_variant

        ssepnet_variant = get_ssepnet_variant(recipe.model_key)
        if ssepnet_variant.branch_ablation == "hrnet_only":
            active_backbones = ("hrnet",)
            head_input_channels = 480
        elif ssepnet_variant.branch_ablation == "dinov3_only":
            active_backbones = ("dinov3",)
            head_input_channels = 352
        else:
            active_backbones = ("hrnet", "dinov3")
            head_input_channels = (
                800 if ssepnet_variant.fusion_mode in {"concat", "gate_concat"} else 480
            )
        boundary_head_enabled = bool(ssepnet_variant.boundary_enabled)
        requires_instance_supervision = ssepnet_variant.pbcl_weight > 0.0
    elif recipe.family in {"ssepnet_candidates", "ssepnet_matched_controls"}:
        from geoseg.models.ssepnet import get_ssepnet_candidate

        candidate_spec = get_ssepnet_candidate(recipe.model_key)
        active_backbones = candidate_spec.active_backbones
        boundary_head_enabled = True
        requires_instance_supervision = True
    needs_instances = (
        requires_instance_supervision or recipe.experiment_id in PBCL_BASE_IDS
    )
    needs_wprf = recipe.loss_kind == "ssepnet_wprf"
    if recipe.family in {
        "ssepnet",
        "ssepnet_candidates",
        "ssepnet_matched_controls",
        "ssepnet_lora_efficiency",
        "ssepnet_ocr_optimization",
        "ssepnet_dual_fpn_ocr",
        "ssepnet_multiscale_dinov3",
        "ssepnet_stage_adapter_fpn_ocr",
        "ssepnet_loss_baselines",
        "ssepnet_multiscale_dinov3_v2",
    }:
        key_by_backbone = {
            "hrnet": "hrnetv2-w32-imagenet",
            "dinov3": "dinov3-vitl16-sat493m",
            "hrformer_s": "hrformer-s-imagenet",
            "pidnet_s_pd": "pidnet-s-imagenet",
            "ddrnet23_slim": "ddrnet23-slim-imagenet",
        }
        pretrained_keys = tuple(key_by_backbone[name] for name in active_backbones)
    elif recipe.experiment_id == "hrnet_swin_b_ocr_boundary_pbcl":
        pretrained_keys = ("geoseg-stseg-base", "hrnetv2-w32-imagenet")
    elif recipe.experiment_id == "hrnet_swinv2_b_ocr_boundary_pbcl":
        pretrained_keys = ("torchgeo-swin-v2-b", "hrnetv2-w32-imagenet")
    elif recipe.experiment_id == "hrnet_vit_b_ocr_boundary_pbcl":
        pretrained_keys = ("torchgeo-vit-base-sentinel2-mae", "hrnetv2-w32-imagenet")
    else:
        pretrained_keys = _PRETRAINED_KEYS_BY_MODEL.get(recipe.model_key, ())
    pretrained_root = default_pretrained_root()
    pretrained_artifacts = {
        key: {
            "path": str(
                (pretrained_root / _PRETRAINED_ARTIFACT_PAYLOADS[key][0]).resolve()
            ),
            "sha256": _PRETRAINED_ARTIFACT_PAYLOADS[key][1],
        }
        for key in pretrained_keys
    }
    run_name = f"{recipe.experiment_id}_seed{seed}"
    output_root = Path("train_out/AquaDateset")
    aquadataset_workspace_root = os.environ.get(
        "GEOSEG_AQUADATASET_WORKSPACE_ROOT"
    )
    if aquadataset_workspace_root:
        output_root = (
            Path(aquadataset_workspace_root).expanduser().resolve()
            / "train_out"
            / "AquaDateset"
        )
    elif recipe.family == "ssepnet_loss_baselines":
        ewc_workspace_root = os.environ.get("GEOSEG_EWC_WORKSPACE_ROOT")
        structural_workspace_root = os.environ.get(
            "GEOSEG_STRUCTURAL_WORKSPACE_ROOT"
        )
        selected_workspace = (
            structural_workspace_root
            if is_structural_loss
            else ewc_workspace_root
        )
        if selected_workspace:
            output_root = (
                Path(selected_workspace).expanduser().resolve()
                / "train_out"
                / "AquaDateset"
            )
    output_dir = str(
        output_root / recipe.family / recipe.experiment_id / f"seed{seed}"
    )
    return ExperimentSpec(
        model={
            "key": recipe.model_key,
            "in_channels": 4,
            "num_classes": 1,
            "pretrained": recipe.pretrained,
            "freeze_backbones": False,
            "activation_checkpoint": recipe.family
            in {
                "ssepnet",
                "ssepnet_candidates",
                "ssepnet_matched_controls",
                "ssepnet_lora_efficiency",
                "ssepnet_ocr_optimization",
                "ssepnet_dual_fpn_ocr",
                "ssepnet_multiscale_dinov3",
                "ssepnet_stage_adapter_fpn_ocr",
                "ssepnet_loss_baselines",
                "ssepnet_multiscale_dinov3_v2",
                "context_hybrid",
            },
            "pretrained_keys": list(pretrained_keys),
            "pretrained_sha256": {
                key: artifact["sha256"]
                for key, artifact in pretrained_artifacts.items()
            },
            "pretrained_artifacts": pretrained_artifacts,
            **(
                {
                    "architecture_revision": (
                        "stage_adapter_standard_fpn_ablation_v1"
                        if stage_adapter_fpn_ocr_spec.feature_adapter
                        == "stage_adapter"
                        else "flat_dinov3_projection_resize_fpn_control_v1"
                    ),
                    "multiscale_architecture": (
                        stage_adapter_fpn_ocr_spec.feature_adapter
                    ),
                    "feature_adapter": stage_adapter_fpn_ocr_spec.feature_adapter,
                    "stage_depths": list(stage_adapter_fpn_ocr_spec.stage_depths),
                    "stage_output_layers": list(
                        stage_adapter_fpn_ocr_spec.output_layers
                    ),
                    "semantic_scales": [4, 8, 16, 32],
                    "stage_dims": list(stage_adapter_fpn_ocr_spec.stage_dims),
                    "dropout": float(stage_adapter_fpn_ocr_spec.dropout),
                    "active_backbones": ["dinov3"],
                    "decoder_channels": int(
                        stage_adapter_fpn_ocr_spec.decoder_channels
                    ),
                    "ocr_channels": int(stage_adapter_fpn_ocr_spec.ocr_channels),
                    "head_input_channels": int(head_input_channels),
                    "segmentation_head": (
                        "standard_fpn_single_ocr_boundary_v1"
                        if stage_adapter_fpn_ocr_spec.head_kind == "ocr"
                        else "standard_fpn_binary_boundary_v1"
                    ),
                    "semantic_fusion": "additive_top_down_p5_to_p2_then_concat",
                    "head_lineage": (
                        "asymmetric_fpn_ocr_decoder_without_soft_layer_mixing"
                        if stage_adapter_fpn_ocr_spec.head_kind == "ocr"
                        else "matched_standard_fpn_binary"
                    ),
                    "ocr_scale": (
                        4 if stage_adapter_fpn_ocr_spec.head_kind == "ocr" else None
                    ),
                    "auxiliary_scale": 16,
                    "boundary_scale": (
                        4 if stage_adapter_fpn_ocr_spec.boundary_enabled else None
                    ),
                    "boundary_feature_source": (
                        stage_adapter_fpn_ocr_spec.boundary_source
                        if stage_adapter_fpn_ocr_spec.boundary_enabled
                        else None
                    ),
                    "head_normalization": (
                        "group_norm_fpn_batch_norm_ocr"
                        if stage_adapter_fpn_ocr_spec.head_kind == "ocr"
                        else "group_norm_fpn_binary"
                    ),
                    "soft_layer_mixing": False,
                    "boundary_head_enabled": bool(
                        stage_adapter_fpn_ocr_spec.boundary_enabled
                    ),
                    "pbcl_weight": float(loss_spec["pbcl_weight"]),
                    "pbcl_band_radius": int(loss_spec["pbcl_band_radius"]),
                    "pbcl_iterations": int(loss_spec["pbcl_iterations"]),
                    "pbcl_threshold": float(loss_spec["pbcl_threshold"]),
                    "pbcl_margin": float(loss_spec["pbcl_margin"]),
                    "pbcl_neighbors": int(loss_spec["pbcl_neighbors"]),
                    "ewc_enabled": recipe.loss_kind == "ssepnet_ewc",
                    "affinity_channels": int(
                        stage_adapter_fpn_ocr_spec.affinity_channels
                    ),
                    "training_only_affinity_head": bool(
                        stage_adapter_fpn_ocr_spec.affinity_channels
                    ),
                    "ewc_w0": loss_spec.get("ewc_w0"),
                    "ewc_sigma": loss_spec.get("ewc_sigma"),
                    "ewc_power": loss_spec.get("ewc_power"),
                    "requires_ewc_supervision": recipe.loss_kind
                    == "ssepnet_ewc",
                    "requires_instance_supervision": (
                        requires_instance_supervision
                    ),
                    "supervision_variant": (
                        "boundary_ewc"
                        if recipe.loss_kind == "ssepnet_ewc"
                        else "boundary_cldice"
                        if recipe.loss_kind == "ssepnet_cldice"
                        else "boundary_cmalis"
                        if recipe.loss_kind == "ssepnet_cmalis"
                        else "boundary_wprf"
                        if recipe.loss_kind == "ssepnet_wprf"
                        else "boundary_pbcl"
                        if stage_adapter_fpn_ocr_spec.boundary_enabled
                        and float(loss_spec["pbcl_weight"]) > 0.0
                        else (
                            "boundary"
                            if stage_adapter_fpn_ocr_spec.boundary_enabled
                            else (
                                "pbcl"
                                if float(loss_spec["pbcl_weight"]) > 0.0
                                else "base"
                            )
                        )
                    ),
                    "ablation_axis": (
                        _stage_adapter_fpn_ablation_axis(
                            recipe.experiment_id,
                            stage_adapter_fpn_ocr_spec,
                        )
                    ),
                    "physical_branch_pruning": True,
                    "dinov3_adaptation": "full",
                    "dinov3_lora_rank": 0,
                    "dinov3_lora_alpha": None,
                    "comparison_axes": [
                        _stage_adapter_fpn_ablation_axis(
                            recipe.experiment_id,
                            stage_adapter_fpn_ocr_spec,
                        )
                    ],
                }
                if stage_adapter_fpn_ocr_spec is not None
                else {}
            ),
            **(
                {
                    "architecture_revision": "trainable_multiscale_dinov3_v2_distill",
                    "multiscale_architecture": multiscale_dinov3_v2_spec.architecture,
                    "stage_depths": list(multiscale_dinov3_v2_spec.stage_depths),
                    "semantic_scales": [4, 8, 16, 32],
                    "stage_dims": list(multiscale_dinov3_v2_spec.stage_dims),
                    "stage_heads": list(multiscale_dinov3_v2_spec.stage_heads),
                    "window_size": int(multiscale_dinov3_v2_spec.window_size),
                    "dynamic_base_depths": (
                        list(multiscale_dinov3_v2_spec.dynamic_base_depths)
                        if multiscale_dinov3_v2_spec.dynamic_base_depths is not None
                        else None
                    ),
                    "pool_stage_depths": (
                        list(multiscale_dinov3_v2_spec.pool_stage_depths)
                        if multiscale_dinov3_v2_spec.pool_stage_depths is not None
                        else None
                    ),
                    "teacher_taps": list(multiscale_dinov3_v2_spec.teacher_taps),
                    "feature_loss_weights": list(
                        multiscale_dinov3_v2_spec.feature_loss_weights
                    ),
                    "pretrained_coverage_policy": multiscale_dinov3_v2_spec.pretrained_policy,
                    "active_backbones": ["dinov3"],
                    "decoder_channels": 128,
                    "ocr_channels": 256,
                    "head_input_channels": 128,
                    "segmentation_head": "p4_anchored_gated_pyramid_ocr_boundary_v2",
                    "semantic_fusion": "p4_anchor_plus_gated_p5_v2",
                    "p5_gate_init_bias": -1.0,
                    "p3_gate_init_bias": -1.0,
                    "p2_detail_gate_init_bias": -2.0,
                    "boundary_head_enabled": True,
                    "requires_instance_supervision": True,
                    "physical_branch_pruning": True,
                    "dinov3_adaptation": "full",
                    "dinov3_lora_rank": 0,
                    "dinov3_lora_alpha": None,
                    "distillation": {
                        "enabled": True,
                        "teacher": "dinov3-vitl16-sat493m",
                        "teacher_microbatch": 3,
                        "feature_cosine_weight": 0.75,
                        "feature_smooth_l1_weight": 0.25,
                        "freeze_student_blocks_epochs": 3,
                        "lambda_start": 1.0,
                        "lambda_end": 0.1,
                        "lambda_decay_end_epoch": 19,
                        "routing_budget_target": 0.5,
                        "routing_budget_weight": 0.02,
                        "routing_entropy_weight": 0.005,
                        "routing_entropy_end_epoch": 19,
                    },
                }
                if multiscale_dinov3_v2_spec is not None
                else {}
            ),
            **(
                {
                    "architecture_revision": "trainable_multiscale_dinov3_v3",
                    "multiscale_architecture": multiscale_dinov3_spec.architecture,
                    "stage_depths": list(multiscale_dinov3_spec.stage_depths),
                    "stage_output_layers": list(multiscale_dinov3_spec.output_layers),
                    "semantic_scales": [4, 8, 16, 32],
                    "stage_dims": list(multiscale_dinov3_spec.stage_dims),
                    "stage_heads": list(multiscale_dinov3_spec.stage_heads),
                    "window_size": int(multiscale_dinov3_spec.window_size),
                    "dynamic_base_depths": (
                        list(multiscale_dinov3_spec.dynamic_base_depths)
                        if multiscale_dinov3_spec.dynamic_base_depths is not None
                        else None
                    ),
                    "pretrained_coverage_policy": (
                        multiscale_dinov3_spec.pretrained_policy
                    ),
                    "active_backbones": ["dinov3"],
                    "decoder_channels": 128,
                    "ocr_channels": 256,
                    "head_input_channels": 128,
                    "segmentation_head": (
                        "semantic_anchor_gated_pyramid_ocr_boundary_v1"
                    ),
                    "semantic_fusion": "balanced_p4_p5_then_gated_p3_p2",
                    "ocr_scale": 8,
                    "auxiliary_scale": 16,
                    "boundary_scale": 4,
                    "head_normalization": "group_norm",
                    "p4_p5_initial_weights": [0.5, 0.5],
                    "p3_gate_init_bias": -1.0,
                    "p2_detail_gate_init_bias": -2.0,
                    "boundary_head_enabled": True,
                    "requires_instance_supervision": True,
                    "physical_branch_pruning": True,
                    "dinov3_adaptation": "full",
                    "dinov3_lora_rank": 0,
                    "dinov3_lora_alpha": None,
                    "comparison_axes": [
                        "multiscale_architecture",
                        "stage_depths",
                    ],
                }
                if multiscale_dinov3_spec is not None
                else {}
            ),
            **(
                {
                    "architecture_revision": "dinov3_detail_fpn_ocr_v1",
                    "detail_backbone": dual_fpn_ocr_spec.detail_backbone,
                    "active_backbones": list(active_backbones),
                    "detail_scales": [4, 8],
                    "semantic_scales": [4, 8, 16, 32],
                    "decoder_channels": int(dual_fpn_ocr_spec.decoder_channels),
                    "ocr_channels": int(dual_fpn_ocr_spec.ocr_channels),
                    "head_input_channels": int(head_input_channels),
                    "boundary_head_enabled": True,
                    "requires_instance_supervision": True,
                    "physical_branch_pruning": True,
                    "fusion_anchor": "dinov3",
                    "fusion_formula": "semantic_plus_gated_detail_residual",
                    "fusion_gate_init_bias": -2.0,
                    "initialization_contract": "shared_dino_fpn_ocr_seed_aligned_v1",
                    "detail_encoder_optimizer_prefix": (
                        "detail_backbone.encoder."
                        if dual_fpn_ocr_spec.detail_backbone != "none"
                        else None
                    ),
                    "pretrained_coverage_policy": (
                        "strict_except_official_pidnet_d_branch_absent"
                        if dual_fpn_ocr_spec.detail_backbone == "pidnet_s_pd"
                        else "strict_active_modules"
                    ),
                    "pretrained_random_init_modules": (
                        ["layer3_d", "layer4_d", "layer5_d", "diff3", "diff4"]
                        if dual_fpn_ocr_spec.detail_backbone == "pidnet_s_pd"
                        else []
                    ),
                    "dinov3_adaptation": recipe.dinov3_adaptation,
                    "dinov3_lora_rank": int(recipe.dinov3_lora_rank),
                    "dinov3_lora_alpha": (
                        float(recipe.dinov3_lora_rank)
                        if recipe.dinov3_adaptation == "lora"
                        else None
                    ),
                    "comparison_axes": ["detail_backbone", "dinov3_adaptation"],
                }
                if dual_fpn_ocr_spec is not None
                else {}
            ),
            **(
                {
                    "physical_pruning": True,
                    "architecture_revision": "physical_ablation_v2",
                    "active_backbones": list(active_backbones),
                    "head_input_channels": int(head_input_channels),
                    "boundary_head_enabled": bool(boundary_head_enabled),
                    "requires_instance_supervision": bool(
                        requires_instance_supervision
                    ),
                }
                if recipe.family in {"ssepnet", "ssepnet_lora_efficiency"}
                else {}
            ),
            **(
                {
                    "architecture_revision": "dinov3_ocr_optimization_v1",
                    "ocr_adapter": ocr_optimization_spec.adapter,
                    "active_backbones": ["dinov3"],
                    "head_input_channels": int(
                        ocr_optimization_spec.head_input_channels
                    ),
                    "layer_projection_channels": int(
                        ocr_optimization_spec.layer_projection_channels
                    ),
                    "boundary_head_enabled": True,
                    "requires_instance_supervision": True,
                    "physical_branch_pruning": True,
                    "dinov3_adaptation": recipe.dinov3_adaptation,
                    "dinov3_lora_rank": int(recipe.dinov3_lora_rank),
                    "dinov3_lora_alpha": (
                        float(recipe.dinov3_lora_rank)
                        if recipe.dinov3_adaptation == "lora"
                        else None
                    ),
                    "comparison_axes": ["ocr_adapter", "dinov3_adaptation"],
                }
                if ocr_optimization_spec is not None
                else {}
            ),
            **(
                {
                    "architecture_revision": "dinov3_adaptation_control_v1",
                    "dinov3_adaptation": recipe.dinov3_adaptation,
                    "dinov3_lora_rank": int(recipe.dinov3_lora_rank),
                    "dinov3_lora_alpha": (
                        float(recipe.dinov3_lora_rank)
                        if recipe.dinov3_adaptation == "lora"
                        else None
                    ),
                    "comparison_axis": "dinov3_adaptation",
                }
                if recipe.family == "ssepnet_lora_efficiency"
                else {}
            ),
            **(
                {
                    "architecture_revision": "dual_backbone_candidates_v1",
                    "candidate_architecture": candidate_spec.architecture,
                    "branch_ablation": candidate_spec.branch_ablation,
                    "base_candidate": candidate_spec.base_candidate,
                    "active_backbones": list(active_backbones),
                    "decoder_channels": int(candidate_spec.decoder_channels),
                    "num_queries": int(candidate_spec.num_queries),
                    "boundary_head_enabled": True,
                    "requires_instance_supervision": True,
                    "physical_branch_pruning": candidate_spec.branch_ablation
                    == "dinov3_only",
                }
                if candidate_spec is not None
                else {}
            ),
            **(
                {
                    "architecture_revision": "official_mask2former_coco_instance_v1",
                    "instance_prediction": True,
                    "num_queries": 100,
                    "pretrained_repo_id": (
                        "facebook/mask2former-swin-tiny-coco-instance"
                    ),
                    "pretrained_revision": (
                        "22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6"
                    ),
                    "input_stem_adaptation": "rgb_to_bgr_plus_nir_rgb_mean",
                    "allowed_reinitialized_modules": ["class_predictor"],
                    "transformers_compatibility": {
                        "legacy_relative_position_index": "ignored_nonpersistent_buffer",
                        "checkpoint_absent_final_swin_layernorm": "identity",
                        "class_dependent_empty_weight": "reinitialized_for_one_class",
                    },
                    "requires_instance_supervision": True,
                    "boundary_head_enabled": False,
                    "pbcl_weight": 0.0,
                }
                if recipe.family == "instance_baselines"
                else {}
            ),
        },
        data={
            "dataset": "aquadataset",
            "root": AQUADATASET_ROOT,
            "dataset_version": AQUADATASET_VERSION,
            "splits": {"train": "train", "val": "val", "test": "test"},
            "split_counts": dict(AQUADATASET_SPLIT_COUNTS),
            "split_hashes": dict(AQUADATASET_SPLIT_HASHES),
            "bands": ["B", "G", "R", "NIR"],
            "band_indices": [0, 1, 2, 3],
            "image_size": 512,
            "raw_clip_min": 100.0,
            "raw_clip_max": 60000.0,
            "raw_offset": 100.0,
            "raw_scale": 59900.0,
            "normalize_mean": list(AQUADATASET_MEAN),
            "normalize_std": list(AQUADATASET_STD),
            "normalization_pixel_count_per_channel": 454_557_696,
            "instance_target_root": "instance_targets/pbcl_gap5"
            if needs_instances
            else None,
            "ewc_weight_root": "instance_targets/ewc_v1"
            if recipe.loss_kind == "ssepnet_ewc"
            else None,
            "wprf_target_root": "instance_targets/wprf_matched_v1"
            if needs_wprf
            else None,
        },
        loader={
            "train_batch_size": recipe.train_batch_size,
            "val_batch_size": 8,
            "test_batch_size": 8,
            "train_workers": 4,
            "eval_workers": 2,
            "pin_memory": True,
            "drop_last": True,
            "balanced_sampler": True,
            "persistent_workers": False,
            "prefetch_factor": None,
            "collate": "aquav3" if needs_instances or needs_wprf else "default",
            "sampler_generator_stream": 1,
            "loader_generator_streams": {"train": 2, "val": 3, "test": 4},
        },
        optimization={
            "loss": loss_spec,
            "optimizer": {
                "name": "AdamW",
                "base_lr": 2e-4,
                "encoder_lr": 2e-5,
                "weight_decay": 1e-4,
                "encoder_prefixes": list(recipe.encoder_prefixes),
                "no_decay_policy": "bias_norm_position_token",
            },
            "scheduler": {
                "name": "warmup_cosine",
                "warmup_epochs": 3,
                "min_lr_ratio": 0.02,
                "schedule_epochs": 40,
            },
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": recipe.accumulate_grad_batches,
        },
        runtime={
            "seed": seed,
            "max_epochs": 40,
            "precision": "bf16-mixed",
            "devices": "auto",
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": 0,
            "compile": False,
            "channels_last": False,
            "cudnn_benchmark": False,
            "ema": False,
            "random_scale_crop": False,
        },
        checkpoint={
            "policy": "three_best_validation_v1",
            "monitors": {
                "region": "val_fg_iou",
                "structure": "val_pq",
                "overall": "val_overall_score",
            },
            "primary_role": "overall",
            "monitor": "val_overall_score",
            "mode": "max",
            "save_top_k": 1,
            "every_n_epochs": 0,
            "start_epoch": 10,
            "save_last": False,
            "declared_save_last": False,
            "selection": "strict_improvement_earliest_tie",
            "save_weights_only": True,
            "deduplicate_same_epoch": True,
            "overall_score": {
                "name": "region_boundary_object_v1",
                "weights": dict(AQUADATASET_OVERALL_SCORE_WEIGHTS),
                "error_terms": ["merge_rate", "split_rate", "count_nmae"],
                "error_clip": [0.0, 1.0],
            },
        },
        metrics={
            "binary": True,
            "threshold": 0.5,
            "deadhesion_validation": True,
            "boundary_width": 2,
            "boundary_tolerance": 2,
            "min_overlap_pixels": 8,
            "min_overlap_ratio": 0.05,
            "instance_segmentation": recipe.family == "instance_baselines",
            "instance_score_threshold": 0.5,
            "instance_mask_threshold": 0.5,
            "instance_ap_score_threshold": None,
            "instance_num_queries": 100,
            "prediction_kind": (
                "query_instances"
                if recipe.family == "instance_baselines"
                else "connected_components"
            ),
        },
        artifacts={
            "run_name": run_name,
            "weights_path": output_dir,
            "log_name": f"aquadataset/{recipe.family}/{recipe.experiment_id}/seed{seed}",
            "save_val_predictions": False,
            "val_prediction_dir": f"{output_dir}/val_predictions",
            "async_writes": False,
        },
        provenance={
            "schema_version": 1,
            "campaign": "aquadataset",
            "family": recipe.family,
            "experiment_id": recipe.experiment_id,
            "model_key": recipe.model_key,
            "treatment": recipe.loss_kind,
            "base_experiment_id": recipe.base_experiment_id,
            "training_protocol": (
                "uniform_40ep_instance_baseline_v1"
                if recipe.family == "instance_baselines"
                else (
                "uniform_40ep_structural_loss_comparison_v1"
                if is_structural_loss
                else (
                "uniform_40ep_ewc_loss_only_v1"
                if recipe.family == "ssepnet_loss_baselines"
                else (
                "uniform_40ep_matched_control_v1"
                if recipe.family == "ssepnet_matched_controls"
                else (
                    "uniform_40ep_stage_adapter_fpn_binary_pbcl_sensitivity_v1"
                    if recipe.experiment_id
                    in _STAGE_ADAPTER_FPN_BINARY_SENSITIVITY_OVERRIDES
                    else (
                        (
                            "uniform_40ep_binary_stage_adapter_ablation_v1"
                            if recipe.experiment_id
                            in _BINARY_STAGE_ADAPTER_ABLATION_RECIPES
                            else "uniform_40ep_stage_adapter_fpn_ocr_v1"
                        )
                        if recipe.family == "ssepnet_stage_adapter_fpn_ocr"
                        else (
                            "uniform_40ep_multiscale_dinov3_v1"
                            if recipe.family == "ssepnet_multiscale_dinov3"
                            else (
                                "multiscale_dinov3_distill_v2"
                                if recipe.family == "ssepnet_multiscale_dinov3_v2"
                                else (
                                    "uniform_40ep_lora_efficiency_v1"
                                    if recipe.family == "ssepnet_lora_efficiency"
                                    else (
                                        "uniform_40ep_dual_fpn_ocr_v1"
                                        if recipe.family == "ssepnet_dual_fpn_ocr"
                                        else (
                                            "uniform_40ep_ocr_optimization_v1"
                                            if recipe.family
                                            == "ssepnet_ocr_optimization"
                                            else "uniform_40ep_v1"
                                        )
                                    )
                                )
                            )
                        )
                    )
                ))
                )
                )
            ),
            "source_config": str(source_config) if source_config else "",
            **(
                {
                    "comparison_axes": [
                        "train_batch_size",
                        "accumulate_grad_batches",
                    ],
                    "reference_experiment_id": recipe.base_experiment_id,
                    "seed_scope": [42],
                    "matched_control": True,
                }
                if recipe.family == "ssepnet_matched_controls"
                else {}
            ),
            **(
                {
                    "comparison_axes": [
                        _stage_adapter_fpn_ablation_axis(
                            recipe.experiment_id,
                            stage_adapter_fpn_ocr_spec,
                        )
                    ],
                    "reference_experiment_id": (
                        recipe.base_experiment_id or recipe.experiment_id
                    ),
                    "seed_scope": [42, 43, 44],
                }
                if recipe.family == "ssepnet_stage_adapter_fpn_ocr"
                else {}
            ),
            **(
                {
                    "comparison_axes": ["loss"],
                    "reference_experiment_ids": [
                        f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_binary_boundary",
                        _STAGE_ADAPTER_FPN_BINARY_FULL,
                    ],
                    "seed_scope": [42, 43, 44],
                    "loss_only": True,
                    "fbeta_enabled": False,
                    "cross_dataset_training": False,
                    "test_used_for_selection": False,
                }
                if recipe.loss_kind == "ssepnet_ewc"
                else {}
            ),
            **(
                {
                    "comparison_axes": ["structural_supervision"],
                    "reference_experiment_ids": [
                        f"{_STAGE_ADAPTER_FPN_OCR_MAIN}_binary_boundary",
                        _EWC_EXPERIMENT,
                        _STAGE_ADAPTER_FPN_BINARY_FULL,
                    ],
                    "seed_scope": [42, 43, 44],
                    "matched_carrier": True,
                    "test_used_for_selection": False,
                    "training_only_affinity_head": recipe.loss_kind
                    in {"ssepnet_cmalis", "ssepnet_wprf"},
                }
                if is_structural_loss
                else {}
            ),
            **(
                {
                    "matched_control": True,
                    "configuration_status": "training_authorized",
                    "treatment_level": "projection_resize",
                    "control_level": "stage_adapter",
                    "controlled_components": [
                        "dinov3_vitl16_sat493m",
                        "layer_taps_5_11_17_23",
                        "p2_p3_p4_p5_widths",
                        "standard_fpn",
                        "binary_head",
                        "raw_p2_boundary",
                        "pbcl",
                        "optimizer",
                        "schedule",
                        "data",
                        "checkpoint_policy",
                    ],
                }
                if recipe.experiment_id
                in _BINARY_STAGE_ADAPTER_ABLATION_RECIPES
                else {}
            ),
            **(
                {
                    "sensitivity_screen_seed": 42,
                    "sensitivity_confirmation_seeds": [43, 44],
                    "sensitivity_reference": {
                        "pbcl_iterations": 9,
                        "pbcl_band_radius": 3,
                        "pbcl_margin": 0.1,
                        "pbcl_weight": 0.1,
                    },
                }
                if recipe.experiment_id
                in _STAGE_ADAPTER_FPN_BINARY_SENSITIVITY_OVERRIDES
                else {}
            ),
            **(
                {
                    "comparison_axes": ["detail_backbone", "dinov3_adaptation"],
                    "reference_experiment_id": (
                        recipe.base_experiment_id or recipe.experiment_id
                    ),
                }
                if recipe.family == "ssepnet_dual_fpn_ocr"
                else {}
            ),
            **(
                {
                    "comparison_axis": "dinov3_adaptation",
                    "reference_experiment_id": "ssepnet_dinov3_full",
                }
                if recipe.family == "ssepnet_lora_efficiency"
                else {}
            ),
            **(
                {
                    "comparison_axes": ["ocr_adapter", "dinov3_adaptation"],
                    "reference_experiment_id": (
                        recipe.base_experiment_id or recipe.experiment_id
                    ),
                }
                if recipe.family == "ssepnet_ocr_optimization"
                else {}
            ),
            **(
                {
                    "comparison_axes": [
                        "multiscale_architecture",
                        "stage_depths",
                    ],
                    "reference_experiment_id": (
                        recipe.base_experiment_id or recipe.experiment_id
                    ),
                }
                if recipe.family == "ssepnet_multiscale_dinov3"
                else {}
            ),
            **(
                {
                    "comparison_axes": [
                        "multiscale_architecture",
                        "stage_depths",
                        "distillation",
                    ],
                    "reference_experiment_id": (
                        recipe.base_experiment_id or recipe.experiment_id
                    ),
                    "seed_scope": [42],
                }
                if recipe.family == "ssepnet_multiscale_dinov3_v2"
                else {}
            ),
        },
    )
