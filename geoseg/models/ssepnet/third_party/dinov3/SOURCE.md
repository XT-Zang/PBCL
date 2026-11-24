# Embedded DINOv3 source notice

The code in this directory is an inference-focused extraction of Meta's official DINOv3 implementation, taken from the local official DINOv3 repository snapshot used by SSepNet on 2026-08-21.

Only the ViT components required to construct DINOv3 ViT-L/16 SAT-493M and return intermediate patch features are retained. Training-only, distributed, evaluation, text, detection, depth, and other backbone code is omitted. Class and parameter names that participate in the pretrained state dictionary remain compatible with the official implementation.

This directory is governed by the adjacent `LICENSE.md`, not relicensed under GeoSeg's GPL license.

`geoseg/models/ssepnet/deformable_mask2former.py` also contains standalone, pure-PyTorch
adaptations of the official DINOv3 segmentation MS-Deformable-Attention pixel decoder and masked
transformer decoder. The external compiled attention extension was intentionally replaced by an
autograd-safe `grid_sample` implementation so the candidate remains portable and CPU-testable.
Those adapted portions are governed by this same DINOv3 license; the HRNet fusion, semantic-output
adapter, and AquaDateset integration around them are GeoSeg-specific.
