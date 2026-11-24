# WPRF source

The target preparation and differentiable max-min propagation are matched ports
of `mituan-ai/WPRF_open` at commit
`418e362b440df08e72ae6da54abeec6559a8f442`:

https://github.com/mituan-ai/WPRF_open

The GeoSeg integration preserves the paper's stride-4, 8-neighbour,
multi-scale reachability configuration. It deliberately keeps GeoSeg's matched
BCE+Dice+Boundary semantic objective instead of replacing it with WPRF's
bottleneck-aware segmentation loss. The affinity head, local edge loss, and
reachability loss are training-only; the primary inference output remains the
binary union mask.
