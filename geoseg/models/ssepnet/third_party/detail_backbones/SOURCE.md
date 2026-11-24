# Detail-backbone source provenance

The standalone feature extractors in this directory's parent package are derived from the
following official MIT-licensed implementations. Only the model blocks needed by the GeoSeg
stride-4/8 detail contract are retained; dataset, training, classification, and original
segmentation-head code is not included.

- HRFormer-S: <https://github.com/HRNet/HRFormer>, commit
  `1245b88b5824fbd8cdb358b5ee909a4e537a2ef5`.
- PIDNet-S: <https://github.com/XuJiacong/PIDNet>, commit
  `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`.
- DDRNet-23-slim: <https://github.com/ydhongHIT/DDRNet>, commit
  `de0db317c5af4b0946231f475cf74ca4f51b8ac2`.

The implementations preserve the official stage widths and internal exchange topology while
using GeoSeg's resolution-safe interpolation and four-band stem inflation contracts.
