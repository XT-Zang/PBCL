from __future__ import annotations

import os
from pathlib import Path


DEFAULT_DATASETS_ROOT = Path("data")
DATASETS_ROOT = Path(
    os.environ.get("PBCL_DATA_ROOT")
    or os.environ.get("GEOSEG_DATA_ROOT")
    or str(DEFAULT_DATASETS_ROOT)
).expanduser().resolve()


__all__ = ["DATASETS_ROOT", "DEFAULT_DATASETS_ROOT"]
