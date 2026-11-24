from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from typing import Mapping


POSIX_TEMP_DIR = Path("/tmp")
TEMP_ENV_NAMES = ("TMPDIR", "TEMP", "TMP")


def build_posix_child_environment(inherited: Mapping[str, str]) -> dict[str, str]:
    """Copy an inherited environment and force POSIX-safe temporary paths."""
    child = dict(inherited)
    for name in TEMP_ENV_NAMES:
        child[name] = str(POSIX_TEMP_DIR)
    return child


def validate_posix_temp_runtime(temp_dir: str | Path = POSIX_TEMP_DIR) -> None:
    """Fail clearly unless the temp directory supports files and AF_UNIX sockets."""
    path = Path(temp_dir)
    if os.name != "posix":
        raise RuntimeError("POSIX child runtime preflight requires a POSIX host")
    if not path.exists():
        raise RuntimeError(f"POSIX temporary directory does not exist: {path}")
    if not path.is_dir():
        raise RuntimeError(f"POSIX temporary path is not a directory: {path}")

    try:
        with tempfile.NamedTemporaryFile(prefix="geoseg-pbcl-", dir=path):
            pass
    except OSError as exc:
        raise RuntimeError(f"POSIX temporary directory is not writable: {path}: {exc}") from exc

    try:
        with tempfile.TemporaryDirectory(prefix="geoseg-pbcl-", dir=path) as socket_dir:
            socket_path = Path(socket_dir) / "worker.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.bind(str(socket_path))
    except (AttributeError, OSError) as exc:
        raise RuntimeError(
            f"POSIX temporary directory does not support AF_UNIX sockets: {path}: {exc}"
        ) from exc


def run_multiworker_dataloader_smoke(
    child_environment: Mapping[str, str],
    *,
    num_workers: int = 2,
    timeout_seconds: float = 30.0,
) -> list[int]:
    """Read one real DataLoader batch in a bounded subprocess."""
    if num_workers <= 0:
        raise ValueError("num_workers must be positive for the multi-worker smoke test")
    invalid = {
        name: child_environment.get(name)
        for name in TEMP_ENV_NAMES
        if child_environment.get(name) != str(POSIX_TEMP_DIR)
    }
    if invalid:
        raise RuntimeError(
            f"Refusing multi-worker smoke with unsafe child temporary environment: {invalid}"
        )

    smoke_program = (
        "import json\n"
        "import torch\n"
        "from torch.utils.data import DataLoader, TensorDataset\n"
        "dataset = TensorDataset(torch.arange(8, dtype=torch.int64))\n"
        f"loader = DataLoader(dataset, batch_size=2, num_workers={num_workers})\n"
        "iterator = iter(loader)\n"
        "batch = next(iterator)[0].tolist()\n"
        "del iterator\n"
        "print(json.dumps(batch))\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", smoke_program],
            env=dict(child_environment),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Multi-worker DataLoader smoke could not run: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no child output"
        raise RuntimeError(
            f"Multi-worker DataLoader smoke failed with exit code "
            f"{completed.returncode}: {detail}"
        )
    try:
        return list(json.loads(completed.stdout.strip().splitlines()[-1]))
    except (IndexError, json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(
            f"Multi-worker DataLoader smoke returned invalid output: {completed.stdout!r}"
        ) from exc


def preflight_posix_child_runtime(child_environment: Mapping[str, str]) -> None:
    """Validate /tmp and prove a real multi-worker first batch can be read."""
    validate_posix_temp_runtime()
    first_batch = run_multiworker_dataloader_smoke(child_environment)
    if first_batch != [0, 1]:
        raise RuntimeError(
            f"Multi-worker DataLoader smoke returned an unexpected first batch: {first_batch}"
        )
