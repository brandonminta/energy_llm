"""Provenance: code commit, library versions, hardware, SLURM context."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = ("torch", "transformers", "tokenizers", "numpy", "pandas", "pyarrow",
             "h5py", "yaml", "huggingface_hub")


def code_version() -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True,
                                  capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def library_versions() -> dict:
    import importlib

    out = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            out[name] = getattr(importlib.import_module(name), "__version__", "?")
        except ImportError:
            out[name] = None
    return out


def hardware() -> dict:
    import torch

    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "slurm": {k: v for k, v in os.environ.items()
                  if k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                           "SLURM_JOB_PARTITION", "SLURM_RESTART_COUNT",
                           "SLURM_JOB_NODELIST", "CUDA_VISIBLE_DEVICES")},
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = {"name": props.name, "total_memory_gb": round(props.total_memory / 1e9, 2)}
    return info


def set_determinism(seed: int) -> None:
    """Seed every RNG and keep float32 matmuls at full precision (no TF32)."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
