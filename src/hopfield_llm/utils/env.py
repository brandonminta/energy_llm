"""Environment detection and configuration loading.

Automatically selects local vs HPC configuration based on the HOPFIELD_ENV
environment variable or the presence of SLURM_JOB_ID.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hopfield_llm.utils.logging import get_logger

log = get_logger("utils.env")


@dataclass
class EnvConfig:
    environment: str        # "local" or "hpc"
    device: str             # "cuda" or "cpu"
    max_vram_gb: float
    max_ram_gb: float
    default_batch_size: int
    use_4bit: bool
    num_workers: int
    cache_dir: Path
    results_dir: Path

    def resolve_batch_size(self, requested: int | None = None) -> int:
        """Return the smaller of requested and environment default."""
        if requested is None:
            return self.default_batch_size
        return min(requested, self.default_batch_size)


def detect_environment() -> str:
    """Detect environment: 'local' or 'hpc'."""
    explicit = os.environ.get("HOPFIELD_ENV")
    if explicit:
        return explicit.lower()
    if os.environ.get("SLURM_JOB_ID"):
        return "hpc"
    return "local"


def _expand_env_vars(value: str) -> str:
    """Expand $VAR and ${VAR} in a string value."""
    return os.path.expandvars(value)


def load_env_config(
    config_path: str | Path | None = None,
    environment: str | None = None,
) -> EnvConfig:
    """Load environment configuration from YAML.

    Resolution order:
        1. Explicit config_path
        2. configs/environments/{environment}.yaml relative to repo root
        3. Auto-detect environment and use default path

    Args:
        config_path: Explicit YAML path (overrides auto-detection).
        environment: Force environment name (overrides auto-detection).

    Returns:
        EnvConfig dataclass.
    """
    if environment is None:
        environment = detect_environment()

    if config_path is None:
        # Look relative to the project root (3 levels up from this file)
        project_root = Path(__file__).resolve().parents[3]
        config_path = project_root / "configs" / "environments" / f"{environment}.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        log.warning(
            "Environment config not found: %s — using defaults for '%s'",
            config_path, environment,
        )
        return _defaults(environment)

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    return EnvConfig(
        environment=raw.get("environment", environment),
        device=raw.get("device", "cuda"),
        max_vram_gb=float(raw.get("max_vram_gb", 4)),
        max_ram_gb=float(raw.get("max_ram_gb", 14)),
        default_batch_size=int(raw.get("default_batch_size", 1)),
        use_4bit=bool(raw.get("use_4bit", True)),
        num_workers=int(raw.get("num_workers", 4)),
        cache_dir=Path(_expand_env_vars(str(raw.get("cache_dir", "./data/cache")))),
        results_dir=Path(_expand_env_vars(str(raw.get("results_dir", "./data/results")))),
    )


def _defaults(environment: str) -> EnvConfig:
    if environment == "hpc":
        return EnvConfig(
            environment="hpc",
            device="cuda",
            max_vram_gb=20,
            max_ram_gb=60,
            default_batch_size=8,
            use_4bit=True,
            num_workers=8,
            cache_dir=Path(os.environ.get("SCRATCH", "/tmp")) / "hopfield_cache",
            results_dir=Path(os.environ.get("SCRATCH", "/tmp")) / "hopfield_results",
        )
    return EnvConfig(
        environment="local",
        device="cuda",
        max_vram_gb=4,
        max_ram_gb=14,
        default_batch_size=1,
        use_4bit=True,
        num_workers=4,
        cache_dir=Path("./data/cache"),
        results_dir=Path("./data/results"),
    )
