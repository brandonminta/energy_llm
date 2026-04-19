"""Experiment configuration: typed dataclass + YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hopfield_llm.utils.logging import get_logger

log = get_logger("pipeline.config")


@dataclass
class ExperimentConfig:
    """Flat representation of an experiment configuration.

    The YAML file is hierarchical for readability; this dataclass flattens
    it for simple access in pipeline code.
    """

    # Experiment metadata
    name: str = "unnamed"
    description: str = ""

    # Model
    model_alias: str = "qwen25_3b"
    load_in_4bit: bool = True

    # Dataset
    dataset_name: str = "truthfulqa"
    max_samples: int | None = None
    seed: int = 42

    # Trajectory parameters
    beta: float = 15.0
    threshold: float = 0.1
    energy_mode: str = "dot"
    max_new_tokens: int = 50
    diagnostic_subset: int = 20

    # Analysis parameters
    divergence_metrics: list[str] = field(
        default_factory=lambda: ["kl_fwd", "js", "hellinger"]
    )
    score_metric: str = "js"
    score_aggregation: str = "mean"
    signal_zone: tuple[int, int] | None = None

    # Sharding (SLURM)
    num_shards: int = 1

    # Output
    base_dir: str = "runs"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for tracking/snapshots."""
        return {
            "experiment": {
                "name": self.name,
                "description": self.description,
            },
            "model": {
                "alias": self.model_alias,
                "load_in_4bit": self.load_in_4bit,
            },
            "dataset": {
                "name": self.dataset_name,
                "max_samples": self.max_samples,
                "seed": self.seed,
            },
            "trajectory": {
                "beta": self.beta,
                "threshold": self.threshold,
                "energy_mode": self.energy_mode,
                "max_new_tokens": self.max_new_tokens,
                "diagnostic_subset": self.diagnostic_subset,
            },
            "analysis": {
                "divergence_metrics": self.divergence_metrics,
                "score_metric": self.score_metric,
                "score_aggregation": self.score_aggregation,
                "signal_zone": list(self.signal_zone) if self.signal_zone else None,
            },
            "sharding": {
                "num_shards": self.num_shards,
            },
            "output": {
                "base_dir": self.base_dir,
            },
        }


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML and return a typed ExperimentConfig.

    Args:
        path: Path to the experiment YAML file.

    Returns:
        ExperimentConfig with all fields populated (defaults for missing keys).

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If required fields are missing or invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping, got {type(raw).__name__}")

    exp = raw.get("experiment", {})
    model = raw.get("model", {})
    dataset = raw.get("dataset", {})
    traj = raw.get("trajectory", {})
    analysis = raw.get("analysis", {})
    sharding = raw.get("sharding", {})
    output = raw.get("output", {})

    # Parse signal_zone: null, or [start, end]
    sz_raw = analysis.get("signal_zone")
    signal_zone = None
    if sz_raw is not None:
        if isinstance(sz_raw, (list, tuple)) and len(sz_raw) == 2:
            signal_zone = (int(sz_raw[0]), int(sz_raw[1]))
        else:
            raise ValueError(
                f"signal_zone must be null or [start, end], got {sz_raw}"
            )

    config = ExperimentConfig(
        name=exp.get("name", "unnamed"),
        description=exp.get("description", ""),
        model_alias=model.get("alias", "qwen25_3b"),
        load_in_4bit=model.get("load_in_4bit", True),
        dataset_name=dataset.get("name", "truthfulqa"),
        max_samples=dataset.get("max_samples"),
        seed=int(dataset.get("seed", 42)),
        beta=float(traj.get("beta", 15.0)),
        threshold=float(traj.get("threshold", 0.1)),
        energy_mode=traj.get("energy_mode", "dot"),
        max_new_tokens=int(traj.get("max_new_tokens", 50)),
        diagnostic_subset=int(traj.get("diagnostic_subset", 20)),
        divergence_metrics=analysis.get(
            "divergence_metrics", ["kl_fwd", "js", "hellinger"]
        ),
        score_metric=analysis.get("score_metric", "js"),
        score_aggregation=analysis.get("score_aggregation", "mean"),
        signal_zone=signal_zone,
        num_shards=int(sharding.get("num_shards", 1)),
        base_dir=output.get("base_dir", "runs"),
    )

    log.info("Loaded experiment config: %s (%s)", config.name, path)
    return config


def apply_overrides(
    config: ExperimentConfig, overrides: list[str]
) -> ExperimentConfig:
    """Apply key=value overrides to an ExperimentConfig.

    Args:
        config: Base config to modify.
        overrides: List of "key=value" strings (e.g. ["beta=20.0", "seed=0"]).

    Returns:
        Modified ExperimentConfig (mutated in-place and returned).
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: '{item}'")
        key, value = item.split("=", 1)
        key = key.strip()
        if not hasattr(config, key):
            raise ValueError(
                f"Unknown config key: '{key}'. "
                f"Valid keys: {[f.name for f in config.__dataclass_fields__.values()]}"
            )

        field_obj = config.__dataclass_fields__[key]
        current = getattr(config, key)

        # Type coercion based on the current value's type
        if current is None:
            # Try int, then float, then leave as string
            for cast in (int, float):
                try:
                    value = cast(value)
                    break
                except ValueError:
                    continue
            if value == "null" or value == "None":
                value = None
        elif isinstance(current, bool):
            value = value.lower() in ("true", "1", "yes")
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, list):
            value = [v.strip() for v in value.split(",")]

        setattr(config, key, value)
        log.info("Override: %s = %r", key, value)

    return config
