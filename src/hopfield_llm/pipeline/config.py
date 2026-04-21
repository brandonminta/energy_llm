"""Experiment configuration: typed dataclass + YAML loader."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hopfield_llm.utils.logging import get_logger

log = get_logger("pipeline.config")


@dataclass
class ProbeConfig:
    """Configuration for a single Hopfield probe (bank + hook placement)."""

    bank: str = "up"
    normalize: bool = True
    hook_target: str = "mlp_input"   # "mlp_input" | "mlp_output"
    normalize_query: bool = True
    label: str = ""                  # human-readable tag used in artifact filenames


@dataclass
class ExperimentConfig:
    """Flat representation of an experiment configuration.

    The YAML file is hierarchical for readability; this dataclass flattens
    it for simple access in pipeline code.
    """

    # Experiment metadata
    name:        str = "unnamed"
    description: str = ""

    # Model
    model_alias:  str  = "qwen25_3b"
    load_in_4bit: bool = True

    # Dataset
    dataset_name: str      = "truthfulqa"
    max_samples:  int | None = None
    seed:         int      = 42

    # Trajectory parameters
    beta:               float = 15.0
    threshold:          float = 0.1
    energy_mode:        str   = "dot"
    max_new_tokens:     int   = 50
    diagnostic_subset:  int   = 20

    # Memory bank (legacy single-probe field)
    bank: str = "down"

    # Analysis parameters
    divergence_metrics: list[str] = field(
        default_factory=lambda: ["delta_energy", "energy_shift_l1", "gen_drift_mean"]
    )
    score_metric:       str               = "delta_energy"
    score_aggregation:  str               = "mean"
    signal_zone:        tuple[int, int] | None = None

    # Sharding (HPC)
    num_shards: int = 1

    # Scores subset (limits which samples get raw score tensors saved)
    scores_subset_size: int = 200

    # Labeling / calibration
    calibration_subset_size: int = 50

    # Output
    base_dir: str = "outputs"

    # Probe configuration (new — optional dual-probe support)
    primary_probe: ProbeConfig = field(default_factory=lambda: ProbeConfig(
        bank="up", normalize=True, hook_target="mlp_input", label="key_space"
    ))
    secondary_probe: ProbeConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a nested dict for config snapshots."""

        def _probe_dict(p: ProbeConfig) -> dict[str, Any]:
            return {
                "bank": p.bank,
                "normalize": p.normalize,
                "hook_target": p.hook_target,
                "normalize_query": p.normalize_query,
                "label": p.label,
            }

        d: dict[str, Any] = {
            "experiment": {"name": self.name, "description": self.description},
            "model":      {"alias": self.model_alias, "load_in_4bit": self.load_in_4bit},
            "dataset":    {"name": self.dataset_name, "max_samples": self.max_samples, "seed": self.seed},
            "trajectory": {
                "beta": self.beta, "threshold": self.threshold,
                "energy_mode": self.energy_mode, "max_new_tokens": self.max_new_tokens,
                "diagnostic_subset": self.diagnostic_subset, "bank": self.bank,
                "scores_subset_size": self.scores_subset_size,
            },
            "analysis": {
                "divergence_metrics": self.divergence_metrics,
                "score_metric": self.score_metric,
                "score_aggregation": self.score_aggregation,
                "signal_zone": list(self.signal_zone) if self.signal_zone else None,
            },
            "sharding": {"num_shards": self.num_shards},
            "labeling": {
                "calibration": {"calibration_subset_size": self.calibration_subset_size}
            },
            "output":   {"base_dir": self.base_dir},
            "primary_probe": _probe_dict(self.primary_probe),
        }
        if self.secondary_probe is not None:
            d["secondary_probe"] = _probe_dict(self.secondary_probe)
        return d


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def _parse_probe_section(raw: dict[str, Any] | None, defaults: ProbeConfig) -> ProbeConfig:
    if raw is None:
        return defaults
    return ProbeConfig(
        bank=raw.get("bank", defaults.bank),
        normalize=bool(raw.get("normalize", defaults.normalize)),
        hook_target=raw.get("hook_target", defaults.hook_target),
        normalize_query=bool(raw.get("normalize_query", defaults.normalize_query)),
        label=raw.get("label", defaults.label),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML and return a typed ExperimentConfig.

    Raises:
        FileNotFoundError: YAML file does not exist.
        ValueError: Invalid structure or values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping, got {type(raw).__name__}")

    exp      = raw.get("experiment", {})
    model    = raw.get("model", {})
    dataset  = raw.get("dataset", {})
    traj     = raw.get("trajectory", {})
    analysis = raw.get("analysis", {})
    sharding = raw.get("sharding", {})
    output   = raw.get("output", {})
    labeling = raw.get("labeling", {})

    sz_raw = analysis.get("signal_zone")
    signal_zone = None
    if sz_raw is not None:
        if isinstance(sz_raw, (list, tuple)) and len(sz_raw) == 2:
            signal_zone = (int(sz_raw[0]), int(sz_raw[1]))
        else:
            raise ValueError(f"signal_zone must be null or [start, end], got {sz_raw}")

    _prim_default = ProbeConfig(bank="up", normalize=True, hook_target="mlp_input", label="key_space")
    primary_probe = _parse_probe_section(raw.get("primary_probe"), _prim_default)

    sec_raw = raw.get("secondary_probe")
    secondary_probe: ProbeConfig | None = None
    if sec_raw is not None:
        _sec_default = ProbeConfig(bank="down_values", normalize=True, hook_target="mlp_output", label="value_space")
        secondary_probe = _parse_probe_section(sec_raw, _sec_default)

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
        bank=traj.get("bank", "down"),
        divergence_metrics=analysis.get(
            "divergence_metrics", ["delta_energy", "energy_shift_l1", "gen_drift_mean"]
        ),
        score_metric=analysis.get("score_metric", "delta_energy"),
        score_aggregation=analysis.get("score_aggregation", "mean"),
        signal_zone=signal_zone,
        num_shards=int(sharding.get("num_shards", 1)),
        scores_subset_size=int(traj.get("scores_subset_size", 200)),
        calibration_subset_size=int(
            labeling.get("calibration", {}).get("calibration_subset_size", 50)
        ),
        base_dir=output.get("base_dir", "outputs"),
        primary_probe=primary_probe,
        secondary_probe=secondary_probe,
    )

    log.info("Loaded experiment config: %s (%s)", config.name, path)
    return config


def apply_overrides(config: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    """Apply key=value overrides to an ExperimentConfig in-place.

    Args:
        config:    Config to modify.
        overrides: List of "key=value" strings (e.g. ["beta=20.0", "seed=0"]).

    Returns:
        The modified config (same object).
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: '{item}'")
        key, value = item.split("=", 1)
        key = key.strip()
        if not hasattr(config, key):
            raise ValueError(
                f"Unknown config key: '{key}'. "
                f"Valid: {[f.name for f in config.__dataclass_fields__.values()]}"
            )

        current = getattr(config, key)
        if current is None:
            for cast in (int, float):
                try:
                    value = cast(value); break
                except ValueError:
                    continue
            if value in ("null", "None"):
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


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_with_value_space_probe(base_config: ExperimentConfig) -> ExperimentConfig:
    """Return a copy of base_config with secondary_probe set to the value-space probe.

    The value-space probe uses bank='down_values' (W_down.T rows) and
    hook_target='mlp_output' (captures FFN output y^l as the query), as
    described in Geva et al. 2021 §4.
    """
    return dataclasses.replace(
        base_config,
        secondary_probe=ProbeConfig(
            bank="down_values",
            normalize=True,
            hook_target="mlp_output",
            label="value_space",
        ),
    )
