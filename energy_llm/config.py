"""Typed experiment configuration and snapshotting.

One YAML per experiment (configs/e1.yaml, e2.yaml, e3.yaml). Every stage
loads the same file; the calibration stage writes ``beta_star`` into the
frozen snapshot at ``<output_dir>/config_snapshot.yaml`` and all later stages
read the snapshot, so a single beta_star flows through the whole run
(subsec:impl_beta).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_BETA_GRID = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]


@dataclass
class ModelConfig:
    alias: str = "qwen25_3b"
    dtype: str = "bfloat16"          # reported runs: bfloat16 on A100
    load_in_4bit: bool = False       # local dev only, never for reported runs
    device: str = "auto"


@dataclass
class DatasetConfig:
    name: str = "truthfulqa"         # truthfulqa | triviaqa
    num_samples: int | None = None   # triviaqa: 500 drawn at seed (subsec:datasets)


@dataclass
class BankConfig:
    bank_id: str = "gate"            # primary and only evaluated bank (subsec:stage1)
    path: str = ""                   # default: <output_dir>/banks.pt


@dataclass
class CalibrationConfig:
    num_samples: int = 30                       # N_c (tab:hyperparameters)
    beta_grid: list[float] = field(default_factory=lambda: list(DEFAULT_BETA_GRID))
    target_norm_entropy: float = 0.55           # H_bar* (tab:hyperparameters)
    beta_star: float | None = None              # written by the calibrate stage


@dataclass
class GenerationConfig:
    max_new_tokens: int = 50         # T_max (tab:hyperparameters)


@dataclass
class TrajectoryConfig:
    cache_score_tensors: bool = False  # E1: True (subsec:an_sensitivity)
    score_dtype: str = "float32"       # dtype of the bank matmul on GPU
    log_memory_every: int = 10


@dataclass
class LabelingConfig:
    judge_model: str = "gpt-4o-mini"
    api_base: str | None = None      # None = api.openai.com
    max_retries: int = 6
    human_subset_size: int = 60      # subsec:an_label_reliability


@dataclass
class ProbeConfig:
    c_grid: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    split: list[float] = field(default_factory=lambda: [0.70, 0.15, 0.15])
    inner_cv_folds: int = 3
    outer_cv_folds: int = 5


@dataclass
class StatsConfig:
    n_bootstrap: int = 10_000        # B (subsec:an_discrimination)
    n_permutation: int = 1_000       # B_perm (subsec:an_localisation)
    alpha: float = 0.05              # family level, Bonferroni over 2 tests


@dataclass
class ExperimentConfig:
    name: str = "e1"
    seed: int = 42
    output_dir: str = "results/e1"
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    bank: BankConfig = field(default_factory=BankConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)

    # ---- derived paths -------------------------------------------------
    @property
    def out(self) -> Path:
        return Path(self.output_dir)

    @property
    def banks_path(self) -> Path:
        return Path(self.bank.path) if self.bank.path else self.out / "banks.pt"

    @property
    def trajectories_dir(self) -> Path:
        return self.out / "trajectories"

    @property
    def labels_dir(self) -> Path:
        return self.out / "labels"

    @property
    def calibration_dir(self) -> Path:
        return self.out / "calibration"

    @property
    def features_dir(self) -> Path:
        return self.out / "features"

    @property
    def figures_dir(self) -> Path:
        return self.out / "figures"

    @property
    def snapshot_path(self) -> Path:
        return self.out / "config_snapshot.yaml"

    # ---- (de)serialisation ----------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        sections = {
            "model": ModelConfig, "dataset": DatasetConfig, "bank": BankConfig,
            "calibration": CalibrationConfig, "generation": GenerationConfig,
            "trajectory": TrajectoryConfig, "labeling": LabelingConfig,
            "probe": ProbeConfig, "stats": StatsConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key in sections:
                known = {f.name for f in dataclasses.fields(sections[key])}
                unknown = set(value) - known
                if unknown:
                    raise KeyError(f"Unknown config keys in '{key}': {sorted(unknown)}")
                kwargs[key] = sections[key](**value)
            elif key in {f.name for f in dataclasses.fields(cls)}:
                kwargs[key] = value
            else:
                raise KeyError(f"Unknown config section: '{key}'")
        return cls(**kwargs)

    def save_snapshot(self) -> Path:
        """Freeze the config (including beta_star once calibrated)."""
        from energy_llm.io_artifacts import save_json_atomic  # noqa: F401 (atomic pattern)
        self.out.mkdir(parents=True, exist_ok=True)
        tmp = self.snapshot_path.with_name(self.snapshot_path.name + ".tmp")
        tmp.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        tmp.replace(self.snapshot_path)
        return self.snapshot_path


def load_config(path: str | Path, prefer_snapshot: bool = True) -> ExperimentConfig:
    """Load an experiment config.

    If the run's frozen snapshot exists and ``prefer_snapshot`` is True, the
    snapshot wins: it carries the calibrated beta_star and is what every
    post-calibration stage must read (subsec:impl_beta).
    """
    raw = yaml.safe_load(Path(path).read_text())
    cfg = ExperimentConfig.from_dict(raw)
    if prefer_snapshot and cfg.snapshot_path.exists():
        snap = yaml.safe_load(cfg.snapshot_path.read_text())
        cfg = ExperimentConfig.from_dict(snap)
    return cfg


def require_beta_star(cfg: ExperimentConfig) -> float:
    if cfg.calibration.beta_star is None:
        raise RuntimeError(
            "beta_star is not calibrated. Run scripts/calibrate_beta.py first; "
            "it writes beta_star into the config snapshot read by all later stages."
        )
    return float(cfg.calibration.beta_star)
