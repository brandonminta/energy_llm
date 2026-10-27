"""Per-run configuration: one versioned YAML per model (configs/*.yaml).

The YAML is the complete scientific specification of a run: model and exact
hub revision, dtype, prompt (single system instruction, no few-shot),
generation cap, calibration target and grid, beta sensitivity multipliers,
probe layers and seed. beta* itself is not in the YAML: ``calibrate`` writes
it to ``<output_dir>/calibration/beta_star.json`` and ``extract`` reads it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from energy_llm.storage import sha256_text


class ConfigError(ValueError):
    pass


@dataclass
class ModelConfig:
    id: str = ""
    revision: str = ""               # exact hub commit hash
    dtype: str = "bfloat16"          # the only accepted value
    device: str = "auto"             # auto | cuda | cpu


@dataclass
class PromptConfig:
    system: str = "Answer the question briefly and directly."
    date_string: str | None = None   # required when the chat template reads it (Llama)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 50


@dataclass
class CalibrationConfig:
    target_norm_entropy: float = 0.55
    log10_beta_min: float = 0.0
    log10_beta_max: float = 4.0
    num_grid: int = 81

    def beta_grid(self) -> list[float]:
        n = self.num_grid
        step = (self.log10_beta_max - self.log10_beta_min) / (n - 1)
        return [10.0 ** (self.log10_beta_min + i * step) for i in range(n)]


@dataclass
class ExtractionConfig:
    beta_multipliers: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    jaccard_top_k: int = 64
    hidden_layers: str | list[int] = "all"
    samples_per_shard: int = 1000
    max_attempts: int = 3


@dataclass
class RunConfig:
    name: str = ""
    seed: int = 42
    output_dir: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)

    # ---- paths -----------------------------------------------------------
    @property
    def out(self) -> Path:
        path = os.path.expandvars(self.output_dir)
        if "$" in path:
            raise ConfigError(f"output_dir {self.output_dir!r} uses an unset environment variable")
        return Path(path)

    @property
    def calibration_dir(self) -> Path:
        return self.out / "calibration"

    @property
    def beta_star_path(self) -> Path:
        return self.calibration_dir / "beta_star.json"

    @property
    def bank_h5_path(self) -> Path:
        return self.calibration_dir / "bank.h5"

    def extraction_dir(self, name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise ConfigError(f"extraction name {name!r} must match [A-Za-z0-9._-]+")
        return self.out / "extract" / name

    # ---- identity ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def sha256(self) -> str:
        """Hash of everything except ``output_dir`` (a location, not part of
        the scientific specification). Shards refuse to resume under a
        different hash."""
        d = self.to_dict()
        d.pop("output_dir")
        return sha256_text(json.dumps(d, sort_keys=True))

    def calibration_context(self) -> dict[str, Any]:
        """The settings beta* depends on; extract refuses a beta* computed
        under a different context."""
        return {
            "model": dataclasses.asdict(self.model) | {"device": None},
            "prompt": dataclasses.asdict(self.prompt),
            "generation": dataclasses.asdict(self.generation),
            "calibration": dataclasses.asdict(self.calibration),
        }

    def hidden_layer_indices(self, n_layers: int) -> list[int]:
        hl = self.extraction.hidden_layers
        if hl == "all":
            return list(range(n_layers))
        bad = [l for l in hl if not 0 <= l < n_layers]
        if bad:
            raise ConfigError(f"hidden_layers out of range for L={n_layers}: {bad}")
        return sorted(set(hl))

    # ---- validation ---------------------------------------------------------
    def validate(self) -> "RunConfig":
        if not self.name:
            raise ConfigError("name is required")
        if not self.output_dir:
            raise ConfigError("output_dir is required")
        if not self.model.id:
            raise ConfigError("model.id is required")
        if not re.fullmatch(r"[0-9a-f]{40}", self.model.revision or ""):
            raise ConfigError("model.revision must be a 40-hex hub commit hash")
        if self.model.dtype != "bfloat16":
            raise ConfigError("model.dtype must be bfloat16")
        if self.model.device not in ("auto", "cuda", "cpu"):
            raise ConfigError("model.device must be auto|cuda|cpu")
        if not self.prompt.system.strip():
            raise ConfigError("prompt.system must be non-empty")
        if self.generation.max_new_tokens < 1:
            raise ConfigError("generation.max_new_tokens must be >= 1")
        c = self.calibration
        if not 0.0 < c.target_norm_entropy < 1.0:
            raise ConfigError("calibration.target_norm_entropy must be in (0, 1)")
        if c.num_grid < 2 or c.log10_beta_max <= c.log10_beta_min:
            raise ConfigError("calibration grid is empty")
        e = self.extraction
        if 1.0 not in e.beta_multipliers or any(m <= 0 for m in e.beta_multipliers):
            raise ConfigError("extraction.beta_multipliers must be positive and include 1.0")
        if e.jaccard_top_k < 1:
            raise ConfigError("extraction.jaccard_top_k must be >= 1")
        if e.hidden_layers != "all" and not (
            isinstance(e.hidden_layers, list) and all(isinstance(l, int) for l in e.hidden_layers)
        ):
            raise ConfigError("extraction.hidden_layers must be 'all' or a list of ints")
        if e.samples_per_shard < 1 or e.max_attempts < 1:
            raise ConfigError("samples_per_shard and max_attempts must be >= 1")
        return self

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        sections = {
            "model": ModelConfig, "prompt": PromptConfig,
            "generation": GenerationConfig, "calibration": CalibrationConfig,
            "extraction": ExtractionConfig,
        }
        top = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key in sections:
                known = {f.name for f in dataclasses.fields(sections[key])}
                unknown = set(value or {}) - known
                if unknown:
                    raise ConfigError(f"unknown keys in '{key}': {sorted(unknown)}")
                kwargs[key] = sections[key](**(value or {}))
            elif key in top:
                kwargs[key] = value
            else:
                raise ConfigError(f"unknown config key '{key}'")
        return cls(**kwargs).validate()


def load_config(path: str | Path) -> RunConfig:
    return RunConfig.from_dict(yaml.safe_load(Path(path).read_text()))
