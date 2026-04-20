"""Experiment tracking: config snapshots, git info, metrics, and replay."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hopfield_llm.utils.logging import get_logger

log = get_logger("utils.tracking")


def _short_hash(text: str, length: int = 6) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def _git_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        info["commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode().strip()
        )
        info["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        )
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL)
        info["dirty"] = dirty != 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["error"] = "git not available"
    return info


@dataclass
class ExperimentTracker:
    """Manage a single experiment run directory."""

    base_dir: Path
    experiment_id: str = ""
    run_dir: Path | None = None

    _start_time: float = 0.0
    _log_path: Path | None = None
    _metrics: list[dict[str, Any]] | None = None

    def start(self, config: dict[str, Any]) -> Path:
        """Create run directory, snapshot config and git state, start timer.

        Args:
            config: Experiment config dict (from ExperimentConfig.to_dict()).

        Returns:
            Path to the created run directory.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Safely extract nested fields from config.to_dict() structure
        model_section   = config.get("model", {})
        dataset_section = config.get("dataset", {})
        traj_section    = config.get("trajectory", {})

        slug_model   = (model_section.get("alias", "unknown")
                        if isinstance(model_section, dict) else str(model_section))
        slug_dataset = (dataset_section.get("name", "unknown")
                        if isinstance(dataset_section, dict) else str(dataset_section))
        bank         = traj_section.get("bank", "down") if isinstance(traj_section, dict) else "down"
        beta         = traj_section.get("beta", 15.0) if isinstance(traj_section, dict) else 15.0

        short = _short_hash(json.dumps(config, sort_keys=True, default=str))
        self.experiment_id = f"{ts}_{slug_model}_{slug_dataset}_{bank}_b{beta}_{short}"

        self.run_dir = self.base_dir / self.experiment_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        (self.run_dir / "config.yaml").write_text(
            yaml.dump(config, default_flow_style=False), encoding="utf-8"
        )
        (self.run_dir / "git_info.json").write_text(
            json.dumps(_git_info(), indent=2), encoding="utf-8"
        )

        self._log_path = self.run_dir / "log.txt"
        self._log_path.write_text(f"Experiment started: {ts}\n", encoding="utf-8")
        self._start_time = time.time()
        self._metrics = []

        log.info("Experiment started: %s", self.experiment_id)
        return self.run_dir

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Append a metric entry to the in-memory log and log.txt."""
        entry = {"key": key, "value": value, "step": step, "time": time.time()}
        if self._metrics is not None:
            self._metrics.append(entry)
        if self._log_path is not None:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(f"  [{key}] step={step} value={value:.6f}\n")

    def finish(self, metrics_df=None) -> Path:
        """Finalise: save environment info, metrics, and duration.

        Args:
            metrics_df: Optional pandas DataFrame to save as parquet.

        Returns:
            Path to the run directory.
        """
        assert self.run_dir is not None, "Call start() before finish()"

        elapsed = time.time() - self._start_time
        env_info: dict[str, Any] = {
            "duration_seconds": round(elapsed, 2),
            "timestamp_finish": datetime.now(timezone.utc).isoformat(),
        }
        try:
            import torch
            if torch.cuda.is_available():
                env_info["gpu_name"] = torch.cuda.get_device_name(0)
                env_info["gpu_memory_total_mb"] = round(
                    torch.cuda.get_device_properties(0).total_mem / 1e6, 0
                )
        except Exception:
            pass

        (self.run_dir / "environment.yaml").write_text(
            yaml.dump(env_info, default_flow_style=False), encoding="utf-8"
        )
        if self._metrics:
            (self.run_dir / "metrics_log.json").write_text(
                json.dumps(self._metrics, indent=2, default=str), encoding="utf-8"
            )
        if metrics_df is not None:
            metrics_df.to_parquet(self.run_dir / "metrics.parquet")
        if self._log_path is not None:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(f"Experiment finished in {elapsed:.1f}s\n")

        log.info("Experiment finished: %s (%.1fs)", self.experiment_id, elapsed)
        return self.run_dir

    @classmethod
    def load(cls, run_dir: str | Path) -> dict[str, Any]:
        """Load a previous experiment run for inspection or replay."""
        run_dir = Path(run_dir)
        result: dict[str, Any] = {"run_dir": str(run_dir)}

        for fname, key, loader in [
            ("config.yaml",      "config",      lambda p: yaml.safe_load(p.read_text(encoding="utf-8"))),
            ("git_info.json",    "git_info",    lambda p: json.loads(p.read_text(encoding="utf-8"))),
            ("environment.yaml", "environment", lambda p: yaml.safe_load(p.read_text(encoding="utf-8"))),
        ]:
            fpath = run_dir / fname
            if fpath.exists():
                result[key] = loader(fpath)

        metrics_path = run_dir / "metrics.parquet"
        if metrics_path.exists():
            result["metrics_path"] = str(metrics_path)

        return result
