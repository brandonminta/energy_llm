"""Experiment tracking: config snapshots, git info, metrics, and replay.

Each experiment run is persisted in its own directory under
``experiments/runs/{experiment_id}/``.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
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
    """Capture current git state."""
    info: dict[str, Any] = {}
    try:
        info["commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        info["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        )
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
            config: Full experiment configuration dict.

        Returns:
            Path to the created run directory.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_id = config.get("model", config.get("model_id", "unknown"))
        dataset_id = config.get("dataset", config.get("dataset_id", "unknown"))
        bank = config.get("bank", {}).get("name", "gate") if isinstance(config.get("bank"), dict) else config.get("bank", "gate")
        beta = config.get("divergence", {}).get("beta", 15.0) if isinstance(config.get("divergence"), dict) else config.get("beta", 15.0)
        short = _short_hash(json.dumps(config, sort_keys=True, default=str))

        slug_model = str(model_id).replace("/", "_")
        self.experiment_id = f"{ts}_{slug_model}_{dataset_id}_{bank}_{beta}_{short}"

        self.run_dir = self.base_dir / self.experiment_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        (self.run_dir / "config.yaml").write_text(
            yaml.dump(config, default_flow_style=False), encoding="utf-8"
        )

        # Save git info
        (self.run_dir / "git_info.json").write_text(
            json.dumps(_git_info(), indent=2), encoding="utf-8"
        )

        # Init log
        self._log_path = self.run_dir / "log.txt"
        self._log_path.write_text(
            f"Experiment started: {ts}\n", encoding="utf-8"
        )

        self._start_time = time.time()
        self._metrics = []

        log.info("Experiment started: %s", self.experiment_id)
        return self.run_dir

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric (appended incrementally)."""
        entry = {"key": key, "value": value, "step": step, "time": time.time()}
        if self._metrics is not None:
            self._metrics.append(entry)
        if self._log_path is not None:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(f"  [{key}] step={step} value={value:.6f}\n")

    def finish(self, metrics_df=None) -> Path:
        """Finalise the experiment: save metrics, compute duration.

        Args:
            metrics_df: Optional pandas DataFrame to save as parquet.

        Returns:
            Path to the run directory.
        """
        assert self.run_dir is not None, "Call start() before finish()"

        elapsed = time.time() - self._start_time

        # Save environment info
        env_info = {
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

        # Save incremental metrics
        if self._metrics:
            (self.run_dir / "metrics_log.json").write_text(
                json.dumps(self._metrics, indent=2, default=str), encoding="utf-8"
            )

        # Save final DataFrame
        if metrics_df is not None:
            metrics_df.to_parquet(self.run_dir / "metrics.parquet")

        if self._log_path is not None:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(f"Experiment finished in {elapsed:.1f}s\n")

        log.info("Experiment finished: %s (%.1fs)", self.experiment_id, elapsed)
        return self.run_dir

    @classmethod
    def load(cls, run_dir: str | Path) -> dict[str, Any]:
        """Load a previous experiment for inspection or replay.

        Returns:
            Dict with config, git_info, environment, and metrics paths.
        """
        run_dir = Path(run_dir)
        result: dict[str, Any] = {"run_dir": str(run_dir)}

        config_path = run_dir / "config.yaml"
        if config_path.exists():
            result["config"] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        git_path = run_dir / "git_info.json"
        if git_path.exists():
            result["git_info"] = json.loads(git_path.read_text(encoding="utf-8"))

        env_path = run_dir / "environment.yaml"
        if env_path.exists():
            result["environment"] = yaml.safe_load(env_path.read_text(encoding="utf-8"))

        metrics_path = run_dir / "metrics.parquet"
        if metrics_path.exists():
            result["metrics_path"] = str(metrics_path)

        return result
