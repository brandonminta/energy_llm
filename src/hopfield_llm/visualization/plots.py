"""Matplotlib visualization for analysis results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hopfield_llm.utils.logging import get_logger

log = get_logger("visualization.plots")


def visualize_analysis(analysis_path: str | Path, output_dir: str | Path) -> Path:
    """Render bar and line plots from a saved analysis JSON.

    Produces:
        - score_by_category.png (if categories present)
        - {metric}_by_layer.png for each layer metric
        - analysis_snapshot.json copy

    Returns:
        Path to the output directory.
    """
    analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Score by category bar chart
    score_by_category = analysis.get("score_by_category", {})
    if score_by_category:
        categories = [
            name for name, payload in score_by_category.items() if payload is not None
        ]
        means = [score_by_category[name]["mean"] for name in categories]
        if categories:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(categories, means)
            ax.set_ylabel("Mean hallucination score")
            ax.set_title("Score by category")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(output_dir / "score_by_category.png")
            plt.close(fig)
            log.info("Saved score_by_category.png")

    # Per-layer metric line plots
    layer_metrics = analysis.get("layer_metrics", {})
    for metric_name, payload in layer_metrics.items():
        mean = np.asarray(payload.get("mean", []), dtype=float)
        std = np.asarray(payload.get("std", []), dtype=float)
        if mean.size == 0:
            continue
        x = np.arange(mean.size)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, mean, label="mean")
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, label="std")
        ax.set_xlabel("Layer index")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} by layer")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric_name}_by_layer.png")
        plt.close(fig)

    # Snapshot
    (output_dir / "analysis_snapshot.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    log.info("Saved %d layer-metric plots to %s", len(layer_metrics), output_dir)
    return output_dir
