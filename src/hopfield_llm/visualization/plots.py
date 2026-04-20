"""Matplotlib visualizations from analysis JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hopfield_llm.utils.logging import get_logger

log = get_logger("visualization.plots")


def visualize_analysis(analysis_path: str | Path, output_dir: str | Path) -> Path:
    """Render plots from a saved analysis JSON.

    Produces:
        score_by_category.png        — bar chart of mean score per dataset
        {metric}_by_layer.png        — per-layer delta metrics (mean ± std)
        global_divergences.png       — bar chart of global KL/JS/Hellinger
        analysis_snapshot.json       — copy of the analysis JSON

    Args:
        analysis_path: Path to the analysis JSON produced by analyze_trajectories.
        output_dir:    Directory to write plot files into.

    Returns:
        Path to the output directory.
    """
    analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Score by category bar chart
    score_by_category = analysis.get("score_by_category", {})
    categories = [k for k, v in score_by_category.items() if v is not None]
    if categories:
        means = [score_by_category[c]["mean"] for c in categories]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(categories, means)
        ax.set_ylabel("Mean hallucination score")
        ax.set_title("Score by category")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(out / "score_by_category.png")
        plt.close(fig)
        log.info("Saved score_by_category.png")

    # Per-layer delta metric line plots
    layer_metrics = analysis.get("layer_metrics", {})
    for metric_name, payload in layer_metrics.items():
        mean = np.asarray(payload.get("mean", []), dtype=float)
        std  = np.asarray(payload.get("std",  []), dtype=float)
        if mean.size == 0:
            continue
        x = np.arange(mean.size)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, mean, label="mean")
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, label="±std")
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Layer index")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} by layer (generation − prefill)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{metric_name}_by_layer.png")
        plt.close(fig)

    # Global divergences bar chart
    global_div = analysis.get("global_divergences", {})
    if global_div:
        names  = list(global_div.keys())
        means  = [global_div[n]["mean"] for n in names]
        stds   = [global_div[n]["std"]  for n in names]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, means, yerr=stds, capsize=4)
        ax.set_ylabel("Divergence (mean ± std over samples)")
        ax.set_title("Global distribution divergences")
        fig.tight_layout()
        fig.savefig(out / "global_divergences.png")
        plt.close(fig)
        log.info("Saved global_divergences.png")

    # Snapshot copy
    (out / "analysis_snapshot.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    log.info("Saved %d per-layer plots to %s", len(layer_metrics), out)
    return out
