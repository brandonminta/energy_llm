"""Tests for analyze_trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hopfield_llm.pipeline.analysis import analyze_trajectories


def test_analyze_produces_json(synthetic_trajectory_dir, tmp_dir):
    output = tmp_dir / "analysis.json"
    result = analyze_trajectories(
        traj_dir=synthetic_trajectory_dir,
        output=output,
    )
    assert result == output
    assert output.exists()

    analysis = json.loads(output.read_text(encoding="utf-8"))
    assert analysis["artifact_type"] == "analysis"


def test_analysis_sample_count(synthetic_trajectory_dir, tmp_dir):
    output = tmp_dir / "analysis.json"
    analyze_trajectories(traj_dir=synthetic_trajectory_dir, output=output)

    analysis = json.loads(output.read_text(encoding="utf-8"))
    assert analysis["n_samples"] == 5
    assert analysis["n_scored"] == 5


def test_analysis_has_layer_metrics(synthetic_trajectory_dir, tmp_dir):
    output = tmp_dir / "analysis.json"
    analyze_trajectories(traj_dir=synthetic_trajectory_dir, output=output)

    analysis = json.loads(output.read_text(encoding="utf-8"))
    lm = analysis["layer_metrics"]

    expected_metrics = [
        "kl_fwd", "kl_rev", "js", "hellinger",
        "delta_entropy", "delta_norm_entropy", "delta_energy",
        "delta_top_activation", "delta_mean_activation",
    ]
    for metric in expected_metrics:
        assert metric in lm, f"Missing metric: {metric}"
        assert "mean" in lm[metric]
        assert "std" in lm[metric]
        assert len(lm[metric]["mean"]) == 4  # 4 layers in synthetic data


def test_analysis_has_score_summary(synthetic_trajectory_dir, tmp_dir):
    output = tmp_dir / "analysis.json"
    analyze_trajectories(traj_dir=synthetic_trajectory_dir, output=output)

    analysis = json.loads(output.read_text(encoding="utf-8"))
    ss = analysis["score_summary"]
    assert "count" in ss
    assert "mean" in ss
    assert "std" in ss
    assert "min" in ss
    assert "max" in ss
    assert ss["count"] == 5


def test_analysis_has_categories(synthetic_trajectory_dir, tmp_dir):
    output = tmp_dir / "analysis.json"
    analyze_trajectories(traj_dir=synthetic_trajectory_dir, output=output)

    analysis = json.loads(output.read_text(encoding="utf-8"))
    cats = analysis["score_by_category"]
    assert "test" in cats  # all synthetic samples have source="test"
    assert cats["test"]["count"] == 5


def test_analysis_different_score_metrics(synthetic_trajectory_dir, tmp_dir):
    for metric in ["js", "kl_fwd", "delta_energy"]:
        output = tmp_dir / f"analysis_{metric}.json"
        analyze_trajectories(
            traj_dir=synthetic_trajectory_dir,
            output=output,
            score_metric=metric,
        )
        analysis = json.loads(output.read_text(encoding="utf-8"))
        assert analysis["score_metric"] == metric
        assert analysis["score_summary"]["count"] == 5


def test_analysis_with_signal_zone(synthetic_trajectory_dir, tmp_dir):
    output = tmp_dir / "analysis.json"
    analyze_trajectories(
        traj_dir=synthetic_trajectory_dir,
        output=output,
        signal_zone=(1, 3),
    )
    analysis = json.loads(output.read_text(encoding="utf-8"))
    assert analysis["n_scored"] == 5


def test_analysis_empty_dir_raises(tmp_dir):
    empty_dir = tmp_dir / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No trajectory samples found"):
        analyze_trajectories(traj_dir=empty_dir, output=tmp_dir / "out.json")


def test_analysis_compatible_with_visualize(synthetic_trajectory_dir, tmp_dir):
    """Verify the output JSON has the format expected by visualize_analysis."""
    output = tmp_dir / "analysis.json"
    analyze_trajectories(traj_dir=synthetic_trajectory_dir, output=output)

    from hopfield_llm.visualization.plots import visualize_analysis
    plots_dir = tmp_dir / "plots"
    visualize_analysis(str(output), str(plots_dir))

    # Should produce at least the layer metric plots
    plot_files = list(plots_dir.glob("*.png"))
    assert len(plot_files) > 0
