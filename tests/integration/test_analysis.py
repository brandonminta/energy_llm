"""Integration tests for analyze_trajectories → visualize_analysis."""

from __future__ import annotations

import json

import pytest

from hopfield_llm.pipeline.analysis import analyze_trajectories


def test_analyze_produces_json(synthetic_trajectory_dir, tmp_dir):
    output = tmp_dir / "analysis.json"
    result = analyze_trajectories(traj_dir=synthetic_trajectory_dir, output=output)
    assert result == output
    assert output.exists()
    analysis = json.loads(output.read_text())
    assert analysis["artifact_type"] == "analysis"


def test_analysis_sample_count(synthetic_trajectory_dir, tmp_dir):
    output = tmp_dir / "analysis.json"
    analyze_trajectories(traj_dir=synthetic_trajectory_dir, output=output)
    analysis = json.loads(output.read_text())
    assert analysis["n_samples"] == 5
    assert analysis["n_scored"] == 5


def test_analysis_layer_metrics_are_delta_arrays(synthetic_trajectory_dir, tmp_dir):
    """layer_metrics must contain per-layer delta arrays, not global scalars."""
    output = tmp_dir / "analysis.json"
    analyze_trajectories(traj_dir=synthetic_trajectory_dir, output=output)
    analysis = json.loads(output.read_text())
    lm = analysis["layer_metrics"]

    expected = [
        "delta_energy", "delta_entropy", "delta_norm_entropy",
        "delta_top_activation", "delta_mean_activation",
    ]
    for metric in expected:
        assert metric in lm, f"Missing per-layer metric: {metric}"
        assert "mean" in lm[metric]
        assert len(lm[metric]["mean"]) == 4  # 4 layers in synthetic data

    # Scalar summaries live in global_divergences, not layer_metrics
    for scalar in ("energy_shift_l1", "energy_shift_l2", "gen_drift_mean"):
        assert scalar not in lm, (
            f"'{scalar}' should be in global_divergences, not layer_metrics"
        )


def test_analysis_global_divergences(synthetic_trajectory_dir, tmp_dir):
    """global_divergences must contain the interpretable scalar summaries."""
    output = tmp_dir / "analysis.json"
    analyze_trajectories(traj_dir=synthetic_trajectory_dir, output=output)
    analysis = json.loads(output.read_text())
    gd = analysis["global_divergences"]
    for metric in ("energy_shift_l1", "energy_shift_l2", "peak_delta_layer",
                   "gen_drift_mean", "gen_drift_max"):
        assert metric in gd, f"Missing scalar summary: {metric}"
        assert "mean" in gd[metric]
        assert "std"  in gd[metric]


def test_analysis_score_summary(synthetic_trajectory_dir, tmp_dir):
    output = tmp_dir / "analysis.json"
    analyze_trajectories(traj_dir=synthetic_trajectory_dir, output=output)
    ss = json.loads(output.read_text())["score_summary"]
    for key in ("count", "mean", "std", "min", "max"):
        assert key in ss
    assert ss["count"] == 5


def test_analysis_categories(synthetic_trajectory_dir, tmp_dir):
    output = tmp_dir / "analysis.json"
    analyze_trajectories(traj_dir=synthetic_trajectory_dir, output=output)
    cats = json.loads(output.read_text())["score_by_category"]
    assert "test" in cats
    assert cats["test"]["count"] == 5


def test_analysis_score_metric_delta_energy(synthetic_trajectory_dir, tmp_dir):
    output = tmp_dir / "analysis.json"
    analyze_trajectories(
        traj_dir=synthetic_trajectory_dir, output=output, score_metric="delta_energy"
    )
    analysis = json.loads(output.read_text())
    assert analysis["score_metric"] == "delta_energy"
    assert analysis["n_scored"] == 5


def test_analysis_with_signal_zone(synthetic_trajectory_dir, tmp_dir):
    output = tmp_dir / "analysis.json"
    analyze_trajectories(
        traj_dir=synthetic_trajectory_dir, output=output, signal_zone=(1, 3)
    )
    assert json.loads(output.read_text())["n_scored"] == 5


def test_analysis_empty_dir_raises(tmp_dir):
    empty = tmp_dir / "empty"; empty.mkdir()
    with pytest.raises(ValueError, match="No trajectory samples found"):
        analyze_trajectories(traj_dir=empty, output=tmp_dir / "out.json")


def test_analysis_compatible_with_visualize(synthetic_trajectory_dir, tmp_dir):
    output = tmp_dir / "analysis.json"
    analyze_trajectories(traj_dir=synthetic_trajectory_dir, output=output)

    from hopfield_llm.visualization.plots import visualize_analysis
    plots_dir = tmp_dir / "plots"
    visualize_analysis(str(output), str(plots_dir))

    assert len(list(plots_dir.glob("*.png"))) > 0
