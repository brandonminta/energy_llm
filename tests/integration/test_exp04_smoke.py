"""Smoke tests for the evaluation pipeline (load → AUROC → probe).

Replicates the critical path of exp-04 without GPU or real data:
  1. load_all_features — reads synthetic trajectory + label files
  2. per_layer_auroc_table — returns a DataFrame with per-layer AUROCs
  3. fit_logreg_probe — runs nested-CV without crashing

All tests use the shared synthetic_trajectory_dir fixture (5 samples, L=4, T=10)
with synthetic labels written inline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hopfield_llm.evaluation.features import load_all_features
from hopfield_llm.evaluation.per_layer_auroc import per_layer_auroc_table, best_single_feature
from hopfield_llm.evaluation.probe import fit_logreg_probe


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def labeled_trajectory_dir(synthetic_trajectory_dir: Path) -> tuple[Path, Path]:
    """Return (traj_dir, labels_dir) with synthetic binary labels."""
    labels_dir = synthetic_trajectory_dir.parent / "labels"
    labels_dir.mkdir()
    # Alternate hallucination labels: 0,1,0,1,0 → balanced enough for AUROC
    for i in range(5):
        sid = f"test_{i}"
        (labels_dir / f"{sid}_label.json").write_text(
            json.dumps({
                "id": sid,
                "is_hallucination": bool(i % 2),
                "correctness_score": 1.0 - (i % 2),
                "label_source": "heuristic",
                "label_version": "1.0",
            }),
            encoding="utf-8",
        )
    return synthetic_trajectory_dir, labels_dir


# ── load_all_features ─────────────────────────────────────────────────────────

def test_load_all_features_returns_labeled_samples(labeled_trajectory_dir):
    traj_dir, labels_dir = labeled_trajectory_dir
    samples = load_all_features(traj_dir, labels_dir)
    assert len(samples) == 5


def test_load_all_features_has_required_keys(labeled_trajectory_dir):
    traj_dir, labels_dir = labeled_trajectory_dir
    samples = load_all_features(traj_dir, labels_dir)
    required = {"is_hallucination", "delta_energy",
                "gen_energy_answer", "gen_entropy_answer",
                "prefill_energy_last"}
    for s in samples:
        for k in required:
            assert k in s, f"Missing key '{k}' in sample {s.get('id')}"


def test_load_all_features_per_layer_shape(labeled_trajectory_dir):
    traj_dir, labels_dir = labeled_trajectory_dir
    samples = load_all_features(traj_dir, labels_dir)
    for s in samples:
        assert s["delta_energy"].shape == (4,), \
            f"Expected shape (4,) got {s['delta_energy'].shape}"


def test_load_all_features_skips_unlabeled(synthetic_trajectory_dir):
    # No label files written → should return empty list
    samples = load_all_features(synthetic_trajectory_dir)
    assert samples == []


# ── per_layer_auroc_table ─────────────────────────────────────────────────────

def test_auroc_table_shape(labeled_trajectory_dir):
    traj_dir, labels_dir = labeled_trajectory_dir
    samples = load_all_features(traj_dir, labels_dir)
    keys = ["delta_energy", "gen_energy_answer"]
    df = per_layer_auroc_table(samples, keys)
    # 4 layer rows + 1 best_layer row
    assert len(df) == 5
    assert list(df.columns) == keys


def test_auroc_table_best_layer_row(labeled_trajectory_dir):
    traj_dir, labels_dir = labeled_trajectory_dir
    samples = load_all_features(traj_dir, labels_dir)
    keys = ["delta_energy", "gen_energy_answer"]
    df = per_layer_auroc_table(samples, keys)
    best = df.loc["best_layer"]
    for k in keys:
        assert 0 <= int(best[k]) < 4, f"best_layer for '{k}' out of range: {best[k]}"


def test_auroc_table_empty_inputs():
    df = per_layer_auroc_table([], ["delta_energy"])
    assert df.empty


def test_best_single_feature(labeled_trajectory_dir):
    traj_dir, labels_dir = labeled_trajectory_dir
    samples = load_all_features(traj_dir, labels_dir)
    keys = ["delta_energy", "gen_energy_answer"]
    df = per_layer_auroc_table(samples, keys)
    name, layer, auroc = best_single_feature(df)
    assert name in keys
    assert 0 <= layer < 4
    assert 0.0 <= auroc <= 1.0


# ── fit_logreg_probe ──────────────────────────────────────────────────────────

def test_fit_logreg_probe_runs(labeled_trajectory_dir):
    traj_dir, labels_dir = labeled_trajectory_dir
    samples = load_all_features(traj_dir, labels_dir)
    # 5 samples is too few for nested-CV; pad to 50 by repeating
    samples_padded = (samples * 10)[:50]
    result = fit_logreg_probe(samples_padded, feature_keys=["delta_energy", "gen_energy_answer"])
    assert "mean_auroc" in result
    assert "std_auroc" in result
    assert 0.0 <= result["mean_auroc"] <= 1.0


def test_fit_logreg_probe_too_few_samples_raises(labeled_trajectory_dir):
    traj_dir, labels_dir = labeled_trajectory_dir
    samples = load_all_features(traj_dir, labels_dir)
    # Fewer than 50 samples → probe raises ValueError
    with pytest.raises(ValueError, match="labeled samples"):
        fit_logreg_probe(samples[:3], feature_keys=["delta_energy"])
