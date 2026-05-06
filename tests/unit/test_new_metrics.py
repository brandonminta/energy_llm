"""Tests for the new temporal, retrieval-gap, and storage-compression metrics."""

from __future__ import annotations

import numpy as np
import pytest

from hopfield_llm.metrics.divergences import (
    SampleDivergenceResult,
    _temporal_energy_metrics,
    compute_sample_divergences,
)
from hopfield_llm.metrics.scoring import (
    ALL_METRICS,
    PER_LAYER_METRICS,
    hallucination_score,
)
from hopfield_llm.pipeline.analysis import (
    _top_act_gap_per_layer,
    _top_act_gap_per_layer_topk,
)
from hopfield_llm.pipeline.stages import _topk_compress


# ------------------------------------------------------------------
# Temporal metrics (slope / variance / spread)
# ------------------------------------------------------------------


def test_temporal_metrics_recover_known_signals():
    L, T = 3, 10
    rng = np.random.default_rng(0)
    # Layer 0: monotone rising line  → positive slope, zero spread above noise
    # Layer 1: flat line              → ~0 slope, near-0 variance
    # Layer 2: pure noise             → ~0 slope, larger variance
    line     = np.linspace(0.0, 1.0, T)
    flat     = np.full(T, 0.5)
    noise    = rng.normal(0.0, 1.0, T)
    gen_e    = np.stack([line, flat, noise], axis=0).astype(np.float32)

    slope, var, spread = _temporal_energy_metrics(gen_e)

    assert slope[0] > 0.05, "rising-line slope must be positive"
    assert abs(slope[1]) < 1e-6, "flat-line slope must be ~0"
    assert var[1] < 1e-6, "flat-line variance must be ~0"
    assert var[2] > 0.1, "noisy line should have non-trivial variance"
    assert spread[0] == pytest.approx(line.max() - line.min(), rel=1e-5)


def test_temporal_metrics_handle_all_nan_row():
    L, T = 2, 5
    gen_e = np.full((L, T), np.nan, dtype=np.float32)
    slope, var, spread = _temporal_energy_metrics(gen_e)
    assert np.all(np.isnan(slope))
    assert np.all(np.isnan(var))
    assert np.all(np.isnan(spread))


def test_compute_sample_divergences_populates_temporal_fields():
    L, T = 4, 6
    rng = np.random.default_rng(7)
    pre = rng.normal(size=L).astype(np.float32)
    gen = rng.normal(size=(L, T)).astype(np.float32)
    div = compute_sample_divergences(prompt_energy=pre, gen_energy=gen)
    assert div.energy_slope is not None  and div.energy_slope.shape  == (L,)
    assert div.energy_var is not None    and div.energy_var.shape    == (L,)
    assert div.energy_spread is not None and div.energy_spread.shape == (L,)


# ------------------------------------------------------------------
# Top1-Top2 gap
# ------------------------------------------------------------------


def test_top_act_gap_one_row_dominates():
    """Big top1 → bigger gap; uniform → tiny gap."""
    L, T, K = 2, 3, 64
    sc = np.zeros((L, T, K), dtype=np.float32)
    # Row 0: spike at index 0
    sc[0, :, 0] = 1.0
    # Row 1: uniform 0.5
    sc[1, :, :] = 0.5

    gap = _top_act_gap_per_layer(sc)
    assert gap.shape == (L,)
    assert gap[0] > 0.5, "spike should produce gap close to 1"
    assert gap[1] == pytest.approx(0.0, abs=1e-6)


def test_top_act_gap_topk_matches_full():
    rng = np.random.default_rng(3)
    L, T, K = 2, 3, 50
    sc = rng.normal(size=(L, T, K)).astype(np.float32)
    gap_full = _top_act_gap_per_layer(sc)

    topv, _idx, _tail = _topk_compress(sc, k=4)
    gap_topk = _top_act_gap_per_layer_topk(topv)

    np.testing.assert_allclose(gap_full, gap_topk, rtol=1e-3, atol=1e-3)


# ------------------------------------------------------------------
# topk_compress round-trip
# ------------------------------------------------------------------


def test_topk_compress_preserves_top_values():
    rng = np.random.default_rng(11)
    sc = rng.normal(size=(2, 3, 30)).astype(np.float32)
    topv, topi, tail_max = _topk_compress(sc, k=5)

    assert topv.shape == (2, 3, 5)
    assert topi.shape == (2, 3, 5)
    assert tail_max.shape == (2, 3)

    # Reconstruct top values via fancy indexing and confirm equality
    for l in range(2):
        for t in range(3):
            recovered = sc[l, t][topi[l, t]]
            np.testing.assert_allclose(recovered, topv[l, t].astype(np.float32), rtol=1e-3)
            # Top values must be ≥ tail_max (descending order maintained)
            assert topv[l, t, -1].astype(np.float32) >= tail_max[l, t].astype(np.float32) - 1e-3


# ------------------------------------------------------------------
# hallucination_score handles None optional fields gracefully
# ------------------------------------------------------------------


def test_hallucination_score_returns_nan_for_missing_optional():
    L = 3
    div = SampleDivergenceResult(
        delta_energy=np.zeros(L, dtype=np.float64),
        delta_entropy=np.zeros(L, dtype=np.float64),
        delta_norm_entropy=np.zeros(L, dtype=np.float64),
        delta_top_activation=np.zeros(L, dtype=np.float64),
        delta_lse=np.zeros(L, dtype=np.float64),
        energy_shift_l1=0.0, energy_shift_l2=0.0, peak_delta_layer=0,
        # All optional fields default to None
    )
    # Optional metrics should return NaN, not crash
    for m in ("top_act_gap_mean", "delta_logit_entropy",
              "gen_logit_entropy_mean", "gen_logit_top_prob_min",
              "delta_energy_null"):
        assert m in PER_LAYER_METRICS
        score = hallucination_score(div, metric=m)
        assert np.isnan(score), f"metric {m} should return NaN when None"


def test_all_metrics_listed_in_choices():
    """Defensive: every metric the CLI offers must be in ALL_METRICS."""
    expected = {
        "delta_energy", "delta_norm_entropy", "delta_entropy",
        "delta_top_activation", "delta_lse",
        "energy_slope", "energy_var", "energy_spread",
        "top_act_gap_mean",
        "delta_logit_entropy", "gen_logit_entropy_mean", "gen_logit_top_prob_min",
        "delta_energy_null",
        "js_mean_per_layer", "js_max_per_layer",
        "hellinger_mean_per_layer", "hellinger_max_per_layer",
        "energy_shift_l1", "energy_shift_l2", "peak_delta_layer",
    }
    assert expected.issubset(set(ALL_METRICS))
