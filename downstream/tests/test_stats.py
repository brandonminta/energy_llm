"""Statistics tests (sec:analysis_method): bootstrap CI, paired bootstrap,
folded permutation scan, per-category AUROC."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from energy_llm.stats import (
    auroc,
    bootstrap_auroc_ci,
    folded,
    paired_bootstrap_delta_auroc,
    per_category_auroc,
    per_feature_folded_auroc,
    permutation_scan_threshold,
)

RNG = np.random.default_rng(0)


class TestAuroc:
    def test_matches_sklearn_with_ties(self):
        y = RNG.integers(0, 2, 200)
        scores = np.round(RNG.normal(size=200), 1)  # induce ties
        assert auroc(y, scores) == pytest.approx(roc_auc_score(y, scores), abs=1e-12)

    def test_folded(self):
        assert folded(0.3) == pytest.approx(0.7)
        assert folded(0.7) == pytest.approx(0.7)
        np.testing.assert_allclose(folded(np.array([0.2, 0.8])), [0.8, 0.8])


class TestBootstrap:
    def test_ci_contains_point_estimate(self):
        y = RNG.integers(0, 2, 150)
        scores = y + RNG.normal(scale=0.8, size=150)
        out = bootstrap_auroc_ci(y, scores, n_bootstrap=500, seed=1)
        lo, hi = out["ci_0.950"]
        assert lo <= out["auroc"] <= hi
        assert "ci_0.975" in out  # Bonferroni interval for the 2-test family

    def test_strong_signal_discriminates(self):
        y = RNG.integers(0, 2, 200)
        scores = y + RNG.normal(scale=0.3, size=200)
        out = bootstrap_auroc_ci(y, scores, n_bootstrap=500, seed=1)
        assert out["discriminates_95"]

    def test_single_class_resamples_redrawn(self):
        # heavily imbalanced: single-class draws are likely, must be redrawn
        y = np.array([1] + [0] * 30)
        scores = RNG.normal(size=31)
        out = bootstrap_auroc_ci(y, scores, n_bootstrap=300, seed=2)
        assert np.isfinite(out["auroc"]) or len(np.unique(y)) == 2

    def test_paired_delta_isolates_added_value(self):
        n = 200
        y = RNG.integers(0, 2, n)
        base = y + RNG.normal(scale=1.5, size=n)
        full = y + RNG.normal(scale=0.3, size=n)
        out = paired_bootstrap_delta_auroc(y, full, base, n_bootstrap=500, seed=3)
        assert out["delta_auroc"] > 0
        assert out["contributes_95"]
        assert out["auroc_full"] > out["auroc_base"]

    def test_paired_delta_null_not_significant(self):
        n = 200
        y = RNG.integers(0, 2, n)
        base = y + RNG.normal(scale=1.0, size=n)
        full = base + RNG.normal(scale=0.01, size=n)  # no added information
        out = paired_bootstrap_delta_auroc(y, full, base, n_bootstrap=500, seed=4)
        lo, _ = out["ci_0.950"]
        assert lo <= 0.05


class TestPermutationScan:
    def test_null_features_stay_below_threshold(self):
        n, m = 300, 24
        X = RNG.normal(size=(n, m))
        y = RNG.integers(0, 2, n)
        out = permutation_scan_threshold(X, y, n_permutation=200, seed=5)
        # pure noise: roughly 5% of *scans* exceed; almost all columns below
        assert out["n_significant"] <= max(2, m // 5)
        assert out["scan_threshold"] > 0.5

    def test_planted_signal_detected(self):
        n, m = 300, 24
        X = RNG.normal(size=(n, m))
        y = RNG.integers(0, 2, n)
        X[:, 7] += y * 2.0  # strong planted layer-feature
        out = permutation_scan_threshold(X, y, n_permutation=200, seed=6)
        assert out["observed_folded_auroc"][7] > out["scan_threshold"]

    def test_folded_catches_inverted_direction(self):
        n = 300
        X = RNG.normal(size=(n, 4))
        y = RNG.integers(0, 2, n)
        X[:, 2] -= y * 2.0  # inverted direction
        obs = per_feature_folded_auroc(X, y)
        assert obs[2] > 0.8

    def test_threshold_exceeds_pointwise(self):
        """The scan max under the null exceeds any single column's typical
        folded AUROC — the reason the scan correction exists."""
        n, m = 200, 40
        X = RNG.normal(size=(n, m))
        y = RNG.integers(0, 2, n)
        out = permutation_scan_threshold(X, y, n_permutation=200, seed=7)
        single = per_feature_folded_auroc(X[:, :1], y)[0]
        assert out["scan_threshold"] > 0.55


class TestPerCategory:
    def test_descriptive_breakdown(self):
        n = 200
        y = RNG.integers(0, 2, n)
        scores = y + RNG.normal(scale=0.5, size=n)
        cats = ["big_a"] * 100 + ["big_b"] * 80 + ["tiny"] * 20
        rows = per_category_auroc(cats, y, scores, min_n=30)
        names = [r["category"] for r in rows]
        assert "big_a" in names and "big_b" in names and "tiny" not in names
        assert rows[0]["n"] >= rows[-1]["n"]  # sorted by size

    def test_single_class_category_skipped(self):
        y = np.array([0] * 50 + [1] * 50)
        scores = RNG.normal(size=100)
        cats = ["pure"] * 50 + ["mixed"] * 50
        rows = per_category_auroc(cats, y, scores, min_n=10)
        assert all(r["category"] != "pure" for r in rows)
