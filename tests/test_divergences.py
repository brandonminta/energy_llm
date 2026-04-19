"""Tests for divergence primitives (KL, JS, Hellinger)."""

from __future__ import annotations

import pytest
import torch

from hopfield_llm.metrics.divergences import hellinger, js_divergence, kl_divergence


@pytest.fixture
def uniform_dist() -> torch.Tensor:
    return torch.ones(10) / 10.0


@pytest.fixture
def peaked_dist() -> torch.Tensor:
    d = torch.zeros(10)
    d[0] = 1.0
    return d


class TestKLDivergence:
    def test_identical_distributions(self, uniform_dist):
        result = kl_divergence(uniform_dist, uniform_dist)
        assert abs(result) < 1e-6

    def test_forward_positive(self, uniform_dist, peaked_dist):
        result = kl_divergence(uniform_dist, peaked_dist)
        assert result > 0.0

    def test_both_directions(self, uniform_dist, peaked_dist):
        fwd, rev = kl_divergence(uniform_dist, peaked_dist, direction="both")
        assert fwd > 0.0
        assert rev > 0.0
        # KL is asymmetric
        assert abs(fwd - rev) > 0.01

    def test_reverse_direction(self, uniform_dist, peaked_dist):
        rev = kl_divergence(uniform_dist, peaked_dist, direction="reverse")
        assert rev > 0.0

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            kl_divergence(torch.ones(5), torch.ones(3))

    def test_rejects_non_1d(self):
        with pytest.raises(ValueError, match="1D"):
            kl_divergence(torch.ones(3, 3), torch.ones(3, 3))


class TestJSDivergence:
    def test_identical_distributions(self, uniform_dist):
        result = js_divergence(uniform_dist, uniform_dist)
        assert abs(result) < 1e-6

    def test_symmetric(self, uniform_dist, peaked_dist):
        js_ab = js_divergence(uniform_dist, peaked_dist)
        js_ba = js_divergence(peaked_dist, uniform_dist)
        assert abs(js_ab - js_ba) < 1e-6

    def test_bounded(self, uniform_dist, peaked_dist):
        result = js_divergence(uniform_dist, peaked_dist)
        # JS divergence is bounded by ln(2) ≈ 0.693
        assert 0.0 <= result <= 0.7


class TestHellinger:
    def test_identical_distributions(self, uniform_dist):
        result = hellinger(uniform_dist, uniform_dist)
        assert abs(result) < 1e-5

    def test_symmetric(self, uniform_dist, peaked_dist):
        h_ab = hellinger(uniform_dist, peaked_dist)
        h_ba = hellinger(peaked_dist, uniform_dist)
        assert abs(h_ab - h_ba) < 1e-6

    def test_bounded_zero_to_one(self, uniform_dist, peaked_dist):
        result = hellinger(uniform_dist, peaked_dist)
        assert 0.0 <= result <= 1.0

    def test_positive_for_different_dists(self, uniform_dist, peaked_dist):
        result = hellinger(uniform_dist, peaked_dist)
        assert result > 0.0
