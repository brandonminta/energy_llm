"""Unit tests for energy_llm.metrics: JS convention (eq:js_convention),
Hellinger^2 (eq:hellinger_squared), entropy, and the separable energy
decomposition (eq:modern_hopfield_energy)."""

import numpy as np
import pytest

from energy_llm import metrics

RNG = np.random.default_rng(42)


def random_dist(k: int) -> np.ndarray:
    p = RNG.random(k) + 1e-3
    return p / p.sum()


class TestJSConvention:
    def test_disjoint_supports_equal_log2(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert metrics.js_nats(p, q) == pytest.approx(np.log(2), abs=1e-12)

    def test_identical_is_zero(self):
        p = random_dist(64)
        assert metrics.js_nats(p, p.copy()) == pytest.approx(0.0, abs=1e-12)

    def test_analytic_value(self):
        # P=(1/2,1/2), Q=(1,0): M=(3/4,1/4);
        # JS = H(M) - H(P)/2 - H(Q)/2 in nats
        p = np.array([0.5, 0.5])
        q = np.array([1.0, 0.0])
        h_m = -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
        expected = h_m - 0.5 * np.log(2)
        assert metrics.js_nats(p, q) == pytest.approx(expected, rel=1e-10)

    def test_symmetric(self):
        p, q = random_dist(128), random_dist(128)
        assert metrics.js_nats(p, q) == pytest.approx(metrics.js_nats(q, p), rel=1e-12)

    def test_bounded_by_log2(self):
        for _ in range(20):
            p, q = random_dist(256), random_dist(256)
            assert 0.0 <= metrics.js_nats(p, q) <= np.log(2) + 1e-12

    def test_matches_scipy_convention_exactly(self):
        from scipy.spatial.distance import jensenshannon
        p, q = random_dist(64), random_dist(64)
        expected = jensenshannon(p, q, base=2) ** 2 * np.log(2)
        assert metrics.js_nats(p, q) == pytest.approx(expected, rel=0, abs=0)

    def test_vectorized_axis(self):
        p = np.stack([random_dist(32) for _ in range(5)])
        q = random_dist(32)
        vec = metrics.js_nats(p, q[None, :], axis=1)
        for t in range(5):
            assert vec[t] == pytest.approx(metrics.js_nats(p[t], q), rel=1e-12)


class TestHellinger:
    def test_disjoint_is_one(self):
        assert metrics.hellinger_sq(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)

    def test_identical_is_zero(self):
        p = random_dist(64)
        assert metrics.hellinger_sq(p, p) == pytest.approx(0.0, abs=1e-15)

    def test_symmetric_and_bounded(self):
        p, q = random_dist(128), random_dist(128)
        h = metrics.hellinger_sq(p, q)
        assert h == pytest.approx(metrics.hellinger_sq(q, p), rel=1e-14)
        assert 0.0 <= h <= 1.0

    def test_bhattacharyya_identity(self):
        p, q = random_dist(64), random_dist(64)
        bc = np.sum(np.sqrt(p * q))
        assert metrics.hellinger_sq(p, q) == pytest.approx(1.0 - bc, rel=1e-12)


class TestEntropy:
    def test_uniform_max(self):
        k = 100
        p = np.full(k, 1.0 / k)
        assert metrics.entropy(p) == pytest.approx(np.log(k), rel=1e-12)
        assert metrics.norm_entropy(p) == pytest.approx(1.0, rel=1e-12)

    def test_point_mass_zero(self):
        p = np.zeros(10)
        p[3] = 1.0
        assert metrics.entropy(p) == pytest.approx(0.0, abs=1e-15)
        assert metrics.norm_entropy(p) == pytest.approx(0.0, abs=1e-15)


class TestEnergyDecomposition:
    def test_terms_sum_to_reference_energy(self):
        s = RNG.normal(size=64)
        r = RNG.normal(size=16)
        beta = 7.0
        naive = -np.log(np.sum(np.exp(beta * s))) / beta + 0.5 * np.sum(r**2)
        lse = metrics.lse_term(s, beta)
        quad = metrics.quadratic_term(r)
        assert lse + quad == pytest.approx(naive, rel=1e-12)
        assert metrics.energy(s, r, beta) == pytest.approx(naive, rel=1e-12)

    def test_lse_stable_at_large_beta(self):
        s = RNG.normal(size=512) * 10
        value = metrics.lse_term(s, beta=1e5)
        assert np.isfinite(value)
        # at huge beta, lse_term -> -max(s)
        assert value == pytest.approx(-s.max(), rel=1e-4)

    def test_retrieval_distribution_normalised(self):
        s = RNG.normal(size=(5, 64))
        p = metrics.retrieval_distribution(s, beta=15.0)
        np.testing.assert_allclose(p.sum(axis=-1), 1.0, rtol=1e-12)
        assert (p > 0).all()
