"""Feature aggregation tests (subsec:feature_design): mean pooling over ALL
generated positions, phi assembly order (eq:feature_vector), and the
beta-recompute path from cached score tensors."""

import numpy as np
import pytest

from energy_llm import metrics
from energy_llm.features import (
    PHI_GROUPS,
    feature_columns,
    recompute_sample_arrays_from_scores,
    sample_baselines_from_arrays,
    sample_features_from_arrays,
)

L, T = 3, 4
RNG = np.random.default_rng(0)


@pytest.fixture()
def arrays():
    return {
        "ref_lse_term": RNG.normal(size=L),
        "ref_quadratic_term": RNG.random(L),
        "ref_entropy": RNG.random(L) * 3,
        "ref_norm_entropy": RNG.random(L),
        "gen_lse_term": RNG.normal(size=(L, T)),
        "gen_quadratic_term": RNG.random((L, T)),
        "gen_entropy": RNG.random((L, T)) * 3,
        "gen_norm_entropy": RNG.random((L, T)),
        "gen_js": RNG.random((L, T)) * 0.5,
        "gen_hellinger_sq": RNG.random((L, T)) * 0.5,
        "token_logprobs": -RNG.random(T),
        "token_predictive_entropy": RNG.random(T) * 2,
    }


class TestPooling:
    def test_delta_energy_is_mean_over_all_tokens_minus_ref(self, arrays):
        feats = sample_features_from_arrays(arrays)
        gen_e = arrays["gen_lse_term"] + arrays["gen_quadratic_term"]
        ref_e = arrays["ref_lse_term"] + arrays["ref_quadratic_term"]
        np.testing.assert_allclose(feats["delta_energy"], gen_e.mean(axis=1) - ref_e)

    def test_mean_js_pools_all_positions(self, arrays):
        feats = sample_features_from_arrays(arrays)
        np.testing.assert_allclose(feats["mean_js"], arrays["gen_js"].mean(axis=1))
        np.testing.assert_allclose(
            feats["mean_hellinger_sq"], arrays["gen_hellinger_sq"].mean(axis=1)
        )

    def test_delta_norm_entropy(self, arrays):
        feats = sample_features_from_arrays(arrays)
        np.testing.assert_allclose(
            feats["delta_norm_entropy"],
            arrays["gen_norm_entropy"].mean(axis=1) - arrays["ref_norm_entropy"],
        )

    def test_baselines(self, arrays):
        base = sample_baselines_from_arrays(arrays)
        assert base["seq_logprob"] == pytest.approx(arrays["token_logprobs"].mean())
        assert base["token_entropy_mean"] == pytest.approx(
            arrays["token_predictive_entropy"].mean()
        )


class TestPhiAssembly:
    def test_phi_is_4L_group_major(self):
        cols = feature_columns(n_layers=L)
        assert len(cols) == 4 * L
        assert cols[0] == "delta_energy_l0"
        assert cols[L] == "delta_norm_entropy_l0"
        assert cols[2 * L] == "mean_js_l0"
        assert cols[3 * L] == "mean_hellinger_sq_l0"
        assert PHI_GROUPS == ["delta_energy", "delta_norm_entropy",
                              "mean_js", "mean_hellinger_sq"]

    def test_delta_entropy_not_in_phi(self):
        assert not any("delta_entropy" in c and "norm" not in c
                       for c in feature_columns(n_layers=L))


class TestBetaRecompute:
    def test_matches_online_computation(self):
        """Recomputing from cached scores at the same beta reproduces the
        online arrays exactly (float64 path, no storage rounding)."""
        K, beta = 32, 9.5
        s_ref = RNG.normal(size=(L, K))
        s_gen = RNG.normal(size=(L, T, K))
        quad_ref = RNG.random(L)
        quad_gen = RNG.random((L, T))
        cache = {
            "ref_retrieval_scores": s_ref,
            "gen_retrieval_scores": s_gen,
            "ref_quadratic_term": quad_ref,
            "gen_quadratic_term": quad_gen,
        }
        arrays = recompute_sample_arrays_from_scores(cache, beta)

        # online-equivalent reference computation, layer by layer
        for l in range(L):
            p_ref = metrics.retrieval_distribution(s_ref[l], beta)
            p_gen = metrics.retrieval_distribution(s_gen[l], beta)
            np.testing.assert_allclose(arrays["ref_lse_term"][l],
                                       metrics.lse_term(s_ref[l], beta))
            np.testing.assert_allclose(arrays["gen_entropy"][l], metrics.entropy(p_gen))
            np.testing.assert_allclose(
                arrays["gen_js"][l],
                metrics.js_nats(p_gen, p_ref[None, :], axis=1), rtol=1e-10)
            np.testing.assert_allclose(
                arrays["gen_hellinger_sq"][l],
                metrics.hellinger_sq(p_gen, p_ref[None, :], axis=1), rtol=1e-10)

    def test_quadratic_terms_pass_through(self):
        K = 16
        gen_quad = np.tile(np.arange(T, dtype=float), (L, 1))
        cache = {
            "ref_retrieval_scores": RNG.normal(size=(L, K)),
            "gen_retrieval_scores": RNG.normal(size=(L, T, K)),
            "ref_quadratic_term": np.arange(L, dtype=float),
            "gen_quadratic_term": gen_quad,
        }
        arrays = recompute_sample_arrays_from_scores(cache, 5.0)
        np.testing.assert_allclose(arrays["ref_quadratic_term"], np.arange(L))
        np.testing.assert_allclose(arrays["gen_quadratic_term"], gen_quad)
