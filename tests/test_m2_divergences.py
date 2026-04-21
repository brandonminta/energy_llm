"""Tests for M2: retrieval-distribution divergences.

Case 1: KL on identical distributions is zero.
Case 2: compute_sample_divergences with None distributions.
Case 3: compute_sample_divergences end-to-end on real data (requires model).
Case 4: skip_first_gen_token correctly reduces the T dimension (requires model).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def llm():
    try:
        from hopfield_llm.models.loader import HFLLM
        return HFLLM("qwen25_1_5b", load_in_4bit=False)
    except Exception as exc:
        pytest.skip(f"qwen25_1_5b not available: {exc}")


@pytest.fixture(scope="module")
def banks(llm):
    from hopfield_llm.memory.banks import extract_banks
    return extract_banks(llm, bank="up", normalize=True)


# ------------------------------------------------------------------
# Case 1: KL on identical distributions is zero
# ------------------------------------------------------------------


def test_kl_identical_is_zero():
    from hopfield_llm.metrics.divergences import kl_divergence

    torch.manual_seed(7)
    p = torch.softmax(torch.randn(256), dim=0)
    result = kl_divergence(p, p.clone(), direction="forward")
    assert result < 1e-6, f"KL(p||p) expected < 1e-6, got {result:.2e}"


def test_kl_forward_nonnegative():
    from hopfield_llm.metrics.divergences import kl_divergence

    torch.manual_seed(42)
    p = torch.softmax(torch.randn(64), dim=0)
    q = torch.softmax(torch.randn(64), dim=0)
    assert kl_divergence(p, q) >= 0.0


# ------------------------------------------------------------------
# Case 2: compute_sample_divergences with None distributions
# ------------------------------------------------------------------


def test_sample_divergences_none_distributions():
    from hopfield_llm.metrics.divergences import compute_sample_divergences

    rng = np.random.RandomState(0)
    L, T = 10, 20
    prompt_energy = rng.randn(L).astype(np.float32)
    gen_energy    = rng.randn(L, T).astype(np.float32)

    result = compute_sample_divergences(
        prompt_distributions=None,
        prompt_energy=prompt_energy,
        gen_distributions=None,
        gen_energy=gen_energy,
    )

    # KL arrays must be all NaN when distributions are unavailable
    assert np.all(np.isnan(result.kl_gen_to_prompt_per_layer)), \
        "kl_gen_to_prompt_per_layer should be all NaN when distributions=None"
    assert np.all(np.isnan(result.js_gen_to_prompt_per_layer))
    assert np.all(np.isnan(result.hellinger_gen_to_prompt_per_layer))

    # Energy scalars must be finite
    assert np.isfinite(result.energy_shift_l1), "energy_shift_l1 should be finite"
    assert np.isfinite(result.energy_shift_l2)
    assert result.energy_shift_l1 >= 0.0
    assert result.energy_shift_l2 >= 0.0

    # gen_drift scalars are NaN (no distributions)
    assert np.isnan(result.gen_drift_mean)
    assert np.isnan(result.gen_drift_max)

    # delta_energy should be finite (computed from energy arrays)
    assert np.all(np.isfinite(result.delta_energy)), \
        "delta_energy should be finite when energy arrays are valid"


def test_sample_divergences_none_shapes():
    """Shape of NaN arrays must be [L, T_gen]."""
    from hopfield_llm.metrics.divergences import compute_sample_divergences

    L, T = 7, 13
    result = compute_sample_divergences(
        prompt_distributions=None,
        prompt_energy=np.zeros(L, dtype=np.float32),
        gen_distributions=None,
        gen_energy=np.zeros((L, T), dtype=np.float32),
    )
    assert result.kl_gen_to_prompt_per_layer.shape == (L, T)
    assert result.delta_energy.shape == (L,)


# ------------------------------------------------------------------
# Case 3: end-to-end on real data
# ------------------------------------------------------------------


def test_sample_divergences_end_to_end(llm, banks):
    from hopfield_llm.hooks.capture import capture_generation, capture_prefill
    from hopfield_llm.metrics.divergences import compute_sample_divergences

    question = "What is the capital of France?"

    prefill = capture_prefill(
        llm, question, banks=banks, beta=1.0,
        hook_target="mlp_input", normalize_query=True,
        save_distributions=True,
    )
    gen = capture_generation(
        llm, question, banks=banks, beta=1.0,
        hook_target="mlp_input", normalize_query=True,
        max_new_tokens=15, save_distributions=True,
    )

    assert prefill.distributions is not None, "prefill.distributions should be set"
    assert gen.distributions     is not None, "gen.distributions should be set"

    n_layers  = llm.n_layers
    T_gen     = gen.distributions.shape[1]

    result = compute_sample_divergences(
        prompt_distributions=prefill.distributions,   # [L, T_prompt, K]
        prompt_energy=prefill.energy_last,            # [L]
        gen_distributions=gen.distributions,          # [L, T_gen, K]
        gen_energy=gen.energy,                        # [L, T_gen]
        skip_first_gen_token=True,
    )

    T_eff = T_gen - 1  # one token dropped by skip_first_gen_token

    # Shape check
    assert result.kl_gen_to_prompt_per_layer.shape == (n_layers, T_eff), (
        f"Expected ({n_layers}, {T_eff}), got {result.kl_gen_to_prompt_per_layer.shape}"
    )

    # No NaNs in KL/JS/Hellinger arrays
    assert not np.any(np.isnan(result.kl_gen_to_prompt_per_layer)), \
        "kl_gen_to_prompt_per_layer should have no NaNs"
    assert not np.any(np.isnan(result.js_gen_to_prompt_per_layer))
    assert not np.any(np.isnan(result.hellinger_gen_to_prompt_per_layer))

    # All KL values must be >= 0
    assert np.all(result.kl_gen_to_prompt_per_layer >= 0.0), \
        "KL divergence must be non-negative"
    assert np.all(result.js_gen_to_prompt_per_layer >= 0.0)
    assert np.all(result.hellinger_gen_to_prompt_per_layer >= 0.0)


# ------------------------------------------------------------------
# Case 4: skip_first_gen_token reduces T dimension by 1
# ------------------------------------------------------------------


def test_skip_first_gen_token_shape(llm, banks):
    from hopfield_llm.hooks.capture import capture_generation, capture_prefill
    from hopfield_llm.metrics.divergences import compute_sample_divergences

    question = "Name a large planet in our solar system."

    prefill = capture_prefill(
        llm, question, banks=banks, beta=1.0,
        save_distributions=True,
    )
    gen = capture_generation(
        llm, question, banks=banks, beta=1.0,
        max_new_tokens=10, save_distributions=True,
    )

    assert gen.distributions is not None
    T_gen = gen.distributions.shape[1]

    result_skip = compute_sample_divergences(
        prompt_distributions=prefill.distributions,
        prompt_energy=prefill.energy_last,
        gen_distributions=gen.distributions,
        gen_energy=gen.energy,
        skip_first_gen_token=True,
    )
    result_no_skip = compute_sample_divergences(
        prompt_distributions=prefill.distributions,
        prompt_energy=prefill.energy_last,
        gen_distributions=gen.distributions,
        gen_energy=gen.energy,
        skip_first_gen_token=False,
    )

    assert result_skip.kl_gen_to_prompt_per_layer.shape[1] == T_gen - 1, (
        f"With skip_first_gen_token=True: expected T={T_gen - 1}, "
        f"got {result_skip.kl_gen_to_prompt_per_layer.shape[1]}"
    )
    assert result_no_skip.kl_gen_to_prompt_per_layer.shape[1] == T_gen, (
        f"With skip_first_gen_token=False: expected T={T_gen}, "
        f"got {result_no_skip.kl_gen_to_prompt_per_layer.shape[1]}"
    )
