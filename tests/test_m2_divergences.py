"""Tests for M2: retrieval-distribution divergences.

Case 1: KL/JS/Hellinger torch primitives (1-D).
Case 2: compute_sample_divergences — energy scalars always finite.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


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
# Case 2: compute_sample_divergences — energy scalars always finite
# ------------------------------------------------------------------


def test_sample_divergences_energy_scalars():
    from hopfield_llm.metrics.divergences import compute_sample_divergences

    rng = np.random.RandomState(0)
    L, T = 10, 20
    result = compute_sample_divergences(
        prompt_energy=rng.randn(L).astype(np.float32),
        gen_energy=rng.randn(L, T).astype(np.float32),
    )

    assert np.isfinite(result.energy_shift_l1)
    assert np.isfinite(result.energy_shift_l2)
    assert result.energy_shift_l1 >= 0.0
    assert result.energy_shift_l2 >= 0.0
    assert result.delta_energy.shape == (L,)
    assert np.all(np.isfinite(result.delta_energy))
