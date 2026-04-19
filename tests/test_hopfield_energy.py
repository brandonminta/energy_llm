"""Tests for hopfield_energy_mlp."""

from __future__ import annotations

import pytest
import torch

from hopfield_llm.metrics.hopfield import LayerEnergyResult, hopfield_energy_mlp


def test_returns_layer_energy_result(synthetic_h_pre, synthetic_banks):
    result = hopfield_energy_mlp(synthetic_h_pre, synthetic_banks[0])
    assert isinstance(result, LayerEnergyResult)


def test_energy_is_finite(synthetic_h_pre, synthetic_banks):
    result = hopfield_energy_mlp(synthetic_h_pre, synthetic_banks[0])
    assert result.energy != float("inf")
    assert result.energy != float("-inf")
    assert result.energy == result.energy  # not NaN


def test_entropy_nonnegative(synthetic_h_pre, synthetic_banks):
    result = hopfield_energy_mlp(synthetic_h_pre, synthetic_banks[0])
    assert result.entropy >= 0.0


def test_norm_entropy_bounded(synthetic_h_pre, synthetic_banks):
    result = hopfield_energy_mlp(synthetic_h_pre, synthetic_banks[0])
    assert 0.0 <= result.norm_entropy <= 1.0


def test_higher_beta_lower_entropy(synthetic_banks):
    """Higher beta (sharper retrieval) should produce lower entropy."""
    torch.manual_seed(99)
    h = torch.randn(128)
    r_low = hopfield_energy_mlp(h, synthetic_banks[0], beta=1.0)
    r_high = hopfield_energy_mlp(h, synthetic_banks[0], beta=50.0)
    assert r_high.entropy < r_low.entropy


def test_cosine_mode(synthetic_h_pre, synthetic_banks):
    result = hopfield_energy_mlp(
        synthetic_h_pre, synthetic_banks[0], energy_mode="cosine"
    )
    assert isinstance(result, LayerEnergyResult)
    assert result.quadratic == 0.5  # cosine mode uses constant quadratic


def test_n_active_nonnegative(synthetic_h_pre, synthetic_banks):
    result = hopfield_energy_mlp(synthetic_h_pre, synthetic_banks[0])
    assert result.n_active >= 0


def test_rejects_wrong_shapes():
    with pytest.raises(ValueError, match="1D"):
        hopfield_energy_mlp(torch.randn(4, 4), torch.randn(8, 4))

    with pytest.raises(ValueError, match="2D"):
        hopfield_energy_mlp(torch.randn(4), torch.randn(4))

    with pytest.raises(ValueError, match="Shape mismatch"):
        hopfield_energy_mlp(torch.randn(4), torch.randn(8, 5))


def test_rejects_unknown_energy_mode(synthetic_h_pre, synthetic_banks):
    with pytest.raises(ValueError, match="Unknown energy_mode"):
        hopfield_energy_mlp(synthetic_h_pre, synthetic_banks[0], energy_mode="bad")
