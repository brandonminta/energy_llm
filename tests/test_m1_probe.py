"""Tests for M1: theoretically-correct Modern Hopfield probe.

Case 1: Bank shape invariants (requires model).
Case 2: Hook captures correct-shape query (requires model).
Case 3: Factual vs gibberish sanity check (requires model + GPU recommended).

Tests that require a model are skipped automatically when the model or a CUDA
device is not available.
"""

from __future__ import annotations

import pytest
import torch

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def llm():
    """Load qwen25_1_5b without 4-bit quantisation for CPU compatibility."""
    try:
        from hopfield_llm.models.loader import HFLLM
        model = HFLLM("qwen25_1_5b", load_in_4bit=False)
        return model
    except Exception as exc:
        pytest.skip(f"qwen25_1_5b not available: {exc}")


# ------------------------------------------------------------------
# Case 1: Bank shape invariants
# ------------------------------------------------------------------


def test_gate_plus_up_raises():
    """Removed composite bank must raise ValueError immediately."""
    from hopfield_llm.memory.banks import extract_banks

    class _FakeLLM:
        class model:
            class config:
                hidden_size = 64
                intermediate_size = 128
        layers = []

    with pytest.raises(ValueError, match="gate_plus_up"):
        extract_banks(_FakeLLM(), bank="gate_plus_up")


def test_gate_up_concat_raises():
    from hopfield_llm.memory.banks import extract_banks

    class _FakeLLM:
        class model:
            class config:
                hidden_size = 64
                intermediate_size = 128
        layers = []

    with pytest.raises(ValueError, match="gate_up_concat"):
        extract_banks(_FakeLLM(), bank="gate_up_concat")


def test_up_bank_shape(llm):
    """bank='up' with normalize=True: rows [intermediate_size, hidden_size], unit norm."""
    from hopfield_llm.memory.banks import extract_banks

    banks = extract_banks(llm, bank="up", normalize=True)
    d_m = llm.model.config.intermediate_size
    d   = llm.model.config.hidden_size

    for l_idx, w in banks.items():
        assert w.shape == (d_m, d), (
            f"Layer {l_idx}: expected ({d_m}, {d}), got {tuple(w.shape)}"
        )
        row_norms = w.norm(dim=-1)
        assert (row_norms - 1.0).abs().max().item() < 1e-5, (
            f"Layer {l_idx}: rows not unit-normalised, max_err={((row_norms-1).abs().max()):.2e}"
        )


def test_down_values_bank_shape(llm):
    """bank='down_values' with normalize=True: rows [intermediate_size, hidden_size]."""
    from hopfield_llm.memory.banks import extract_banks

    banks = extract_banks(llm, bank="down_values", normalize=True)
    d_m = llm.model.config.intermediate_size
    d   = llm.model.config.hidden_size

    for l_idx, w in banks.items():
        assert w.shape == (d_m, d), (
            f"Layer {l_idx}: expected ({d_m}, {d}), got {tuple(w.shape)}"
        )


def test_bank_metadata_has_M(llm):
    """Metadata dict must contain 'M' key with a positive float for every layer."""
    from hopfield_llm.memory.banks import extract_banks

    banks, meta = extract_banks(llm, bank="up", normalize=True, return_metadata=True)
    for l_idx in banks:
        assert "M" in meta[l_idx], f"Layer {l_idx}: 'M' missing from metadata"
        assert meta[l_idx]["M"] > 0.0, f"Layer {l_idx}: M={meta[l_idx]['M']} is not positive"


def test_down_bank_emits_future_warning(llm):
    """bank='down' must emit FutureWarning pointing to down_values."""
    import warnings
    from hopfield_llm.memory.banks import extract_banks

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        extract_banks(llm, bank="down", normalize=True)

    future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
    assert future_warnings, "No FutureWarning emitted for bank='down'"
    assert "down_values" in str(future_warnings[0].message)


# ------------------------------------------------------------------
# Case 2: Hook captures correct-shape query
# ------------------------------------------------------------------


def test_prefill_hook_mlp_input_shape(llm):
    """hook_target='mlp_input' must produce x_stacked of shape [n_layers, T_prompt, hidden_size]."""
    from hopfield_llm.hooks.capture import capture_prefill
    from hopfield_llm.memory.banks import extract_banks

    banks = extract_banks(llm, bank="up", normalize=True)
    n_layers = llm.n_layers
    hidden_size = llm.model.config.hidden_size

    _, x_stacked = capture_prefill(
        llm, "The quick brown fox",
        banks=banks, beta=1.0,
        hook_target="mlp_input",
        return_x=True,
    )
    assert x_stacked.ndim == 3, f"Expected 3-D x_stacked, got shape {x_stacked.shape}"
    assert x_stacked.shape[0] == n_layers, (
        f"mlp_input: expected L={n_layers}, got {x_stacked.shape[0]}"
    )
    assert x_stacked.shape[2] == hidden_size, (
        f"mlp_input: expected d={hidden_size}, got {x_stacked.shape[2]}"
    )


def test_prefill_hook_mlp_output_shape(llm):
    """hook_target='mlp_output' must produce x_stacked of shape [n_layers, T_prompt, hidden_size]."""
    from hopfield_llm.hooks.capture import capture_prefill
    from hopfield_llm.memory.banks import extract_banks

    banks = extract_banks(llm, bank="down_values", normalize=True)
    n_layers = llm.n_layers
    hidden_size = llm.model.config.hidden_size

    _, x_stacked = capture_prefill(
        llm, "The quick brown fox",
        banks=banks, beta=1.0,
        hook_target="mlp_output",
        return_x=True,
    )
    assert x_stacked.ndim == 3, f"Expected 3-D x_stacked, got shape {x_stacked.shape}"
    assert x_stacked.shape[0] == n_layers, (
        f"mlp_output: expected L={n_layers}, got {x_stacked.shape[0]}"
    )
    assert x_stacked.shape[2] == hidden_size, (
        f"mlp_output: expected d={hidden_size}, got {x_stacked.shape[2]}"
    )


# ------------------------------------------------------------------
# Case 3: Factual vs gibberish top_act sanity check
# ------------------------------------------------------------------


def test_factual_higher_top_act_than_gibberish(llm):
    """Middle-third layers must show higher top_act for factual than gibberish text."""
    import numpy as np
    from hopfield_llm.hooks.capture import capture_prefill
    from hopfield_llm.memory.banks import extract_banks

    banks = extract_banks(llm, bank="up", normalize=True)
    n_layers = llm.n_layers

    factual    = "The capital of France is"
    gibberish  = "xkcdflq mlqpwo zznxn qqwerzz"

    r_fact = capture_prefill(
        llm, factual, banks=banks, beta=1.0,
        hook_target="mlp_input", normalize_query=True,
    )
    r_gibb = capture_prefill(
        llm, gibberish, banks=banks, beta=1.0,
        hook_target="mlp_input", normalize_query=True,
    )

    lo = n_layers // 3
    hi = (2 * n_layers) // 3
    middle_layers = range(lo, hi)

    # Use last-token top_act for comparison (backward-compat with [L, T] shape)
    fact_top_act = r_fact.top_act[:, -1]
    gibb_top_act = r_gibb.top_act[:, -1]

    margin = 0.05
    passed_any = any(
        fact_top_act[l] > gibb_top_act[l] + margin
        for l in middle_layers
    )

    if not passed_any:
        print("\nFactual  top_act (last token):", fact_top_act)
        print("Gibberish top_act (last token):", gibb_top_act)
        diff = fact_top_act[lo:hi] - gibb_top_act[lo:hi]
        raise AssertionError(
            f"No middle-third layer (layers {lo}–{hi-1}) has "
            f"top_act[factual] > top_act[gibberish] + {margin}. "
            f"Middle-layer diffs: {diff}"
        )
