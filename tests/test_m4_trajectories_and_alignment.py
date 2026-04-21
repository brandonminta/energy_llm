"""Tests for M4: full-trajectory prefill capture and answer-span alignment.

Case 1: PrefillResult shape contract — fields are [L, T_prompt]; energy_last and
        energy_mean are derived [L] scalars.
Case 2: capture_prefill returns full [L, T_prompt] trajectory (mock LLM).
Case 3: find_answer_span locates a gold answer in generated text.
Case 4: content_token_mask excludes leading/trailing whitespace tokens.
Case 5: extract_answer_features — span vs. fallback mask pooling.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from hopfield_llm.analysis.alignment import (
    content_token_mask,
    extract_answer_features,
    find_answer_span,
)
from hopfield_llm.hooks.capture import PrefillResult, capture_prefill


# ------------------------------------------------------------------
# Minimal fakes for capture_prefill tests
# ------------------------------------------------------------------


class _FakeMLP(nn.Module):
    """Identity MLP that lets pre-hooks fire with the input tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _FakeMLP()


class _FakeModel:
    """Calls each layer's mlp with a deterministic activation, triggering hooks."""

    def __init__(self, layers: list[_FakeLayer], T: int, d: int) -> None:
        self._layers = layers
        self._T = T
        self._d = d

    def __call__(self, **kwargs) -> None:  # noqa: ARG002
        x = torch.ones(1, self._T, self._d)
        for layer in self._layers:
            _ = layer.mlp(x)


class _FakeTokenizer:
    def __init__(self, T: int) -> None:
        self._T = T
        self.eos_token_id = 2
        self.pad_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt",
                 add_special_tokens: bool = False) -> dict:
        ids = torch.ones(1, self._T, dtype=torch.long)
        return {"input_ids": ids}


class _FakeLLM:
    def __init__(self, n_layers: int = 3, T: int = 5, d: int = 8) -> None:
        self.layers = [_FakeLayer() for _ in range(n_layers)]
        self.device = "cpu"
        self.tokenizer = _FakeTokenizer(T)
        self.model = _FakeModel(self.layers, T, d)


def _make_banks(n_layers: int, K: int = 4, d: int = 8) -> dict[int, torch.Tensor]:
    torch.manual_seed(0)
    return {i: torch.randn(K, d) for i in range(n_layers)}


# ------------------------------------------------------------------
# Minimal char-level tokenizer for alignment tests
# ------------------------------------------------------------------


class _CharTokenizer:
    """Maps each character to one token; offset_mapping is trivial."""

    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = False,
    ) -> dict:
        n = len(text)
        result: dict = {"input_ids": list(range(n))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(n)]
        return result


# ------------------------------------------------------------------
# Case 1: PrefillResult shape contract
# ------------------------------------------------------------------


def test_prefill_result_2d_shape():
    L, T = 4, 6
    energy = np.random.rand(L, T).astype(np.float32)
    ones = np.ones((L, T), dtype=np.float32)
    result = PrefillResult(
        energy=energy,
        entropy=ones,
        norm_entropy=ones,
        lse=ones,
        quadratic=ones,
        top_act=ones,
        n_active=np.zeros((L, T), dtype=np.int32),
        energy_last=energy[:, -1].copy(),
        energy_mean=np.nanmean(energy, axis=1),
    )
    assert result.energy.shape == (L, T)
    assert result.energy_last.shape == (L,)
    assert result.energy_mean.shape == (L,)
    np.testing.assert_allclose(result.energy_last, energy[:, -1])
    np.testing.assert_allclose(result.energy_mean, energy.mean(axis=1), rtol=1e-5)


def test_prefill_result_energy_last_equals_last_column():
    L, T = 3, 7
    energy = np.arange(L * T, dtype=np.float32).reshape(L, T)
    result = PrefillResult(
        energy=energy,
        entropy=np.zeros_like(energy),
        norm_entropy=np.zeros_like(energy),
        lse=np.zeros_like(energy),
        quadratic=np.zeros_like(energy),
        top_act=np.zeros_like(energy),
        n_active=np.zeros((L, T), dtype=np.int32),
        energy_last=energy[:, -1].copy(),
        energy_mean=np.nanmean(energy, axis=1),
    )
    np.testing.assert_array_equal(result.energy_last, energy[:, T - 1])


# ------------------------------------------------------------------
# Case 2: capture_prefill returns full [L, T_prompt] trajectory
# ------------------------------------------------------------------


def test_capture_prefill_trajectory_shape():
    n_layers, T, d, K = 3, 5, 8, 4
    llm = _FakeLLM(n_layers=n_layers, T=T, d=d)
    banks = _make_banks(n_layers, K=K, d=d)

    result = capture_prefill(llm, "test question", banks, normalize_query=False)

    assert isinstance(result, PrefillResult)
    assert result.energy.shape == (n_layers, T), (
        f"Expected [L={n_layers}, T={T}], got {result.energy.shape}"
    )
    assert result.energy_last.shape == (n_layers,)
    assert result.energy_mean.shape == (n_layers,)
    np.testing.assert_allclose(
        result.energy_last, result.energy[:, -1], rtol=1e-5,
    )


def test_capture_prefill_energy_mean_consistency():
    n_layers, T, d = 2, 4, 6
    llm = _FakeLLM(n_layers=n_layers, T=T, d=d)
    banks = _make_banks(n_layers, K=3, d=d)

    result = capture_prefill(llm, "q", banks, normalize_query=True)

    expected_mean = np.nanmean(result.energy, axis=1)
    np.testing.assert_allclose(result.energy_mean, expected_mean, rtol=1e-5)


def test_capture_prefill_reduce_in_hook_deprecation():
    """reduce_in_hook=True should emit a DeprecationWarning."""
    n_layers, T, d = 2, 3, 8
    llm = _FakeLLM(n_layers=n_layers, T=T, d=d)
    banks = _make_banks(n_layers, K=4, d=d)

    with pytest.warns(DeprecationWarning, match="reduce_in_hook"):
        result = capture_prefill(llm, "q", banks, reduce_in_hook=True)

    # With reduce_in_hook=True the query collapses to [1, d] → T_effective == 1
    assert result.energy.shape[0] == n_layers
    assert result.energy_last.shape == (n_layers,)


# ------------------------------------------------------------------
# Case 3: find_answer_span
# ------------------------------------------------------------------


def test_find_answer_span_exact():
    tok = _CharTokenizer()
    text = "Paris is the capital."
    gold = ["Paris"]
    span = find_answer_span(text, gold, tok)
    assert span is not None
    start, end = span
    assert text[start:end].lower() == "paris"


def test_find_answer_span_no_match():
    tok = _CharTokenizer()
    span = find_answer_span("Berlin is the capital.", ["Paris"], tok)
    assert span is None


def test_find_answer_span_case_insensitive():
    tok = _CharTokenizer()
    span = find_answer_span("PARIS is the answer.", ["paris"], tok)
    assert span is not None


def test_find_answer_span_first_gold_wins():
    tok = _CharTokenizer()
    text = "Paris and London are cities."
    # gold_answers list order determines which match wins, not position in text
    span = find_answer_span(text, ["London", "Paris"], tok)
    assert span is not None
    start, end = span
    assert text[start:end].lower() == "london"


# ------------------------------------------------------------------
# Case 4: content_token_mask
# ------------------------------------------------------------------


def test_content_token_mask_strips_whitespace():
    tok = _CharTokenizer()
    text = "  hi  "
    mask = content_token_mask(text, tok)
    assert mask.shape == (len(text),)
    # Leading 2 spaces and trailing 2 spaces should be False
    assert not mask[0] and not mask[1]
    assert not mask[-1] and not mask[-2]
    # "hi" characters should be True
    assert mask[2] and mask[3]


def test_content_token_mask_no_whitespace():
    tok = _CharTokenizer()
    text = "hello"
    mask = content_token_mask(text, tok)
    assert mask.all()


def test_content_token_mask_blank():
    tok = _CharTokenizer()
    mask = content_token_mask("   ", tok)
    assert not mask.any()


# ------------------------------------------------------------------
# Case 5: extract_answer_features — span vs fallback
# ------------------------------------------------------------------


def test_extract_answer_features_uses_span():
    L, T = 4, 10
    rng = np.random.default_rng(0)
    metrics = rng.random((L, T)).astype(np.float32)
    span = (2, 5)
    mask = np.ones(T, dtype=bool)

    result = extract_answer_features(metrics, span, mask)

    assert result.shape == (L,)
    expected = np.nanmean(metrics[:, 2:5], axis=1)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_extract_answer_features_fallback_when_span_none():
    L, T = 3, 8
    metrics = np.ones((L, T), dtype=np.float32)
    mask = np.zeros(T, dtype=bool)
    mask[1:4] = True  # content tokens 1–3

    result = extract_answer_features(metrics, None, mask)

    np.testing.assert_allclose(result, np.ones(L), rtol=1e-5)


def test_extract_answer_features_fallback_when_span_oob():
    L, T = 2, 5
    metrics = np.arange(L * T, dtype=np.float32).reshape(L, T)
    span = (10, 15)  # out of bounds
    mask = np.array([False, True, True, False, False])

    result = extract_answer_features(metrics, span, mask)

    expected = np.nanmean(metrics[:, mask], axis=1)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_extract_answer_features_full_fallback_when_mask_empty():
    """When both span and mask are empty, falls back to nanmean over all T."""
    L, T = 2, 4
    metrics = np.ones((L, T), dtype=np.float32) * 3.0
    mask = np.zeros(T, dtype=bool)

    result = extract_answer_features(metrics, None, mask)

    np.testing.assert_allclose(result, np.full(L, 3.0), rtol=1e-5)
