"""Tests for hopfield_llm.models — profiles, arch resolution, and HFLLM logic.

Test tiers
----------
1. Profile tests         — always run, zero dependencies (pure Python)
2. Arch resolution tests — always run, use lightweight nn.Module stubs
3. HFLLM logic tests     — always run, exercise static methods only (no GPU/network)
4. Real model test       — skipped unless:
       HOPFIELD_TEST_REAL_MODEL=1  (any value)
     AND a CUDA device is available.
   Override the model alias with HOPFIELD_TEST_MODEL_ALIAS (default: qwen25_3b).
   On HPC, a good choice is: HOPFIELD_TEST_MODEL_ALIAS=qwen25_7b

   Run with:
       HOPFIELD_TEST_REAL_MODEL=1 pytest tests/test_models.py -v -s
       HOPFIELD_TEST_REAL_MODEL=1 HOPFIELD_TEST_MODEL_ALIAS=qwen25_7b pytest tests/test_models.py::test_real_model_generate -v -s
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from hopfield_llm.models.arch import (
    BACKBONE_PATHS,
    LAYER_ATTRS,
    resolve_backbone,
    resolve_layers,
)
from hopfield_llm.models.loader import HFLLM
from hopfield_llm.models.profiles import (
    DEFAULT_MODEL_ALIAS,
    MODEL_PROFILES,
    ModelProfile,
    resolve_model_profile,
)

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

_REAL_MODEL_REQUESTED = bool(os.getenv("HOPFIELD_TEST_REAL_MODEL"))
_REAL_MODEL_ALIAS = os.getenv("HOPFIELD_TEST_MODEL_ALIAS", DEFAULT_MODEL_ALIAS)

_needs_real_model = pytest.mark.skipif(
    not (_REAL_MODEL_REQUESTED and torch.cuda.is_available()),
    reason=(
        "Set HOPFIELD_TEST_REAL_MODEL=1 and ensure a CUDA device is present. "
        "Override model with HOPFIELD_TEST_MODEL_ALIAS (default: qwen25_3b). "
        "HPC recommended alias: qwen25_7b"
    ),
)


class _FakeLayers(nn.ModuleList):
    """Minimal ModuleList used by arch stubs."""


class _FakeBackbone(nn.Module):
    """Backbone that exposes a .layers attribute (Qwen/Llama family)."""

    def __init__(self, n: int = 4):
        super().__init__()
        self.layers = _FakeLayers([nn.Linear(8, 8) for _ in range(n)])


class _FakeBackboneH(nn.Module):
    """Backbone that exposes a .h attribute (GPT-2 family)."""

    def __init__(self, n: int = 3):
        super().__init__()
        self.h = _FakeLayers([nn.Linear(4, 4) for _ in range(n)])


class _FakeBackboneNoLayers(nn.Module):
    """Backbone with no recognized layer attribute."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)


class _FakeModelWithModel(nn.Module):
    """Top-level model with .model backbone (Qwen/Llama style)."""

    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.model = _FakeBackbone(n_layers)


class _FakeModelWithTransformer(nn.Module):
    """Top-level model with .transformer backbone (GPT-2 style)."""

    def __init__(self):
        super().__init__()
        self.transformer = _FakeBackboneH(3)


class _FakeModelNoBackbone(nn.Module):
    """Top-level model with no recognized backbone attribute."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 4)


# ---------------------------------------------------------------------------
# 1. Profile tests
# ---------------------------------------------------------------------------


def test_default_alias_resolves():
    p = resolve_model_profile(DEFAULT_MODEL_ALIAS)
    assert p is not None
    assert p.model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert p.L == 36
    assert p.D == 2048


def test_all_aliases_round_trip():
    """Every key in MODEL_PROFILES must resolve back to a valid ModelProfile."""
    for alias in MODEL_PROFILES:
        p = resolve_model_profile(alias)
        assert p is not None, f"Alias '{alias}' failed to resolve"
        assert isinstance(p, ModelProfile)


def test_all_profiles_l_d_positive():
    for alias, p in MODEL_PROFILES.items():
        assert p.L > 0, f"{alias}: L={p.L} must be > 0"
        assert p.D > 0, f"{alias}: D={p.D} must be > 0"
        assert isinstance(p.safe_4bit, bool), f"{alias}: safe_4bit must be bool"


def test_resolve_case_insensitive():
    lower = resolve_model_profile("qwen25_3b")
    upper = resolve_model_profile("QWEN25_3B")
    assert lower == upper


def test_resolve_unknown_returns_none():
    assert resolve_model_profile("not_a_real_model_xyz") is None


def test_hpc_models_present():
    """7B+ and 14B models required for HPC A100 20GB experiments."""
    required = ["qwen25_7b", "llama31_8b", "gemma2_9b", "qwen25_14b", "phi3_medium"]
    for alias in required:
        assert alias in MODEL_PROFILES, f"HPC model '{alias}' missing from MODEL_PROFILES"


def test_hpc_14b_profile():
    p = resolve_model_profile("qwen25_14b")
    assert p is not None
    assert p.L == 48
    assert p.D == 5120
    assert p.safe_4bit is True


def test_phi3_medium_profile():
    p = resolve_model_profile("phi3_medium")
    assert p is not None
    assert p.L == 40
    assert p.D == 5120
    assert p.safe_4bit is True


def test_fast_local_model_present():
    """llama32_1b is the fast smoke-test model for local development."""
    assert "llama32_1b" in MODEL_PROFILES
    p = MODEL_PROFILES["llama32_1b"]
    assert p.L == 16
    assert p.D == 2048


def test_mistral_safe_4bit_false():
    """Mistral7b is intentionally excluded from 4-bit (bitsandbytes compat)."""
    p = resolve_model_profile("mistral7b")
    assert p is not None
    assert p.safe_4bit is False


# ---------------------------------------------------------------------------
# 2. Architecture resolution tests
# ---------------------------------------------------------------------------


def test_backbone_paths_nonempty():
    assert len(BACKBONE_PATHS) >= 3


def test_layer_attrs_nonempty():
    assert len(LAYER_ATTRS) >= 4


def test_resolve_backbone_model_attr():
    """Qwen/Llama-style: backbone lives at model.model."""
    m = _FakeModelWithModel(n_layers=6)
    backbone = resolve_backbone(m)
    assert backbone is m.model


def test_resolve_backbone_transformer_attr():
    """GPT-2 style: backbone lives at model.transformer."""
    m = _FakeModelWithTransformer()
    backbone = resolve_backbone(m)
    assert backbone is m.transformer


def test_resolve_backbone_raises_on_unknown():
    m = _FakeModelNoBackbone()
    with pytest.raises(AttributeError, match="Cannot resolve backbone"):
        resolve_backbone(m)


def test_resolve_layers_uses_layers_attr():
    b = _FakeBackbone(n=5)
    layers = resolve_layers(b)
    assert len(layers) == 5
    assert layers is b.layers


def test_resolve_layers_uses_h_attr():
    b = _FakeBackboneH(n=3)
    layers = resolve_layers(b)
    assert len(layers) == 3
    assert layers is b.h


def test_resolve_layers_raises_on_unknown():
    b = _FakeBackboneNoLayers()
    with pytest.raises(AttributeError, match="Cannot resolve layer list"):
        resolve_layers(b)


def test_arch_roundtrip_full_stack():
    """resolve_backbone + resolve_layers together, end-to-end with stubs."""
    m = _FakeModelWithModel(n_layers=8)
    backbone = resolve_backbone(m)
    layers = resolve_layers(backbone)
    assert len(layers) == 8


# ---------------------------------------------------------------------------
# 3. HFLLM static / class-method logic tests  (no GPU, no network)
# ---------------------------------------------------------------------------


def test_resolve_device_explicit():
    assert HFLLM._resolve_device("cpu") == "cpu"
    assert HFLLM._resolve_device("cuda") == "cuda"


def test_resolve_device_auto_no_cuda():
    with patch("hopfield_llm.models.loader.torch.cuda.is_available", return_value=False):
        assert HFLLM._resolve_device(None) == "cpu"


def test_resolve_device_auto_with_cuda():
    with patch("hopfield_llm.models.loader.torch.cuda.is_available", return_value=True):
        assert HFLLM._resolve_device(None) == "cuda"


def test_resolve_dtype_explicit():
    assert HFLLM._resolve_dtype(torch.bfloat16, "cpu") == torch.bfloat16


def test_resolve_dtype_cpu_default():
    assert HFLLM._resolve_dtype(None, "cpu") == torch.float32


def test_resolve_dtype_cuda_default():
    assert HFLLM._resolve_dtype(None, "cuda") == torch.float16


def test_resolve_effective_4bit_not_requested():
    assert HFLLM._resolve_effective_4bit(False, "cuda", None) is False


def test_resolve_effective_4bit_no_cuda():
    assert HFLLM._resolve_effective_4bit(True, "cpu", None) is False


def test_resolve_effective_4bit_unsafe_profile():
    bad = ModelProfile(model_id="test/model", L=32, D=4096, safe_4bit=False)
    assert HFLLM._resolve_effective_4bit(True, "cuda", bad) is False


def test_resolve_effective_4bit_happy_path():
    good = ModelProfile(model_id="test/model", L=32, D=4096, safe_4bit=True)
    assert HFLLM._resolve_effective_4bit(True, "cuda", good) is True


def test_resolve_effective_4bit_no_profile():
    """No profile → treat as safe (caller's responsibility)."""
    assert HFLLM._resolve_effective_4bit(True, "cuda", None) is True


# ---------------------------------------------------------------------------
# 4. Real model integration test  (GPU + HOPFIELD_TEST_REAL_MODEL=1 required)
# ---------------------------------------------------------------------------


@_needs_real_model
def test_real_model_load_summary():
    """Load the model and verify the summary dict is well-formed."""
    llm = HFLLM(model_id=_REAL_MODEL_ALIAS, load_in_4bit=True)
    s = llm.summary()

    assert s["model_id"], "model_id must be non-empty"
    assert s["n_layers_resolved"] > 0, "at least one layer must be resolved"
    assert s["hidden_size"] is not None, "hidden_size must be populated"
    assert s["load_in_4bit_effective"] is True, "4-bit should be effective on GPU"
    assert s["device"] == "cuda"

    profile = resolve_model_profile(_REAL_MODEL_ALIAS)
    if profile is not None:
        assert s["n_layers_resolved"] == profile.L, (
            f"resolved layers ({s['n_layers_resolved']}) != profile.L ({profile.L})"
        )
        assert s["hidden_size"] == profile.D, (
            f"hidden_size ({s['hidden_size']}) != profile.D ({profile.D})"
        )

    print(f"\n[summary] {s}")


@_needs_real_model
def test_real_model_generate():
    """Load the model and run a real greedy generation pass."""
    llm = HFLLM(model_id=_REAL_MODEL_ALIAS, load_in_4bit=True)

    prompt = "What is the capital of France? Answer in one word."
    response = llm.generate(prompt, max_new_tokens=32, do_sample=False)

    assert isinstance(response, str), "generate() must return a string"
    assert len(response) > len(prompt), "response must be longer than the prompt"

    print(f"\n[generate] model   : {llm.model_id}")
    print(f"[generate] prompt  : {prompt}")
    print(f"[generate] response: {response}")
