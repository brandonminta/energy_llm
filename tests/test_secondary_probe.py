"""Tests for secondary probe configuration and dual-probe pipeline stages.

Case 1: ProbeConfig defaults are consistent.
Case 2: make_with_value_space_probe sets the correct fields.
Case 3: build_banks stage creates two bank files when secondary_probe is set.
Case 4: run_trajectory stage produces two artifact sets.
Case 5: single-probe path preserves legacy filenames.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from hopfield_llm.datasets.base import DataSample
from hopfield_llm.pipeline.config import ExperimentConfig, ProbeConfig, make_with_value_space_probe
from hopfield_llm.pipeline.stages import build_banks, run_trajectory

# ── Stub LLM (minimal PyTorch modules to satisfy hook registration) ────────────

_D, _DM, _VOCAB, _NL = 8, 16, 50, 2


class _StubMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(_D, _DM, bias=False)
        self.down_proj = nn.Linear(_DM, _D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.relu(self.up_proj(x)))


class _StubLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _StubMLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class _StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _D)
        self.layers = nn.ModuleList([_StubLayer() for _ in range(_NL)])
        self.config = type("Config", (), {
            "hidden_size": _D, "intermediate_size": _DM
        })()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kw):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return type("Out", (), {"logits": None, "past_key_values": None})()

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens: int = 2,
        eos_token_id=None,
        pad_token_id=None,
        do_sample: bool = False,
        **kw,
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        # Prefill: run full prompt through layers (hooks see T > 1 → skip)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        # Decode: one token at a time (hooks see T == 1 → capture)
        cur = input_ids.clone()
        for _ in range(max_new_tokens):
            h = self.embed_tokens(cur[:, -1:])  # [B, 1, D]
            for layer in self.layers:
                h = layer(h)
            cur = torch.cat([cur, torch.zeros(B, 1, dtype=torch.long)], dim=1)
        return cur


class _StubTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text: str, return_tensors=None, add_special_tokens=True):
        # Character-level tokenisation (no truncation) for predictable T_prompt
        ids = [(ord(c) % (_VOCAB - 3) + 3) for c in text] or [3]
        if return_tensors == "pt":
            t = torch.tensor([ids], dtype=torch.long)
            return {"input_ids": t, "attention_mask": torch.ones_like(t)}
        return {"input_ids": ids}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def encode(self, text: str, add_special_tokens: bool = True):
        return [(ord(c) % (_VOCAB - 3) + 3) for c in text[:4]]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "ans"


class _StubLLM:
    n_layers = _NL
    alias = "stub"
    device = torch.device("cpu")

    def __init__(self, **kw):
        self.model = _StubModel()
        self.layers = list(self.model.layers)
        self.tokenizer = _StubTokenizer()

    def summary(self) -> dict:
        return {"alias": "stub", "device": "cpu", "n_layers": _NL}


_FAKE_SAMPLE = DataSample(
    id="test_q0",
    source="test",
    question="What?",
    gold_answers=["42"],
)


# ── Case 1: ProbeConfig defaults ───────────────────────────────────────────────


def test_probe_config_defaults():
    pc = ProbeConfig()
    assert pc.bank == "up"
    assert pc.hook_target == "mlp_input"
    assert pc.normalize is True
    assert pc.normalize_query is True
    assert pc.label == ""


# ── Case 2: make_with_value_space_probe ────────────────────────────────────────


def test_make_with_value_space_probe():
    base = ExperimentConfig()
    cfg = make_with_value_space_probe(base)

    assert cfg.secondary_probe is not None
    assert cfg.secondary_probe.bank == "down_values"
    assert cfg.secondary_probe.hook_target == "mlp_output"
    assert cfg.secondary_probe.label == "value_space"
    # Primary probe is unchanged
    assert cfg.primary_probe == base.primary_probe
    # Base config is not mutated
    assert base.secondary_probe is None


# ── Case 3: build_banks creates two labeled bank files ─────────────────────────


def test_build_banks_creates_labeled_files(tmp_path):
    primary = ProbeConfig(bank="up", normalize=True, hook_target="mlp_input", label="key_space")
    secondary = ProbeConfig(bank="down_values", normalize=True, hook_target="mlp_output", label="value_space")

    stub = _StubLLM()
    with patch("hopfield_llm.pipeline.stages.HFLLM", return_value=stub):
        build_banks(
            model="stub",
            output=str(tmp_path / "banks.pt"),
            primary_probe=primary,
            secondary_probe=secondary,
        )

    assert (tmp_path / "banks_key_space.pt").exists()
    assert (tmp_path / "banks_value_space.pt").exists()
    assert (tmp_path / "banks_key_space_metadata.json").exists()
    assert (tmp_path / "banks_value_space_metadata.json").exists()

    # Legacy filenames must NOT be present
    assert not (tmp_path / "banks.pt").exists()
    assert not (tmp_path / "banks_metadata.json").exists()


# ── Case 4: run_trajectory produces two artifact sets ──────────────────────────


def test_run_trajectory_creates_labeled_artifacts(tmp_path):
    primary = ProbeConfig(bank="up", normalize=True, hook_target="mlp_input", label="key_space")
    secondary = ProbeConfig(bank="down_values", normalize=True, hook_target="mlp_output", label="value_space")

    banks_dir = tmp_path / "banks"
    banks_dir.mkdir()
    traj_dir = tmp_path / "traj"

    stub = _StubLLM()
    with patch("hopfield_llm.pipeline.stages.HFLLM", return_value=stub):
        build_banks(
            model="stub",
            output=str(banks_dir / "banks.pt"),
            primary_probe=primary,
            secondary_probe=secondary,
        )
        with patch("hopfield_llm.pipeline.stages.TruthfulQADataset", return_value=[_FAKE_SAMPLE]):
            run_trajectory(
                model="stub",
                dataset="truthfulqa",
                banks=str(banks_dir / "banks.pt"),
                output=str(traj_dir),
                max_new_tokens=2,
                h_pre_mode="off",
                primary_probe=primary,
                secondary_probe=secondary,
            )

    sid = _FAKE_SAMPLE.id

    assert (traj_dir / f"{sid}_key_space.npz").exists()
    assert (traj_dir / f"{sid}_value_space.npz").exists()
    assert (traj_dir / f"{sid}_key_space.json").exists()
    assert (traj_dir / f"{sid}_value_space.json").exists()

    # generated_text must be identical in both JSON files
    prim_json = json.loads((traj_dir / f"{sid}_key_space.json").read_text())
    sec_json = json.loads((traj_dir / f"{sid}_value_space.json").read_text())
    assert prim_json["generated_text"] == sec_json["generated_text"]

    # Legacy filenames must NOT be present
    assert not (traj_dir / f"{sid}.npz").exists()
    assert not (traj_dir / f"{sid}.json").exists()


# ── Case 5: single-probe path preserves legacy filenames ──────────────────────


def test_single_probe_uses_legacy_filenames(tmp_path):
    banks_path = tmp_path / "banks.pt"
    traj_dir = tmp_path / "traj"

    stub = _StubLLM()
    with patch("hopfield_llm.pipeline.stages.HFLLM", return_value=stub):
        # Use bank="up" — rows are hidden-space vectors matching mlp_input queries
        build_banks(model="stub", output=str(banks_path), bank="up")
        with patch("hopfield_llm.pipeline.stages.TruthfulQADataset", return_value=[_FAKE_SAMPLE]):
            run_trajectory(
                model="stub",
                dataset="truthfulqa",
                banks=str(banks_path),
                output=str(traj_dir),
                max_new_tokens=2,
                h_pre_mode="off",
            )

    # Legacy bank filenames
    assert banks_path.exists()
    assert (tmp_path / "banks_metadata.json").exists()
    assert not (tmp_path / "banks_key_space.pt").exists()
    assert not (tmp_path / "banks_value_space.pt").exists()

    # Legacy trajectory filenames
    sid = _FAKE_SAMPLE.id
    assert (traj_dir / f"{sid}.npz").exists()
    assert (traj_dir / f"{sid}.json").exists()
    assert not (traj_dir / f"{sid}_key_space.npz").exists()
    assert not (traj_dir / f"{sid}_value_space.npz").exists()
