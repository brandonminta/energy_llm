"""Tests for three pipeline completeness gaps (Request 2).

Case A1: generation_config is written to {id}.json with correct keys.
Case A2: generation_config is stored in {id}.npz as generation_config_json.
Case B1: scores_subset_size=0 → no {id}_scores.npz files created.
Case B2: scores_subset_size=2 with 3 samples → only first 2 have scores files.
Case B3: scores_subset_size=-1 → all samples have scores + WARNING logged.
Case C1: label_trajectory writes {id}_labeled.json for each trajectory JSON.
Case C2: calibration.json has kappa=1.0 when heuristic and judge agree perfectly.
Case C3: calibration.json has a non-null warning when kappa < 0.5.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from hopfield_llm.datasets.base import DataSample
from hopfield_llm.pipeline.stages import build_banks, label_trajectory, run_trajectory

# ── Stub LLM ──────────────────────────────────────────────────────────────────

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
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        cur = input_ids.clone()
        for _ in range(max_new_tokens):
            h = self.embed_tokens(cur[:, -1:])
            for layer in self.layers:
                h = layer(h)
            cur = torch.cat([cur, torch.zeros(B, 1, dtype=torch.long)], dim=1)
        return cur


class _StubTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text: str, return_tensors=None, add_special_tokens=True):
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


# ── Stub judge ────────────────────────────────────────────────────────────────


class _StubJudge:
    """Returns a predefined sequence of judge verdicts (0=hallucination, 1=correct)."""

    model_id = "stub_judge"
    _prompt_hash = "stubhash0"

    def __init__(self, verdicts: list[int]):
        self._iter = iter(verdicts)

    def judge(self, question, generated_text, gold_answers):
        v = next(self._iter)
        return {"raw": str(v), "parsed": v, "success": True}


# ── Shared fixtures ───────────────────────────────────────────────────────────

_FAKE_SAMPLE = DataSample(
    id="test_q0",
    source="test",
    question="What?",
    gold_answers=["42"],
)

_FAKE_SAMPLES_3 = [
    DataSample(id="test_q0", source="test", question="What?", gold_answers=["42"]),
    DataSample(id="test_q1", source="test", question="When?", gold_answers=["now"]),
    DataSample(id="test_q2", source="test", question="Where?", gold_answers=["here"]),
]


def _run_single(tmp_path, **traj_kwargs):
    """Run build_banks + run_trajectory (legacy mode) with stub LLM; return (banks_path, traj_dir)."""
    banks_path = tmp_path / "banks.pt"
    traj_dir = tmp_path / "traj"
    stub = _StubLLM()
    with patch("hopfield_llm.pipeline.stages.HFLLM", return_value=stub):
        build_banks(model="stub", output=str(banks_path), bank="up")
        with patch("hopfield_llm.pipeline.stages.TruthfulQADataset", return_value=[_FAKE_SAMPLE]):
            run_trajectory(
                model="stub",
                dataset="truthfulqa",
                banks=str(banks_path),
                output=str(traj_dir),
                max_new_tokens=2,
                h_pre_mode="off",
                **traj_kwargs,
            )
    return banks_path, traj_dir


def _run_three(tmp_path, **traj_kwargs):
    """Run build_banks + run_trajectory for 3 samples; return (banks_path, traj_dir)."""
    banks_path = tmp_path / "banks.pt"
    traj_dir = tmp_path / "traj"
    stub = _StubLLM()
    with patch("hopfield_llm.pipeline.stages.HFLLM", return_value=stub):
        build_banks(model="stub", output=str(banks_path), bank="up")
        with patch(
            "hopfield_llm.pipeline.stages.TruthfulQADataset",
            return_value=_FAKE_SAMPLES_3,
        ):
            run_trajectory(
                model="stub",
                dataset="truthfulqa",
                banks=str(banks_path),
                output=str(traj_dir),
                max_new_tokens=2,
                h_pre_mode="off",
                **traj_kwargs,
            )
    return banks_path, traj_dir


# ── Case A1: generation_config written to .json ───────────────────────────────


def test_a1_generation_config_in_json(tmp_path):
    _, traj_dir = _run_single(tmp_path)
    meta = json.loads((traj_dir / "test_q0.json").read_text())
    assert "generation_config" in meta, "generation_config key missing from JSON"
    cfg = meta["generation_config"]
    assert "max_new_tokens" in cfg
    assert cfg["max_new_tokens"] == 2


# ── Case A2: generation_config stored in .npz ─────────────────────────────────


def test_a2_generation_config_in_npz(tmp_path):
    _, traj_dir = _run_single(tmp_path)
    data = np.load(traj_dir / "test_q0.npz", allow_pickle=False)
    assert "generation_config_json" in data, "generation_config_json missing from npz"
    cfg = json.loads(str(data["generation_config_json"]))
    assert isinstance(cfg, dict)
    assert cfg["max_new_tokens"] == 2


# ── Case B1: scores_subset_size=0 → no _scores.npz ───────────────────────────


def test_b1_no_scores_when_subset_zero(tmp_path):
    _, traj_dir = _run_single(tmp_path, scores_subset_size=0)
    scores_files = list(traj_dir.glob("*_scores.npz"))
    assert len(scores_files) == 0, f"Expected no scores files, found: {scores_files}"


# ── Case B2: scores_subset_size=2 with 3 samples → first 2 have scores ────────


def test_b2_scores_saved_for_first_n(tmp_path):
    _, traj_dir = _run_three(tmp_path, scores_subset_size=2)
    assert (traj_dir / "test_q0_scores.npz").exists(), "test_q0 should have scores"
    assert (traj_dir / "test_q1_scores.npz").exists(), "test_q1 should have scores"
    assert not (traj_dir / "test_q2_scores.npz").exists(), "test_q2 should NOT have scores"


# ── Case B3: scores_subset_size=-1 → all samples + WARNING ────────────────────


def test_b3_all_scores_saved_with_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="hopfield_llm.pipeline.stages"):
        _, traj_dir = _run_three(tmp_path, scores_subset_size=-1)

    for sample in _FAKE_SAMPLES_3:
        assert (traj_dir / f"{sample.id}_scores.npz").exists(), (
            f"{sample.id} should have scores"
        )

    warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("scores" in m.lower() or "disk" in m.lower() for m in warning_msgs), (
        f"Expected a disk-usage warning, got: {warning_msgs}"
    )


# ── Helpers for C tests ───────────────────────────────────────────────────────


def _write_fake_trajs(traj_dir: Path, samples: list[dict]) -> None:
    traj_dir.mkdir(parents=True, exist_ok=True)
    for s in samples:
        path = traj_dir / f"{s['id']}.json"
        path.write_text(json.dumps(s), encoding="utf-8")


# ── Case C1: label_trajectory creates {id}_labeled.json ──────────────────────


def test_c1_labeled_files_created(tmp_path):
    traj_dir = tmp_path / "traj"
    out_dir = tmp_path / "out"
    samples = [
        {"id": "test_q0", "source": "test", "question": "Q?",
         "gold_answers": ["forty-two"], "generated_text": "forty-two"},
        {"id": "test_q1", "source": "test", "question": "Q?",
         "gold_answers": ["forty-two"], "generated_text": "forty-two"},
        {"id": "test_q2", "source": "test", "question": "Q?",
         "gold_answers": ["forty-two"], "generated_text": "forty-two"},
    ]
    _write_fake_trajs(traj_dir, samples)

    judge = _StubJudge([1, 1, 1])
    label_trajectory(traj_dir, out_dir, judge=judge, calibration_subset_size=3)

    for i in range(3):
        assert (out_dir / f"test_q{i}_labeled.json").exists()
    assert (out_dir / "calibration.json").exists()

    # Verify the labeled JSON contains required keys
    labeled = json.loads((out_dir / "test_q0_labeled.json").read_text())
    assert "is_hallucination" in labeled
    assert "metadata" in labeled
    assert "heuristic_is_hallucination" in labeled["metadata"]


# ── Case C2: kappa=1.0 when heuristic and judge agree ────────────────────────


def test_c2_perfect_kappa(tmp_path):
    traj_dir = tmp_path / "traj"
    out_dir = tmp_path / "out"

    # 3 hallucination samples (generated_text won't match gold → F1=0 → heuristic=True)
    # 7 correct samples (exact match → F1=1.0 → heuristic=False)
    samples = (
        [{"id": f"test_q{i}", "source": "t", "question": "Q?",
          "gold_answers": ["forty-two"], "generated_text": "xyz"}
         for i in range(3)]
        + [{"id": f"test_q{i}", "source": "t", "question": "Q?",
            "gold_answers": ["forty-two"], "generated_text": "forty-two"}
           for i in range(3, 10)]
    )
    _write_fake_trajs(traj_dir, samples)

    # Judge agrees with heuristic: 0 (hallucination) for first 3, 1 (correct) for last 7
    judge = _StubJudge([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    label_trajectory(traj_dir, out_dir, judge=judge, calibration_subset_size=10)

    cal = json.loads((out_dir / "calibration.json").read_text())
    assert abs(cal["cohen_kappa"] - 1.0) < 1e-6, f"Expected kappa=1.0, got {cal['cohen_kappa']}"
    assert cal["warning"] is None, f"Expected no warning, got: {cal['warning']}"
    assert cal["n_paired_samples"] == 10


# ── Case C3: warning when kappa < 0.5 ────────────────────────────────────────


def test_c3_low_kappa_warning(tmp_path):
    traj_dir = tmp_path / "traj"
    out_dir = tmp_path / "out"

    # heuristic = [T,T,T, F,F,F,F, F,F,F] (3 hallucination, 7 correct by F1)
    samples = (
        [{"id": f"test_q{i}", "source": "t", "question": "Q?",
          "gold_answers": ["forty-two"], "generated_text": "xyz"}
         for i in range(3)]
        + [{"id": f"test_q{i}", "source": "t", "question": "Q?",
            "gold_answers": ["forty-two"], "generated_text": "forty-two"}
           for i in range(3, 10)]
    )
    _write_fake_trajs(traj_dir, samples)

    # judge = [T,T,T, T,T,T,T, F,F,F] → 7 hallucination, 3 correct
    # kappa = (6/10 - 0.42) / 0.58 ≈ 0.310
    judge = _StubJudge([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    label_trajectory(traj_dir, out_dir, judge=judge, calibration_subset_size=10)

    cal = json.loads((out_dir / "calibration.json").read_text())
    assert cal["cohen_kappa"] < 0.5, f"Expected kappa < 0.5, got {cal['cohen_kappa']}"
    assert cal["warning"] is not None, "Expected a warning for low kappa"
    assert "0.3" in cal["warning"], (
        f"Expected kappa value '0.3' in warning, got: {cal['warning']!r}"
    )
