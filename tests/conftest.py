"""Shared fixtures for hopfield_llm tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def synthetic_banks() -> dict[int, torch.Tensor]:
    """Synthetic W_down banks: 4 layers, shape [64, 128]."""
    torch.manual_seed(42)
    return {i: torch.randn(64, 128) for i in range(4)}


@pytest.fixture
def synthetic_h_pre() -> torch.Tensor:
    """Synthetic h_pre query vector of shape [128]."""
    torch.manual_seed(0)
    return torch.randn(128)


@pytest.fixture
def synthetic_trajectory_dir(tmp_path: Path) -> Path:
    """Synthetic trajectory directory with 5 samples (L=4, T=10)."""
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    rng = np.random.RandomState(42)
    L, T = 4, 10

    for i in range(5):
        sid = f"test_{i}"
        np.savez_compressed(
            traj_dir / f"{sid}.npz",
            prefill_energy=rng.randn(L).astype(np.float32),
            prefill_entropy=rng.rand(L).astype(np.float32),
            prefill_norm_entropy=rng.rand(L).astype(np.float32),
            prefill_lse=rng.randn(L).astype(np.float32),
            prefill_quadratic=rng.randn(L).astype(np.float32),
            prefill_top_act=rng.randn(L).astype(np.float32),
            prefill_n_active=rng.randint(0, 100, L).astype(np.int32),
            gen_energy=rng.randn(L, T).astype(np.float32),
            gen_entropy=rng.rand(L, T).astype(np.float32),
            gen_norm_entropy=rng.rand(L, T).astype(np.float32),
            gen_lse=rng.randn(L, T).astype(np.float32),
            gen_quadratic=rng.randn(L, T).astype(np.float32),
            gen_top_act=rng.randn(L, T).astype(np.float32),
            gen_n_active=rng.randint(0, 100, (L, T)).astype(np.int32),
            token_ids=rng.randint(0, 32000, T).astype(np.int32),
        )
        (traj_dir / f"{sid}.json").write_text(json.dumps({
            "id": sid, "source": "test",
            "question": f"Test question {i}?",
            "gold_answers": [f"Answer {i}"],
            "generated_text": f"Generated answer {i}",
            "model": "test_model", "n_layers": L, "n_tokens_generated": T,
        }, indent=2), encoding="utf-8")

    return traj_dir
