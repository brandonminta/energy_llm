"""Shared fixtures for hopfield_llm tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def synthetic_banks() -> dict[int, torch.Tensor]:
    """Create synthetic W_down banks for 4 layers: [d, d_m] = [64, 128]."""
    torch.manual_seed(42)
    return {i: torch.randn(64, 128) for i in range(4)}


@pytest.fixture
def synthetic_h_pre() -> torch.Tensor:
    """Create a synthetic h_pre vector of shape [128]."""
    torch.manual_seed(0)
    return torch.randn(128)


@pytest.fixture
def synthetic_trajectory_dir(tmp_path: Path) -> Path:
    """Create a synthetic trajectory directory with .npz + .json files."""
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()

    rng = np.random.RandomState(42)
    L, T = 4, 10

    for i in range(5):
        sample_id = f"test_{i}"

        arrays = {
            "prefill_energy": rng.randn(L).astype(np.float32),
            "prefill_entropy": rng.rand(L).astype(np.float32),
            "prefill_norm_entropy": rng.rand(L).astype(np.float32),
            "prefill_lse": rng.randn(L).astype(np.float32),
            "prefill_quadratic": rng.randn(L).astype(np.float32),
            "prefill_top_act": rng.randn(L).astype(np.float32),
            "prefill_n_active": rng.randint(0, 100, size=L).astype(np.int32),
            "gen_energy": rng.randn(L, T).astype(np.float32),
            "gen_entropy": rng.rand(L, T).astype(np.float32),
            "gen_norm_entropy": rng.rand(L, T).astype(np.float32),
            "gen_lse": rng.randn(L, T).astype(np.float32),
            "gen_quadratic": rng.randn(L, T).astype(np.float32),
            "gen_top_act": rng.randn(L, T).astype(np.float32),
            "gen_n_active": rng.randint(0, 100, size=(L, T)).astype(np.int32),
            "token_ids": rng.randint(0, 32000, size=T).astype(np.int32),
        }
        np.savez_compressed(traj_dir / f"{sample_id}.npz", **arrays)

        meta = {
            "id": sample_id,
            "source": "test",
            "question": f"Test question {i}?",
            "gold_answers": [f"Answer {i}"],
            "generated_text": f"Generated answer {i}",
            "model": "test_model",
            "n_layers": L,
            "n_tokens_generated": T,
        }
        (traj_dir / f"{sample_id}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    return traj_dir
