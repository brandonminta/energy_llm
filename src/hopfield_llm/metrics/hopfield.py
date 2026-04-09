"""Modern Hopfield energy computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass
class LayerEnergyResult:
    energy: float
    retrieval_entropy: float
    normalized_entropy: float
    top_activation: float
    mean_activation: float
    n_active_memories: int
    lse: float
    K: int
    softmax_w: torch.Tensor          # [K] on CPU
    similarities: torch.Tensor | None = None  # [K] on CPU if requested


def hopfield_energy_mlp(
    xi: torch.Tensor,
    W_keys: torch.Tensor,
    beta: float = 15.0,
    active_threshold: float = 0.1,
    energy_mode: Literal["cosine", "dot"] = "cosine",
    return_similarities: bool = False,
) -> LayerEnergyResult:
    """Compute Modern Hopfield energy for a query state against a memory bank.

    Args:
        xi: Query hidden state, shape [D].
        W_keys: Memory bank, shape [K, D].
        beta: Inverse temperature (sharpness of retrieval).
        active_threshold: Similarity threshold for counting active memories.
        energy_mode: 'cosine' (normalised) or 'dot' (raw).
        return_similarities: Include raw similarity vector in result.

    Returns:
        LayerEnergyResult with energy, entropy, and activation metrics.
    """
    if xi.ndim != 1:
        raise ValueError(f"xi must be 1D [D], got shape={tuple(xi.shape)}")
    if W_keys.ndim != 2:
        raise ValueError(f"W_keys must be 2D [K, D], got shape={tuple(W_keys.shape)}")
    if W_keys.shape[1] != xi.shape[0]:
        raise ValueError(
            f"Shape mismatch: W_keys={tuple(W_keys.shape)} vs xi={tuple(xi.shape)}"
        )

    xi = xi.to(torch.float32)
    W_keys = W_keys.to(torch.float32)
    K = W_keys.shape[0]
    if K <= 0:
        raise ValueError("W_keys must contain at least one memory (K > 0)")

    logK_over_beta = float(
        torch.log(torch.tensor(float(K), dtype=torch.float32)) / beta
    )

    if energy_mode == "cosine":
        xi_eff = F.normalize(xi, dim=0)
        W_eff = F.normalize(W_keys, dim=1)
        quadratic_term = 0.5
        C = logK_over_beta + 0.5
    elif energy_mode == "dot":
        xi_eff = xi
        W_eff = W_keys
        quadratic_term = 0.5 * float(xi.pow(2).sum())
        C = logK_over_beta
    else:
        raise ValueError(f"Unknown energy_mode: {energy_mode}")

    v = W_eff @ xi_eff
    lse_t = torch.logsumexp(beta * v, dim=0) / beta
    lse = float(lse_t)
    energy = float(-lse + quadratic_term + C)

    softmax_w = F.softmax(beta * v, dim=0)
    entropy_t = -(softmax_w.clamp_min(1e-9).log() * softmax_w).sum()
    retrieval_entropy = float(entropy_t)
    normalized_entropy = (
        float(entropy_t / torch.log(torch.tensor(float(K), dtype=torch.float32)))
        if K > 1
        else 0.0
    )

    top_activation = float(v.max())
    mean_activation = float(v.mean())
    n_active_memories = int((v > active_threshold).sum())

    return LayerEnergyResult(
        energy=energy,
        retrieval_entropy=retrieval_entropy,
        normalized_entropy=normalized_entropy,
        top_activation=top_activation,
        mean_activation=mean_activation,
        n_active_memories=n_active_memories,
        lse=lse,
        K=K,
        softmax_w=softmax_w.detach().cpu(),
        similarities=v.detach().cpu() if return_similarities else None,
    )
