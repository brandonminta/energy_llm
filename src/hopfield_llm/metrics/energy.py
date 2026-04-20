"""Energy computation over MLP activations.

The core primitive: given a query vector and a memory matrix, compute the
Modern Hopfield retrieval energy, entropy, and related statistics.

The name ``compute_energy`` is deliberately generic — the formula is grounded
in Modern Hopfield Networks but the architecture of the memory (W_down, W_gate,
etc.) is supplied externally and can be changed without touching this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LayerEnergyResult:
    """Scalar metrics produced by one energy computation."""
    energy:       float   # Modern Hopfield energy
    entropy:      float   # Shannon entropy of attention weights
    norm_entropy: float   # Normalised entropy in [0, 1]
    lse:          float   # log-sum-exp / beta
    quadratic:    float   # 0.5 ||h||^2
    top_act:      float   # max similarity score
    n_active:     int     # number of scores above threshold


def compute_energy(
    query: torch.Tensor,
    memory: torch.Tensor,
    beta: float = 15.0,
    threshold: float = 0.1,
    mode: str = "dot",
) -> LayerEnergyResult:
    """Compute Modern Hopfield energy for a query against a memory matrix.

    Args:
        query:    Query vector of shape [d_m].
                  For MLP-based banks this is ``h_pre = SiLU(gate) ⊙ up``.
        memory:   Memory matrix of shape [K, d_m] (K patterns, each of dim d_m).
                  For W_down banks: shape [d_out, d_m].
        beta:     Inverse temperature (controls retrieval sharpness).
        threshold: Threshold for counting active similarity scores.
        mode:     'dot' (default) or 'cosine' (normalises query before scoring).

    Returns:
        LayerEnergyResult with scalar metrics.
    """
    if query.ndim != 1:
        raise ValueError(f"query must be 1D [d_m], got shape={tuple(query.shape)}")
    if memory.ndim != 2:
        raise ValueError(f"memory must be 2D [K, d_m], got shape={tuple(memory.shape)}")
    if memory.shape[1] != query.shape[0]:
        raise ValueError(
            f"Dimension mismatch: memory={tuple(memory.shape)}, query={tuple(query.shape)}"
        )

    query = query.to(torch.float32)
    memory = memory.to(torch.float32)

    K = memory.shape[0]
    if K <= 0:
        raise ValueError("memory must have at least one row (K > 0)")

    if mode == "dot":
        scores = memory @ query          # [K]
        quadratic = 0.5 * float(query.pow(2).sum())
    elif mode == "cosine":
        q_norm = F.normalize(query, dim=0)
        scores = memory @ q_norm         # [K]
        quadratic = 0.5
    else:
        raise ValueError(f"Unknown mode='{mode}'. Use 'dot' or 'cosine'.")

    log_K = math.log(K)
    log_K_over_beta = log_K / beta
    lse = float(torch.logsumexp(beta * scores, dim=0) / beta)
    energy = -lse + quadratic + log_K_over_beta

    attn_weights = F.softmax(beta * scores, dim=0)
    entropy_t = -(attn_weights.clamp_min(1e-9).log() * attn_weights).sum()
    entropy = float(entropy_t)
    norm_entropy = float(entropy_t) / log_K if K > 1 else 0.0

    return LayerEnergyResult(
        energy=energy,
        entropy=entropy,
        norm_entropy=norm_entropy,
        lse=lse,
        quadratic=quadratic,
        top_act=float(scores.max()),
        n_active=int((scores > threshold).sum()),
    )
