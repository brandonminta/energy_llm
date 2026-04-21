"""Energy computation over MLP activations.

The core primitive: given a query vector and a memory matrix, compute the
Modern Hopfield retrieval energy, entropy, and related statistics.

The name ``compute_energy`` is deliberately generic — the formula is grounded
in Modern Hopfield Networks but the architecture of the memory (W_up, down_values,
etc.) is supplied externally and can be changed without touching this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from hopfield_llm.utils.logging import get_logger

log = get_logger("metrics.energy")


@dataclass
class LayerEnergyResult:
    """Scalar metrics produced by one energy computation.

    Energy decomposition (Ramsauer et al. 2020, Eq. 1):
        energy = -lse_term + quadratic_term + log_N_over_beta + M_squared_term
    """
    energy:          float   # Full MHN energy (sum of four terms below)
    entropy:         float   # Shannon entropy of softmax(beta * scores)
    norm_entropy:    float   # entropy / log(K), normalised to [0, 1]
    lse:             float   # log-sum-exp / beta (backward-compat alias for lse_term)
    quadratic:       float   # 0.5 ||xi||^2 (backward-compat alias for quadratic_term)
    top_act:         float   # max similarity score
    n_active:        int     # number of scores above threshold

    # Explicit decomposed energy terms (added in M1 refactor)
    lse_term:        float = 0.0   # logsumexp(beta * scores) / beta
    quadratic_term:  float = 0.0   # 0.5 * ||xi||^2  (0.5 in cosine mode)
    log_N_over_beta: float = 0.0   # log(K) / beta
    M_squared_term:  float = 0.0   # 0.5 * M^2; zero when M not supplied
    energy_lse_only: float = 0.0   # -lse_term; for within-layer comparisons

    # Optional dense outputs — None by default to avoid memory overhead
    scores:       torch.Tensor | None = field(default=None, compare=False)
    distribution: torch.Tensor | None = field(default=None, compare=False)


def compute_energy(
    query: torch.Tensor,
    memory: torch.Tensor,
    beta: float = 15.0,
    threshold: float = 0.1,
    mode: str = "dot",
    M: float | None = None,
    return_scores: bool = False,
    return_distribution: bool = False,
) -> LayerEnergyResult:
    """Compute Modern Hopfield energy for a query against a memory matrix.

    Args:
        query:              Query vector of shape [d].
                            For key-space probe: FFN input x^l (shape [d]).
                            For value-space probe: FFN output y^l (shape [d]).
        memory:             Memory matrix of shape [K, d].
                            For key-space: rows of W_up, shape [d_m, d].
                            For value-space: rows of W_down.T, shape [d_m, d].
        beta:               Inverse temperature (controls retrieval sharpness).
        threshold:          Threshold for counting active similarity scores.
        mode:               'dot' or 'cosine' (normalises query before scoring).
        M:                  Max row norm of the non-normalised bank (max_i ||x_i||).
                            When provided the 0.5*M^2 term makes energy cross-layer
                            comparable. When None, M_squared_term is 0 and a debug
                            message is logged.
        return_scores:      If True, attach raw scores tensor to result (K floats).
        return_distribution: If True, attach softmax(beta * scores) to result.

    Returns:
        LayerEnergyResult with scalar metrics and optional dense tensors.
    """
    if query.ndim != 1:
        raise ValueError(f"query must be 1D [d], got shape={tuple(query.shape)}")
    if memory.ndim != 2:
        raise ValueError(f"memory must be 2D [K, d], got shape={tuple(memory.shape)}")
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
        scores = memory @ query
        quadratic_term = 0.5 * float(query.pow(2).sum())
    elif mode == "cosine":
        q_norm = F.normalize(query, dim=0)
        scores = memory @ q_norm
        quadratic_term = 0.5  # ||q_norm||^2 == 1
    else:
        raise ValueError(f"Unknown mode='{mode}'. Use 'dot' or 'cosine'.")

    log_K = math.log(K)
    log_N_over_beta = log_K / beta
    lse_term = float(torch.logsumexp(beta * scores, dim=0) / beta)

    if M is not None:
        M_squared_term = 0.5 * M * M
    else:
        M_squared_term = 0.0
        log.debug(
            "M not provided; energy does not include 0.5*M^2 and is not "
            "cross-layer comparable. Pass M from bank metadata to fix this."
        )

    energy = -lse_term + quadratic_term + log_N_over_beta + M_squared_term
    energy_lse_only = -lse_term

    attn_weights = F.softmax(beta * scores, dim=0)
    entropy_t = -(attn_weights.clamp_min(1e-9).log() * attn_weights).sum()
    entropy = float(entropy_t)
    norm_entropy = float(entropy_t) / log_K if K > 1 else 0.0

    return LayerEnergyResult(
        energy=energy,
        entropy=entropy,
        norm_entropy=norm_entropy,
        lse=lse_term,
        quadratic=quadratic_term,
        top_act=float(scores.max()),
        n_active=int((scores > threshold).sum()),
        lse_term=lse_term,
        quadratic_term=quadratic_term,
        log_N_over_beta=log_N_over_beta,
        M_squared_term=M_squared_term,
        energy_lse_only=energy_lse_only,
        scores=scores.clone() if return_scores else None,
        distribution=attn_weights if return_distribution else None,
    )
