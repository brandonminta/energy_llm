"""Modern Hopfield energy over MLP h_pre activations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LayerEnergyResult:
    energy:       float
    entropy:      float
    norm_entropy: float
    lse:          float
    quadratic:    float
    top_act:      float
    n_active:     int


def hopfield_energy_mlp(
    h_pre: torch.Tensor,
    W_down: torch.Tensor,
    beta: float = 15.0,
    threshold: float = 0.1,
    energy_mode: str = "dot",
) -> LayerEnergyResult:
    """Compute Modern Hopfield energy for h_pre against W_down.

    Args:
        h_pre: Pre-projection MLP activation, shape [d_m].
              Captured as ``args[0]`` from a forward_pre_hook on down_proj.
        W_down: Down-projection weight matrix, shape [d, d_m].
        beta: Inverse temperature (sharpness of retrieval).
        threshold: Activation threshold for counting active neurons.
        energy_mode: 'dot' (default) or 'cosine'.

    Returns:
        LayerEnergyResult with scalar metrics.
    """
    if h_pre.ndim != 1:
        raise ValueError(f"h_pre must be 1D [d_m], got shape={tuple(h_pre.shape)}")
    if W_down.ndim != 2:
        raise ValueError(f"W_down must be 2D [d, d_m], got shape={tuple(W_down.shape)}")
    if W_down.shape[1] != h_pre.shape[0]:
        raise ValueError(
            f"Shape mismatch: W_down={tuple(W_down.shape)}, h_pre={tuple(h_pre.shape)}"
        )

    h_pre = h_pre.to(torch.float32)
    W_down = W_down.to(torch.float32)

    K = W_down.shape[0]  # = d, output dim of down_proj
    if K <= 0:
        raise ValueError("W_down must have at least one row (d > 0)")

    if energy_mode == "dot":
        s = W_down @ h_pre                        # [d]
        quadratic = 0.5 * float(h_pre.pow(2).sum())
    elif energy_mode == "cosine":
        h_norm = F.normalize(h_pre, dim=0)
        s = W_down @ h_norm                       # [d]
        quadratic = 0.5
    else:
        raise ValueError(f"Unknown energy_mode: '{energy_mode}'. Use 'dot' or 'cosine'.")

    logK_over_beta = float(torch.tensor(float(K)).log() / beta)
    lse = float(torch.logsumexp(beta * s, dim=0) / beta)
    energy = -lse + quadratic + logK_over_beta

    softmax_w = F.softmax(beta * s, dim=0)
    entropy_t = -(softmax_w.clamp_min(1e-9).log() * softmax_w).sum()
    entropy = float(entropy_t)
    norm_entropy = (
        float(entropy_t / torch.tensor(float(K)).log()) if K > 1 else 0.0
    )

    top_act = float(s.max())
    n_active = int((s > threshold).sum())

    return LayerEnergyResult(
        energy=energy,
        entropy=entropy,
        norm_entropy=norm_entropy,
        lse=lse,
        quadratic=quadratic,
        top_act=top_act,
        n_active=n_active,
    )
