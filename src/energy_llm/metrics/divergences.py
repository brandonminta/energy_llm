from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Distancias / divergencias entre distribuciones de retrieval
# ============================================================

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    p = p.clamp(min=eps)
    p = p / p.sum()

    q = q.clamp(min=eps)
    q = q / q.sum()

    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum()
    kl_qm = (q * (q / m).log()).sum()
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def hellinger(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    p = p.clamp(min=eps)
    p = p / p.sum()

    q = q.clamp(min=eps)
    q = q / q.sum()

    return float((0.5 * (p.sqrt() - q.sqrt()).pow(2).sum()).sqrt())


def kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-10,
    direction: Literal["forward", "reverse", "both"] = "forward",
) -> float | tuple[float, float]:
    """
    KL divergence entre distribuciones de retrieval.

    forward: KL(p || q)
    reverse: KL(q || p)
    both   : devuelve ambas
    """
    if p.shape != q.shape:
        raise ValueError(f"shapes distintos: {p.shape} vs {q.shape}")
    if p.ndim != 1:
        raise ValueError(f"esperado tensor 1D, recibido {p.ndim}D")

    p = p.clamp(min=eps)
    p = p / p.sum()

    q = q.clamp(min=eps)
    q = q / q.sum()

    kl_fwd = float((p * (p / q).log()).sum())

    if direction == "forward":
        return kl_fwd

    kl_rev = float((q * (q / p).log()).sum())

    if direction == "reverse":
        return kl_rev

    return kl_fwd, kl_rev


# ============================================================
# Energía tipo Modern Hopfield sobre banco MLP
# ============================================================

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
    softmax_w: torch.Tensor
    similarities: torch.Tensor | None = None


def hopfield_energy_mlp(
    xi: torch.Tensor,
    W_keys: torch.Tensor,
    beta: float = 15.0,
    active_threshold: float = 0.1,
    energy_mode: Literal["cosine", "dot"] = "cosine",
    return_similarities: bool = False,
) -> LayerEnergyResult:
    """
    Evalúa un vector de estado xi contra un banco de memorias W_keys.

    Args:
        xi:
            Tensor [D]
        W_keys:
            Tensor [K, D]
        beta:
            Temperatura inversa
        active_threshold:
            Umbral para contar memorias activas
        energy_mode:
            - 'cosine': surrogate esférico con xi y W normalizados
            - 'dot'   : producto punto crudo
        return_similarities:
            Si True, devuelve también el vector de similitudes v

    Nota importante:
        - En modo 'cosine', usamos una surrogate energy angular/esférica.
        - En modo 'dot', usamos:
              E = -lse + 0.5 * ||xi||^2 + log(K)/beta
          sin el +0.5 extra en C.
    """
    if xi.ndim != 1:
        raise ValueError(f"xi debe ser 1D [D], recibido shape={tuple(xi.shape)}")

    if W_keys.ndim != 2:
        raise ValueError(f"W_keys debe ser 2D [K, D], recibido shape={tuple(W_keys.shape)}")

    if W_keys.shape[1] != xi.shape[0]:
        raise ValueError(
            f"Incompatibilidad de shapes: W_keys={tuple(W_keys.shape)} vs xi={tuple(xi.shape)}"
        )

    xi = xi.to(torch.float32)
    W_keys = W_keys.to(torch.float32)

    K = W_keys.shape[0]
    if K <= 0:
        raise ValueError("W_keys debe contener al menos una memoria (K > 0)")

    logK_over_beta = float(torch.log(torch.tensor(float(K), dtype=torch.float32)) / beta)

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
        raise ValueError(f"energy_mode desconocido: {energy_mode}")

    v = W_eff @ xi_eff  # [K]

    lse_t = torch.logsumexp(beta * v, dim=0) / beta
    lse = float(lse_t)

    energy = float(-lse + quadratic_term + C)

    softmax_w = F.softmax(beta * v, dim=0)

    entropy_t = -(softmax_w.clamp_min(1e-9).log() * softmax_w).sum()
    retrieval_entropy = float(entropy_t)

    if K > 1:
        normalized_entropy = float(
            entropy_t / torch.log(torch.tensor(float(K), dtype=torch.float32))
        )
    else:
        normalized_entropy = 0.0

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


# ============================================================
# Divergencias layer-wise entre dos estados
# ============================================================

@dataclass
class LayerDivergenceResult:
    kl_fwd: np.ndarray
    kl_rev: np.ndarray
    js: np.ndarray
    hellinger: np.ndarray
    delta_entropy: np.ndarray
    delta_norm_entropy: np.ndarray
    delta_energy: np.ndarray
    delta_top_activation: np.ndarray
    delta_mean_activation: np.ndarray
    reference_label: str = "ref"
    target_label: str = "cmp"


@torch.no_grad()
def compute_layer_divergences(
    h_ref: torch.Tensor,
    h_cmp: torch.Tensor,
    mlp_keys: dict[int, torch.Tensor],
    beta: float = 15.0,
    energy_mode: Literal["cosine", "dot"] = "cosine",
    active_threshold: float = 0.1,
    reference_label: str = "ref",
    target_label: str = "cmp",
) -> LayerDivergenceResult:
    """
    Compara dos estados layer-wise contra el mismo banco MLP.

    Convención de signo para deltas:
        positivo = cmp tiene mayor valor que ref
        negativo = cmp tiene menor valor que ref

    Capas ausentes en mlp_keys quedan como np.nan
    para evitar sesgo en agregaciones posteriores.
    """
    if h_ref.ndim != 2 or h_cmp.ndim != 2:
        raise ValueError(f"h_ref y h_cmp deben ser [L, D], shapes: {h_ref.shape}, {h_cmp.shape}")

    if h_ref.shape != h_cmp.shape:
        raise ValueError(f"h_ref y h_cmp deben tener mismo shape: {h_ref.shape} vs {h_cmp.shape}")

    L = h_ref.shape[0]

    kl_fwd            = np.full(L, np.nan, dtype=np.float32)
    kl_rev            = np.full(L, np.nan, dtype=np.float32)
    js                = np.full(L, np.nan, dtype=np.float32)
    hell              = np.full(L, np.nan, dtype=np.float32)
    delta_entropy     = np.full(L, np.nan, dtype=np.float32)
    delta_norm_entropy= np.full(L, np.nan, dtype=np.float32)
    delta_energy      = np.full(L, np.nan, dtype=np.float32)
    delta_top_activation  = np.full(L, np.nan, dtype=np.float32)
    delta_mean_activation = np.full(L, np.nan, dtype=np.float32)

    for i in range(L):
        if i not in mlp_keys:
            continue

        W = mlp_keys[i].float()

        r_ref = hopfield_energy_mlp(
            h_ref[i].float(), W,
            beta=beta, active_threshold=active_threshold,
            energy_mode=energy_mode, return_similarities=False,
        )
        r_cmp = hopfield_energy_mlp(
            h_cmp[i].float(), W,
            beta=beta, active_threshold=active_threshold,
            energy_mode=energy_mode, return_similarities=False,
        )

        p_ref = r_ref.softmax_w
        p_cmp = r_cmp.softmax_w

        kl_fwd[i], kl_rev[i]  = kl_divergence(p_ref, p_cmp, direction="both")
        js[i]                  = js_divergence(p_ref, p_cmp)
        hell[i]                = hellinger(p_ref, p_cmp)

        delta_entropy[i]       = r_cmp.retrieval_entropy   - r_ref.retrieval_entropy
        delta_norm_entropy[i]  = r_cmp.normalized_entropy  - r_ref.normalized_entropy
        delta_energy[i]        = r_cmp.energy              - r_ref.energy
        delta_top_activation[i]  = r_cmp.top_activation   - r_ref.top_activation
        delta_mean_activation[i] = r_cmp.mean_activation  - r_ref.mean_activation

    return LayerDivergenceResult(
        kl_fwd=kl_fwd, kl_rev=kl_rev, js=js, hellinger=hell,
        delta_entropy=delta_entropy, delta_norm_entropy=delta_norm_entropy,
        delta_energy=delta_energy, delta_top_activation=delta_top_activation,
        delta_mean_activation=delta_mean_activation,
        reference_label=reference_label, target_label=target_label,
    )

