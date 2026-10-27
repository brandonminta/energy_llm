"""Retrieval metrics, all in float32 torch.

Per layer and token, with x the residual at the input of
post_attention_layernorm and the unit-row bank of :mod:`energy_llm.bank`:

    x_hat  = x / sqrt(mean(x^2) + eps)
    s      = (bank @ x_hat) / sqrt(D)          cosines in [-1, 1]
    p      = softmax(beta * s)
    E      = -(1/beta) * logsumexp(beta * s)   (1/2 ||x_hat||^2 ~ D/2 is constant, omitted)
    H_bar  = H(p) / log(D_ff)
    margin = top1(s) - top2(s)

Divergences between two distributions over the D_ff memories: Jensen-Shannon
in nats (bounded by log 2), squared Hellinger, and Jaccard of the top-k sets.
"""

from __future__ import annotations

import math

import torch

LN2 = math.log(2.0)


def rms_normalise(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def cosine_scores(x_hat: torch.Tensor, bank_rows: torch.Tensor) -> torch.Tensor:
    """[T, D] x [D_ff, D] -> [T, D_ff]."""
    d = x_hat.shape[-1]
    return (x_hat.float() @ bank_rows.T) / math.sqrt(d)


def energy(s: torch.Tensor, beta: float) -> torch.Tensor:
    return -torch.logsumexp(beta * s, dim=-1) / beta


def log_distribution(s: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.log_softmax(beta * s, dim=-1)


def norm_entropy(logp: torch.Tensor) -> torch.Tensor:
    p = logp.exp()
    plogp = torch.where(p > 0, p * logp, torch.zeros_like(p))
    return -plogp.sum(dim=-1) / math.log(logp.shape[-1])


def margin(s: torch.Tensor) -> torch.Tensor:
    top = torch.topk(s, 2, dim=-1).values
    return top[..., 0] - top[..., 1]


def _kl_to(a: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    terms = torch.where(a > 0, a * (torch.log(a) - torch.log(m)), torch.zeros_like(a))
    return terms.sum(dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence in nats. Clamped to [0, log 2] to absorb
    float32 rounding at the bounds."""
    m = 0.5 * (p + q)
    js = 0.5 * _kl_to(p, m) + 0.5 * _kl_to(q, m)
    return js.clamp(0.0, LN2)


def hellinger_sq(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return 0.5 * (p.sqrt() - q.sqrt()).pow(2).sum(dim=-1)


def jaccard_topk(p: torch.Tensor, q: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, p.shape[-1])
    a = torch.topk(p, k, dim=-1).indices
    b = torch.topk(q, k, dim=-1).indices
    inter = torch.isin(a, b).sum(dim=-1).float()
    return inter / (2 * k - inter)
