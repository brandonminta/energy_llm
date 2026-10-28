"""Retrieval metrics, all in float32 torch.

Per layer and token, with x the residual at the input of
post_attention_layernorm and the unit-row bank of :mod:`energy_llm.core.bank`:

    x_hat  = x / sqrt(mean(x^2) + eps)
    s      = (bank @ x_hat) / sqrt(D)          cosines in [-1, 1]  (computed in bank.Bank.scores)
    p      = softmax(beta * s)
    E      = -(1/beta) * logsumexp(beta * s)   (1/2 ||x_hat||^2 ~ D/2 is constant, omitted)
    H_bar  = H(p) / log(D_ff)
    margin = top1(s) - top2(s)

Divergences between two distributions over the D_ff memories: Jensen-Shannon
in nats (bounded by log 2), squared Hellinger, and Jaccard of the top-k sets.
They take LOG-probabilities: at large beta most float32 probabilities
underflow to 0 (or to denormals whose half underflows), which made
m = (p+q)/2 vanish, JS jump to its log 2 clamp, and top-k pick arbitrary
indices among exact zeros. log_softmax is always finite, so every
divergence is computed in the log domain.
"""

from __future__ import annotations

import math

import torch

LN2 = math.log(2.0)


def rms_normalise(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


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


def log_mean_distribution(logp: torch.Tensor) -> torch.Tensor:
    """log of the mean of the distributions in the rows of ``logp`` [n, K]."""
    return torch.logsumexp(logp, dim=0) - math.log(logp.shape[0])


def js_divergence(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence in nats from log-probabilities, with
    log m = logaddexp(log p, log q) - log 2 (never -inf). Clamped to
    [0, log 2] to absorb float32 rounding at the bounds."""
    log_m = torch.logaddexp(logp, logq) - LN2
    js = 0.5 * (logp.exp() * (logp - log_m)).sum(dim=-1) \
        + 0.5 * (logq.exp() * (logq - log_m)).sum(dim=-1)
    return js.clamp(0.0, LN2)


def hellinger_sq(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((0.5 * logp).exp() - (0.5 * logq).exp()).pow(2).sum(dim=-1)


def jaccard_topk(logp: torch.Tensor, logq: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k ranked on log-probabilities (same order as the probabilities,
    without ties at underflowed zeros)."""
    k = min(k, logp.shape[-1])
    a = torch.topk(logp, k, dim=-1).indices
    b = torch.topk(logq, k, dim=-1).indices
    inter = torch.isin(a, b).sum(dim=-1).float()
    return inter / (2 * k - inter)
