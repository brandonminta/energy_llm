"""Retrieval metrics, all in float32 torch.

Per layer and token, with x the residual at the input of
post_attention_layernorm and the unit-row bank of :mod:`energy_llm.core.bank`:

    x_hat  = x / sqrt(mean(x^2) + eps)
    s      = (bank @ x_hat) / sqrt(D)          cosines in [-1, 1]  (computed in bank.Bank.scores)
    p      = softmax(beta * s)
    E      = -(1/beta) * logsumexp(beta * s)   (1/2 ||x_hat||^2 ~ D/2 is constant, omitted)
    H_bar  = H(p) / log(D_ff)
    margin = top1(s) - top2(s)

Divergences between two distributions over the D_ff memories (Jensen-Shannon
in nats, bounded by log 2) take LOG-probabilities: at large beta most float32
probabilities underflow to 0, which made m = (p+q)/2 vanish and JS jump to
its log 2 clamp. log_softmax is always finite, so every divergence is
computed in the log domain.
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


def js_divergence(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence in nats from log-probabilities, with
    log m = logaddexp(log p, log q) - log 2 (never -inf). Clamped to
    [0, log 2] to absorb float32 rounding at the bounds."""
    log_m = torch.logaddexp(logp, logq) - LN2
    js = 0.5 * (logp.exp() * (logp - log_m)).sum(dim=-1) \
        + 0.5 * (logq.exp() * (logq - log_m)).sum(dim=-1)
    return js.clamp(0.0, LN2)
