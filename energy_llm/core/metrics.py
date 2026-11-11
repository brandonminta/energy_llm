"""Retrieval metrics, all in float32 torch.

Per layer and token, with x the residual at the input of
post_attention_layernorm and the unit-row bank of :mod:`energy_llm.core.bank`:

    x_hat  = x / sqrt(mean(x^2) + eps)
    s      = (bank @ x_hat) / sqrt(D)          cosines in [-1, 1]  (computed in bank.Bank.scores)
    p      = softmax(beta * s)
    E      = -(1/beta) * logsumexp(beta * s)   (1/2 ||x_hat||^2 ~ D/2 is constant, omitted)
    H_bar  = H(p) / log(D_ff)
    margin = top1(s) - top2(s)
    mean_score = sum p s,   var_score = sum p (s - mean_score)^2
    topk_mass  = sum of the k largest p,   topk_indices = their indices
    xhat_top1_frac = max_j x_hat_j^2 / D   (share of ||x_hat||^2 ~ D in one coordinate)

Divergences between two distributions over the D_ff memories (Jensen-Shannon
in nats, bounded by log 2; KL in nats) take LOG-probabilities: at large beta
most float32 probabilities underflow to 0, which made m = (p+q)/2 vanish and
JS jump to its log 2 clamp. log_softmax is always finite, so every divergence
is computed in the log domain; topk_mass sums the top-k in the log domain too.
JS is written pointwise in d = log p - log q so that JS(p, p) == 0 and
KL(p || p) == 0 hold exactly (verify: self_divergence).
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


def mean_score(logp: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return (logp.exp() * s).sum(dim=-1)


def var_score(logp: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    p = logp.exp()
    mu = (p * s).sum(dim=-1, keepdim=True)
    return (p * (s - mu).pow(2)).sum(dim=-1)


def topk_mass(logp: torch.Tensor, k: int) -> torch.Tensor:
    return torch.topk(logp, k, dim=-1).values.logsumexp(dim=-1).exp()


def topk_indices(logp: torch.Tensor, k: int) -> torch.Tensor:
    """int32 indices of the k largest probabilities, in descending order."""
    return torch.topk(logp, k, dim=-1, sorted=True).indices.to(torch.int32)


def xhat_top1_frac(x_hat: torch.Tensor) -> torch.Tensor:
    return x_hat.pow(2).max(dim=-1).values / x_hat.shape[-1]


def kl_divergence(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """KL(p || q) in nats from log-probabilities, clamped at 0 (float32
    rounding can make it slightly negative when p ~ q). Callers pass both
    rows of ONE log_softmax(beta s) tensor, so KL(p || p) is exactly 0: every
    term is p * (logp - logp) = 0."""
    return (logp.exp() * (logp - logq)).sum(dim=-1).clamp_min(0.0)


def js_divergence(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence in nats from log-probabilities, pointwise:
    with d = log p - log q,
        log(p/m) = log 2 - softplus(-d),   log(q/m) = log 2 - softplus(d),
        JS = 1/2 sum p log(p/m) + 1/2 sum q log(q/m).
    log 2 is taken as softplus(0) in the same dtype, so every term vanishes
    exactly where d = 0 and JS(p, p) == 0 bit-exactly (the former
    logaddexp(log p, log q) - log 2 form left one float32 rounding). Terms
    with p_i = 0 (resp. q_i = 0) are 0 by the xlogy convention: masked, so
    log p = -inf never produces 0 * inf. Clamped to [0, log 2] to absorb
    float32 rounding at the bounds."""
    d = logp - logq
    ln2 = torch.nn.functional.softplus(torch.zeros((), dtype=d.dtype, device=d.device))
    p, q = logp.exp(), logq.exp()
    zero = torch.zeros((), dtype=d.dtype, device=d.device)
    tp = torch.where(p > 0, p * (ln2 - torch.nn.functional.softplus(-d)), zero)
    tq = torch.where(q > 0, q * (ln2 - torch.nn.functional.softplus(d)), zero)
    return (0.5 * tp.sum(dim=-1) + 0.5 * tq.sum(dim=-1)).clamp(0.0, LN2)
