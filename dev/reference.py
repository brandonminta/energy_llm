"""Deliberately naive float64 reference implementation (verification only).

Re-implements every quantity of docs/MAPPING.md from the raw model weights
and the captured residuals, with explicit Python loops over layers, memories
and positions and Python floats (float64). It does NOT import anything from
``energy_llm``: the only inputs are the model (to read ``gate_proj.weight``,
``post_attention_layernorm.weight`` and ``config.rms_norm_eps``) and the
residual activations, which the caller reads with whatever hook it likes.

It is slow on purpose (no vectorised softmax, no fused kernels, a different
algebraic route: explicit unit rows of W_tilde instead of folding gamma into
x_hat), so agreement with the production pipeline is evidence that both
compute the specification, not that they share a bug.
"""

from __future__ import annotations

import math

import numpy as np

# Written out independently of energy_llm.core.capture.FEATURES; a test checks the order.
FEATURES = (
    "energy_prompt", "energy_resp", "delta_energy",
    "norm_entropy_prompt", "norm_entropy_resp", "delta_norm_entropy",
    "js", "hellinger_sq", "jaccard_topk",
    "margin_prompt", "margin_resp",
)
VARIANTS = ("mean", "first")


# ---------------------------------------------------------------- primitives

def x_hat(x: list[float], eps: float) -> list[float]:
    """x / sqrt(mean(x^2) + eps)."""
    ms = math.fsum(v * v for v in x) / len(x)
    inv = 1.0 / math.sqrt(ms + eps)
    return [v * inv for v in x]


def softmax_and_lse(s: list[float], beta: float) -> tuple[list[float], float]:
    m = max(s)
    e = [math.exp(beta * (v - m)) for v in s]
    z = math.fsum(e)
    return [v / z for v in e], beta * m + math.log(z)


def energy(lse: float, beta: float) -> float:
    return -lse / beta


def norm_entropy(p: list[float]) -> float:
    h = -math.fsum(v * math.log(v) for v in p if v > 0.0)
    return h / math.log(len(p))


def margin(s: list[float]) -> float:
    a, b = sorted(s, reverse=True)[:2]
    return a - b


def js(p: list[float], q: list[float]) -> float:
    total = []
    for a, b in zip(p, q):
        m = 0.5 * (a + b)
        if a > 0.0:
            total.append(0.5 * a * math.log(a / m))
        if b > 0.0:
            total.append(0.5 * b * math.log(b / m))
    return math.fsum(total)


def hellinger_sq(p: list[float], q: list[float]) -> float:
    return 0.5 * math.fsum((math.sqrt(a) - math.sqrt(b)) ** 2 for a, b in zip(p, q))


def top_k(p: list[float], k: int) -> set[int]:
    return set(sorted(range(len(p)), key=lambda i: -p[i])[:k])


def jaccard(p: list[float], q: list[float], k: int) -> float:
    a, b = top_k(p, k), top_k(q, k)
    return len(a & b) / len(a | b)


def mean_distribution(ps: list[list[float]]) -> list[float]:
    n = len(ps)
    return [math.fsum(col) / n for col in zip(*ps)]


# ---------------------------------------------------------------- per layer

def layer_scores(w_g: np.ndarray, gamma: np.ndarray, xs: list[list[float]]) -> tuple[list[list[float]], list[float]]:
    """Explicit W_tilde rows, their norms, unit rows, and s = unit . x_hat / sqrt(D).
    Returns (scores [T][K], row norms [K])."""
    d = len(gamma)
    gamma_l = [float(g) for g in gamma]
    scores = [[0.0] * w_g.shape[0] for _ in xs]
    norms = []
    for i in range(w_g.shape[0]):
        w_tilde = [float(w) * g for w, g in zip(w_g[i], gamma_l)]
        n = math.sqrt(math.fsum(v * v for v in w_tilde))
        norms.append(n)
        unit = np.array([v / n for v in w_tilde], dtype=np.float64)
        col = np.asarray(xs, dtype=np.float64) @ unit   # one memory, all positions
        for t in range(len(xs)):
            scores[t][i] = float(col[t]) / math.sqrt(d)
    return scores, norms


def phase_features(s_sel: list[list[float]], beta: float, k: int, variant: str) -> list[float]:
    """s_sel[0] = last prompt position, s_sel[1:] = response positions."""
    nan = float("nan")
    p0, lse0 = softmax_and_lse(s_sel[0], beta)
    e0, h0, m0 = energy(lse0, beta), norm_entropy(p0), margin(s_sel[0])
    if len(s_sel) == 1:
        return [e0, nan, nan, h0, nan, nan, nan, nan, nan, m0, nan]
    ps, es, hs, ms = [], [], [], []
    for s in s_sel[1:]:
        p, lse = softmax_and_lse(s, beta)
        ps.append(p); es.append(energy(lse, beta)); hs.append(norm_entropy(p)); ms.append(margin(s))
    if variant == "mean":
        n = len(ps)
        e_r, h_r, m_r = math.fsum(es) / n, math.fsum(hs) / n, math.fsum(ms) / n
        p_r = mean_distribution(ps)
    else:
        e_r, h_r, m_r, p_r = es[0], hs[0], ms[0], ps[0]
    return [e0, e_r, e_r - e0, h0, h_r, h_r - h0,
            js(p_r, p0), hellinger_sq(p_r, p0), jaccard(p_r, p0, k), m0, m_r]


def reference_item(model, residuals: dict, n_prompt: int, gen_ids: list[int],
                   stop_ids: list[int], betas: list[float], top_k_: int,
                   layers: list[int], trace_beta: float) -> dict:
    """Everything for one question on the given layers.

    residuals[l]: array-like [T, D] (any dtype), T = n_prompt + len(gen_ids).
    Returns scores {l: [T][K]}, features [len(layers), len(betas), 2, 11],
    energy / norm_entropy / margin traces [len(layers), T] at ``trace_beta``.
    """
    stops = set(int(t) for t in stop_ids)
    resp = [n_prompt + j for j, t in enumerate(gen_ids) if int(t) not in stops]
    sel = [n_prompt - 1] + resp
    eps = float(model.config.rms_norm_eps)
    T = n_prompt + len(gen_ids)

    feats = np.empty((len(layers), len(betas), len(VARIANTS), len(FEATURES)))
    tr_e = np.empty((len(layers), T))
    tr_h = np.empty((len(layers), T))
    tr_m = np.empty((len(layers), T))
    scores_out = {}
    for li, l in enumerate(layers):
        layer = model.model.layers[l]
        w_g = layer.mlp.gate_proj.weight.detach().double().cpu().numpy()
        gamma = layer.post_attention_layernorm.weight.detach().double().cpu().numpy()
        x = np.asarray(residuals[l], dtype=np.float64)
        xs = [x_hat([float(v) for v in x[t]], eps) for t in range(T)]
        s, _ = layer_scores(w_g, gamma, xs)
        scores_out[l] = s
        s_sel = [s[t] for t in sel]
        for bi, beta in enumerate(betas):
            for vi, variant in enumerate(VARIANTS):
                feats[li, bi, vi] = phase_features(s_sel, beta, top_k_, variant)
        for t in range(T):
            p, lse = softmax_and_lse(s[t], trace_beta)
            tr_e[li, t] = energy(lse, trace_beta)
            tr_h[li, t] = norm_entropy(p)
            tr_m[li, t] = margin(s[t])
    return {"scores": scores_out, "features": feats, "energy": tr_e,
            "norm_entropy": tr_h, "margin": tr_m, "response_positions": resp}
