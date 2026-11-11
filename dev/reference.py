"""Deliberately naive float64 reference implementation (verification only).

Re-implements every quantity of docs/ARCHITECTURE.md §5 from the raw model weights
and the captured residuals, with explicit Python loops over layers, memories
and positions and Python floats (float64). It does NOT import anything from
``energy_llm``: the only inputs are the model (to read ``gate_proj.weight``,
``post_attention_layernorm.weight`` and ``config.rms_norm_eps``) and the
residual activations, which the caller reads with whatever hook it likes.

It is slow on purpose (no vectorised softmax, no fused kernels, a different
algebraic route: explicit unit rows instead of one shared pre-activation), so
agreement with the production pipeline is evidence that both compute the
specification, not that they share a bug. Two score variants, written out
independently:

    cos_gamma_query  c_i = (w_i / ||w_i||) . (q / ||q||),  q = gamma * x_hat
    cos_gamma_key    s_i = (w_tilde_i / ||w_tilde_i||) . x_hat / sqrt(D),
                     w_tilde_i = w_i * gamma

and the query gain g = ||gamma * x_hat|| / sqrt(D).
"""

from __future__ import annotations

import math

import numpy as np

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
    """log(a / m) written as log 2 + log a - log(a + b): m = (a + b)/2 rounds to
    0 when a is the smallest float64 subnormal (beta ~ 1e4), a + b never does."""
    total = []
    for a, b in zip(p, q):
        if a > 0.0:
            total.append(0.5 * a * (math.log(2.0) + math.log(a) - math.log(a + b)))
        if b > 0.0:
            total.append(0.5 * b * (math.log(2.0) + math.log(b) - math.log(a + b)))
    return math.fsum(total)


def mean_score(p: list[float], s: list[float]) -> float:
    return math.fsum(a * b for a, b in zip(p, s))


def var_score(p: list[float], s: list[float]) -> float:
    mu = mean_score(p, s)
    return math.fsum(a * (b - mu) ** 2 for a, b in zip(p, s))


def topk_indices(p: list[float], k: int) -> list[int]:
    """Indices of the k largest probabilities, descending (ties: lower index first)."""
    order = sorted(range(len(p)), key=lambda i: (-p[i], i))
    return order[:k]


def topk_mass(p: list[float], k: int) -> float:
    return math.fsum(p[i] for i in topk_indices(p, k))


def kl(logp: list[float], logq: list[float]) -> float:
    """KL(p || q) from log-probabilities (log p_i = beta s_i - lse is exact in
    float64 even where p_i itself underflows)."""
    return max(0.0, math.fsum(math.exp(a) * (a - b) for a, b in zip(logp, logq)))


def xhat_top1_frac(xh: list[float]) -> float:
    return max(v * v for v in xh) / len(xh)


# ---------------------------------------------------------------- per layer

def layer_scores(w_g: np.ndarray, gamma: np.ndarray, xs: list[list[float]]) -> tuple[list[list[float]], list[float]]:
    """cos_gamma_key: explicit W_tilde rows, their norms, unit rows, and
    s = unit . x_hat / sqrt(D). Returns (scores [T][K], row norms [K])."""
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


def query_scores(w_g: np.ndarray, gamma: np.ndarray, xs: list[list[float]]) -> tuple[list[list[float]], list[float]]:
    """cos_gamma_query: explicit queries q_t = gamma * x_hat_t and unit queries,
    explicit unit rows of W_g (no gamma), c = unit_w . unit_q.
    Returns (scores [T][K], query gains [T] = ||q_t|| / sqrt(D))."""
    d = len(gamma)
    qs, gains = [], []
    for x in xs:
        q = [float(g) * v for g, v in zip(gamma, x)]
        nq = math.sqrt(math.fsum(v * v for v in q))
        gains.append(nq / math.sqrt(d))
        qs.append([v / nq for v in q])
    qs = np.asarray(qs, dtype=np.float64)
    scores = [[0.0] * w_g.shape[0] for _ in xs]
    for i in range(w_g.shape[0]):
        w = [float(v) for v in w_g[i]]
        n = math.sqrt(math.fsum(v * v for v in w))
        col = qs @ np.array([v / n for v in w], dtype=np.float64)
        for t in range(len(xs)):
            scores[t][i] = float(col[t])
    return scores, gains


REFERENCES = ("question_last", "prompt_last", "response_first")   # written out independently
VARIANTS = ("cos_gamma_query", "cos_gamma_key")


def _variant_traces(s: list[list[float]], refs: dict, betas: list[float],
                    topk_mass_k: int, topk_ids_k: int) -> dict:
    """Every per-token trace of one layer from its scores s [T][K]."""
    T, nb = len(s), len(betas)
    per_beta = ["energy", "norm_entropy", "topk_mass", "mean_score", "var_score"] + \
        [f"{d}/{r}" for d in ("js", "dkl") for r in REFERENCES]
    out = {k: np.full((T, nb), np.nan) for k in per_beta}
    out["margin"] = np.array([margin(s[t]) for t in range(T)])
    out["topk_ids"] = np.array([topk_indices(s[t], topk_ids_k) for t in range(T)], dtype=np.int64)
    for bi, beta in enumerate(betas):
        ps, logps = [], []
        for t in range(T):
            p, lse = softmax_and_lse(s[t], beta)
            ps.append(p)
            logps.append([beta * v - lse for v in s[t]])
            out["energy"][t, bi] = energy(lse, beta)
            out["norm_entropy"][t, bi] = norm_entropy(p)
            out["topk_mass"][t, bi] = topk_mass(p, topk_mass_k)
            out["mean_score"][t, bi] = mean_score(p, s[t])
            out["var_score"][t, bi] = var_score(p, s[t])
        for r, pos in refs.items():
            if pos is None:
                continue
            for t in range(T):
                out[f"js/{r}"][t, bi] = js(ps[t], ps[pos])
                out[f"dkl/{r}"][t, bi] = kl(logps[t], logps[pos]) - kl(logps[pos], logps[t])
    return out


def reference_item(model, residuals: dict, n_prompt: int, gen_ids: list[int],
                   stop_ids: list[int], layers: list[int], betas: dict,
                   question_last: int, topk_mass_k: int, topk_ids_k: int) -> dict:
    """Every per-token trace of both score variants for one question on the
    given layers.

    residuals[l]: array-like [T, D] (any dtype), T = n_prompt + len(gen_ids).
    ``betas[variant]``: that variant's beta list. ``question_last``: position
    of the last question token (the caller reads it from the role mask).
    Returns scores {variant: {l: [T][K]}}, query_gain and xhat_top1_frac
    [len(layers), T], the reference positions, and traces {variant: {name:
    array}} named as in production: [len(layers), T, n_beta] for energy,
    norm_entropy, topk_mass, mean_score, var_score, js/<ref>, dkl/<ref>
    (dkl = KL(p_t||p_ref) - KL(p_ref||p_t)); [len(layers), T] for margin;
    topk_ids [len(layers), T, k] (as sets: ties may order differently).
    """
    stops = set(int(t) for t in stop_ids)
    resp = [n_prompt + j for j, t in enumerate(gen_ids) if int(t) not in stops]
    refs = {"question_last": question_last, "prompt_last": n_prompt - 1,
            "response_first": resp[0] if resp else None}
    eps = float(model.config.rms_norm_eps)
    T = n_prompt + len(gen_ids)
    scores = {v: {} for v in VARIANTS}
    per_layer = {v: [] for v in VARIANTS}
    xfrac, gains = [], []
    for l in layers:
        layer = model.model.layers[l]
        w_g = layer.mlp.gate_proj.weight.detach().double().cpu().numpy()
        gamma = layer.post_attention_layernorm.weight.detach().double().cpu().numpy()
        x = np.asarray(residuals[l], dtype=np.float64)
        xs = [x_hat([float(v) for v in x[t]], eps) for t in range(T)]
        xfrac.append([xhat_top1_frac(xs[t]) for t in range(T)])
        scores["cos_gamma_key"][l], _ = layer_scores(w_g, gamma, xs)
        scores["cos_gamma_query"][l], g = query_scores(w_g, gamma, xs)
        gains.append(g)
        for v in VARIANTS:
            per_layer[v].append(_variant_traces(scores[v][l], refs, betas[v], topk_mass_k, topk_ids_k))
    traces = {v: {k: np.stack([d[k] for d in per_layer[v]]) for k in per_layer[v][0]} for v in VARIANTS}
    return {"scores": scores, "reference_positions": refs, "response_positions": resp,
            "traces": traces, "query_gain": np.asarray(gains), "xhat_top1_frac": np.asarray(xfrac)}
