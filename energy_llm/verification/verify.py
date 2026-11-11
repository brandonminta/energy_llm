"""Verification of the extraction (``energy-llm verify``): not part of the
production path.

On a few questions it runs the production extract + merge, then checks:

  reference      every per-token trace vs dev/reference.py (naive float64,
                 explicit loops), at every beta and reference position
  gate           real gate pre-activation g == ||w|| ||gamma x_hat|| c
                 (cos_gamma_query) == ||w_tilde|| sqrt(D) s (cos_gamma_key), with a
                 no-gamma negative control that must fail by a wide margin
  tf_vs_cache    residual x captured during cached token-by-token generation
                 vs the teacher-forced pass, propagated to the traces; PASS on
                 the analysis positions, broken down by role and special position
  alignment      last prompt position emits the first generated token; response
                 positions are exactly the generated non-stop tokens
  hidden         hidden.h5 / traces.h5 hold the same x and values the metrics
                 computed
  permutation    permuting the bank rows (and W_g consistently) changes nothing
                 (topk_ids compared after mapping through the permutation)
  limits         beta -> 0: H_bar -> 1; beta -> inf: H_bar -> 0, E -> -max(s),
                 var_score -> 0, topk_mass -> 1
  monotone       H_bar non-increasing in beta on real activations
  calibration    H_bar at beta* on the warm-up set lies in the calibration bracket
  self           JS(p, p) = KL(p || p) = 0
  determinism    two identical runs agree (bit-exact on CPU)

Every check that involves the bank runs for BOTH score variants
(cos_gamma_query, cos_gamma_key; core.bank) and must pass for both. Every
tolerance is explicit and justified next to its definition; the report gives
the maximum observed error and the tolerance for each check.
"""

from __future__ import annotations

import importlib.util
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from energy_llm.config import RunConfig
from energy_llm.core import metrics
from energy_llm.core.bank import CONTROL, PRIMARY, VARIANTS, Bank, build_bank
from energy_llm.core.capture import (
    REFERENCES, SHARED, ResidualHooks, compute_item, forward_item, norm_entropy_curve, variant_traces,
)
from energy_llm.core.model import ModelBundle, decoder_layers, ffn_norm
from energy_llm.core.prompting import ROLE_QUESTION, ROLE_RESPONSE, build_prompt
from energy_llm.pipeline.extract import load_beta_star, variant_betas, variant_topk_ids
from energy_llm.storage import Item, read_items, sha256_file, write_json_atomic

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "dev" / "reference.py"

# Unit roundoff u = 2^-p (p = significand bits incl. the implicit one).
UNIT_ROUNDOFF = {"bfloat16": 2.0 ** -8, "float16": 2.0 ** -11, "float32": 2.0 ** -24}
# Floor for float32 reductions over D_ff terms: torch reduces pairwise, error
# ~ log2(D_ff) * u ~ 1e-6; 1e-5 leaves a 10x margin.
F32_FLOOR = 1e-5


def load_reference():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"{REFERENCE_PATH} not found (verify needs the repository checkout)")
    spec = importlib.util.spec_from_file_location("energy_llm_dev_reference", REFERENCE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Check:
    name: str
    status: str                 # PASS | FAIL | SKIP
    max_error: float | None
    tolerance: float | None
    detail: dict = field(default_factory=dict)


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _nan_aware_max_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, bool]:
    """Max |a-b| over entries finite in both, and whether NaNs coincide."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    m = ~(np.isnan(a) | np.isnan(b))
    return (float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0), same_nan


def _f16_ulp_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, bool]:
    """Max |a-b| in float16 ulps of b over entries finite in both, and whether
    NaNs coincide. Stored traces and residuals are float16: a recomputation
    that differs by float32 rounding (GPU) can flip the last float16 bit."""
    a, b = np.asarray(a, dtype=np.float16), np.asarray(b, dtype=np.float16)
    same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    m = ~(np.isnan(a) | np.isnan(b))
    if not m.any():
        return 0.0, same_nan
    a64, b64 = a[m].astype(np.float64), b[m].astype(np.float64)
    return float((np.abs(a64 - b64) / np.spacing(np.abs(b[m])).astype(np.float64)).max()), same_nan


def _topk_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Min over rows of |A ∩ B| / |A ∪ B| for index arrays [..., k]."""
    a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    return min((len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a.tolist(), b.tolist())),
               default=1.0)


# ----------------------------------------------------------------------
# Reference comparison (also used by tests/test_reference.py)
# ----------------------------------------------------------------------

def reference_tolerances(delta_s: float, beta: float, s_range: float, k_mass: int,
                         d_model: int) -> dict[str, float]:
    """Per-trace tolerance at one beta from the observed max score error
    delta_s, via first-order bounds in the max norm (|d log p_i| <= 2 beta
    delta_s, hence |dp_i| <= 2 beta delta_s p_i and ||dp||_1 <= 2 beta delta_s):
      E      = -(1/beta) lse(beta s):  |dE| <= delta_s
      margin = top1 - top2:            |dm| <= 2 delta_s
      H_bar:  sum_i |dH/ds_i| <= beta sum p_i |log p_i + H| <= 2 beta log D_ff,
              so |dH_bar| <= 2 beta delta_s
      mean_score = sum p s: sum dp (s - mid) + sum p ds
              <= beta delta_s range + delta_s; used: 2 beta delta_s range + delta_s
      var_score: sum dp (s-mu)^2 + 2 sum p (s-mu) ds <= beta delta_s range^2 + 2 delta_s range;
              used (range <= 2): 2 beta delta_s range + 2 delta_s range
      topk_mass: sum over the top k of |dp_i| <= 2 beta delta_s (a swap at the
              k-th boundary exchanges near-equal p); used max(k, 2) beta delta_s
      JS: 4 beta delta_s (conservative constant; log(p/m) is bounded by log 2)
      dkl = KL(p||q) - KL(q||p): KL has no bounded log ratio, so
              |dKL(p||q)| <= sum |dp| |log p - log q| + sum p |dlog p - dlog q|
                          <= 2 beta delta_s sum p |log p - log q| + 4 beta delta_s;
              the per-element weight A = sum p|log p - log q| + sum q|log q - log p|
              is computed by compare_with_reference from the float64 reference
              scores: 2 beta delta_s A + 8 beta delta_s (here only the 8 beta delta_s term)
      xhat_top1_frac, query_gain: from x_hat (and gamma) only, within score_tolerance
    ``s_range`` = max_t (max s_t - min s_t). compare_with_reference adds a
    float32 reduction floor F32_FLOOR * max(1, |value|) per element (dkl
    reaches hundreds at beta ~ 1e4). topk_ids are compared as sets."""
    b, d = beta, delta_s
    tol = {"energy": d, "margin": 2 * d, "norm_entropy": 2 * b * d,
           "mean_score": 2 * b * d * s_range + d, "var_score": 2 * b * d * s_range + 2 * d * s_range,
           "topk_mass": max(k_mass, 2) * b * d, "xhat_top1_frac": score_tolerance(d_model),
           "query_gain": score_tolerance(d_model)}
    for r in REFERENCES:
        tol[f"js/{r}"] = 4 * b * d
        tol[f"dkl/{r}"] = 8 * b * d
    return tol


# topk_ids vs the reference: near-ties at the k-th score may swap one index
TOPK_MIN_JACCARD = 0.98


def self_divergence_tolerance(d_ff: int, c: float = 4.0) -> float:
    """JS(p, p) and KL(p || p) are computed pointwise from ONE log_softmax(beta s)
    (metrics.js_divergence, metrics.kl_divergence), so every term is exactly 0
    and the expected value is 0 bit-exactly. The tolerance is the bound a
    non-exact implementation would have to meet: each of the D_ff terms carries
    at most a relative rounding u32 on a log-ratio of size <= log D_ff (the
    worst log-probability gap of a near-uniform p), hence c * u32 * log D_ff
    with c = 4 (two roundings per term, the 1/2 weights, one of margin).
    At D_ff = 11008: 2.2e-6."""
    return c * UNIT_ROUNDOFF["float32"] * math.log(d_ff)


def score_tolerance(d_model: int) -> float:
    """Sequential float32 dot product of D terms: |fl(sum) - sum| <= D u sum|terms|
    (first order), and sum|w_tilde_ij x_hat_j| / (||w_tilde_i|| sqrt(D)) <= 1
    by Cauchy-Schwarz with ||x_hat|| ~ sqrt(D); so |ds| <= D u."""
    return d_model * UNIT_ROUNDOFF["float32"]


def divergence_scale(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """A = sum p |log p - log q| + sum q |log q - log p| per row: the sum of
    the absolute terms of KL(p||q) and KL(q||p), i.e. the condition scale of
    dkl = KL(p||q) - KL(q||p). dkl is a difference of two KLs of up to
    hundreds of nats at large beta, so its absolute rounding error scales
    with A, not with |dkl| (ill-conditioned at high beta)."""
    d = (logp - logq).abs()
    return (logp.exp() * d).sum(dim=-1) + (logq.exp() * d).sum(dim=-1)


def view(traces: dict, variant: str) -> dict:
    """One variant's traces (prefix removed) plus the shared ones."""
    return variant_traces(traces, variant) | {k: traces[k] for k in SHARED if k in traces}


def compare_with_reference(bundle: ModelBundle, bank: Bank, trace, traces: dict, ref, layers: list[int],
                           betas: list[float], k_mass: int, variant: str) -> dict:
    """``traces``: one variant's production traces plus the shared ones
    (``view``); ``ref``: the full dev/reference.py record (both variants)."""
    eps = bundle.dims.rms_norm_eps
    scores = ref["scores"][variant]
    ref_traces = ref["traces"][variant] | {"xhat_top1_frac": ref["xhat_top1_frac"],
                                           "query_gain": ref["query_gain"]}
    ds = 0.0
    for l in layers:
        s = bank.scores(l, metrics.rms_normalise(trace.residuals[l].to(bank.device), eps),
                        variant).double().cpu().numpy()
        ds = max(ds, float(np.abs(s - np.asarray(scores[l])).max()))
    s_range = max(float(np.ptp(np.asarray(scores[l]), axis=-1).max()) for l in layers)
    tol_s = score_tolerance(bundle.dims.d_model)
    tols = [reference_tolerances(ds, b, s_range, k_mass, bundle.dims.d_model) for b in betas]
    # dkl: add the data-dependent term 2 beta ds A (see reference_tolerances)
    T = len(scores[layers[0]])
    dkl_tol = {r: np.zeros((len(layers), T, len(betas))) for r in REFERENCES}
    for li, l in enumerate(layers):
        sc = torch.as_tensor(np.asarray(scores[l], dtype=np.float64))
        for bi, b in enumerate(betas):
            lp = torch.log_softmax(b * sc, dim=-1)
            for r, pos in ref["reference_positions"].items():
                if pos is not None:
                    a = divergence_scale(lp, lp[pos]).numpy()
                    dkl_tol[r][li, :, bi] = 2 * b * ds * a + tols[bi][f"dkl/{r}"]
    ratio, nan_ok, per = 0.0, True, {}
    for name, r in ref_traces.items():
        if name not in traces:          # topk_ids not stored for this variant
            continue
        p = traces[name][layers]
        if name == "topk_ids":
            jac = _topk_jaccard(p, r)
            per[name] = {"min_row_jaccard": jac, "required": TOPK_MIN_JACCARD}
            continue
        if name.startswith("dkl/"):
            t = dkl_tol[name[4:]]
        else:
            t = np.asarray([tl[name] for tl in tols]) if r.ndim == 3 else tols[-1][name]
        nan_ok &= bool(np.array_equal(np.isnan(p), np.isnan(r)))
        m = ~(np.isnan(p) | np.isnan(r))
        err = np.abs(p.astype(np.float64) - r)
        q = err / (t + F32_FLOOR * np.maximum(1.0, np.abs(r)))
        worst = float(q[m].max()) if m.any() else 0.0
        per[name] = {"max_error": float(err[m].max()) if m.any() else 0.0,
                     "max_error_over_tolerance": worst}
        ratio = max(ratio, worst)
    ids_ok = "topk_ids" not in per or per["topk_ids"]["min_row_jaccard"] >= TOPK_MIN_JACCARD
    return {"score_error": ds, "score_tolerance": tol_s, "nan_pattern_equal": nan_ok,
            "max_error_over_tolerance": ratio, "traces": per,
            "ok": ds <= tol_s and nan_ok and ratio <= 1.0 and ids_ok}


# ----------------------------------------------------------------------
# Checks against the model's own computation
# ----------------------------------------------------------------------

class _Capture:
    """Minimal hooks owned by verify: append every call (cached generation),
    and the gate_proj output."""

    def __init__(self, model, gate: bool):
        self.x: dict[int, list[torch.Tensor]] = {}
        self.g: dict[int, torch.Tensor] = {}
        self.handles = []
        for l, layer in enumerate(decoder_layers(model)):
            self.handles.append(ffn_norm(model, layer).register_forward_pre_hook(
                lambda m, a, k, l=l: self.x.setdefault(l, []).append(
                    (a[0] if a else k["hidden_states"])[0].detach().clone()), with_kwargs=True))
            if gate:
                self.handles.append(layer.mlp.gate_proj.register_forward_hook(
                    lambda m, a, o, l=l: self.g.__setitem__(l, o[0].detach().float())))

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.inference_mode()
def check_gate(bundle: ModelBundle, bank: Bank, trace) -> tuple[dict[str, float], float, float]:
    """Per variant, max normalised error e = |g - g_rec| / (|g| + ||w_tilde ⊙ x_hat||_2)
    over layers, positions, memories, with g_rec = ||w|| ||gamma * x_hat|| c
    (cos_gamma_query) and ||w_tilde|| sqrt(D) s (cos_gamma_key); the same for
    the no-gamma control, and the tolerance. In the model g = W_g (gamma * x_hat)
    involves three roundings in the model dtype (x_hat, gamma * x_hat, the
    output), each <= u relative to the scale |g| + ||w_tilde ⊙ x_hat||_2;
    tol = 4u (one u of margin), floored at F32_FLOOR for float32 models. The
    gate is read from the trace model (where the traced residuals come from)."""
    model = bundle.trace_model
    cap = _Capture(model, gate=True)
    full = torch.tensor([trace.prompt.input_ids + trace.gen_ids], device=bundle.device)
    try:
        model(input_ids=full, use_cache=False, logits_to_keep=1)
    finally:
        cap.remove()
    eps, d = bundle.dims.rms_norm_eps, bundle.dims.d_model
    err, ctrl = dict.fromkeys(VARIANTS, 0.0), 0.0
    for l in range(bundle.dims.n_layers):
        x_hat = metrics.rms_normalise(cap.x[l][0].to(bank.device), eps)
        g_rec = {CONTROL: bank.scores(l, x_hat, CONTROL) * bank.row_norms[l] * math.sqrt(d),
                 PRIMARY: bank.scores(l, x_hat, PRIMARY) * bank.key_norms[l]
                 * bank.query_norm(l, x_hat)[:, None]}
        g = cap.g[l].to(bank.device)
        w = bank.gate_weights[l].float()
        scale = g.abs() + torch.sqrt((x_hat * bank.gamma[l]).pow(2) @ w.pow(2).T)
        g_ctrl = x_hat @ w.T                      # gamma omitted: must NOT reconstruct g
        for v, gr in g_rec.items():
            err[v] = max(err[v], ((g - gr).abs() / scale).max().item())
        ctrl = max(ctrl, ((g - g_ctrl).abs() / scale).max().item())
        del w
    tol = max(4 * UNIT_ROUNDOFF[bundle.trace_dtype], F32_FLOOR)
    return err, ctrl, tol


def cpu64(t: torch.Tensor) -> torch.Tensor:
    """Convention of every comparison between computation paths (tf, cache,
    fp32): both operands go to CPU float64 inside the comparing helper, never
    in the caller, so operands kept on different devices (CPU-saved residuals
    vs live GPU tensors) cannot meet in one op."""
    return t.detach().to("cpu", torch.float64)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Per row ||a - b|| / ||b||, on CPU in float64 (cpu64)."""
    a, b = cpu64(a), cpu64(b)
    return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)).numpy()


def dist_stats(v) -> dict:
    """median / p99 / max of a flat sample (None when empty)."""
    v = np.concatenate([np.ravel(x) for x in v]) if isinstance(v, list) else np.ravel(v)
    if v.size == 0:
        return {"median": None, "p99": None, "max": None}
    return {"median": float(np.median(v)), "p99": float(np.percentile(v, 99)), "max": float(v.max())}


# Trace quantities the residual error is propagated to, and the dataset whose
# spread between questions is the signal it is compared with.
PROPAGATED = {"norm_entropy": "norm_entropy", "energy": "energy", "js": "js/question_last"}


@torch.inference_mode()
def propagate(bank: Bank, eps: float, l: int, x_a: torch.Tensor, x_b: torch.Tensor,
              beta_star: dict[str, float]) -> dict[str, dict[str, np.ndarray]]:
    """Per variant v and position at beta*_v: |H_bar(a) - H_bar(b)|,
    |E(a) - E(b)| and JS(p_a, p_b), with p = softmax(beta s_v) from the
    residuals x_a, x_b of layer l. Each path's scores and log p are computed as
    in production (float32 on the bank device, one preactivation per path);
    the comparison (differences, JS) runs on CPU in float64 (cpu64)."""
    side = {v: [] for v in VARIANTS}
    for x in (x_a, x_b):
        x_hat = metrics.rms_normalise(x.to(bank.device), eps)
        z = bank.preactivation(l, x_hat)
        for v in VARIANTS:
            s = bank.from_preactivation(l, z, x_hat, v)
            lp = metrics.log_distribution(s, beta_star[v])
            side[v].append((cpu64(metrics.norm_entropy(lp)), cpu64(metrics.energy(s, beta_star[v])),
                            cpu64(lp)))
            del s, lp
        del x_hat, z
    out = {}
    for v, ((h_a, e_a, la), (h_b, e_b, lb)) in side.items():
        out[v] = {"norm_entropy": (h_a - h_b).abs().numpy(), "energy": (e_a - e_b).abs().numpy(),
                  "js": metrics.js_divergence(la, lb).numpy()}
    return out


# Positions on which TF-vs-cache and TF-vs-fp32 noise is judged: the ones the
# analysis reads (response tokens and the three references). Template tokens
# and position 0 (attention sink: the pilot's max of ||x_cache - x_tf||) are
# reported apart and do not enter the PASS criterion.
ANALYSIS = "analysis"


def position_groups(roles: list[int], refs: dict, n: int) -> dict[str, np.ndarray]:
    """Boolean masks over positions 0..n-1: ``analysis`` (response ∪
    references), ``role/<name>`` per role and ``position/<name>`` for
    position 0 and the references (empty when outside 0..n-1)."""
    from energy_llm.core.prompting import ROLE_NAMES

    r = np.asarray(roles[:n])

    def at(p):
        m = np.zeros(n, dtype=bool)
        if p is not None and p < n:
            m[p] = True
        return m

    special = {"position_0": at(0)} | {k: at(p) for k, p in refs.items()}
    analysis = r == ROLE_RESPONSE
    for k in REFERENCES:
        analysis = analysis | special[k]
    return ({ANALYSIS: analysis} | {f"role/{ROLE_NAMES[c]}": r == c for c in sorted(ROLE_NAMES)}
            | {f"position/{k}": m for k, m in special.items()})


def _masked(per_layer: list, masks: list[dict], key: str) -> list:
    """per_layer[l]: list over questions of per-position arrays -> the same
    restricted to the positions of group ``key``."""
    return [[a[m[key]] for a, m in zip(qs, masks)] for qs in per_layer]


def grouped_noise(noise: dict, signal: dict, masks: list[dict]) -> dict:
    """noise_to_signal per position group (same sigma_l: the signal is the
    response spread of the trace at beta*)."""
    return {k: noise_to_signal({m: _masked(noise[m], masks, k) for m in noise}, signal) for k in masks[0]}


def grouped_dist(per_layer: list, masks: list[dict]) -> dict:
    """dist_stats per layer and position group."""
    return {k: [dist_stats(qs) for qs in _masked(per_layer, masks, k)] for k in masks[0]}


def noise_to_signal(noise: dict, signal: dict) -> dict:
    """noise[metric][l], signal[metric][l]: lists of per-question arrays.
    sigma_l = std over (response positions x questions) of the metric at beta*;
    ratio = p99(|delta|) / sigma_l (the PASS statistic) and max(|delta|) / sigma_l
    (reported apart), both next to sigma_l per layer (a large ratio can be a
    small sigma, not a large noise). sigma_l = 0 gives ratio 0 when the noise
    is 0, else inf."""
    def ratio(x, sd):
        return 0.0 if x == 0 else (float("inf") if sd == 0 else x / sd)

    per_layer, worst_p99, worst_max, where = [], 0.0, 0.0, None
    for l in range(len(next(iter(noise.values())))):
        row = {"layer": l}
        for m in noise:
            st = dist_stats(noise[m][l])
            sig = np.concatenate([np.ravel(x) for x in signal[m][l]]) if signal[m][l] else np.zeros(0)
            sd = float(np.std(sig)) if sig.size > 1 else 0.0
            rp, rm = ratio(st["p99"] or 0.0, sd), ratio(st["max"] or 0.0, sd)
            row[m] = st | {"sigma": sd, "ratio_p99": rp, "ratio_max": rm}
            if rp > worst_p99:
                worst_p99, where = rp, {"layer": l, "metric": m, "sigma": sd}
            worst_max = max(worst_max, rm)
        per_layer.append(row)
    return {"max_ratio_p99": worst_p99, "max_ratio_max": worst_max, "worst_p99_at": where,
            "per_layer": per_layer}


@torch.inference_mode()
def check_tf_vs_cache(bundle: ModelBundle, trace) -> tuple[dict, dict]:
    """Regenerate with the KV cache while recording x at every call. Positions
    0 .. n_prompt + T_g - 2 are compared with the teacher-forced x (the last
    generated token is never fed back during generation). Returns the cached
    residuals per layer ([n_cmp, D], model dtype, CPU) and info (tokens_equal,
    positions_compared, the first-step logits for the alignment check)."""
    model = bundle.model
    cap = _Capture(model, gate=False)
    ids = torch.tensor([trace.prompt.input_ids], device=bundle.device)
    try:
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             generation_config=bundle.generation_config)
    finally:
        cap.remove()
    gen = out.sequences[0, ids.shape[1]:].tolist()
    x_cache = {l: torch.cat(cap.x[l], dim=0).cpu() for l in range(bundle.dims.n_layers)}
    del cap
    return x_cache, {"tokens_equal": gen == trace.gen_ids, "positions_compared": x_cache[0].shape[0],
                     "first_step_logits": out.logits[0][0].float()}


def tf_cache_tolerance(dtype: str) -> float:
    """INFORMATIVE threshold on ||x_cache - x_tf|| / ||x_tf|| (not the PASS
    criterion since step 9). KV-cached and full-sequence attention use different
    kernels and reduction orders; in float32 the difference stays ~1e-6 relative
    (1e-4 allows for depth); in bfloat16 rounding differences compound over
    layers (pilot: 0.09 at Qwen2.5-3B). PASS is decided on the traces: see
    noise_to_signal."""
    return 1e-4 if dtype == "float32" else 5e-2


def load_fp32_model(cfg: RunConfig, device: torch.device):
    """A separate float32 instance of the same checkpoint (local cache)."""
    from energy_llm.core.model import load_model

    model, _ = load_model(cfg.model.id, cfg.model.revision, device=str(device), dtype=torch.float32)
    model.eval()
    return model


# Pre-registered precision rule (pilot): if the TF bf16 traces are farther
# from float32 than this fraction of sigma_l on the analysis positions at any
# layer, the traces should be computed in float32 (model.trace_dtype).
RECOMMEND_FP32_RATIO = 0.1


@torch.inference_mode()
def fp32_reference(cfg: RunConfig, bundle: ModelBundle, bank: Bank, saved: list[dict],
                   beta_star: dict[str, float], signal: dict) -> dict:
    """Recompute x with a float32 model on the same tokens (one teacher-forced
    pass per question) and compare the traced residuals with it, per layer:
    d(x_tf, x_fp32) at every position, d(x_cache, x_fp32) at the cached
    positions, and both propagated to the traces of each variant at beta*_v
    (noise_to_signal), all broken down by position group (position_groups).
    ``signal[v]``: the per-variant sigma samples of tf_vs_cache. The float32
    instance is loaded separately from the cache (never an in-place cast of
    the production model) and freed afterwards; a float32 bundle (tests) is
    its own reference, and so is a float32 trace model (trace_dtype float32).
    Full float32 matmul precision is forced during the
    check (no TF32) and restored after. Precision rule: if max over variants,
    layers and metrics of p99/sigma (TF vs float32, analysis positions) exceeds
    RECOMMEND_FP32_RATIO, the report recommends trace_dtype=float32."""
    from energy_llm.core.model import release_cuda

    prev = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    separate = bundle.trace_dtype != "float32"
    x32 = []
    try:
        model = load_fp32_model(cfg, bundle.device) if separate else bundle.trace_model
        for rec in saved:
            cap = _Capture(model, gate=False)
            try:
                model(input_ids=torch.tensor([rec["ids"]], device=bundle.device), use_cache=False,
                      logits_to_keep=1)
            finally:
                cap.remove()
            x32.append({l: v[0].float().cpu() for l, v in cap.x.items()})
            del cap
        if separate:
            del model
            release_cuda()
    finally:
        torch.set_float32_matmul_precision(prev[0])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev[1], prev[2]

    L, eps = bundle.dims.n_layers, bundle.dims.rms_norm_eps
    d_tf = [[] for _ in range(L)]
    d_cache = [[] for _ in range(L)]
    n_tf = {v: {m: [[] for _ in range(L)] for m in PROPAGATED} for v in VARIANTS}
    n_cache = {v: {m: [[] for _ in range(L)] for m in PROPAGATED} for v in VARIANTS}
    m_tf, m_cache = [], []
    for rec, x in zip(saved, x32):
        n = rec["x_cache"][0].shape[0]
        m_tf.append(position_groups(rec["roles"], rec["refs"], len(rec["ids"])))
        m_cache.append(position_groups(rec["roles"], rec["refs"], n))
        for l in range(L):
            d_tf[l].append(rel_err(rec["x_tf"][l], x[l]))
            d_cache[l].append(rel_err(rec["x_cache"][l], x[l][:n]))
            for v, d in propagate(bank, eps, l, rec["x_tf"][l], x[l], beta_star).items():
                for m, a in d.items():
                    n_tf[v][m][l].append(a)
            for v, d in propagate(bank, eps, l, rec["x_cache"][l], x[l][:n], beta_star).items():
                for m, a in d.items():
                    n_cache[v][m][l].append(a)
    tr_tf = {v: grouped_noise(n_tf[v], signal[v], m_tf) for v in VARIANTS}
    tr_cache = {v: grouped_noise(n_cache[v], signal[v], m_cache) for v in VARIANTS}
    worst = max(((tr_tf[v][ANALYSIS]["max_ratio_p99"], v) for v in VARIANTS), key=lambda t: t[0])
    rule = {"statistic": "max over variants, layers and metrics of p99(|trace(x_tf) - trace(x_fp32)|) / "
                         "sigma_l on the analysis positions",
            "threshold": RECOMMEND_FP32_RATIO, "value": worst[0],
            "at": {"variant": worst[1]} | (tr_tf[worst[1]][ANALYSIS]["worst_p99_at"] or {}),
            "recommend": "trace_dtype=float32" if worst[0] > RECOMMEND_FP32_RATIO else None}
    return {
        "separate_fp32_instance": separate,
        "precision": {"float32_matmul_precision": "highest", "cuda_matmul_allow_tf32": False,
                      "cudnn_allow_tf32": False, "restored_after": True},
        "x_rel_tf_vs_fp32": grouped_dist(d_tf, m_tf),
        "x_rel_cache_vs_fp32": grouped_dist(d_cache, m_cache),
        "traces_tf_vs_fp32": tr_tf,
        "traces_cache_vs_fp32": tr_cache,
        "precision_rule": rule,
    }


class PermutedBank(Bank):
    """Bank whose memories are permuted (rows of W_g and both key norms
    together), computed on the fly: scores come out permuted, nothing else changes."""

    def __init__(self, base: Bank, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.perm = [torch.randperm(n.shape[0], generator=g).to(n.device) for n in base.row_norms]
        super().__init__(gate_weights=base.gate_weights, gamma=base.gamma,
                         key_norms=[k[p] for k, p in zip(base.key_norms, self.perm)],
                         row_norms=[n[p] for n, p in zip(base.row_norms, self.perm)],
                         d_model=base.d_model)

    def preactivation(self, l, x_hat):
        w = self.gate_weights[l].detach()[self.perm[l]].float()
        z = (x_hat * self.gamma[l]) @ w.T
        del w
        return z


# Safety factor of the permutation tolerances: each term of a reduction also
# carries a few roundings of its own (exp, product, difference: <= 3u, small
# against ceil(log2 D_ff) u = 14u at D_ff = 11008) plus second-order terms.
PERM_C = 2.0


def gamma_shuffled_bank(base: Bank, seed: int = 1) -> Bank:
    """Negative control: gamma permuted over the D input dims WITHOUT permuting
    W_g. The memories change, so the permutation tolerances must be exceeded."""
    g = torch.Generator().manual_seed(seed)
    return Bank(gate_weights=base.gate_weights,
                gamma=[gm[torch.randperm(gm.shape[0], generator=g).to(gm.device)] for gm in base.gamma],
                key_norms=base.key_norms, row_norms=base.row_norms, d_model=base.d_model)


@torch.inference_mode()
def permutation_tolerances(bundle: ModelBundle, bank: Bank, pbank: "PermutedBank", trace, tr: dict,
                           betas: list[float], k_mass: int, variant: str,
                           c: float = PERM_C) -> dict[str, np.ndarray]:
    """Per-element tolerance of every float trace under a permutation of the
    memories (derivation: ARCHITECTURE section 13.5). The scores are the same
    numbers in another order; what changes is the order of every reduction over
    D_ff. Assumed: tree (pairwise) reductions, as torch does on CPU (cascade)
    and CUDA (tree), so a sum of n terms is off by <= g_n sum|terms| with
    g_n = ceil(log2 D_ff) u32 (first order). Quantities:
      delta  = g_n + u (1 + log D_ff): a priori bound on the shift of every
               log p_i = (z_i - max z) - log S, S = sum exp(z - max z) in
               [1, D_ff] (log_softmax subtracts the max first, so the shift does
               not grow with |lse|): g_n from the order of the sum S, u log D_ff
               from rounding log S, u from the final subtraction's carry;
      eps_i  <= u |log p_i|: once lse moves, each log p_i = beta s_i - lse is
               rounded on its own (NOT a uniform shift), bounded with
               Lambda_t = beta range_t(s) + log D_ff >= max_i |log p_i|;
      H      = entropy in nats (= sum p |log p|), A = divergence_scale(p_t, p_ref),
               A2 = sum p |log p| |d| + sum q |log q| |d| (d = log p - log q): the
               exact weight of the per-element roundings in the divergences;
      J      = sum p |log(p/m)| + sum q |log(q/m)|: sum of the absolute JS terms;
      ds     = max |s_perm - s[perm]| (0 when the matmul is permutation-exact)
               feeds the Lipschitz terms of reference_tolerances.
    Then, times c (PERM_C), plus the Lipschitz terms and a floor u32 (the
    float16 storage resolution near 0):
      energy        delta/beta + 2u|E|          (lse = max + log S, rounded at |lse|)
      norm_entropy  (delta (1+H) + u H (1+Lambda) + g_n H) / log D_ff
      mean_score    delta + u H + g_n           (sum p|s| <= 1)
      var_score     (delta + u Lambda + g_n) var   (centred form, nonnegative terms)
      topk_mass     (delta + u Lambda + g_n) mass
      js/<ref>      (delta_t + delta_r) 2 JS + u (H_t + H_r)(1 + ln2) + u A + u A2 + g_n J
                    (a uniform shift of d = log p - log q cancels at first order,
                    p sigmoid(-d) = q sigmoid(d) = pq/(p+q); a uniform rescaling
                    of p multiplies the signed sum KL(p||m) <= 2 JS)
      dkl/<ref>     (delta_t + delta_r)(2 + A) + u (2 H_t + 2 H_r + A)
                    + u A2 + g_n A        (+ 2 beta ds A)
      margin        2 ds;   xhat_top1_frac, query_gain 0 (independent of the memories).
    ``tr``: one variant's traces (``view``), ``variant`` its score variant.
    dkl is the worst-conditioned trace: its terms add up to A (hundreds of
    nats at beta ~ 1e4) while |dkl| can be small."""
    dims = bundle.dims
    L, eps, logd = dims.n_layers, dims.rms_norm_eps, math.log(dims.d_ff)
    u = UNIT_ROUNDOFF["float32"]
    gn = math.ceil(math.log2(dims.d_ff)) * u
    refs = trace.reference_positions
    T, nb = tr["energy"].shape[1], len(betas)
    b = np.asarray(betas, dtype=np.float64)[None, None, :]
    delta = gn + u * (1 + logd)
    A = {r: np.full((L, T, nb), np.nan) for r in REFERENCES}
    A2 = {r: np.full((L, T, nb), np.nan) for r in REFERENCES}
    J = {r: np.full((L, T, nb), np.nan) for r in REFERENCES}
    ds, srange, srow = np.zeros(L), np.zeros(L), np.zeros((L, T))
    for l in range(L):
        x_hat = metrics.rms_normalise(trace.residuals[l].to(bank.device), eps)
        s, sp = bank.scores(l, x_hat, variant), pbank.scores(l, x_hat, variant)
        ds[l] = (sp - s[:, pbank.perm[l]]).abs().max().item()
        srow[l] = (s.max(-1).values - s.min(-1).values).cpu().numpy()
        srange[l] = srow[l].max()
        for bi, beta in enumerate(betas):
            lp = metrics.log_distribution(s, beta)
            for r, pos in refs.items():
                if pos is not None:
                    A[r][l, :, bi] = divergence_scale(lp, lp[pos]).cpu().numpy()
                    d = (lp - lp[pos]).abs()
                    A2[r][l, :, bi] = ((lp.exp() * lp.abs() * d).sum(-1)
                                       + (lp[pos].exp() * lp[pos].abs() * d).sum(-1)).cpu().numpy()
                    sd = lp - lp[pos]
                    sp_ = torch.nn.functional.softplus
                    J[r][l, :, bi] = ((lp.exp() * (metrics.LN2 - sp_(-sd)).abs()).sum(-1)
                                      + (lp[pos].exp() * (metrics.LN2 - sp_(sd)).abs()).sum(-1)).cpu().numpy()
        del x_hat, s, sp
    lip = {}
    for name in ("energy", "norm_entropy", "mean_score", "var_score", "topk_mass", "js", "dkl"):
        key = name if name not in ("js", "dkl") else f"{name}/{REFERENCES[0]}"
        lip[name] = np.asarray([[reference_tolerances(ds[l], bt, srange[l], k_mass, dims.d_model)[key]
                                 for bt in betas] for l in range(L)])[:, None, :]
    f64 = {k: v.astype(np.float64) for k, v in tr.items() if k not in ("topk_ids", "query_gain")}
    E, H = f64["energy"], f64["norm_entropy"] * logd
    lam = b * srow[:, :, None] + logd                       # >= max_i |log p_i|
    tol = {
        "energy": c * (delta / b + 2 * u * np.abs(E)) + lip["energy"],
        "norm_entropy": c * (delta * (1 + H) + u * H * (1 + lam) + gn * H) / logd + lip["norm_entropy"],
        "mean_score": c * (delta + u * H + gn) + lip["mean_score"],
        "var_score": c * (delta + u * lam + gn) * f64["var_score"] + lip["var_score"],
        "topk_mass": c * (delta + u * lam + gn) * f64["topk_mass"] + lip["topk_mass"],
        "margin": np.broadcast_to(2 * ds[:, None], (L, T)).copy(),
        "xhat_top1_frac": np.zeros((L, T)),
        "query_gain": np.zeros((L, T)),
    }
    for r, pos in refs.items():
        if pos is None:
            tol[f"js/{r}"] = tol[f"dkl/{r}"] = np.full((L, T, nb), np.nan)
            continue
        dd = 2 * delta
        hh = H + H[:, pos:pos + 1, :]
        a, a2, l2 = A[r], A2[r], metrics.LN2
        js = f64[f"js/{r}"]
        tol[f"js/{r}"] = c * (dd * 2 * js + u * hh * (1 + l2) + u * a + u * a2 + gn * J[r]) + lip["js"]
        tol[f"dkl/{r}"] = (c * (dd * (2 + a) + u * (2 * hh + a) + u * a2 + gn * a) + lip["dkl"]
                           + 2 * b * ds[:, None, None] * a)
    return {k: v + u for k, v in tol.items()}


def trace_ratios(a: dict, b: dict, tol: dict, betas: list[float]) -> dict:
    """Per float trace: max |a - b|, max |a - b| / tol and where (layer,
    position, beta) the ratio peaks; plus the overall worst."""
    per, worst = {}, {"ratio": 0.0}
    for k, t in tol.items():
        x, y = a[k].astype(np.float64), b[k].astype(np.float64)
        err = np.abs(x - y)
        m = np.isfinite(err) & np.isfinite(t)
        if not m.any():
            continue
        q = np.where(m, err / t, -np.inf)
        idx = np.unravel_index(int(np.argmax(q)), q.shape)
        at = {"layer": int(idx[0]), "position": int(idx[1])}
        if len(idx) == 3:
            at["beta"] = float(betas[idx[2]])
        per[k] = {"max_error": float(err[m].max()), "max_ratio": float(q[idx]), "at": at,
                  "tolerance_at_max": float(t[idx])}
        if per[k]["max_ratio"] > worst["ratio"]:
            worst = {"ratio": per[k]["max_ratio"], "trace": k} | at
    return {"traces": per, "worst": worst}


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def _read_outputs(ext_dir: Path, item_id: str):
    """traces.h5, hidden.h5 and answers.parquet records of one id."""
    import h5py
    import pandas as pd

    def group(g) -> dict:
        out = {}
        g.visititems(lambda name, obj: out.__setitem__(name, obj[()])
                     if isinstance(obj, h5py.Dataset) else None)
        return out

    with h5py.File(ext_dir / "traces.h5") as tf, h5py.File(ext_dir / "hidden.h5") as hf:
        tr, hid = group(tf[item_id]), group(hf[item_id])
    ans = pd.read_parquet(ext_dir / "answers.parquet").set_index("id").loc[item_id]
    return tr, hid, ans


TF_MAX_NOISE_RATIO = 0.1     # default PASS threshold of tf_vs_cache (p99 noise / sigma)


def run_verify(cfg: RunConfig, bundle: ModelBundle, items: list[Item], n: int = 4,
               ref_layers: list[int] | None = None, ref_n: int = 2,
               name: str | None = None, tf_max_noise_ratio: float = TF_MAX_NOISE_RATIO,
               fp32_check: bool = True) -> dict:
    """``ref_n`` questions x ``ref_layers`` go through the float64 reference
    (~75 s per layer, question and variant at Qwen2.5-3B size on one CPU core);
    every other check runs on all ``n`` questions and all layers. Every check
    involving the bank runs for both score variants and must pass for both.
    ``tf_max_noise_ratio`` is the tf_vs_cache PASS threshold (analysis
    positions); ``fp32_check`` runs fp32_reference."""
    from energy_llm.core.prompting import ROLE_NAMES

    from energy_llm.pipeline.smoke import run_single_process

    started = time.time()
    reference = load_reference()
    beta_rec = load_beta_star(cfg, items)
    beta_star = {v: float(beta_rec["variants"][v]["beta_star"]) for v in VARIANTS}
    all_betas, k_ids = variant_betas(cfg, beta_rec), variant_topk_ids(cfg)
    k_mass = cfg.extraction.topk_mass_k
    L = bundle.dims.n_layers
    eps = bundle.dims.rms_norm_eps
    ref_layers = ref_layers if ref_layers is not None else sorted({0, L // 2, L - 1})
    hidden_layers = cfg.hidden_layer_indices(L)
    sel = items[:n]
    on_cpu = bundle.device.type == "cpu"
    det_tol = 0.0 if on_cpu else 1.0          # float16 ulps: bit-exact on CPU
    ids_min_jaccard = 1.0 if on_cpu else 0.98   # GPU: near-ties at the k-th score

    run_cfg, ext_dir, _, merged = run_single_process(cfg, bundle, sel, "verify", name)
    bank = build_bank(bundle.model, bundle.dims)
    hooks = ResidualHooks(bundle.trace_model)
    stops = set(int(t) for t in np.atleast_1d(bundle.generation_config.eos_token_id))
    if bundle.tokenizer.eos_token_id is not None:
        stops.add(int(bundle.tokenizer.eos_token_id))

    acc = {key: [] for key in ("reference", "gate", "gate_ctrl", "tf_tokens", "align",
                               "hidden", "files", "limits", "monotone", "self", "det_tokens")}
    ref_details = []
    perm = None
    # tf_vs_cache: x relative error per layer, its argmax location, the error
    # propagated to the traces of each variant at beta*_v (noise), the trace
    # spread (signal) and the position groups of each question
    x_rel = [[] for _ in range(L)]
    x_worst = {"value": -1.0}
    tf_noise = {v: {m: [[] for _ in range(L)] for m in PROPAGATED} for v in VARIANTS}
    tf_signal = {v: {m: [[] for _ in range(L)] for m in PROPAGATED} for v in VARIANTS}
    tf_masks = []
    saved = []                                   # residuals kept for fp32_reference
    grid = cfg.calibration.beta_grid()
    try:
        for qi, it in enumerate(sel):
            prompt = build_prompt(bundle.tokenizer, cfg.prompt.system, it.question, cfg.prompt.date_string)
            trace = forward_item(bundle, hooks, prompt)
            prod = compute_item(bundle, bank, trace, it.id, all_betas, hidden_layers, k_mass, k_ids)

            # reference (float64, naive), both variants
            if qi < ref_n:
                q_last = max(i for i, r in enumerate(trace.prompt.roles) if r == ROLE_QUESTION)
                ref = reference.reference_item(
                    bundle.model, {l: trace.residuals[l].float().cpu().numpy() for l in ref_layers},
                    trace.n_prompt, trace.gen_ids, sorted(stops), ref_layers, all_betas, q_last,
                    k_mass, max(k_ids.values()))
                pos_ok = ref["reference_positions"] == prod.reference_positions
                per_v = {}
                for v in VARIANTS:
                    cmp = compare_with_reference(bundle, bank, trace, view(prod.traces, v), ref,
                                                 ref_layers, all_betas[v], k_mass, v)
                    cmp["ok"] &= pos_ok
                    per_v[v] = cmp
                ref_details.append({"id": it.id, "reference_positions_equal": pos_ok}
                                   | {v: {k_: x for k_, x in c.items() if k_ != "ok"} for v, c in per_v.items()})
                acc["reference"].append({v: (c["max_error_over_tolerance"], c["ok"]) for v, c in per_v.items()})
                del ref

            # (a) gate reconstruction, both variants
            err, ctrl, gtol = check_gate(bundle, bank, trace)
            acc["gate"].append(err)
            acc["gate_ctrl"].append(ctrl)

            # (b) teacher forcing vs cached generation; (c) alignment
            x_cache, info = check_tf_vs_cache(bundle, trace)
            n_cmp = info["positions_compared"]
            roles_all = trace.roles
            resp = trace.response_positions
            tf_masks.append(position_groups(roles_all, trace.reference_positions, n_cmp))
            for l in range(L):
                x_tf = trace.residuals[l][:n_cmp]
                rel = rel_err(x_cache[l], x_tf)
                x_rel[l].append(rel)
                j = int(rel.argmax())
                if rel[j] > x_worst["value"]:
                    x_worst = {"value": float(rel[j]), "id": it.id, "layer": l, "position": j,
                               "role": ROLE_NAMES[roles_all[j]]}
                for v, d in propagate(bank, eps, l, x_cache[l], x_tf, beta_star).items():
                    for m, a in d.items():
                        tf_noise[v][m][l].append(a)
                for v in VARIANTS:
                    for m, ds_name in PROPAGATED.items():
                        tf_signal[v][m][l].append(prod.traces[f"{v}/{ds_name}"][l, resp, -1])
            acc["tf_tokens"].append(info["tokens_equal"])
            if fp32_check:
                saved.append({"ids": trace.prompt.input_ids + trace.gen_ids, "x_cache": x_cache,
                              "x_tf": {l: v.cpu() for l, v in trace.residuals.items()},
                              "roles": roles_all, "refs": trace.reference_positions})
            del x_cache
            full = torch.tensor([trace.prompt.input_ids + trace.gen_ids], device=bundle.device)
            with torch.inference_mode():
                tf_logits = bundle.trace_model(input_ids=full, use_cache=False,
                                         logits_to_keep=len(trace.gen_ids) + 1).logits[0, 0].float()
            n_p = trace.n_prompt
            indep_resp = [n_p + j for j, t in enumerate(trace.gen_ids) if t not in stops]
            roles = np.asarray(trace.roles)
            align = {
                "tf_argmax_at_last_prompt_is_first_token": int(tf_logits.argmax()) == trace.gen_ids[0],
                "generation_step0_argmax_is_first_token":
                    int(info["first_step_logits"].argmax()) == trace.gen_ids[0],
                "response_positions_match": trace.response_positions == indep_resp,
                "no_stop_token_in_response": all(trace.gen_ids[p - n_p] not in stops
                                                 for p in trace.response_positions),
                "role_mask_response_exact": list(np.flatnonzero(roles == ROLE_RESPONSE)) == indep_resp,
                "last_prompt_position_not_response": roles[n_p - 1] != ROLE_RESPONSE,
            }
            logit_diff = (cpu64(tf_logits) - cpu64(info["first_step_logits"])).abs().max().item()
            acc["align"].append((all(align.values()), align, logit_diff))

            # (d) hidden = the x the metrics read; files = in-memory values
            # positions derived independently: roles of the prompt, indep_resp
            q_pos = [i for i, r in enumerate(roles[:n_p]) if r == ROLE_QUESTION]
            positions = {"question_last": q_pos[-1], "prompt_last": n_p - 1,
                         "response_first": indep_resp[0] if indep_resp else None,
                         "response_last": indep_resp[-1] if indep_resp else None}
            h_err = 0.0
            for name_, pos in positions.items():
                x = (torch.stack([trace.residuals[l][pos].float().cpu() for l in hidden_layers]).half().numpy()
                     if pos is not None else np.full_like(prod.hidden["prompt_last"], np.nan))
                h_err = max(h_err, _nan_aware_max_diff(prod.hidden[name_], x)[0])
            acc["hidden"].append(h_err)
            # files hold float16(traces), topk_ids int32 and the float16 residuals
            f_tr, f_hid, f_ans = _read_outputs(ext_dir, it.id)
            floats = {k: v for k, v in prod.traces.items() if not k.endswith("topk_ids")}
            diffs = [*(_f16_ulp_diff(f_tr[k_], v) for k_, v in floats.items()),
                     *(_f16_ulp_diff(f_hid[k_], v) for k_, v in prod.hidden.items())]
            acc["files"].append((max(d for d, _ in diffs), all(s_ for _, s_ in diffs)
                                 and all(_topk_jaccard(f_tr[k_], v) >= ids_min_jaccard
                                         for k_, v in prod.traces.items() if k_.endswith("topk_ids"))
                                 and set(f_tr) == set(prod.traces) | {"role_mask"}
                                 and set(f_hid) == set(prod.hidden)
                                 and np.array_equal(f_tr["role_mask"], prod.role_mask)))
            acc["det_tokens"].append(list(f_ans["generated_token_ids"]) == trace.gen_ids)

            # invariance under permutation of the memories, per-trace tolerances
            # from float32 reduction bounds; negative control: gamma shuffled
            if qi == 0:
                pbank = PermutedBank(bank)
                pres = compute_item(bundle, pbank, trace, it.id, all_betas, hidden_layers, k_mass, k_ids)
                cres = compute_item(bundle, gamma_shuffled_bank(bank), trace, it.id, all_betas,
                                    hidden_layers, k_mass, k_ids)
                perm = {}
                for v in VARIANTS:
                    pv = view(prod.traces, v)
                    ptol = permutation_tolerances(bundle, bank, pbank, trace, pv, all_betas[v], k_mass, v)
                    jac = None
                    if "topk_ids" in pv:
                        # topk_ids are memory indices: permuted index j is original perm[j]
                        ids = pres.traces[f"{v}/topk_ids"]
                        mapped = np.stack([pbank.perm[l].cpu().numpy()[ids[l]] for l in range(L)])
                        jac = _topk_jaccard(mapped, pv["topk_ids"])
                    perm[v] = (trace_ratios(view(pres.traces, v), pv, ptol, all_betas[v]),
                               trace_ratios(view(cres.traces, v), pv, ptol, all_betas[v]), jac)
                del pres, cres

            # limits, monotonicity, self-divergence on real scores, both variants
            sel_pos = torch.tensor([n_p - 1] + trace.response_positions, device=bank.device)
            lim_worst, mono_worst, self_worst = (dict.fromkeys(VARIANTS, 0.0) for _ in range(3))
            for l in range(L):
                x_hat = metrics.rms_normalise(trace.residuals[l].to(bank.device), eps)
                for v in VARIANTS:
                    s = bank.scores(l, x_hat, v)
                    s0 = s[n_p - 1]
                    h0 = metrics.norm_entropy(metrics.log_distribution(s0, 1e-8)).item()
                    m0 = metrics.margin(s0).item()
                    if m0 > 0:
                        b_inf = 50.0 / m0                       # e^-50 ~ 2e-22 per runner-up
                        lp_inf = metrics.log_distribution(s0, b_inf)
                        h_inf = metrics.norm_entropy(lp_inf).item()
                        e_gap = abs(metrics.energy(s0, b_inf).item() + s0.max().item())
                        lim_worst[v] = max(lim_worst[v], 1 - h0, h_inf,
                                           e_gap - math.log(s0.shape[0]) / b_inf,
                                           metrics.var_score(lp_inf, s0).item(),
                                           1 - metrics.topk_mass(lp_inf, k_mass).item())
                    else:
                        lim_worst[v] = max(lim_worst[v], 1 - h0)
                    hs = torch.stack([metrics.norm_entropy(metrics.log_distribution(s[sel_pos], b))
                                      for b in grid])
                    mono_worst[v] = max(mono_worst[v], (hs[1:] - hs[:-1]).max().item())
                    lp = metrics.log_distribution(s0, beta_star[v])
                    self_worst[v] = max(self_worst[v], metrics.js_divergence(lp, lp).item(),
                                        abs(metrics.kl_divergence(lp, lp).item()))
            acc["limits"].append(lim_worst)
            acc["monotone"].append(mono_worst)
            acc["self"].append(self_worst)
            del trace, prod
    finally:
        hooks.remove()

    tokens_equal = all(acc["tf_tokens"])
    tf_g = {v: grouped_noise(tf_noise[v], tf_signal[v], tf_masks) for v in VARIANTS}
    fp32 = fp32_reference(cfg, bundle, bank, saved, beta_star, tf_signal) if fp32_check else None
    del saved

    def worst(key):             # max over questions, per variant -> {v: value}
        return {v: max(q[v] for q in acc[key]) for v in VARIANTS}

    self_tol = self_divergence_tolerance(bundle.dims.d_ff)
    ref_w = {v: max(q[v][0] for q in acc["reference"]) for v in VARIANTS}
    ref_ok = all(q[v][1] for q in acc["reference"] for v in VARIANTS)
    gate_w, gate_ctrl = worst("gate"), min(acc["gate_ctrl"])
    tf_w = {v: tf_g[v][ANALYSIS]["max_ratio_p99"] for v in VARIANTS}
    perm_ok = all(p["worst"]["ratio"] <= 1.0 and c["worst"]["ratio"] > 1.0
                  and (j is None or j >= TOPK_MIN_JACCARD) for p, c, j in perm.values())
    lim, mono, selfd = worst("limits"), worst("monotone"), worst("self")
    checks = [
        Check("reference", _status(ref_ok), max(ref_w.values()), 1.0,
              {"layers": ref_layers, "n_questions": len(acc["reference"]),
               "metric": "max error / tolerance (tolerances from "
               "Lipschitz bounds on the observed score error, see reference_tolerances)",
               "per_variant": ref_w, "per_question": ref_details}),
        Check("gate", _status(max(gate_w.values()) <= gtol and gate_ctrl > 10 * gtol),
              max(gate_w.values()), gtol,
              {"per_variant": gate_w,
               "reconstruction": {PRIMARY: "||w_i|| ||gamma * x_hat|| c_i",
                                  CONTROL: "||w_tilde_i|| sqrt(D) s_i"},
               "no_gamma_control_min_error": gate_ctrl,
               "control_must_exceed": 10 * gtol, "trace_dtype": bundle.trace_dtype}),
        Check("tf_vs_cache", _status(max(tf_w.values()) <= tf_max_noise_ratio and tokens_equal),
              max(tf_w.values()), tf_max_noise_ratio,
              {"metric": "per variant, max over layers and traces of p99(|delta|) / sigma_l at beta*_v "
                         "on the analysis positions: delta = |H_bar|, |E| and JS(p_cache, p_tf); "
                         "sigma_l = std over (response positions x questions) of norm_entropy, "
                         "energy, js/question_last at beta*_v (reported per layer next to the ratio)",
               "pass_positions": "analysis = response tokens + question_last, prompt_last, response_first "
                                 "(of the compared positions 0 .. n_prompt + T_g - 2); template tokens "
                                 "and position 0 are reported apart and do not enter the criterion",
               "tokens_equal": tokens_equal,
               "per_variant": {v: {"max_ratio_p99": tf_w[v],
                                   "max_ratio_max": tf_g[v][ANALYSIS]["max_ratio_max"],
                                   "worst_p99_at": tf_g[v][ANALYSIS]["worst_p99_at"],
                                   "groups": tf_g[v]} for v in VARIANTS},
               "x_rel_informative": {"metric": "||x_cache - x_tf|| / ||x_tf||",
                                     "threshold": tf_cache_tolerance(bundle.dtype),
                                     "max": x_worst["value"],
                                     "within_threshold": x_worst["value"] <= tf_cache_tolerance(bundle.dtype),
                                     "max_at": {k: v for k, v in x_worst.items() if k != "value"},
                                     "groups": grouped_dist(x_rel, tf_masks)}}),
        (Check("fp32_reference", "INFO", fp32["precision_rule"]["value"], RECOMMEND_FP32_RATIO,
               fp32 | {"max_error_is": "p99 noise / sigma of the TF traces vs float32 on the analysis "
                                       "positions (informative; above the tolerance: RECOMMEND "
                                       "trace_dtype=float32)"})
         if fp32 is not None else
         Check("fp32_reference", "SKIP", None, None, {"reason": "disabled (--no-fp32-reference)"})),
        Check("alignment", _status(all(ok for ok, _, _ in acc["align"])),
              max(d for _, _, d in acc["align"]), None,
              {"per_question": [a for _, a, _ in acc["align"]],
               "max_error_is": "max |TF logits[n_prompt-1] - generation step-0 logits| (informative)"}),
        Check("hidden", _status(max(acc["hidden"]) == 0.0), max(acc["hidden"]), 0.0,
              {"metric": "hidden arrays vs float16(x) the metrics read at those positions"}),
        Check("files_vs_memory", _status(max(e for e, _ in acc["files"]) <= det_tol
                                         and all(ok for _, ok in acc["files"])),
              max(e for e, _ in acc["files"]), det_tol,
              {"metric": "traces.h5 / hidden.h5 vs an in-memory recomputation, in float16 ulps; "
                         "topk_ids exact on CPU, row Jaccard >= 0.98 on GPU"}),
        Check("permutation", _status(perm_ok),
              max(p["worst"]["ratio"] for p, _, _ in perm.values()), 1.0,
              {"metric": "per variant, max over traces and elements of |change| / tolerance; per-trace "
                         "tolerances from float32 reduction bounds (permutation_tolerances)",
               "reduction_model": "tree (pairwise): ceil(log2 D_ff) * u32 per reduction over D_ff",
               "c": PERM_C,
               "per_variant": {v: {"worst": p["worst"], "per_trace": p["traces"],
                                   "negative_control": {
                                       "what": "gamma permuted over D without permuting W_g; must exceed 1",
                                       "worst": c["worst"],
                                       "per_trace_max_ratio": {k: x["max_ratio"] for k, x in c["traces"].items()}},
                                   "topk_ids_min_row_jaccard": j}
                               for v, (p, c, j) in perm.items()},
               "topk_ids_required": TOPK_MIN_JACCARD}),
        Check("limits", _status(max(lim.values()) <= F32_FLOOR), max(lim.values()), F32_FLOOR,
              {"per_variant": lim,
               "metric": "max of 1 - H_bar(beta=1e-8), H_bar(beta=50/margin), "
                         "|E + max s| - log(D_ff)/beta, var_score and 1 - topk_mass at beta=50/margin"}),
        Check("monotone", _status(max(mono.values()) <= F32_FLOOR), max(mono.values()), F32_FLOOR,
              {"per_variant": mono, "metric": "max increase of H_bar between consecutive grid betas"}),
        Check("self_divergence", _status(max(selfd.values()) <= self_tol), max(selfd.values()), self_tol,
              {"per_variant": selfd,
               "metric": "max of JS(p, p) and |KL(p || p)| (expected exactly 0: pointwise forms)",
               "tolerance_is": "4 * u32 * log(D_ff), see self_divergence_tolerance"}),
        Check("determinism", _status(all(acc["det_tokens"]) and max(e for e, _ in acc["files"]) <= det_tol),
              max(e for e, _ in acc["files"]), det_tol,
              {"tokens_equal": all(acc["det_tokens"]), "device": bundle.device.type,
               "criterion": "bit-exact on CPU, <= 1 float16 ulp on GPU"}),
        _calibration_check(cfg, bundle, bank, beta_rec),
    ]
    report = {
        "extraction_dir": str(ext_dir),
        "model": {"id": cfg.model.id, "revision": cfg.model.revision, "dtype": bundle.dtype,
                  "trace_dtype": bundle.trace_dtype, "device": str(bundle.device)},
        "beta_star": beta_star,
        "primary": PRIMARY,
        "n_questions": len(sel),
        "options": {"n": n, "reference_n": ref_n, "reference_layers": ref_layers,
                    "tf_max_noise_ratio": tf_max_noise_ratio, "fp32_reference": fp32_check},
        "elapsed_s": round(time.time() - started, 1),
        "passed": all(c.status != "FAIL" for c in checks),
        "recommend": None if fp32 is None else fp32["precision_rule"]["recommend"],
        "checks": [asdict(c) for c in checks],
    }
    write_json_atomic(ext_dir / "verify_report.json", report)
    return report


def _calibration_check(cfg: RunConfig, bundle: ModelBundle, bank: Bank, rec: dict) -> Check:
    """Per variant: H_bar is monotone in beta, so the true mean H_bar at beta*_v
    must lie between the curve values of the two grid points that bracket it."""
    path = Path(rec.get("input_path", ""))
    if not path.exists() or sha256_file(path) != rec["input_sha256"]:
        return Check("calibration", "SKIP", None, None, {"reason": f"warm-up file {path} not available"})
    grid = np.asarray(rec["beta_grid"])
    hooks = ResidualHooks(bundle.trace_model)
    vals = {v: [] for v in VARIANTS}
    try:
        for it in read_items(path):
            if it.id not in set(rec["warmup_ids"]):
                continue
            prompt = build_prompt(bundle.tokenizer, cfg.prompt.system, it.question, cfg.prompt.date_string)
            trace = forward_item(bundle, hooks, prompt)
            for v in VARIANTS:
                b = rec["variants"][v]["beta_star"]
                vals[v].append(float(norm_entropy_curve(bundle, bank, trace, [b])[v][0]))
    finally:
        hooks.remove()
    per, ok, err = {}, True, 0.0
    for v in VARIANTS:
        rv = rec["variants"][v]
        curve = np.asarray(rv["mean_norm_entropy_curve"])
        i = int(np.searchsorted(grid, rv["beta_star"])) - 1
        hi, lo = float(curve[i]), float(curve[i + 1])
        h = float(np.mean(vals[v]))
        ok &= lo - F32_FLOOR <= h <= hi + F32_FLOOR
        err = max(err, abs(h - rec["target_norm_entropy"]))
        per[v] = {"mean_norm_entropy_at_beta_star": h, "bracket": [lo, hi], "beta_star": rv["beta_star"],
                  "grid_position": rv["grid_position"]}
    return Check("calibration", _status(ok), err, None,
                 {"per_variant": per, "target": rec["target_norm_entropy"],
                  "n_warmup": len(vals[PRIMARY]),
                  "max_error_is": "max over variants of |H_bar(beta*_v) - H*| (informative; "
                                  "PASS = inside the bracket for both)"})


def format_report(r: dict) -> str:
    stars = "  ".join(f"{v}={b:.6g}" for v, b in r["beta_star"].items())
    lines = [f"verify {r['model']['id']} ({r['model']['dtype']}, traces {r['model']['trace_dtype']} "
             f"on {r['model']['device']}), "
             f"beta* {stars}, {r['n_questions']} questions, {r['elapsed_s']} s",
             f"  {'check':<17} {'status':<6} {'max error':>12} {'tolerance':>12}"]
    fmt = lambda v: "-" if v is None else f"{v:.3e}"
    for c in r["checks"]:
        lines.append(f"  {c['name']:<17} {c['status']:<6} {fmt(c['max_error']):>12} {fmt(c['tolerance']):>12}")
    for c in r["checks"]:
        d = c["detail"]
        if c["name"] == "tf_vs_cache":
            for v, pv in d["per_variant"].items():
                w = pv["worst_p99_at"] or {}
                g = pv["groups"]
                lines.append(f"  tf_vs_cache[{v}]: analysis p99/sigma {pv['max_ratio_p99']:.3e} at layer "
                             f"{w.get('layer')} {w.get('metric')} (sigma {fmt(w.get('sigma'))}); "
                             f"template {g['role/template']['max_ratio_p99']:.3e}, "
                             f"position 0 {g['position/position_0']['max_ratio_p99']:.3e} (not in PASS)")
            xi = d["x_rel_informative"]
            lines.append(f"  tf_vs_cache: x rel max {xi['max']:.3e} at {xi['max_at']}")
        if c["name"] == "fp32_reference" and c["status"] == "INFO":
            pr = d["precision_rule"]
            lines.append(f"  fp32_reference: max p99/sigma TF vs float32 (analysis) {pr['value']:.3e} "
                         f"at {pr['at']} (threshold {pr['threshold']})")
    if r.get("recommend"):
        lines.append(f"  RECOMMEND {r['recommend']}")
    lines.append(f"  options: {r.get('options')}")
    lines.append(f"  => {'ALL PASS' if r['passed'] else 'FAILURES'}   report: "
                 f"{r['extraction_dir']}/verify_report.json")
    return "\n".join(lines)
