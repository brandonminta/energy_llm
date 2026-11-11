"""Verification of the extraction (``energy-llm verify``): not part of the
production path.

On a few questions it runs the production extract + merge, then checks:

  reference      every per-token trace vs dev/reference.py (naive float64,
                 explicit loops), at every beta and reference position
  gate           real gate pre-activation g == ||w_tilde|| * sqrt(D) * s, with a
                 no-gamma negative control that must fail by a wide margin
  tf_vs_cache    residual x captured during cached token-by-token generation
                 vs the teacher-forced pass, position by position
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

Every tolerance is explicit and justified next to its definition; the report
gives the maximum observed error and the tolerance for each check.
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
from energy_llm.core.bank import Bank, build_bank
from energy_llm.core.capture import REFERENCES, ResidualHooks, compute_item, forward_item, norm_entropy_curve
from energy_llm.core.model import ModelBundle, decoder_layers
from energy_llm.core.prompting import ROLE_QUESTION, ROLE_RESPONSE, build_prompt
from energy_llm.pipeline.extract import load_beta_star
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
      xhat_top1_frac: from x_hat only, within score_tolerance
    ``s_range`` = max_t (max s_t - min s_t). compare_with_reference adds a
    float32 reduction floor F32_FLOOR * max(1, |value|) per element (dkl
    reaches hundreds at beta ~ 1e4). topk_ids are compared as sets."""
    b, d = beta, delta_s
    tol = {"energy": d, "margin": 2 * d, "norm_entropy": 2 * b * d,
           "mean_score": 2 * b * d * s_range + d, "var_score": 2 * b * d * s_range + 2 * d * s_range,
           "topk_mass": max(k_mass, 2) * b * d, "xhat_top1_frac": score_tolerance(d_model)}
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


def compare_with_reference(bundle: ModelBundle, bank: Bank, trace, prod, ref, layers: list[int],
                           betas: list[float], k_mass: int) -> dict:
    eps = bundle.dims.rms_norm_eps
    ds = 0.0
    for l in layers:
        s = bank.scores(l, metrics.rms_normalise(trace.residuals[l], eps)).double().cpu().numpy()
        ds = max(ds, float(np.abs(s - np.asarray(ref["scores"][l])).max()))
    s_range = max(float(np.ptp(np.asarray(ref["scores"][l]), axis=-1).max()) for l in layers)
    tol_s = score_tolerance(bundle.dims.d_model)
    tols = [reference_tolerances(ds, b, s_range, k_mass, bundle.dims.d_model) for b in betas]
    # dkl: add the data-dependent term 2 beta ds A (see reference_tolerances)
    T = len(ref["scores"][layers[0]])
    dkl_tol = {r: np.zeros((len(layers), T, len(betas))) for r in REFERENCES}
    for li, l in enumerate(layers):
        sc = np.asarray(ref["scores"][l], dtype=np.float64)
        for bi, b in enumerate(betas):
            z = b * sc
            lp = z - (z.max(1, keepdims=True) + np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)))
            p = np.exp(lp)
            for r, pos in ref["reference_positions"].items():
                if pos is not None:
                    d = np.abs(lp - lp[pos])
                    a = (p * d).sum(1) + (p[pos] * d).sum(1)
                    dkl_tol[r][li, :, bi] = 2 * b * ds * a + tols[bi][f"dkl/{r}"]
    ratio, nan_ok, per = 0.0, True, {}
    for name, r in ref["traces"].items():
        p = prod.traces[name][layers]
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
    ids_ok = per["topk_ids"]["min_row_jaccard"] >= TOPK_MIN_JACCARD
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
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
                lambda m, a, k, l=l: self.x.setdefault(l, []).append(
                    (a[0] if a else k["hidden_states"])[0].detach().clone()), with_kwargs=True))
            if gate:
                self.handles.append(layer.mlp.gate_proj.register_forward_hook(
                    lambda m, a, o, l=l: self.g.__setitem__(l, o[0].detach().float())))

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.inference_mode()
def check_gate(bundle: ModelBundle, bank: Bank, trace) -> tuple[float, float, float]:
    """Max normalised error e = |g - ||w_tilde|| sqrt(D) s| / (|g| + ||w_tilde ⊙ x_hat||_2)
    over layers, positions, memories, the same for the no-gamma control, and
    the tolerance. In the model g = W_g (gamma * x_hat) involves three roundings
    in the model dtype (x_hat, gamma * x_hat, the output), each <= u relative to
    the scale |g| + ||w_tilde ⊙ x_hat||_2; tol = 4u (one u of margin), floored
    at F32_FLOOR for float32 models."""
    model = bundle.model
    cap = _Capture(model, gate=True)
    full = torch.tensor([trace.prompt.input_ids + trace.gen_ids], device=bundle.device)
    try:
        model(input_ids=full, use_cache=False, logits_to_keep=1)
    finally:
        cap.remove()
    eps, d = bundle.dims.rms_norm_eps, bundle.dims.d_model
    err = ctrl = 0.0
    for l in range(bundle.dims.n_layers):
        x_hat = metrics.rms_normalise(cap.x[l][0], eps)
        s = bank.scores(l, x_hat)
        g_rec = s * bank.row_norms[l] * math.sqrt(d)
        g = cap.g[l]
        w = bank.gate_weights[l].float()
        scale = g.abs() + torch.sqrt((x_hat * bank.gamma[l]).pow(2) @ w.pow(2).T)
        g_ctrl = x_hat @ w.T                      # gamma omitted: must NOT reconstruct g
        err = max(err, ((g - g_rec).abs() / scale).max().item())
        ctrl = max(ctrl, ((g - g_ctrl).abs() / scale).max().item())
        del w
    tol = max(4 * UNIT_ROUNDOFF[bundle.dtype], F32_FLOOR)
    return err, ctrl, tol


@torch.inference_mode()
def check_tf_vs_cache(bundle: ModelBundle, trace) -> tuple[float, dict]:
    """Regenerate with the KV cache while recording x at every call, and compare
    with the teacher-forced x at positions 0 .. n_prompt + T_g - 2 (the last
    generated token is never fed back during generation). Error per position:
    ||x_cache - x_tf|| / ||x_tf||. Also returns the first-step logits for the
    alignment check."""
    model = bundle.model
    cap = _Capture(model, gate=False)
    ids = torch.tensor([trace.prompt.input_ids], device=bundle.device)
    try:
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             generation_config=bundle.generation_config)
    finally:
        cap.remove()
    gen = out.sequences[0, ids.shape[1]:].tolist()
    worst = 0.0
    n_cmp = 0
    for l in range(bundle.dims.n_layers):
        x_cache = torch.cat(cap.x[l], dim=0).float()
        x_tf = trace.residuals[l][: x_cache.shape[0]].float()
        n_cmp = x_cache.shape[0]
        rel = (x_cache - x_tf).norm(dim=-1) / x_tf.norm(dim=-1).clamp_min(1e-30)
        worst = max(worst, rel.max().item())
    return worst, {"tokens_equal": gen == trace.gen_ids, "positions_compared": n_cmp,
                   "first_step_logits": out.logits[0][0].float()}


def tf_cache_tolerance(dtype: str) -> float:
    """KV-cached and full-sequence attention use different kernels and
    reduction orders; in float32 the difference stays ~1e-6 relative (1e-4
    allows for depth). In bfloat16 it is PROVISIONAL and EMPIRICAL (5e-2):
    rounding differences compound over layers; the decisive criterion is that
    the tokens agree. Revisit after the first cluster run (report shows the max)."""
    return 1e-4 if dtype == "float32" else 5e-2


class PermutedBank(Bank):
    """Bank whose memories are permuted (rows of W_g and the norms together),
    computed on the fly: scores come out permuted, nothing else changes."""

    def __init__(self, base: Bank, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.perm = [torch.randperm(n.shape[0], generator=g).to(n.device) for n in base.row_norms]
        super().__init__(gate_weights=base.gate_weights, gamma=base.gamma,
                         row_norms=[n[p] for n, p in zip(base.row_norms, self.perm)],
                         d_model=base.d_model)

    def scores(self, l, x_hat):
        w = self.gate_weights[l].detach()[self.perm[l]].float()
        s = (x_hat * self.gamma[l]) @ w.T
        del w
        return s / (self.row_norms[l] * math.sqrt(self.d_model))


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


def run_verify(cfg: RunConfig, bundle: ModelBundle, items: list[Item], n: int = 4,
               ref_layers: list[int] | None = None, ref_n: int = 2,
               name: str | None = None) -> dict:
    """``ref_n`` questions x ``ref_layers`` go through the float64 reference
    (~75 s per layer and question at Qwen2.5-3B size on one CPU core); every
    other check runs on all ``n`` questions and all layers."""
    from energy_llm.pipeline.smoke import run_single_process

    started = time.time()
    reference = load_reference()
    beta_rec = load_beta_star(cfg, items)
    beta_star = float(beta_rec["beta_star"])
    betas = cfg.extraction.betas(beta_star)
    k_mass, k_ids = cfg.extraction.topk_mass_k, cfg.extraction.topk_ids_k
    L = bundle.dims.n_layers
    ref_layers = ref_layers if ref_layers is not None else sorted({0, L // 2, L - 1})
    hidden_layers = cfg.hidden_layer_indices(L)
    sel = items[:n]
    on_cpu = bundle.device.type == "cpu"
    det_tol = 0.0 if on_cpu else 1.0          # float16 ulps: bit-exact on CPU
    ids_min_jaccard = 1.0 if on_cpu else 0.98   # GPU: near-ties at the k-th score

    run_cfg, ext_dir, _, merged = run_single_process(cfg, bundle, sel, "verify", name)
    bank = build_bank(bundle.model, bundle.dims)
    hooks = ResidualHooks(bundle.model)
    stops = set(int(t) for t in np.atleast_1d(bundle.generation_config.eos_token_id))
    if bundle.tokenizer.eos_token_id is not None:
        stops.add(int(bundle.tokenizer.eos_token_id))

    acc = {key: [] for key in ("reference", "gate", "gate_ctrl", "tf_cache", "align",
                               "hidden", "files", "perm", "limits", "monotone", "self", "det_tokens")}
    ref_details = []
    try:
        for qi, it in enumerate(sel):
            prompt = build_prompt(bundle.tokenizer, cfg.prompt.system, it.question, cfg.prompt.date_string)
            trace = forward_item(bundle, hooks, prompt)
            prod = compute_item(bundle, bank, trace, it.id, betas, hidden_layers, k_mass, k_ids)

            # reference (float64, naive)
            if qi < ref_n:
                q_last = max(i for i, r in enumerate(trace.prompt.roles) if r == ROLE_QUESTION)
                ref = reference.reference_item(
                    bundle.model, {l: trace.residuals[l].float().cpu().numpy() for l in ref_layers},
                    trace.n_prompt, trace.gen_ids, sorted(stops), ref_layers, betas, q_last,
                    k_mass, k_ids)
                cmp = compare_with_reference(bundle, bank, trace, prod, ref, ref_layers, betas, k_mass)
                cmp["reference_positions_equal"] = ref["reference_positions"] == prod.reference_positions
                cmp["ok"] &= cmp["reference_positions_equal"]
                ref_details.append({"id": it.id} | {k_: v for k_, v in cmp.items() if k_ != "ok"})
                acc["reference"].append((cmp["max_error_over_tolerance"], cmp["ok"]))
                del ref

            # (a) gate reconstruction
            err, ctrl, gtol = check_gate(bundle, bank, trace)
            acc["gate"].append(err)
            acc["gate_ctrl"].append(ctrl)

            # (b) teacher forcing vs cached generation; (c) alignment
            tf_err, info = check_tf_vs_cache(bundle, trace)
            acc["tf_cache"].append((tf_err, info["tokens_equal"]))
            full = torch.tensor([trace.prompt.input_ids + trace.gen_ids], device=bundle.device)
            with torch.inference_mode():
                tf_logits = bundle.model(input_ids=full, use_cache=False,
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
            logit_diff = (tf_logits - info["first_step_logits"]).abs().max().item()
            acc["align"].append((all(align.values()), align, logit_diff))

            # (d) hidden = the x the metrics read; files = in-memory values
            # positions derived independently: roles of the prompt, indep_resp
            q_pos = [i for i, r in enumerate(roles[:n_p]) if r == ROLE_QUESTION]
            positions = {"question_last": q_pos[-1], "prompt_last": n_p - 1,
                         "response_first": indep_resp[0] if indep_resp else None,
                         "response_last": indep_resp[-1] if indep_resp else None}
            h_err = 0.0
            for name, pos in positions.items():
                x = (torch.stack([trace.residuals[l][pos].float().cpu() for l in hidden_layers]).half().numpy()
                     if pos is not None else np.full_like(prod.hidden["prompt_last"], np.nan))
                h_err = max(h_err, _nan_aware_max_diff(prod.hidden[name], x)[0])
            acc["hidden"].append(h_err)
            # files hold float16(traces), topk_ids int32 and the float16 residuals
            f_tr, f_hid, f_ans = _read_outputs(ext_dir, it.id)
            floats = {k: v for k, v in prod.traces.items() if k != "topk_ids"}
            diffs = [*(_f16_ulp_diff(f_tr[k_], v) for k_, v in floats.items()),
                     *(_f16_ulp_diff(f_hid[k_], v) for k_, v in prod.hidden.items())]
            acc["files"].append((max(d for d, _ in diffs), all(s_ for _, s_ in diffs)
                                 and _topk_jaccard(f_tr["topk_ids"], prod.traces["topk_ids"]) >= ids_min_jaccard
                                 and set(f_tr) == set(prod.traces) | {"role_mask"}
                                 and set(f_hid) == set(prod.hidden)
                                 and np.array_equal(f_tr["role_mask"], prod.role_mask)))
            acc["det_tokens"].append(list(f_ans["generated_token_ids"]) == trace.gen_ids)

            # invariance under permutation of the memories
            if qi == 0:
                pbank = PermutedBank(bank)
                pres = compute_item(bundle, pbank, trace, it.id, betas, hidden_layers, k_mass, k_ids)
                # Summation-order error is relative to the magnitude (dkl reaches
                # hundreds at beta ~ 1e4): error measured as |a - b| / max(1, |b|)
                perm_err = max(_nan_aware_max_diff(pres.traces[k_] / np.maximum(1.0, np.abs(v)),
                                                   v / np.maximum(1.0, np.abs(v)))[0]
                               for k_, v in prod.traces.items() if k_ != "topk_ids")
                # topk_ids are memory indices: permuted index j is original perm[j]
                mapped = np.stack([pbank.perm[l].cpu().numpy()[pres.traces["topk_ids"][l]] for l in range(L)])
                acc["perm"].append((perm_err, _topk_jaccard(mapped, prod.traces["topk_ids"])))

            # limits, monotonicity, self-divergence on real scores
            grid = cfg.calibration.beta_grid()
            sel_pos = torch.tensor([n_p - 1] + trace.response_positions, device=bank.device)
            lim_worst, mono_worst, self_worst = 0.0, 0.0, 0.0
            for l in range(L):
                s = bank.scores(l, metrics.rms_normalise(trace.residuals[l], bundle.dims.rms_norm_eps))
                s0 = s[n_p - 1]
                h0 = metrics.norm_entropy(metrics.log_distribution(s0, 1e-8)).item()
                m0 = metrics.margin(s0).item()
                if m0 > 0:
                    b_inf = 50.0 / m0                       # e^-50 ~ 2e-22 per runner-up
                    h_inf = metrics.norm_entropy(metrics.log_distribution(s0, b_inf)).item()
                    lp_inf = metrics.log_distribution(s0, b_inf)
                    e_gap = abs(metrics.energy(s0, b_inf).item() + s0.max().item())
                    lim_worst = max(lim_worst, 1 - h0, h_inf,
                                    e_gap - math.log(s0.shape[0]) / b_inf,
                                    metrics.var_score(lp_inf, s0).item(),
                                    1 - metrics.topk_mass(lp_inf, k_mass).item())
                else:
                    lim_worst = max(lim_worst, 1 - h0)
                hs = torch.stack([metrics.norm_entropy(metrics.log_distribution(s[sel_pos], b)) for b in grid])
                mono_worst = max(mono_worst, (hs[1:] - hs[:-1]).max().item())
                lp = metrics.log_distribution(s0, beta_star)
                self_worst = max(self_worst, metrics.js_divergence(lp, lp).item(),
                                 abs(metrics.kl_divergence(lp, lp).item()))
            acc["limits"].append(lim_worst)
            acc["monotone"].append(mono_worst)
            acc["self"].append(self_worst)
            del trace, prod
    finally:
        hooks.remove()

    self_tol = self_divergence_tolerance(bundle.dims.d_ff)
    checks = [
        Check("reference", _status(all(ok for _, ok in acc["reference"])),
              max(r for r, _ in acc["reference"]), 1.0,
              {"layers": ref_layers, "n_questions": len(acc["reference"]),
               "metric": "max error / tolerance (tolerances from "
               "Lipschitz bounds on the observed score error, see reference_tolerances)",
               "per_question": ref_details}),
        Check("gate", _status(max(acc["gate"]) <= gtol and min(acc["gate_ctrl"]) > 10 * gtol),
              max(acc["gate"]), gtol,
              {"no_gamma_control_min_error": min(acc["gate_ctrl"]),
               "control_must_exceed": 10 * gtol, "model_dtype": bundle.dtype}),
        Check("tf_vs_cache", _status(max(e for e, _ in acc["tf_cache"]) <= tf_cache_tolerance(bundle.dtype)
                                     and all(t for _, t in acc["tf_cache"])),
              max(e for e, _ in acc["tf_cache"]), tf_cache_tolerance(bundle.dtype),
              {"metric": "max ||x_cache - x_tf|| / ||x_tf||", "tokens_equal": all(t for _, t in acc["tf_cache"]),
               "provisional": bundle.dtype != "float32"}),
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
        Check("permutation", _status(max(e for e, _ in acc["perm"]) <= F32_FLOOR
                                     and min(j for _, j in acc["perm"]) >= TOPK_MIN_JACCARD),
              max(e for e, _ in acc["perm"]), F32_FLOOR,
              {"metric": "max |change| / max(1, |value|) of every float trace "
                         "(float32 summation order)",
               "topk_ids_min_row_jaccard": min(j for _, j in acc["perm"]),
               "topk_ids_required": TOPK_MIN_JACCARD}),
        Check("limits", _status(max(acc["limits"]) <= F32_FLOOR), max(acc["limits"]), F32_FLOOR,
              {"metric": "max of 1 - H_bar(beta=1e-8), H_bar(beta=50/margin), "
                         "|E + max s| - log(D_ff)/beta, var_score and 1 - topk_mass at beta=50/margin"}),
        Check("monotone", _status(max(acc["monotone"]) <= F32_FLOOR), max(acc["monotone"]), F32_FLOOR,
              {"metric": "max increase of H_bar between consecutive grid betas"}),
        Check("self_divergence", _status(max(acc["self"]) <= self_tol), max(acc["self"]), self_tol,
              {"metric": "max of JS(p, p) and |KL(p || p)| (expected exactly 0: pointwise forms)",
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
                  "device": str(bundle.device)},
        "beta_star": beta_star,
        "n_questions": len(sel),
        "elapsed_s": round(time.time() - started, 1),
        "passed": all(c.status != "FAIL" for c in checks),
        "checks": [asdict(c) for c in checks],
    }
    write_json_atomic(ext_dir / "verify_report.json", report)
    return report


def _calibration_check(cfg: RunConfig, bundle: ModelBundle, bank: Bank, rec: dict) -> Check:
    """H_bar is monotone in beta, so the true mean H_bar at beta* must lie
    between the curve values of the two grid points that bracket beta*."""
    path = Path(rec.get("input_path", ""))
    if not path.exists() or sha256_file(path) != rec["input_sha256"]:
        return Check("calibration", "SKIP", None, None, {"reason": f"warm-up file {path} not available"})
    grid, curve = np.asarray(rec["beta_grid"]), np.asarray(rec["mean_norm_entropy_curve"])
    i = int(np.searchsorted(grid, rec["beta_star"])) - 1
    hi, lo = float(curve[i]), float(curve[i + 1])
    hooks = ResidualHooks(bundle.model)
    vals = []
    try:
        for it in read_items(path):
            if it.id not in set(rec["warmup_ids"]):
                continue
            prompt = build_prompt(bundle.tokenizer, cfg.prompt.system, it.question, cfg.prompt.date_string)
            trace = forward_item(bundle, hooks, prompt)
            vals.append(float(norm_entropy_curve(bundle, bank, trace, [rec["beta_star"]])[0]))
    finally:
        hooks.remove()
    h = float(np.mean(vals))
    ok = lo - F32_FLOOR <= h <= hi + F32_FLOOR
    return Check("calibration", _status(ok), abs(h - rec["target_norm_entropy"]), None,
                 {"mean_norm_entropy_at_beta_star": h, "target": rec["target_norm_entropy"],
                  "bracket": [lo, hi], "n_warmup": len(vals),
                  "max_error_is": "|H_bar(beta*) - H*| (informative; PASS = inside the bracket)"})


def format_report(r: dict) -> str:
    lines = [f"verify {r['model']['id']} ({r['model']['dtype']} on {r['model']['device']}), "
             f"beta*={r['beta_star']:.6g}, {r['n_questions']} questions, {r['elapsed_s']} s",
             f"  {'check':<17} {'status':<6} {'max error':>12} {'tolerance':>12}"]
    for c in r["checks"]:
        fmt = lambda v: "-" if v is None else f"{v:.3e}"
        lines.append(f"  {c['name']:<17} {c['status']:<6} {fmt(c['max_error']):>12} {fmt(c['tolerance']):>12}")
    lines.append(f"  => {'ALL PASS' if r['passed'] else 'FAILURES'}   report: "
                 f"{r['extraction_dir']}/verify_report.json")
    return "\n".join(lines)
