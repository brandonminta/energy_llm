"""Smoke test: the real extract + merge path on the first N questions, in one
process (no array), followed by a summary of the checks that matter before a
full run. Outputs go to ``<output_dir>/extract/smoke-<UTC time>/`` and
include ``smoke_report.json``. Requires a calibrated beta*.

The summary reports TF==greedy with the list of disagreements against the
margin criterion (tf_disagreements), and the report-only retrieval
diagnostics of retrieval_diagnostics: per layer and per band (thirds, and the
same bands without layer 0) xhat_top1_frac at the three references, var_score
at beta*, the gamma top-k fraction, the pilot criteria 3 and 4, the outlier
dimension d*, ablations (top-k |gamma| dims, d*) at beta* and at a re-matched
beta, beta_l per layer, and H_bar(prompt_last) against the target band and the
calibration bracket. Everything is recomputed here from freshly captured
residuals of the smoke questions: nothing of it reaches a production output.
"""

from __future__ import annotations

import copy
import datetime
import json
import math

import numpy as np
import torch

from energy_llm.config import RunConfig
from energy_llm.core import metrics
from energy_llm.core.bank import Bank, build_bank, gamma_diagnostic
from energy_llm.core.capture import REFERENCES, ResidualHooks, forward_item
from energy_llm.core.model import ModelBundle
from energy_llm.core.prompting import ROLE_RESPONSE, build_prompt
from energy_llm.pipeline.extract import run_extract, shard_path
from energy_llm.pipeline.calibrate import CalibrationError, interpolate_log_beta
from energy_llm.pipeline.merge import run_merge
from energy_llm.pipeline.signals import GracefulStop
from energy_llm.storage import Item, load_json, read_jsonl, write_json_atomic


def run_single_process(cfg: RunConfig, bundle: ModelBundle, sel: list[Item], prefix: str,
                       name: str | None = None, stopper: GracefulStop | None = None,
                       model_load_s: float | None = None):
    """The production extract + merge path on ``sel`` as one shard in this
    process. Returns (cfg used, extraction dir, extract report, merge manifest)."""
    name = name or prefix + "-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    cfg = copy.deepcopy(cfg)
    cfg.extraction.samples_per_shard = len(sel)        # one shard, one process
    ext_dir = cfg.extraction_dir(name)
    ext_dir.mkdir(parents=True, exist_ok=True)
    inp = ext_dir / "input.jsonl"
    inp.write_text("".join(json.dumps({"id": it.id, "question": it.question}) + "\n" for it in sel))
    report = run_extract(cfg, bundle, sel, inp, name, 0, stopper=stopper, model_load_s=model_load_s)
    m = run_merge(cfg, sel, inp, name, allow_incomplete=True)
    return cfg, ext_dir, report, m


def bf16_ulp(x: float) -> float:
    """One bfloat16 ulp at |x| (8 significand bits)."""
    x = abs(float(x))
    return 2.0 ** (math.floor(math.log2(x)) - 7) if x > 0 else 0.0


def tf_disagreements(ans) -> dict:
    """Every TF/greedy disagreement with its top1 - top2 logit margins, against
    delta = max(p99 over the NON-divergent generated steps of the smoke of
    max_v |logit_gen - logit_tf|, 3 bfloat16 ulps of |top1 logit| at that step).
    Criterion: every disagreement has max(margin_tf, margin_gen) < delta, i.e.
    it is a near-tie flipped by the TF/decoding numerical discrepancy."""
    nondiv = [d for diffs, pos in zip(ans["tf_logit_diff"], ans["tf_mismatch_positions"])
              for j, d in enumerate(diffs) if j not in set(int(p) for p in pos)]
    p99 = float(np.percentile(nondiv, 99)) if nondiv else 0.0
    rows = []
    for r in ans.itertuples():
        for k, j in enumerate(r.tf_mismatch_positions):
            delta = max(p99, 3 * bf16_ulp(r.tf_mismatch_top_logit[k]))
            m_tf, m_gen = float(r.tf_mismatch_margin_tf[k]), float(r.tf_mismatch_margin_gen[k])
            rows.append({"id": r.id, "step": int(j), "token_gen": int(r.generated_token_ids[j]),
                         "token_tf": int(r.tf_mismatch_tf_tokens[k]), "margin_tf": m_tf,
                         "margin_gen": m_gen, "top_logit": float(r.tf_mismatch_top_logit[k]),
                         "delta": delta, "below_delta": max(m_tf, m_gen) < delta})
    return {"delta_rule": "max(p99 of max|logit_gen - logit_tf| over non-divergent steps, "
                          "3 bf16 ulp(|top1 logit|))",
            "p99_logit_diff_non_divergent": p99,
            "max_logit_diff_step0": float(ans["tf_logit_diff_step0"].max()),
            "n": len(rows), "all_below_delta": all(r["below_delta"] for r in rows),
            "list": rows}


def zeroed_gamma_bank(bank: Bank, dims_per_layer) -> Bank:
    """The bank with gamma[l][dims] = 0 per layer and the row norms of the
    ablated W_tilde recomputed (report-only ablations: top-k |gamma| dims, d*)."""
    gammas, norms = [], []
    for w_g, gamma, dims in zip(bank.gate_weights, bank.gamma, dims_per_layer):
        g0 = gamma.clone()
        g0[list(dims)] = 0.0
        w = w_g.detach().float()
        norms.append(torch.linalg.vector_norm(w * g0, dim=1))
        del w
        gammas.append(g0)
    return Bank(gate_weights=bank.gate_weights, gamma=gammas, key_norms=bank.key_norms,
                row_norms=norms, d_model=bank.d_model)


def interp_beta(grid, curve, target) -> float | None:
    """beta where the (decreasing) H_bar curve crosses target, interpolated in
    log beta on the grid; None when the grid does not bracket it."""
    try:
        return interpolate_log_beta(list(grid), np.asarray(curve), target)
    except CalibrationError:
        return None


@torch.inference_mode()
def _ablation_metrics(bank: Bank, traces, eps: float, beta: float) -> dict:
    """Per id and layer at one beta: H_bar(prompt_last), dH = mean H_bar over the
    response - H_bar(prompt_last), and the mean over the response of
    JS(p_t, p_ref) for ref in question_last, prompt_last (NaN for an empty response)."""
    L = len(traces[0].residuals)
    out = {k: np.full((len(traces), L), np.nan) for k in
           ("h_prompt_last", "delta_norm_entropy", "js_question_last", "js_prompt_last")}
    for qi, tr in enumerate(traces):
        refs, resp, n_p = tr.reference_positions, tr.response_positions, tr.n_prompt
        for l in range(L):
            lp = metrics.log_distribution(bank.scores(l, metrics.rms_normalise(
                tr.residuals[l].to(bank.device), eps)), beta)
            h = metrics.norm_entropy(lp)
            out["h_prompt_last"][qi, l] = h[n_p - 1].item()
            if resp:
                out["delta_norm_entropy"][qi, l] = (h[resp].mean() - h[n_p - 1]).item()
                for r in ("question_last", "prompt_last"):
                    out[f"js_{r}"][qi, l] = metrics.js_divergence(lp[resp], lp[refs[r]]).mean().item()
            del lp, h
    return out


@torch.inference_mode()
def _grid_curves(bank: Bank, traces, eps: float, grid) -> tuple[np.ndarray, np.ndarray]:
    """[n_ids, L, len(grid)]: H_bar at prompt_last, and the mean H_bar over the
    calibration positions (prompt_last + response tokens)."""
    L = len(traces[0].residuals)
    hp, hc = np.zeros((len(traces), L, len(grid))), np.zeros((len(traces), L, len(grid)))
    for qi, tr in enumerate(traces):
        sel = [tr.n_prompt - 1] + tr.response_positions
        for l in range(L):
            s = bank.scores(l, metrics.rms_normalise(tr.residuals[l][sel].to(bank.device), eps))
            for gi, beta in enumerate(grid):
                h = metrics.norm_entropy(metrics.log_distribution(s, beta))
                hp[qi, l, gi], hc[qi, l, gi] = h[0].item(), h.mean().item()
    return hp, hc


def retrieval_diagnostics(cfg: RunConfig, bundle: ModelBundle, sel: list[Item], ids: list[str],
                          beta_star: float, xf: dict, h_prompt: list, var_p: list, var_r: list) -> dict:
    """Report-only diagnostics of the smoke (nothing reaches a production output).

    - d*: per layer, the most frequent argmax_j |x_hat_j| over every position of
      every smoke question, and the share of positions where it is the argmax.
    - Ablations, bank rebuilt with gamma zeroed on (a) the top-k |gamma| dims and
      (b) d*: dH = H_bar(response mean) - H_bar(prompt_last) and the response
      mean of JS to question_last / prompt_last, at beta* and at the RE-MATCHED
      beta (the beta at which the ablated bank's mean H_bar(prompt_last) over ids
      and layers returns to its intact value at beta*, interpolated on the
      calibration grid; None if not bracketed).
    - Criteria of the pilot, per layer and per band (thirds, and the same bands
      without layer 0): 3 = median xhat_top1_frac at prompt_last > 0.5;
      4 = gamma top-k fraction > 0.5 or dH changes sign under the gamma ablation.
    - beta_l per layer with H_bar_l(beta_l) = target (calibration positions,
      mean over ids), interpolated on the calibration grid; min / max.
    - Mean H_bar at prompt_last at beta* vs target +- 0.05 and vs the
      calibration bracket (beta_star.json curve at the grid points around beta*).
    Residuals are captured afresh (a second forward per question) and kept on CPU."""
    L, D, eps = bundle.dims.n_layers, bundle.dims.d_model, bundle.dims.rms_norm_eps
    rec = load_json(cfg.beta_star_path)
    target = float(rec["target_norm_entropy"])
    grid = cfg.calibration.beta_grid()
    bank = build_bank(bundle.model, bundle.dims)
    diag = gamma_diagnostic(bank)
    by_id = {it.id: it for it in sel}
    hooks = ResidualHooks(bundle.model)
    traces = []
    try:
        for item_id in ids:
            tr = forward_item(bundle, hooks, build_prompt(
                bundle.tokenizer, cfg.prompt.system, by_id[item_id].question, cfg.prompt.date_string))
            tr.residuals = {l: v.cpu() for l, v in tr.residuals.items()}
            traces.append(tr)
    finally:
        hooks.remove()

    # d*: most frequent argmax |x_hat| per layer
    dstar, dshare = [], []
    for l in range(L):
        am = np.concatenate([metrics.rms_normalise(tr.residuals[l], eps).abs().argmax(-1).numpy()
                             for tr in traces])
        counts = np.bincount(am, minlength=D)
        dstar.append(int(counts.argmax()))
        dshare.append(float(counts.max() / am.size))
    banks = {"intact": bank, "top_gamma_zeroed": zeroed_gamma_bank(bank, diag["indices"]),
             "dstar_zeroed": zeroed_gamma_bank(bank, [[d] for d in dstar])}

    at_star = {k: _ablation_metrics(b, traces, eps, beta_star) for k, b in banks.items()}
    h_target = float(np.mean(at_star["intact"]["h_prompt_last"]))
    curves = {k: _grid_curves(b, traces, eps, grid) for k, b in banks.items()}
    matched_beta = {"intact": beta_star}
    at_match = {"intact": at_star["intact"]}
    for k in ("top_gamma_zeroed", "dstar_zeroed"):
        matched_beta[k] = interp_beta(grid, curves[k][0].mean(axis=(0, 1)), h_target)
        at_match[k] = (_ablation_metrics(banks[k], traces, eps, matched_beta[k])
                       if matched_beta[k] is not None else None)

    # beta_l per layer (intact, calibration positions)
    beta_l = [interp_beta(grid, curves["intact"][1][:, l].mean(axis=0), target) for l in range(L)]
    bl = [b for b in beta_l if b is not None]

    def mean(a, layers):
        a = np.asarray(a, dtype=np.float64)[:, layers]
        return float(np.nanmean(a)) if a.size and np.isfinite(a).any() else None

    gfrac = np.asarray(diag["fraction"])
    xf_med_prompt = (np.median(np.asarray(xf["prompt_last"]), axis=0) if xf["prompt_last"]
                     else np.full(L, np.nan))

    def ablation_block(layers):
        out = {}
        for k in banks:
            out[k] = {"beta_star": {m_: mean(at_star[k][key], layers) for m_, key in
                                    (("delta_norm_entropy", "delta_norm_entropy"),
                                     ("js_question_last", "js_question_last"),
                                     ("js_prompt_last", "js_prompt_last"))},
                      "beta_matched": None if at_match[k] is None else {
                          m_: mean(at_match[k][m_], layers)
                          for m_ in ("delta_norm_entropy", "js_question_last", "js_prompt_last")}}
        return out

    def criteria(layers):
        dh_i = mean(at_star["intact"]["delta_norm_entropy"], layers)
        dh_g = mean(at_star["top_gamma_zeroed"]["delta_norm_entropy"], layers)
        med = float(np.median(np.asarray(xf["prompt_last"])[:, layers])) if xf["prompt_last"] else None
        gf = float(np.mean(gfrac[layers]))
        flip = None if dh_i is None or dh_g is None else bool(np.sign(dh_i) != np.sign(dh_g))
        return {"criterion3_xhat_prompt_last_median_gt_0.5": None if med is None else med > 0.5,
                "criterion4_gamma_fraction_gt_0.5_or_sign_flip": gf > 0.5 or bool(flip),
                "gamma_ablation_sign_flip": flip}

    def band_entry(layers):
        if len(layers) == 0:
            return None
        return {
            "layers": [int(layers[0]), int(layers[-1])],
            "xhat_top1_frac": {r: {"median": float(np.median(np.asarray(xf[r])[:, layers])) if xf[r] else None,
                                   "max": float(np.max(np.asarray(xf[r])[:, layers])) if xf[r] else None}
                               for r in REFERENCES},
            "var_score_beta_star": {"prompt": mean(var_p, layers) if var_p else None,
                                    "response": mean(var_r, layers) if var_r else None},
            "gamma_top_fraction": float(np.mean(gfrac[layers])),
            "delta_norm_entropy_response_minus_prompt_last": {
                k: mean(at_star[k]["delta_norm_entropy"], layers) for k in banks},
            "ablation": ablation_block(layers),
            "criteria": criteria(layers),
        }

    thirds = np.array_split(np.arange(L), 3)
    per_layer = []
    for l in range(L):
        e = band_entry(np.asarray([l]))
        per_layer.append({"layer": l, "dstar": dstar[l], "dstar_share": dshare[l],
                          "gamma_top_fraction": e["gamma_top_fraction"],
                          "xhat_top1_frac_prompt_last_median": float(xf_med_prompt[l]),
                          "delta_norm_entropy": e["delta_norm_entropy_response_minus_prompt_last"],
                          "criteria": e["criteria"], "beta_l": beta_l[l]})

    hp_mean = float(np.mean(np.asarray(h_prompt))) if h_prompt else None
    rgrid, rcurve = np.asarray(rec["beta_grid"]), np.asarray(rec["mean_norm_entropy_curve"])
    i = int(np.searchsorted(rgrid, rec["beta_star"])) - 1
    bracket = sorted([float(rcurve[i]), float(rcurve[i + 1])])
    return {
        "gamma_top_k": diag["k"],
        "bands": [band_entry(b) for b in thirds],
        "bands_excluding_layer0": [band_entry(b[b != 0]) for b in thirds],
        "per_layer": per_layer,
        "ablation_betas": {"target_mean_h_prompt_last_intact": h_target, "beta_matched": matched_beta,
                           "grid": "calibration grid"},
        "beta_per_layer": {"target": target, "per_layer": beta_l,
                           "min": min(bl) if bl else None, "max": max(bl) if bl else None,
                           "argmin_layer": beta_l.index(min(bl)) if bl else None,
                           "argmax_layer": beta_l.index(max(bl)) if bl else None,
                           "n_unbracketed": sum(b is None for b in beta_l)},
        "prompt_last_entropy": {
            "mean_h_prompt_last_beta_star": hp_mean,
            "target_band": [target - 0.05, target + 0.05],
            "in_target_band": None if hp_mean is None else abs(hp_mean - target) <= 0.05,
            "calibration_bracket": bracket,
            "in_calibration_bracket": None if hp_mean is None else bracket[0] <= hp_mean <= bracket[1]},
    }


def run_smoke(cfg: RunConfig, bundle: ModelBundle, items: list[Item], n: int = 20,
              stopper: GracefulStop | None = None, model_load_s: float | None = None,
              name: str | None = None) -> dict:
    import h5py
    import pandas as pd

    sel = items[:n]
    cfg, ext_dir, report, m = run_single_process(cfg, bundle, sel, "smoke", name, stopper,
                                                 model_load_s)
    name = ext_dir.name
    run = read_jsonl(shard_path(ext_dir, 0) / "runs.jsonl")[-1]
    failed = read_jsonl(shard_path(ext_dir, 0) / "failed.jsonl")

    ans = pd.read_parquet(ext_dir / "answers.parquet") if m["counts"]["completed"] else None
    summary: dict = {
        "name": name,
        "extraction_dir": str(ext_dir),
        "n_questions": len(sel),
        "n_completed": m["counts"]["completed"],
        "n_failed": len({r["id"] for r in failed}),
        "failures": [{"id": r["id"], "error": r["error"]} for r in failed],
        "model_dtype": bundle.dtype,
        "beta_star": m["beta_star"],
        "model_load_s": model_load_s,
        "sec_per_question": run.get("sec_per_item"),
        "peak_gpu_mem_gb": run.get("peak_gpu_mem_gb"),
        "stopped_by_signal": report["stopped_by_signal"],
    }
    if ans is not None:
        n_gen = int(ans["n_generated_tokens"].sum())
        n_mis = int(ans["tf_mismatch_count"].sum())
        # H_bar at beta* per layer: last prompt token, and the mean over each
        # id's response positions (empty responses skipped); then mean over ids
        h_prompt, h_resp = [], []
        xf = {r: [] for r in REFERENCES}          # per id: [L] xhat_top1_frac at the reference
        var_p, var_r = [], []                     # per id: [L] mean var_score at beta*
        with h5py.File(ext_dir / "traces.h5") as f:
            for item_id, n_p in zip(ans["id"], ans["n_prompt_tokens"]):
                g = f[item_id]
                h = g["norm_entropy"][..., -1]                            # beta* is last
                v = g["var_score"][..., -1]
                resp = g["role_mask"][()] == ROLE_RESPONSE
                h_prompt.append(h[:, n_p - 1])
                var_p.append(v[:, :n_p].mean(axis=1))
                if resp.any():
                    h_resp.append(h[:, resp].mean(axis=1))
                    var_r.append(v[:, resp].mean(axis=1))
                for r, pos in zip(REFERENCES, g.attrs["reference_positions"]):
                    if pos >= 0:
                        xf[r].append(g["xhat_top1_frac"][:, pos])

        diagnostics = retrieval_diagnostics(cfg, bundle, sel, list(ans["id"]), float(m["beta_star"]),
                                            xf, h_prompt, var_p, var_r)
        summary |= {
            "tf_token_match_pct": 100.0 * (1 - n_mis / n_gen) if n_gen else None,
            "tf_exact_answers_pct": 100.0 * float(ans["tf_exact"].mean()),
            "tf_disagreements": tf_disagreements(ans),
            "xhat_norm_max_rel_dev": float(ans["xhat_norm_max_rel_dev"].max()),
            "n_empty_response": int(ans["empty_response"].sum()),
            "n_truncated": int(ans["truncated"].sum()),
            "mean_response_tokens": float(ans["n_response_tokens"].mean()),
            **diagnostics,
            "mean_norm_entropy_per_layer": {
                k: np.round(np.mean(v, axis=0), 4).tolist() if v else []
                for k, v in (("prompt", h_prompt), ("response", h_resp))},
        }
    write_json_atomic(ext_dir / "smoke_report.json", summary)
    return summary


def format_summary(s: dict) -> str:
    def f(v, fmt):
        return "n/a" if v is None else format(v, fmt)

    peak = s.get("peak_gpu_mem_gb") or {}
    lines = [
        f"smoke {s['name']}: {s['n_completed']}/{s['n_questions']} completed, {s['n_failed']} failed",
        f"  model dtype            {s['model_dtype']}",
        f"  beta*                  {s['beta_star']:.6g}",
        f"  TF == greedy           {f(s.get('tf_token_match_pct'), '.2f')} % of tokens, "
        f"{f(s.get('tf_exact_answers_pct'), '.1f')} % of answers",
        f"  max | ||x_hat||/sqrt(D) - 1 |  {f(s.get('xhat_norm_max_rel_dev'), '.2e')}",
    ]
    d = s.get("tf_disagreements")
    if d:
        lines.append(f"  TF/greedy disagreements {d['n']}: all margins < delta: {d['all_below_delta']}   "
                     f"(p99 non-divergent |dlogit| {d['p99_logit_diff_non_divergent']:.3g}, "
                     f"max step-0 {d['max_logit_diff_step0']:.3g})")
        for r in d["list"]:
            lines.append(f"    {r['id']} step {r['step']}: gen {r['token_gen']} tf {r['token_tf']}  margin "
                         f"tf {r['margin_tf']:.4g} gen {r['margin_gen']:.4g}  delta {r['delta']:.4g}  "
                         f"{'ok' if r['below_delta'] else 'ABOVE'}")
    lines += [
        f"  peak GPU memory        {f(peak.get('allocated'), '.2f')} GB allocated, "
        f"{f(peak.get('reserved'), '.2f')} GB reserved",
        f"  seconds / question     {f(s.get('sec_per_question'), '.2f')}   (model load "
        f"{f(s.get('model_load_s'), '.0f')} s)",
        f"  empty / truncated      {s.get('n_empty_response', 'n/a')} / {s.get('n_truncated', 'n/a')}",
    ]
    pl = s.get("prompt_last_entropy")
    if pl:
        lines.append(f"  mean H_bar(prompt_last) at beta*  {f(pl['mean_h_prompt_last_beta_star'], '.4f')}   "
                     f"target band {pl['target_band'][0]:.2f}-{pl['target_band'][1]:.2f}: {pl['in_target_band']}   "
                     f"calibration bracket {pl['calibration_bracket'][0]:.4f}-{pl['calibration_bracket'][1]:.4f}: "
                     f"{pl['in_calibration_bracket']}")
    bl = s.get("beta_per_layer")
    if bl:
        lines.append(f"  beta_l (H_bar_l = {bl['target']}): min {f(bl['min'], '.4g')} (layer {bl['argmin_layer']})  "
                     f"max {f(bl['max'], '.4g')} (layer {bl['argmax_layer']})  unbracketed {bl['n_unbracketed']}")
    ab = s.get("ablation_betas")
    if ab:
        lines.append("  re-matched beta: " + "   ".join(
            f"{k} {f(v, '.4g')}" for k, v in ab["beta_matched"].items()))
    k = s.get("gamma_top_k")
    for title, key in (("", "bands"), (" (without layer 0)", "bands_excluding_layer0")):
        for b in s.get(key, []):
            if b is None:
                continue
            xf, v = b["xhat_top1_frac"], b["var_score_beta_star"]
            c = b["criteria"]
            lines += [
                f"  layers {b['layers'][0]}-{b['layers'][1]}{title}:",
                "    xhat_top1_frac median / max   " + "   ".join(
                    f"{r} {f(xf[r]['median'], '.3f')} / {f(xf[r]['max'], '.3f')}" for r in xf),
                f"    var_score at beta*           prompt {f(v['prompt'], '.3e')}   "
                f"response {f(v['response'], '.3e')}",
                f"    gamma top-{k} fraction        {b['gamma_top_fraction']:.3f}",
                f"    criteria: 3 (xhat prompt_last median > 0.5) {c['criterion3_xhat_prompt_last_median_gt_0.5']}   "
                f"4 (gamma fraction > 0.5 or sign flip) {c['criterion4_gamma_fraction_gt_0.5_or_sign_flip']}",
            ]
            for var, ab_ in b["ablation"].items():
                st, mt = ab_["beta_star"], ab_["beta_matched"]
                lines.append(
                    f"    ablation {var:<17} beta*: dH {f(st['delta_norm_entropy'], '+.4f')} "
                    f"JS q/p {f(st['js_question_last'], '.4f')}/{f(st['js_prompt_last'], '.4f')}"
                    + ("" if mt is None else
                       f"   re-matched: dH {f(mt['delta_norm_entropy'], '+.4f')} "
                       f"JS q/p {f(mt['js_question_last'], '.4f')}/{f(mt['js_prompt_last'], '.4f')}"))
    if s.get("per_layer"):
        lines.append("  per layer: d* (share) | gamma frac | xhat prompt_last median | dH intact / gamma / d* | beta_l")
        for r in s["per_layer"]:
            dh = r["delta_norm_entropy"]
            lines.append(f"    layer {r['layer']:>3}  {r['dstar']:>5} ({r['dstar_share']:.2f}) | "
                         f"{r['gamma_top_fraction']:.3f} | {r['xhat_top1_frac_prompt_last_median']:.3f} | "
                         f"{f(dh['intact'], '+.4f')} / {f(dh['top_gamma_zeroed'], '+.4f')} / "
                         f"{f(dh['dstar_zeroed'], '+.4f')} | {f(r['beta_l'], '.4g')}")
    h = s.get("mean_norm_entropy_per_layer")
    if h:
        lines.append("  mean H_bar at beta* per layer (prompt | response):")
        for l, (a, b) in enumerate(zip(h["prompt"], h["response"])):
            lines.append(f"    layer {l:>3}  {a:.4f} | {b:.4f}")
    for r in s["failures"]:
        lines.append(f"  FAILED {r['id']}: {r['error']}")
    if s.get("extraction_dir"):     # hpc/pilot_bundle.sh finds the run of a job by this line
        lines.append(f"  report: {s['extraction_dir']}/smoke_report.json")
    return "\n".join(lines)
