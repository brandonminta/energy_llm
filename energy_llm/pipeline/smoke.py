"""Smoke test: the real extract + merge path on the first N questions, in one
process (no array), followed by a summary of the checks that matter before a
full run. Outputs go to ``<output_dir>/extract/smoke-<UTC time>/`` and
include ``smoke_report.json``. Requires a calibrated beta*.

Per layer band (early / middle / late thirds) the summary reports
xhat_top1_frac at the three references, var_score at beta* on prompt vs
response tokens, the gamma top-k fraction, and an ablation line: the mean
H_bar(response) - H_bar(prompt_last) at beta* with the top-k |gamma|
dimensions zeroed in the bank vs intact, recomputed here from freshly
captured residuals of the smoke questions (report only: nothing of it
reaches a production output).
"""

from __future__ import annotations

import copy
import datetime
import json

import numpy as np
import torch

from energy_llm.config import RunConfig
from energy_llm.core import metrics
from energy_llm.core.bank import Bank, build_bank, gamma_diagnostic
from energy_llm.core.capture import REFERENCES, ResidualHooks, forward_item
from energy_llm.core.model import ModelBundle
from energy_llm.core.prompting import ROLE_RESPONSE, build_prompt
from energy_llm.pipeline.extract import run_extract, shard_path
from energy_llm.pipeline.merge import run_merge
from energy_llm.pipeline.signals import GracefulStop
from energy_llm.storage import Item, read_jsonl, write_json_atomic


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

        # ablation: H_bar(response mean) - H_bar(prompt_last) at beta*, bank intact vs
        # top-k |gamma| dims zeroed (row norms of the ablated W_tilde recomputed)
        beta_star = float(m["beta_star"])
        bank = build_bank(bundle.model, bundle.dims)
        diag = gamma_diagnostic(bank)
        gammas, norms = [], []
        for w_g, gamma, top in zip(bank.gate_weights, bank.gamma, diag["indices"]):
            g0 = gamma.clone()
            g0[top] = 0.0
            w = w_g.detach().float()
            norms.append(torch.linalg.vector_norm(w * g0, dim=1))
            del w
            gammas.append(g0)
        ablated = Bank(gate_weights=bank.gate_weights, gamma=gammas, row_norms=norms,
                       d_model=bank.d_model)
        dh = {"intact": [], "top_gamma_zeroed": []}       # per id: [L]
        by_id = {it.id: it for it in sel}
        hooks = ResidualHooks(bundle.model)
        try:
            for item_id in ans["id"]:
                it = by_id[item_id]
                tr = forward_item(bundle, hooks, build_prompt(
                    bundle.tokenizer, cfg.prompt.system, it.question, cfg.prompt.date_string))
                if tr.empty_response:
                    continue
                rows = {k: [] for k in dh}
                with torch.inference_mode():
                    for l in range(bundle.dims.n_layers):
                        x_hat = metrics.rms_normalise(tr.residuals[l].to(bank.device),
                                                      bundle.dims.rms_norm_eps)
                        for k, b in (("intact", bank), ("top_gamma_zeroed", ablated)):
                            h = metrics.norm_entropy(metrics.log_distribution(b.scores(l, x_hat), beta_star))
                            rows[k].append((h[tr.response_positions].mean() - h[tr.n_prompt - 1]).item())
                for k in dh:
                    dh[k].append(rows[k])
                del tr
        finally:
            hooks.remove()

        def band(per_id, layers, stat=np.mean):
            return float(stat(np.asarray(per_id)[:, layers])) if per_id else None

        bands = []
        for layers in np.array_split(np.arange(bundle.dims.n_layers), 3):
            bands.append({
                "layers": [int(layers[0]), int(layers[-1])],
                "xhat_top1_frac": {r: {"median": band(xf[r], layers, np.median),
                                       "max": band(xf[r], layers, np.max)} for r in REFERENCES},
                "var_score_beta_star": {"prompt": band(var_p, layers), "response": band(var_r, layers)},
                "gamma_top_fraction": float(np.mean(np.asarray(diag["fraction"])[layers])),
                "delta_norm_entropy_response_minus_prompt_last": {
                    k: band(v, layers) for k, v in dh.items()},
            })
        summary |= {
            "tf_token_match_pct": 100.0 * (1 - n_mis / n_gen) if n_gen else None,
            "tf_exact_answers_pct": 100.0 * float((ans["tf_mismatch_count"] == 0).mean()),
            "xhat_norm_max_rel_dev": float(ans["xhat_norm_max_rel_dev"].max()),
            "n_empty_response": int(ans["empty_response"].sum()),
            "n_truncated": int(ans["truncated"].sum()),
            "mean_response_tokens": float(ans["n_response_tokens"].mean()),
            "gamma_top_k": diag["k"],
            "bands": bands,
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
        f"  peak GPU memory        {f(peak.get('allocated'), '.2f')} GB allocated, "
        f"{f(peak.get('reserved'), '.2f')} GB reserved",
        f"  seconds / question     {f(s.get('sec_per_question'), '.2f')}   (model load "
        f"{f(s.get('model_load_s'), '.0f')} s)",
        f"  empty / truncated      {s.get('n_empty_response', 'n/a')} / {s.get('n_truncated', 'n/a')}",
    ]
    for b in s.get("bands", []):
        xf, v = b["xhat_top1_frac"], b["var_score_beta_star"]
        dh = b["delta_norm_entropy_response_minus_prompt_last"]
        lines += [
            f"  layers {b['layers'][0]}-{b['layers'][1]}:",
            "    xhat_top1_frac median / max   " + "   ".join(
                f"{r} {f(xf[r]['median'], '.3f')} / {f(xf[r]['max'], '.3f')}" for r in xf),
            f"    var_score at beta*           prompt {f(v['prompt'], '.3e')}   "
            f"response {f(v['response'], '.3e')}",
            f"    gamma top-{s['gamma_top_k']} fraction        {b['gamma_top_fraction']:.3f}",
            f"    ablation: mean H_bar(resp) - H_bar(prompt_last) at beta*   intact "
            f"{f(dh['intact'], '+.4f')}   top-{s['gamma_top_k']} gamma zeroed "
            f"{f(dh['top_gamma_zeroed'], '+.4f')}",
        ]
    h = s.get("mean_norm_entropy_per_layer")
    if h:
        lines.append("  mean H_bar at beta* per layer (prompt | response):")
        for l, (a, b) in enumerate(zip(h["prompt"], h["response"])):
            lines.append(f"    layer {l:>3}  {a:.4f} | {b:.4f}")
    for r in s["failures"]:
        lines.append(f"  FAILED {r['id']}: {r['error']}")
    return "\n".join(lines)
