"""Smoke test: the real extract + merge path on the first N questions, in one
process (no array), followed by a summary of the checks that matter before a
full run. Outputs go to ``<output_dir>/extract/smoke-<UTC time>/`` and
include ``smoke_report.json``. Requires a calibrated beta*.
"""

from __future__ import annotations

import copy
import datetime
import json

import numpy as np

from energy_llm.config import RunConfig
from energy_llm.core.model import ModelBundle
from energy_llm.core.prompting import ROLE_RESPONSE
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
        with h5py.File(ext_dir / "traces.h5") as f:
            for item_id, n_p in zip(ans["id"], ans["n_prompt_tokens"]):
                h = f[item_id]["norm_entropy"][..., -1]                  # beta* is last
                resp = f[item_id]["role_mask"][()] == ROLE_RESPONSE
                h_prompt.append(h[:, n_p - 1])
                if resp.any():
                    h_resp.append(h[:, resp].mean(axis=1))
        summary |= {
            "tf_token_match_pct": 100.0 * (1 - n_mis / n_gen) if n_gen else None,
            "tf_exact_answers_pct": 100.0 * float((ans["tf_mismatch_count"] == 0).mean()),
            "xhat_norm_max_rel_dev": float(ans["xhat_norm_max_rel_dev"].max()),
            "n_empty_response": int(ans["empty_response"].sum()),
            "n_truncated": int(ans["truncated"].sum()),
            "mean_response_tokens": float(ans["n_response_tokens"].mean()),
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
    h = s.get("mean_norm_entropy_per_layer")
    if h:
        lines.append("  mean H_bar at beta* per layer (prompt | response):")
        for l, (a, b) in enumerate(zip(h["prompt"], h["response"])):
            lines.append(f"    layer {l:>3}  {a:.4f} | {b:.4f}")
    for r in s["failures"]:
        lines.append(f"  FAILED {r['id']}: {r['error']}")
    return "\n".join(lines)
