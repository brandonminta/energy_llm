"""``energy-llm inspect``: manual review of one question of a finished run.

Reads only the merged outputs of an extraction directory (no model). Prints
the position table (position, token id, token text when the tokenizer is in
the local cache, role, phase) and, per layer, E, H_bar and margin of each
phase at beta*, recomputed from traces.h5 and cross-checked against
features.parquet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from energy_llm.prompting import ROLE_NAMES, ROLE_RESPONSE
from energy_llm.storage import load_json


def _tokenizer(manifest: dict):
    try:
        from transformers import AutoTokenizer

        m = manifest["config"]["model"]
        return AutoTokenizer.from_pretrained(m["id"], revision=m["revision"], local_files_only=True)
    except Exception:
        return None


def inspect_item(run_dir: str | Path, item_id: str, tokenizer=None) -> str:
    import h5py
    import pandas as pd

    run_dir = Path(run_dir)
    manifest = load_json(run_dir / "manifest.json")
    ans = pd.read_parquet(run_dir / "answers.parquet").set_index("id")
    if item_id not in ans.index:
        raise KeyError(f"id {item_id!r} not in {run_dir}/answers.parquet")
    a = ans.loc[item_id]
    with h5py.File(run_dir / "traces.h5") as f:
        g = f[item_id]
        energy, entropy, margin = g["energy"][()], g["norm_entropy"][()], g["margin"][()]
        roles = g["role_mask"][()]
    ids = list(a["prompt_token_ids"]) + list(a["generated_token_ids"])
    n_p = int(a["n_prompt_tokens"])
    resp = [i for i, r in enumerate(roles) if r == ROLE_RESPONSE]
    tok = tokenizer if tokenizer is not None else _tokenizer(manifest)
    stop = a["stop_token_id"]
    stop = "none" if stop is None or (isinstance(stop, float) and np.isnan(stop)) else int(stop)

    out = [f"id {item_id}   beta*={manifest['beta_star']:.6g}   prompt tokens {n_p}, "
           f"generated {int(a['n_generated_tokens'])}, response {int(a['n_response_tokens'])}, "
           f"empty_response={bool(a['empty_response'])}, truncated={bool(a['truncated'])}, "
           f"stop_token_id={stop}",
           f"response: {a['response']!r}", "",
           f"{'pos':>5} {'token_id':>9}  {'role':<12} {'phase':<9} token"]
    for i, t in enumerate(ids):
        phase = "prompt" if i == n_p - 1 else ("response" if i in resp else "")
        text = repr(tok.decode([int(t)])) if tok is not None else ""
        out.append(f"{i:>5} {int(t):>9}  {ROLE_NAMES[int(roles[i])]:<12} {phase:<9} {text}")

    feats = pd.read_parquet(run_dir / "features.parquet")
    f = feats[(feats["id"] == item_id) & (feats["beta_multiplier"] == 1.0)].set_index(["layer", "variant"])
    out += ["", "per layer at beta* (prompt | response mean | response first);  "
                "'*' = traces.h5 and features.parquet disagree",
            f"{'layer':>5}  {'E':^32}  {'H_bar':^26}  {'margin':^26}"]
    nan = float("nan")
    worst = 0.0
    for l in range(energy.shape[0]):
        row = [f"{l:>5}"]
        for name, tr, col in (("E", energy, "energy"), ("H", entropy, "norm_entropy"),
                              ("m", margin, "margin")):
            p0 = tr[l, n_p - 1]
            rm = tr[l, resp].mean() if resp else nan
            rf = tr[l, resp[0]] if resp else nan
            fm = f.loc[(l, "mean"), f"{col}_resp"]
            fp = f.loc[(l, "mean"), f"{col}_prompt"]
            ff = f.loc[(l, "first"), f"{col}_resp"]
            d = np.nanmax(np.abs(np.array([p0 - fp, rm - fm, rf - ff], dtype=float)), initial=0.0)
            worst = max(worst, d)
            flag = "*" if d > 1e-4 * max(1.0, abs(p0)) else " "
            row.append(f"{p0:>9.5f} | {rm:>9.5f} | {rf:>9.5f}{flag}")
        out.append("  ".join(row))
    out.append(f"max |traces - features| = {worst:.2e}")
    return "\n".join(out)
