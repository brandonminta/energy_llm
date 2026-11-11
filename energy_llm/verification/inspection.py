"""``energy-llm inspect``: manual review of one question of a finished run.

Reads only the merged outputs of an extraction directory (no model). Prints
the position table (position, token id, token text when the tokenizer is in
the local cache, role, reference position / response) and, per layer, E,
H_bar, margin, var_score and topk_mass at beta*_v of one score variant
(default the primary, cos_gamma_query) and the query gain g_t from
traces.h5, at the three reference positions and averaged over the response.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from energy_llm.core.bank import PRIMARY
from energy_llm.core.capture import REFERENCES
from energy_llm.core.prompting import ROLE_NAMES, ROLE_RESPONSE
from energy_llm.storage import load_json


def _tokenizer(manifest: dict):
    try:
        from transformers import AutoTokenizer

        m = manifest["config"]["model"]
        return AutoTokenizer.from_pretrained(m["id"], revision=m["revision"], local_files_only=True)
    except Exception:
        return None


def inspect_item(run_dir: str | Path, item_id: str, tokenizer=None, variant: str = PRIMARY) -> str:
    import h5py
    import pandas as pd

    run_dir = Path(run_dir)
    manifest = load_json(run_dir / "manifest.json")
    ans = pd.read_parquet(run_dir / "answers.parquet").set_index("id")
    if item_id not in ans.index:
        raise KeyError(f"id {item_id!r} not in {run_dir}/answers.parquet")
    a = ans.loc[item_id]
    with h5py.File(run_dir / "traces.h5") as f:
        if variant not in f.attrs["variants"]:
            raise KeyError(f"variant {variant!r} not in {list(f.attrs['variants'])}")
        g = f[item_id]
        # beta*_v is the last beta; margin and query_gain are beta-free
        tables = {name: g[f"{variant}/{name}"][()] if name == "margin" else g[f"{variant}/{name}"][..., -1]
                  for name in ("energy", "norm_entropy", "margin", "var_score", "topk_mass")}
        tables["query_gain"] = g["query_gain"][()]
        roles = g["role_mask"][()]
        refs = {r: int(p) for r, p in zip(REFERENCES, g.attrs["reference_positions"]) if p >= 0}
    ids = list(a["prompt_token_ids"]) + list(a["generated_token_ids"])
    n_p = int(a["n_prompt_tokens"])
    resp = [i for i, r in enumerate(roles) if r == ROLE_RESPONSE]
    tok = tokenizer if tokenizer is not None else _tokenizer(manifest)
    stop = a["stop_token_id"]
    stop = "none" if stop is None or (isinstance(stop, float) and np.isnan(stop)) else int(stop)

    out = [f"id {item_id}   variant {variant}   beta*={manifest['beta_star'][variant]:.6g}   "
           f"prompt tokens {n_p}, "
           f"generated {int(a['n_generated_tokens'])}, response {int(a['n_response_tokens'])}, "
           f"empty_response={bool(a['empty_response'])}, truncated={bool(a['truncated'])}, "
           f"stop_token_id={stop}",
           f"response: {a['response']!r}", "",
           f"{'pos':>5} {'token_id':>9}  {'role':<12} {'position':<15} token"]
    ref_at = {p: r for r, p in refs.items()}
    for i, t in enumerate(ids):
        where = ref_at.get(i, "response" if i in resp else "")
        text = repr(tok.decode([int(t)])) if tok is not None else ""
        out.append(f"{i:>5} {int(t):>9}  {ROLE_NAMES[int(roles[i])]:<12} {where:<15} {text}")

    nan = float("nan")
    cols = list(REFERENCES) + ["response_mean"]
    for name, tr in tables.items():
        out += ["", f"{name} per layer" + ("" if name in ("margin", "query_gain") else " at beta*"),
                f"{'layer':>5}  " + "  ".join(f"{c:>14}" for c in cols)]
        for l in range(tr.shape[0]):
            vals = [tr[l, refs[r]] if r in refs else nan for r in REFERENCES]
            vals.append(tr[l, resp].mean() if resp else nan)
            out.append(f"{l:>5}  " + "  ".join(f"{v:>14.6g}" for v in vals))
    return "\n".join(out)
