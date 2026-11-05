"""Shard status reporting and the final merge.

``merge`` reads every shard's item files in input order and writes, each
atomically, into ``<output_dir>/extract/<name>/``:

    answers.parquet    id, prompt, response, token ids, lengths, truncation, ...
    baselines.parquet  mean/min token log-probability, mean token entropy
    traces.h5          /<id>/{energy, norm_entropy, margin} [L, T], /<id>/role_mask [T]
    hidden.h5          /<id>/{prompt_last, response_last} [layers, D] float16
    bank.h5            gamma and W_tilde row norms per layer (copied from calibration)
    manifest.json      written last
"""

from __future__ import annotations

import datetime
import shutil
from pathlib import Path

import numpy as np

from energy_llm import manifest
from energy_llm.config import RunConfig
from energy_llm.core.prompting import ROLE_NAMES
from energy_llm.pipeline.extract import (
    ShardState, item_path, plan_shards, read_item, shard_path, shard_state,
)
from energy_llm.storage import Item, atomic_path, load_json, read_jsonl, sha256_file, write_json_atomic


def collect_states(cfg: RunConfig, items: list[Item], name: str,
                   input_path: str | Path) -> list[ShardState]:
    ext_dir = cfg.extraction_dir(name)
    input_sha = sha256_file(input_path)
    states = []
    for k, work in enumerate(plan_shards(items, cfg.extraction.samples_per_shard)):
        sdir = shard_path(ext_dir, k)
        ident = sdir / "shard.json"
        if ident.exists():
            meta = load_json(ident)
            if meta["input_sha256"] != input_sha or meta["config_sha256"] != cfg.sha256():
                raise RuntimeError(f"{ident} was produced from a different input or config")
        states.append(shard_state(sdir, k, [it.id for it in work], cfg.extraction.max_attempts))
    return states


def _shard_runs(ext_dir: Path, states: list[ShardState]) -> list[dict]:
    """Every process-run record (``runs.jsonl``) of every shard, in shard order."""
    return [r for st in states for r in read_jsonl(shard_path(ext_dir, st.index) / "runs.jsonl")]


def _exhausted_ids(states: list[ShardState]) -> dict[str, str]:
    """id -> last error, for ids out of attempts."""
    return {i: st.last_error.get(i, "") for st in states for i in sorted(st.exhausted)}


def status_report(cfg: RunConfig, items: list[Item], name: str, input_path: str | Path) -> dict:
    states = collect_states(cfg, items, name, input_path)
    runs = _shard_runs(cfg.extraction_dir(name), states)
    rates = [r["sec_per_item"] for r in runs if r.get("sec_per_item")]
    loads = [r["model_load_s"] for r in runs if r.get("model_load_s")]
    report = {
        "num_shards": len(states),
        "missing_shards": [st.index for st in states if not st.done],
        "shards": [{"index": st.index, "ids": len(st.ids), "completed": len(st.completed),
                    "exhausted": len(st.exhausted), "pending": len(st.pending)} for st in states],
        "exhausted_ids": _exhausted_ids(states),
    }
    if rates:
        spi = float(np.median(rates))
        load = float(np.median(loads)) if loads else 0.0
        report["throughput"] = {
            "median_sec_per_item": spi, "median_model_load_s": load,
            # a 2 h target inside the 3 h walltime
            "suggested_samples_per_shard": max(1, int((7200 - load) / spi)),
        }
    return report


def _write_parquet(df, path: Path) -> None:
    with atomic_path(path) as tmp:
        df.to_parquet(tmp, index=False, engine="pyarrow")


def run_merge(cfg: RunConfig, items: list[Item], input_path: str | Path, name: str,
              allow_incomplete: bool = False) -> dict:
    import h5py
    import pandas as pd

    states = collect_states(cfg, items, name, input_path)
    missing = [st.index for st in states if not st.done]
    if missing and not allow_incomplete:
        raise RuntimeError(f"shards not finished: {missing}; run `energy-llm status` and resubmit")

    ext_dir = cfg.extraction_dir(name)
    beta_rec = load_json(cfg.beta_star_path)
    completed = set().union(*(st.completed for st in states))
    shard_of = {i: st.index for st in states for i in st.ids}
    order = [it.id for it in items if it.id in completed]
    with h5py.File(cfg.bank_h5_path, "r") as f:
        n_layers = int(f.attrs["n_layers"])
        # absent in a bank.h5 calibrated before the diagnostic existed (v0.2 step 2)
        gamma_diag = {"k": int(f.attrs["gamma_top_k"]),
                      "indices": f.attrs["gamma_top_indices"].tolist(),
                      "fraction": f.attrs["gamma_top_fraction"].tolist()} \
            if "gamma_top_k" in f.attrs else None

    answers, baselines = [], []
    betas = None
    with atomic_path(ext_dir / "traces.h5") as t_tmp, atomic_path(ext_dir / "hidden.h5") as h_tmp:
        with h5py.File(t_tmp, "w") as tf, h5py.File(h_tmp, "w") as hf:
            tf.attrs["role_codes"] = np.asarray(sorted(ROLE_NAMES), dtype=np.int8)
            tf.attrs["role_names"] = [ROLE_NAMES[k] for k in sorted(ROLE_NAMES)]
            tf.attrs["beta_star"] = float(beta_rec["beta_star"])
            tf.attrs["layout"] = "energy/norm_entropy at beta*, margin: [L, T]; role_mask: [T]"
            hf.attrs["layers"] = np.asarray(cfg.hidden_layer_indices(n_layers))
            hf.attrs["layout"] = ("residual at the input of post_attention_layernorm; "
                                  "prompt_last = last prompt token, response_last = last "
                                  "non-stop response token: [layers, D] float16")
            for item_id in order:
                rec = read_item(item_path(shard_path(ext_dir, shard_of[item_id]), item_id))
                if betas is None:
                    betas = rec["betas"]
                elif not np.array_equal(rec["betas"], betas):
                    raise RuntimeError(f"item {item_id} was computed with betas {rec['betas'].tolist()}, "
                                       f"others with {betas.tolist()}")
                answers.append(rec["answer"])
                baselines.append(rec["baselines"])
                g = tf.create_group(item_id)
                for k in ("energy", "norm_entropy", "margin"):
                    g.create_dataset(k, data=rec[k], compression="gzip", compression_opts=4)
                g.create_dataset("role_mask", data=rec["role_mask"])
                g = hf.create_group(item_id)
                g.create_dataset("prompt_last", data=rec["hidden_prompt_last"],
                                 compression="gzip", compression_opts=4)
                g.create_dataset("response_last", data=rec["hidden_response_last"],
                                 compression="gzip", compression_opts=4)
            if betas is not None:
                tf.attrs["betas"] = np.asarray(betas, dtype=np.float64)   # last = beta*

    _write_parquet(pd.DataFrame(answers), ext_dir / "answers.parquet")
    _write_parquet(pd.DataFrame(baselines), ext_dir / "baselines.parquet")
    with atomic_path(ext_dir / "bank.h5") as tmp:
        shutil.copyfile(cfg.bank_h5_path, tmp)

    run_keys = ("shard_index", "started", "elapsed_s", "model_load_s", "sec_per_item",
                "model_dtype", "peak_gpu_mem_gb", "n_done_this_run", "n_failed_this_run",
                "stopped_by_signal", "hardware", "code")
    runs = [{k: r.get(k) for k in run_keys} for r in _shard_runs(ext_dir, states)]
    dtypes = sorted({r["model_dtype"] for r in runs if r.get("model_dtype")})
    out = {
        "name": name,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": cfg.to_dict(),
        "config_sha256": cfg.sha256(),
        "beta_star": beta_rec["beta_star"],
        "calibration": {k: beta_rec[k] for k in (
            "beta_star", "target_norm_entropy", "n_questions", "input_sha256",
            "bank_fingerprint", "code", "created")},
        "input": {"path": str(input_path), "sha256": sha256_file(input_path),
                  "n_items": len(items)},
        "counts": {"completed": len(order), "exhausted": sum(len(s.exhausted) for s in states),
                   "pending": sum(len(s.pending) for s in states)},
        "exhausted_ids": _exhausted_ids(states),
        "incomplete_shards": missing,
        "checks": {
            "xhat_norm_max_rel_dev": max((a["xhat_norm_max_rel_dev"] for a in answers), default=None),
            "tf_mismatch_ids": [a["id"] for a in answers if a["tf_mismatch_count"]],
            "n_truncated": sum(bool(a["truncated"]) for a in answers),
            "n_empty_response": sum(bool(a["empty_response"]) for a in answers),
        },
        "gamma_diagnostic": gamma_diag,
        "model_dtype": dtypes,
        "role_names": ROLE_NAMES,
        "code": manifest.code_version(),
        "libraries": manifest.library_versions(),
        "merge_host": manifest.hardware(),
        "shard_runs": runs,
    }
    write_json_atomic(ext_dir / "manifest.json", out)
    return out

