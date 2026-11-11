"""Sharded, resumable extraction.

Layout of one extraction run (``<output_dir>/extract/<name>/``)::

    shards/shard_KKKKK/
        shard.json        identity: ids, input/config hashes, beta*, bank fingerprint
        items/<key>.h5    every output of one id, written atomically
        completed.jsonl   completion index; an id is appended only AFTER its
                          item file has been fsynced and renamed into place
        failed.jsonl      one record per failed attempt (id, attempt, error)
        runs.jsonl        one record per process run (timings, hardware)

Shards are contiguous blocks of ``samples_per_shard`` ids in input order.
Each SLURM array task owns exactly one shard directory, so no file is ever
written concurrently. A failed id (exception, OOM, NaN) is never marked
completed; it is retried on the next run until ``max_attempts``.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from energy_llm import manifest
from energy_llm.config import RunConfig
from energy_llm.core.bank import CONTROL, PRIMARY, VARIANTS, build_bank, read_bank_fingerprint
from energy_llm.core.capture import REFERENCES, ItemResult, ResidualHooks, compute_item, forward_item
from energy_llm.core.model import ModelBundle, release_cuda
from energy_llm.core.prompting import build_prompt, check_template
from energy_llm.pipeline.signals import GracefulStop
from energy_llm.storage import (
    Item, append_failure, append_jsonl, atomic_path, item_key, load_json,
    normalise_question, read_failures, read_jsonl, sha256_file, sha256_text,
    write_array, write_json_atomic,
)

log = logging.getLogger("energy_llm.pipeline.extract")


def plan_shards(items: list[Item], samples_per_shard: int) -> list[list[Item]]:
    return [items[i: i + samples_per_shard] for i in range(0, len(items), samples_per_shard)]


def shard_path(ext_dir: Path, k: int) -> Path:
    return ext_dir / "shards" / f"shard_{k:05d}"


def item_path(sdir: Path, item_id: str) -> Path:
    """The one HDF5 file holding every output of ``item_id`` in shard ``sdir``."""
    return sdir / "items" / f"{item_key(item_id)}.h5"


def load_beta_star(cfg: RunConfig, items: list[Item]) -> dict:
    """Read the beta* record and refuse it if it lacks a beta* for any score
    variant, was calibrated under another context, or if any evaluation
    question is in the warm-up set."""
    path = cfg.beta_star_path
    if not path.exists():
        raise RuntimeError(f"{path} not found: run `energy-llm calibrate` first")
    rec = load_json(path)
    missing = [v for v in VARIANTS if "beta_star" not in rec.get("variants", {}).get(v, {})]
    if missing:
        raise RuntimeError(f"{path} has no beta* for {missing}: re-run `energy-llm calibrate` "
                           "in a new output_dir (one beta* per score variant)")
    if rec["context"] != json.loads(json.dumps(cfg.calibration_context())):
        raise RuntimeError(f"{path} was calibrated under a different model/prompt/generation config")
    warm_ids = set(rec["warmup_ids"])
    warm_q = set(rec["warmup_question_sha256"])
    overlap = [it.id for it in items
               if it.id in warm_ids or sha256_text(normalise_question(it.question)) in warm_q]
    if overlap:
        raise RuntimeError(f"{len(overlap)} evaluation items overlap the warm-up set, e.g. {overlap[:5]}")
    return rec


def variant_betas(cfg: RunConfig, beta_rec: dict) -> dict[str, list[float]]:
    """Per score variant: the extraction grid around its beta*, beta* last."""
    return {v: cfg.extraction.betas(beta_rec["variants"][v]["beta_star"]) for v in VARIANTS}


def variant_topk_ids(cfg: RunConfig) -> dict[str, int]:
    """topk_ids size per variant (0 = not stored)."""
    return {PRIMARY: cfg.extraction.topk_ids_k, CONTROL: cfg.extraction.control_topk_ids_k}


# ----------------------------------------------------------------------
# Per-id item files
# ----------------------------------------------------------------------

def write_item(sdir: Path, res: ItemResult) -> None:
    """One HDF5 file per id: traces/<variant>/<name> and traces/<shared name>
    (floats as float16, topk_ids int32), hidden/<position> float16, role_mask
    (``storage.write_array``: gzip 4, one chunk per layer); attrs topk_mass_k
    and reference_positions (in ``REFERENCES`` order, -1 for None); attrs
    betas and topk_ids_k on each traces/<variant> group."""
    import h5py

    with atomic_path(item_path(sdir, res.item_id)) as tmp:
        with h5py.File(tmp, "w") as f:
            f.attrs["id"] = res.item_id
            f.attrs["topk_mass_k"] = res.topk_mass_k
            f.attrs["reference_positions"] = np.asarray(
                [-1 if res.reference_positions[r] is None else res.reference_positions[r]
                 for r in REFERENCES], dtype=np.int64)
            # datasets, not attributes: HDF5 attributes are limited to 64 KB
            f.create_dataset("answer_json", data=json.dumps(res.answer, ensure_ascii=False))
            f.create_dataset("baselines_json", data=json.dumps(res.baselines))
            for name, arr in res.traces.items():
                write_array(f, f"traces/{name}", arr if arr.dtype == np.int32 else arr.astype(np.float16))
            for name, arr in res.hidden.items():
                write_array(f, f"hidden/{name}", arr)
            write_array(f, "role_mask", res.role_mask)
            for v in VARIANTS:
                f["traces"][v].attrs["betas"] = np.asarray(res.betas[v], dtype=np.float64)
                f["traces"][v].attrs["topk_ids_k"] = res.topk_ids_k[v]
    append_jsonl(sdir / "completed.jsonl",
                 {"id": res.item_id, "key": item_key(res.item_id), "time": time.time()})


def read_item(path: Path) -> dict:
    import h5py

    def group(g) -> dict:
        out = {}
        g.visititems(lambda name, obj: out.__setitem__(name, obj[()])
                     if isinstance(obj, h5py.Dataset) else None)
        return out

    with h5py.File(path, "r") as f:
        pos = f.attrs["reference_positions"]
        return {
            "id": str(f.attrs["id"]),
            "betas": {v: np.asarray(f["traces"][v].attrs["betas"]) for v in VARIANTS},
            "topk_mass_k": int(f.attrs["topk_mass_k"]),
            "topk_ids_k": {v: int(f["traces"][v].attrs["topk_ids_k"]) for v in VARIANTS},
            "reference_positions": {r: None if p < 0 else int(p) for r, p in zip(REFERENCES, pos)},
            "answer": json.loads(f["answer_json"][()].decode("utf-8")),
            "baselines": json.loads(f["baselines_json"][()].decode("utf-8")),
            "traces": group(f["traces"]),
            "hidden": group(f["hidden"]),
            "role_mask": f["role_mask"][()],
        }


@dataclass
class ShardState:
    index: int
    ids: list[str]
    completed: set[str]
    exhausted: set[str]
    attempts: dict[str, int]
    last_error: dict[str, str]

    @property
    def pending(self) -> list[str]:
        return [i for i in self.ids if i not in self.completed and i not in self.exhausted]

    @property
    def done(self) -> bool:
        return not self.pending


def shard_state(sdir: Path, index: int, ids: list[str], max_attempts: int) -> ShardState:
    completed = {r["id"] for r in read_jsonl(sdir / "completed.jsonl")
                 if item_path(sdir, r["id"]).exists()}
    attempts, last_error = read_failures(sdir / "failed.jsonl")
    exhausted = {i for i in ids if i not in completed and attempts.get(i, 0) >= max_attempts}
    return ShardState(index, ids, completed & set(ids), exhausted, attempts, last_error)


def _ensure_identity(sdir: Path, identity: dict) -> None:
    path = sdir / "shard.json"
    if path.exists():
        old = load_json(path)
        diff = sorted(k for k in identity if old.get(k) != identity[k])
        if diff:
            raise RuntimeError(f"{path} differs in {diff}: the input, config or beta* changed; "
                               "use a new extraction name")
    else:
        write_json_atomic(path, identity)


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

def run_extract(cfg: RunConfig, bundle: ModelBundle, items: list[Item], input_path: str | Path,
                name: str, shard_index: int, stopper: GracefulStop | None = None,
                model_load_s: float | None = None, stop_after: int | None = None,
                compute=compute_item) -> dict:
    """Process one shard. Returns a report whose ``state`` is ``done`` (every
    id completed or out of attempts) or ``pending``. ``stop_after`` and
    ``compute`` exist for tests (simulated interruption / failures)."""
    stopper = stopper or GracefulStop(install=False)
    started = time.time()
    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)
    ext_dir = cfg.extraction_dir(name)
    shards = plan_shards(items, cfg.extraction.samples_per_shard)
    if not 0 <= shard_index < len(shards):
        raise ValueError(f"shard index {shard_index} out of range: {len(shards)} shards")
    work = shards[shard_index]
    sdir = shard_path(ext_dir, shard_index)

    beta_rec = load_beta_star(cfg, items)
    _ensure_identity(sdir, {
        "shard_index": shard_index, "num_shards": len(shards),
        "ids": [it.id for it in work],
        "input_sha256": sha256_file(input_path),
        "config_sha256": cfg.sha256(),
        "beta_star": {v: beta_rec["variants"][v]["beta_star"] for v in VARIANTS},
        "bank_fingerprint": beta_rec["bank_fingerprint"],
    })

    max_attempts = cfg.extraction.max_attempts
    st = shard_state(sdir, shard_index, [it.id for it in work], max_attempts)
    pending = set(st.pending)
    todo = [it for it in work if it.id in pending]
    log.info("shard %d/%d: %d ids, %d completed, %d exhausted, %d to do",
             shard_index, len(shards), len(work), len(st.completed), len(st.exhausted), len(todo))

    n_done = n_failed = 0
    if todo and not stopper.requested:
        check_template(bundle.tokenizer, cfg.prompt.date_string)
        bank = build_bank(bundle.trace_model, bundle.dims)
        fingerprint = bank.fingerprint()
        if fingerprint != beta_rec["bank_fingerprint"] or \
                read_bank_fingerprint(cfg.bank_h5_path) != fingerprint:
            raise RuntimeError("model weights differ from the calibrated bank")
        layers_h = cfg.hidden_layer_indices(bundle.dims.n_layers)
        betas, k_ids = variant_betas(cfg, beta_rec), variant_topk_ids(cfg)
        hooks = ResidualHooks(bundle.trace_model)
        t_items = time.time()
        try:
            for it in todo:
                if stopper.requested or (stop_after is not None and n_done + n_failed >= stop_after):
                    break
                try:
                    prompt = build_prompt(bundle.tokenizer, cfg.prompt.system, it.question,
                                          cfg.prompt.date_string)
                    trace = forward_item(bundle, hooks, prompt)
                    res = compute(bundle, bank, trace, it.id, betas, layers_h,
                                  cfg.extraction.topk_mass_k, k_ids)
                    del trace
                    write_item(sdir, res)
                    n_done += 1
                except Exception as exc:
                    attempt = st.attempts.get(it.id, 0) + 1
                    append_failure(sdir / "failed.jsonl", it.id, attempt, max_attempts, exc)
                    n_failed += 1
                    log.error("id %s failed (attempt %d/%d): %r", it.id, attempt, max_attempts, exc)
                finally:
                    release_cuda()
        finally:
            hooks.remove()
        items_s = time.time() - t_items
    else:
        items_s = 0.0

    st = shard_state(sdir, shard_index, [it.id for it in work], max_attempts)
    report = {
        "shard_index": shard_index, "num_shards": len(shards),
        "state": "done" if st.done else "pending",
        "n_done_this_run": n_done, "n_failed_this_run": n_failed,
        "n_completed": len(st.completed), "n_exhausted": len(st.exhausted),
        "n_pending": len(st.pending),
        "stopped_by_signal": stopper.requested, "signal": stopper.signal_name,
    }
    append_jsonl(sdir / "runs.jsonl", report | {
        "started": datetime.datetime.fromtimestamp(started, datetime.timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - started, 2),
        "model_load_s": model_load_s,
        "sec_per_item": round(items_s / (n_done + n_failed), 3) if n_done + n_failed else None,
        "model_dtype": bundle.dtype,
        "trace_dtype": bundle.trace_dtype,
        "peak_gpu_mem_gb": manifest.peak_gpu_memory_gb(bundle.device),
        "hardware": manifest.hardware(),
        "code": manifest.code_version(),
    })
    log.info("shard %d: %s", shard_index, report)
    return report
