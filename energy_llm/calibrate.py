"""beta* calibration on a warm-up JSONL disjoint from every evaluation set.

For each warm-up question the full extraction forward runs (greedy +
teacher forcing); for every beta on a log grid the normalised entropy is
averaged over layers and over the calibration positions (last prompt token
+ response tokens, stop tokens excluded). Questions are weighted equally.
beta* is the log-linear interpolation where the mean curve crosses H_bar*.
If the target is not bracketed by the grid, calibration fails (widen the
grid) — beta* is never clamped.

Resumable: each question's curve is its own atomic file; failures are
retried up to ``extraction.max_attempts``. ``extract`` only reads
``beta_star.json``; it never recalibrates.
"""

from __future__ import annotations

import datetime
import logging
import math
import time
import traceback
from pathlib import Path

import numpy as np

from energy_llm import manifest
from energy_llm.bank import build_bank, read_bank_fingerprint, write_bank_h5
from energy_llm.capture import ResidualHooks, forward_item, norm_entropy_curve
from energy_llm.config import RunConfig
from energy_llm.model import ModelBundle
from energy_llm.prompting import build_prompt, check_template
from energy_llm.signals import GracefulStop
from energy_llm.storage import (
    Item, append_jsonl, item_key, load_json, normalise_question, read_jsonl,
    sha256_file, sha256_text, write_json_atomic,
)

log = logging.getLogger("energy_llm.calibrate")


class CalibrationError(RuntimeError):
    pass


def interpolate_log_beta(grid: list[float], curve: np.ndarray, target: float) -> float:
    """Linear interpolation in log(beta) between the first pair of grid
    points whose curve values bracket ``target`` (the curve decreases in beta)."""
    for i in range(len(grid) - 1):
        hi, lo = float(curve[i]), float(curve[i + 1])
        if hi >= target >= lo:
            if hi == lo:
                return float(grid[i])
            frac = (hi - target) / (hi - lo)
            return float(math.exp(math.log(grid[i]) + frac * (math.log(grid[i + 1]) - math.log(grid[i]))))
    raise CalibrationError(
        f"target H_bar*={target} not bracketed: curve spans "
        f"[{float(np.min(curve)):.4f}, {float(np.max(curve)):.4f}] on beta in "
        f"[{grid[0]:.3g}, {grid[-1]:.3g}]; widen calibration.log10_beta_min/max"
    )


def ensure_bank_record(cfg: RunConfig, bank, bundle: ModelBundle) -> None:
    """Write calibration/bank.h5 once; later runs must see the same weights."""
    path = cfg.bank_h5_path
    if path.exists():
        if read_bank_fingerprint(path) != bank.fingerprint():
            raise RuntimeError(f"{path} was built from different weights")
        return
    d = bundle.dims
    write_bank_h5(path, bank, {
        "model_id": cfg.model.id, "revision": cfg.model.revision,
        "n_layers": d.n_layers, "d_model": d.d_model, "d_ff": d.d_ff,
        "rms_norm_eps": d.rms_norm_eps,
        "definition": "W_tilde = gate_proj.weight * post_attention_layernorm.weight "
                      "(column-wise); row_norms = ||W_tilde_i||",
    })


def run_calibrate(cfg: RunConfig, bundle: ModelBundle, items: list[Item],
                  input_path: str | Path, stopper: GracefulStop | None = None) -> dict | None:
    """Returns the beta* record when calibration is complete, or None when
    work remains (interrupted, or failures still below max_attempts)."""
    stopper = stopper or GracefulStop(install=False)
    if cfg.beta_star_path.exists():
        rec = load_json(cfg.beta_star_path)
        log.info("beta* already calibrated: %.6g (%s)", rec["beta_star"], cfg.beta_star_path)
        return rec

    cal = cfg.calibration_dir
    grid = cfg.calibration.beta_grid()
    input_sha = sha256_file(input_path)
    state = {"input_sha256": input_sha, "context": cfg.calibration_context(),
             "ids": [it.id for it in items]}
    state_path = cal / "state.json"
    if state_path.exists():
        if load_json(state_path) != state:
            raise RuntimeError(
                f"{state_path} belongs to a different warm-up input or configuration; "
                "use a new output_dir")
    else:
        write_json_atomic(state_path, state)

    check_template(bundle.tokenizer, cfg.prompt.date_string)
    bank = build_bank(bundle.model, bundle.dims)
    ensure_bank_record(cfg, bank, bundle)

    failed_path = cal / "failed.jsonl"
    attempts: dict[str, int] = {}
    for rec in read_jsonl(failed_path):
        attempts[rec["id"]] = attempts.get(rec["id"], 0) + 1
    max_attempts = cfg.extraction.max_attempts

    hooks = ResidualHooks(bundle.model)
    try:
        for it in items:
            if stopper.requested:
                break
            path = cal / "items" / f"{item_key(it.id)}.json"
            if path.exists() or attempts.get(it.id, 0) >= max_attempts:
                continue
            try:
                prompt = build_prompt(bundle.tokenizer, cfg.prompt.system, it.question,
                                      cfg.prompt.date_string)
                trace = forward_item(bundle, hooks, prompt)
                curve = norm_entropy_curve(bundle, bank, trace, grid)
                write_json_atomic(path, {
                    "id": it.id, "curve": curve.tolist(),
                    "n_positions": 1 + len(trace.response_positions),
                })
                del trace
            except Exception as exc:
                attempts[it.id] = attempts.get(it.id, 0) + 1
                append_jsonl(failed_path, {
                    "id": it.id, "attempt": attempts[it.id], "error": repr(exc),
                    "traceback": traceback.format_exc(), "time": time.time(),
                })
                log.error("warm-up %s failed (attempt %d): %r", it.id, attempts[it.id], exc)
                _release_cuda()
    finally:
        hooks.remove()

    done = {it.id: cal / "items" / f"{item_key(it.id)}.json" for it in items}
    done = {k: p for k, p in done.items() if p.exists()}
    pending = [it.id for it in items
               if it.id not in done and attempts.get(it.id, 0) < max_attempts]
    if pending:
        log.info("calibration pending: %d questions left", len(pending))
        return None
    if not done:
        raise CalibrationError("every warm-up question failed")

    curves = np.stack([np.asarray(load_json(p)["curve"]) for p in done.values()])
    mean_curve = curves.mean(axis=0)
    beta_star = interpolate_log_beta(grid, mean_curve, cfg.calibration.target_norm_entropy)
    by_id = {it.id: it for it in items}
    record = {
        "beta_star": beta_star,
        "target_norm_entropy": cfg.calibration.target_norm_entropy,
        "beta_grid": grid,
        "mean_norm_entropy_curve": mean_curve.tolist(),
        "n_questions": len(done),
        "failed_ids": sorted(set(by_id) - set(done)),
        "warmup_ids": sorted(done),
        "warmup_question_sha256": sorted(
            sha256_text(normalise_question(by_id[i].question)) for i in done),
        "input_path": str(input_path),
        "input_sha256": input_sha,
        "context": cfg.calibration_context(),
        "bank_fingerprint": bank.fingerprint(),
        "code": manifest.code_version(),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    write_json_atomic(cfg.beta_star_path, record)
    log.info("beta* = %.6g written to %s", beta_star, cfg.beta_star_path)
    return record


def _release_cuda() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
