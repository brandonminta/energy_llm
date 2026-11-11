"""beta* calibration on a warm-up JSONL disjoint from every evaluation set.

For each warm-up question the full extraction forward runs (greedy +
teacher forcing); for every beta on a log grid and for EACH score variant
(cos_gamma_query, cos_gamma_key) the normalised entropy is averaged over
layers and over the calibration positions (last prompt token + response
tokens, stop tokens excluded). Questions are weighted equally. beta*_v is
the log-linear interpolation where variant v's mean curve crosses H_bar*.
If the target is not bracketed by the grid for either variant, calibration
fails (widen the grid) — beta* is never clamped. A beta* within
EDGE_DECADES of either end of the grid is flagged (``near_edge``) and logged.

Resumable: each question's curve is its own atomic file; failures are
retried up to ``extraction.max_attempts``. ``extract`` only reads
``beta_star.json``; it never recalibrates.
"""

from __future__ import annotations

import datetime
import logging
import math
from pathlib import Path

import numpy as np

from energy_llm import manifest
from energy_llm.config import RunConfig
from energy_llm.core.bank import (
    PRIMARY, VARIANTS, build_bank, gamma_diagnostic, read_bank_fingerprint, write_bank_h5,
)
from energy_llm.core.capture import ResidualHooks, forward_item, norm_entropy_curve
from energy_llm.core.model import ModelBundle, release_cuda
from energy_llm.core.prompting import build_prompt, check_template
from energy_llm.pipeline.signals import GracefulStop
from energy_llm.storage import (
    Item, append_failure, item_key, load_json, normalise_question, read_failures,
    sha256_file, sha256_text, write_json_atomic,
)

log = logging.getLogger("energy_llm.pipeline.calibrate")


class CalibrationError(RuntimeError):
    pass


EDGE_DECADES = 0.5     # beta* closer than this to an end of the calibration grid is flagged


def grid_position(grid: list[float], beta_star: float) -> dict:
    """Where beta* sits on the calibration grid: log10 beta*, the decades to
    each end, the bracketing grid indices and whether it is near an edge."""
    lb, lo, hi = math.log10(beta_star), math.log10(grid[0]), math.log10(grid[-1])
    i = int(np.searchsorted(grid, beta_star)) - 1
    return {"log10_beta_star": lb, "decades_from_min": lb - lo, "decades_from_max": hi - lb,
            "bracket_indices": [i, i + 1],
            "near_edge": min(lb - lo, hi - lb) < EDGE_DECADES}


def format_beta_stars(rec: dict) -> str:
    """One line per variant: beta*, its position on the calibration grid and a
    WARNING when it is within EDGE_DECADES of an end."""
    grid = rec["beta_grid"]
    lines = [f"calibration grid beta in [{grid[0]:.4g}, {grid[-1]:.4g}] "
             f"(log10 {math.log10(grid[0]):.2f} .. {math.log10(grid[-1]):.2f}), "
             f"target H_bar* = {rec['target_norm_entropy']}"]
    for v, r in rec["variants"].items():
        p = r["grid_position"]
        lines.append(f"  beta*[{v}]{' (primary)' if v == rec.get('primary') else ''} = {r['beta_star']:.6g}"
                     f"   log10 {p['log10_beta_star']:.3f}: {p['decades_from_min']:.2f} decades above "
                     f"the min, {p['decades_from_max']:.2f} below the max"
                     + (f"   WARNING: within {EDGE_DECADES} decades of the grid edge" if p["near_edge"] else ""))
    return "\n".join(lines)


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
    diag = gamma_diagnostic(bank)
    write_bank_h5(path, bank, {
        "model_id": cfg.model.id, "revision": cfg.model.revision,
        "n_layers": d.n_layers, "d_model": d.d_model, "d_ff": d.d_ff,
        "rms_norm_eps": d.rms_norm_eps,
        "definition": "gamma = effective gain of the FFN norm; W_tilde = gate_proj.weight * gamma "
                      "(column-wise); key_norms = ||gate_proj.weight_i||, row_norms = ||W_tilde_i||",
        "gamma_top_k": diag["k"],
        "gamma_top_indices": np.asarray(diag["indices"], dtype=np.int64),   # [L, k]
        "gamma_top_fraction": np.asarray(diag["fraction"], dtype=np.float64),  # [L]
    })


def run_calibrate(cfg: RunConfig, bundle: ModelBundle, items: list[Item],
                  input_path: str | Path, stopper: GracefulStop | None = None) -> dict | None:
    """Returns the beta* record when calibration is complete, or None when
    work remains (interrupted, or failures still below max_attempts)."""
    stopper = stopper or GracefulStop(install=False)
    if cfg.beta_star_path.exists():
        rec = load_json(cfg.beta_star_path)
        log.info("beta* already calibrated: %s", cfg.beta_star_path)
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
    attempts, _ = read_failures(failed_path)
    max_attempts = cfg.extraction.max_attempts

    hooks = ResidualHooks(bundle.trace_model)
    try:
        for it in items:
            if stopper.requested:
                break
            path = _curve_path(cal, it.id)
            if path.exists() or attempts.get(it.id, 0) >= max_attempts:
                continue
            try:
                prompt = build_prompt(bundle.tokenizer, cfg.prompt.system, it.question,
                                      cfg.prompt.date_string)
                trace = forward_item(bundle, hooks, prompt)
                curves = norm_entropy_curve(bundle, bank, trace, grid)
                write_json_atomic(path, {
                    "id": it.id, "curves": {v: c.tolist() for v, c in curves.items()},
                    "n_positions": 1 + len(trace.response_positions),
                })
                del trace
            except Exception as exc:
                attempts[it.id] = attempts.get(it.id, 0) + 1
                append_failure(failed_path, it.id, attempts[it.id], max_attempts, exc)
                log.error("warm-up %s failed (attempt %d): %r", it.id, attempts[it.id], exc)
                release_cuda()
    finally:
        hooks.remove()

    done = {it.id: _curve_path(cal, it.id) for it in items}
    done = {k: p for k, p in done.items() if p.exists()}
    pending = [it.id for it in items
               if it.id not in done and attempts.get(it.id, 0) < max_attempts]
    if pending:
        log.info("calibration pending: %d questions left", len(pending))
        return None
    if not done:
        raise CalibrationError("every warm-up question failed")

    per_q = [load_json(p)["curves"] for p in done.values()]
    target = cfg.calibration.target_norm_entropy
    variants = {}
    for v in VARIANTS:
        mean_curve = np.stack([np.asarray(c[v]) for c in per_q]).mean(axis=0)
        try:
            b = interpolate_log_beta(grid, mean_curve, target)
        except CalibrationError as exc:
            raise CalibrationError(f"{v}: {exc}") from None
        variants[v] = {"beta_star": b, "mean_norm_entropy_curve": mean_curve.tolist(),
                       "grid_position": grid_position(grid, b)}
    by_id = {it.id: it for it in items}
    record = {
        "primary": PRIMARY,
        "variants": variants,
        "target_norm_entropy": target,
        "beta_grid": grid,
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
    for line in format_beta_stars(record).splitlines():
        (log.warning if "WARNING" in line else log.info)("%s", line)
    log.info("beta* written to %s", cfg.beta_star_path)
    return record


def _curve_path(cal_dir: Path, item_id: str) -> Path:
    """Per-question H_bar curve (atomic file, the unit of resumption)."""
    return cal_dir / "items" / f"{item_key(item_id)}.json"
