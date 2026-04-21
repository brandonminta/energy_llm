"""Feature loading for the evaluation pipeline.

Bridges M3 label artifacts and M4 trajectory artifacts into a flat dict
consumed by the AUROC, probe, and report modules.

Disk layout convention
----------------------
artifact_dir/
    {sample_id}.json          trajectory metadata (question, gold_answers, ...)
    {sample_id}.npz           prefill/generation energy arrays
    {sample_id}_scores.npz    raw K-dim dot-product scores (optional, for beta sweep)
    {sample_id}_kl.npz        per-layer KL arrays (optional, requires distributions)
    {sample_id}_baselines.json seq_logprob / entropy baselines (optional)

labels_dir/ (may equal artifact_dir)
    {sample_id}_label.json    M3 label: is_hallucination, correctness_score, ...
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Optional

from hopfield_llm.analysis.alignment import (
    content_token_mask,
    extract_answer_features,
    find_answer_span,
)
from hopfield_llm.utils.logging import get_logger

log = get_logger("evaluation.features")


# ------------------------------------------------------------------
# Label persistence
# ------------------------------------------------------------------


def save_sample_label(sample, labels_dir: Path) -> Path:
    """Persist M3 labeling fields for one DataSample to {sample.id}_label.json.

    Args:
        sample:     DataSample with labeling fields populated.
        labels_dir: Directory to write the label file into.

    Returns:
        Path of the written file.
    """
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "id": sample.id,
        "is_hallucination": sample.is_hallucination,
        "correctness_score": sample.correctness_score,
        "label_source": sample.label_source,
        "label_version": sample.label_version,
    }
    path = labels_dir / f"{sample.id}_label.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# ------------------------------------------------------------------
# Per-sample feature loading
# ------------------------------------------------------------------

_NAN = float("nan")


def load_sample_features(
    artifact_dir: Path,
    sample_id: str,
    labels_dir: Optional[Path] = None,
    tokenizer=None,
) -> Optional[dict]:
    """Load trajectory and label files for one sample into a flat feature dict.

    Samples where ``is_hallucination`` is None are skipped (returns None).

    Args:
        artifact_dir: Directory containing {sample_id}.npz / .json.
        sample_id:    Sample identifier string.
        labels_dir:   Directory containing {sample_id}_label.json. Defaults to
                      artifact_dir when None.
        tokenizer:    Optional HuggingFace tokenizer. When provided, answer-span
                      extraction is used for gen_*_answer features; otherwise
                      features are mean-pooled over all generated tokens.

    Returns:
        Flat dict of features or None.
    """
    artifact_dir = Path(artifact_dir)
    _ldir = Path(labels_dir) if labels_dir is not None else artifact_dir

    # ── Trajectory metadata ────────────────────────────────────────────────
    json_path = artifact_dir / f"{sample_id}.json"
    if not json_path.exists():
        return None
    meta = json.loads(json_path.read_text(encoding="utf-8"))

    # ── Trajectory arrays ──────────────────────────────────────────────────
    npz_path = artifact_dir / f"{sample_id}.npz"
    if not npz_path.exists():
        return None
    arrays = dict(np.load(npz_path))

    # ── Label ─────────────────────────────────────────────────────────────
    label_path = _ldir / f"{sample_id}_label.json"
    if not label_path.exists():
        return None
    label = json.loads(label_path.read_text(encoding="utf-8"))
    if label.get("is_hallucination") is None:
        return None

    # ── Text fields ────────────────────────────────────────────────────────
    generated_text = meta.get("generated_text", "")
    gold_answers   = meta.get("gold_answers", [])

    # ── Prefill [L] ────────────────────────────────────────────────────────
    prefill_energy_last: np.ndarray = arrays["prefill_energy"].astype(np.float32)
    prefill_energy_mean: np.ndarray = arrays.get(
        "prefill_energy_mean", arrays["prefill_energy"]
    ).astype(np.float32)
    L = prefill_energy_last.shape[0]

    # ── Generation [L, T] → [L] via answer-span extraction ─────────────────
    gen_energy  = arrays["gen_energy"].astype(np.float32)   # [L, T]
    gen_entropy = arrays["gen_entropy"].astype(np.float32)  # [L, T]
    gen_top_act = arrays["gen_top_act"].astype(np.float32)  # [L, T]
    gen_lse     = arrays["gen_lse"].astype(np.float32)      # [L, T]
    T = gen_energy.shape[1]

    if tokenizer is not None and generated_text:
        span = find_answer_span(generated_text, gold_answers, tokenizer)
        mask = content_token_mask(generated_text, tokenizer)
        if not mask.any():
            mask = np.ones(T, dtype=bool)
    else:
        span = None
        mask = np.ones(T, dtype=bool)

    gen_energy_answer   = extract_answer_features(gen_energy,  span, mask)
    gen_entropy_answer  = extract_answer_features(gen_entropy, span, mask)
    gen_top_act_answer  = extract_answer_features(gen_top_act, span, mask)
    gen_mean_act_answer = extract_answer_features(gen_lse,     span, mask)

    delta_energy = (gen_energy_answer - prefill_energy_last).astype(np.float32)

    # ── KL divergence [L] (NaN unless pre-computed with save_distributions) ─
    kl_path = artifact_dir / f"{sample_id}_kl.npz"
    if kl_path.exists():
        kl_arr = dict(np.load(kl_path)).get("kl_gen_to_prompt_per_layer")
        kl_gen_to_prompt_mean = (
            np.nanmean(kl_arr, axis=1).astype(np.float32)
            if kl_arr is not None
            else np.full(L, np.nan, np.float32)
        )
    else:
        kl_gen_to_prompt_mean = np.full(L, np.nan, np.float32)

    # ── Scalar energy summaries (recomputed from arrays) ───────────────────
    gen_energy_layer_mean = np.nanmean(gen_energy, axis=1)  # [L]
    de = gen_energy_layer_mean - prefill_energy_last
    de_abs = np.abs(np.where(np.isnan(de), 0.0, de))
    energy_shift_l1 = float(np.sum(de_abs))
    gen_drift_mean  = _NAN  # requires per-layer KL over time

    # ── Pre-computed baselines (NaN when absent) ───────────────────────────
    baselines_path = artifact_dir / f"{sample_id}_baselines.json"
    bl: dict = (
        json.loads(baselines_path.read_text(encoding="utf-8"))
        if baselines_path.exists()
        else {}
    )

    # ── Raw scores for beta sweep (optional) ──────────────────────────────
    scores_path = artifact_dir / f"{sample_id}_scores.npz"
    gen_scores: Optional[np.ndarray] = None
    if scores_path.exists():
        gen_scores = dict(np.load(scores_path)).get("gen_scores")

    return {
        "id":               sample_id,
        "source":           meta.get("source", ""),
        "model":            meta.get("model", ""),
        "is_hallucination": bool(label["is_hallucination"]),
        "correctness_score": float(label.get("correctness_score") or 0.0),
        # Per-layer [L]
        "prefill_energy_last":    prefill_energy_last,
        "prefill_energy_mean":    prefill_energy_mean,
        "gen_energy_answer":      gen_energy_answer.astype(np.float32),
        "gen_entropy_answer":     gen_entropy_answer.astype(np.float32),
        "gen_top_act_answer":     gen_top_act_answer.astype(np.float32),
        "gen_mean_act_answer":    gen_mean_act_answer.astype(np.float32),
        "delta_energy":           delta_energy,
        "kl_gen_to_prompt_mean":  kl_gen_to_prompt_mean,
        # Scalar divergences
        "energy_shift_l1": energy_shift_l1,
        "gen_drift_mean":  gen_drift_mean,
        # Baselines (scalar, NaN when not pre-computed)
        "seq_logprob_mean":   float(bl.get("seq_logprob_mean",   _NAN)),
        "token_entropy_mean": float(bl.get("token_entropy_mean", _NAN)),
        "token_entropy_max":  float(bl.get("token_entropy_max",  _NAN)),
        "p_true":             bl.get("p_true"),  # float | None
        # Text
        "generated_text": generated_text,
        "gold_answers":   gold_answers,
        # Dense scores for beta sweep (None when not captured with save_scores=True)
        "gen_scores": gen_scores,
    }


def load_all_features(
    artifact_dir: Path,
    labels_dir: Optional[Path] = None,
    tokenizer=None,
) -> list[dict]:
    """Load features for all samples that have both a trajectory and a label.

    Silently skips samples missing either artifact or with None labels.

    Args:
        artifact_dir: Trajectory artifact directory.
        labels_dir:   Label directory (defaults to artifact_dir).
        tokenizer:    Optional tokenizer for answer-span extraction.

    Returns:
        List of feature dicts (labeled samples only).
    """
    artifact_dir = Path(artifact_dir)
    sample_ids = sorted(
        jf.stem
        for jf in artifact_dir.glob("*.json")
        if not jf.stem.endswith(("_label", "_baselines", "_hpre"))
        and (artifact_dir / f"{jf.stem}.npz").exists()
    )
    samples: list[dict] = []
    n_skipped = 0
    for sid in sample_ids:
        feat = load_sample_features(artifact_dir, sid, labels_dir, tokenizer)
        if feat is not None:
            samples.append(feat)
        else:
            n_skipped += 1
    if n_skipped:
        log.debug("load_all_features: skipped %d samples (missing label or None)", n_skipped)
    log.info("load_all_features: loaded %d labeled samples from %s", len(samples), artifact_dir)
    return samples
