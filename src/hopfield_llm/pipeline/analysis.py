"""Analysis stage: aggregate trajectory artifacts into an analysis JSON.

CPU-only — no model or GPU required.  Reads the per-sample .npz and .json
files produced by ``run_trajectory`` and computes:

  1. Per-layer delta metrics (generation_mean[l] − prefill[l])
  2. Interpretable scalar summaries per sample (energy_shift_l1, etc.)
  3. Scalar hallucination scores per sample
  4. Dataset-level aggregation (mean/std per layer, score summary)

JS/Hellinger divergences are computed inline during trajectory collection and
stored in the .npz.  The legacy distribution-based KL/JS/Hellinger per-token
fields have been removed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hopfield_llm.analysis.alignment import (
    content_token_mask,
    find_answer_span,
)
from hopfield_llm.metrics.divergences import (
    SampleDivergenceResult,
    compute_sample_divergences,
)
from hopfield_llm.metrics.scoring import hallucination_score
from hopfield_llm.utils.logging import get_logger

log = get_logger("pipeline.analysis")


# Generation arrays masked along the T axis to NaN outside the answer span.
_GEN_ARRAY_KEYS = (
    "gen_energy", "gen_entropy", "gen_norm_entropy",
    "gen_top_act", "gen_lse",
    "gen_js", "gen_hellinger",
)


def _answer_token_mask(
    meta: dict[str, Any],
    T_gen: int,
    tokenizer,
) -> np.ndarray | None:
    """Build a boolean [T_gen] mask selecting the answer-span tokens.

    Falls back to the content-token mask when no gold answer matches.
    Returns None when generated_text is empty (caller should keep the full mean).
    """
    text = meta.get("generated_text") or ""
    if not text or tokenizer is None:
        return None
    gold = meta.get("gold_answers") or []
    span = find_answer_span(text, gold, tokenizer)
    if span is not None:
        s, e = span
        if 0 <= s < e <= T_gen:
            mask = np.zeros(T_gen, dtype=bool)
            mask[s:e] = True
            return mask
    cmask = content_token_mask(text, tokenizer)
    if cmask.shape[0] >= T_gen:
        return cmask[:T_gen]
    out = np.zeros(T_gen, dtype=bool)
    out[: cmask.shape[0]] = cmask
    return out


def _apply_answer_mask(
    arrays: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    """Return a shallow copy of arrays with non-answer columns NaNed in gen_*."""
    masked = dict(arrays)
    keep = mask.astype(bool)
    if not keep.any():
        return masked
    for k in _GEN_ARRAY_KEYS:
        a = arrays.get(k)
        if a is None or a.ndim != 2:
            continue
        T = a.shape[1]
        m = keep[:T] if keep.shape[0] >= T else np.pad(keep, (0, T - keep.shape[0]))
        a2 = a.astype(np.float32, copy=True)
        a2[:, ~m] = np.nan
        masked[k] = a2
    return masked


def _load_sample(
    traj_dir: Path, sample_id: str
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
]:
    """Load metadata, main npz, and the optional sidecar arrays.

    Sidecars (each may be missing):
      {id}_scores.npz       — raw [L, T, K] scores for beta sweep / top-k gap
      {id}_logit_lens.npz   — per-layer logit-lens entropy and top_prob
      {id}_null_control.npz — per-layer energy under shuffled bank
    """
    meta   = json.loads((traj_dir / f"{sample_id}.json").read_text(encoding="utf-8"))
    arrays = dict(np.load(traj_dir / f"{sample_id}.npz"))

    def _load_optional(name: str) -> dict[str, np.ndarray] | None:
        p = traj_dir / f"{sample_id}_{name}.npz"
        return dict(np.load(p)) if p.exists() else None

    return (
        meta,
        arrays,
        _load_optional("scores"),
        _load_optional("logit_lens"),
        _load_optional("null_control"),
    )


def _top_act_gap_per_layer(gen_scores: np.ndarray) -> np.ndarray:
    """Mean over T of (top1 − top2) of gen_scores[L, T, K].

    A small gap means retrieval is "confused" — many bank rows are similarly
    activated.  A large gap means one row dominates.
    """
    sc = gen_scores.astype(np.float32)            # [L, T, K]
    if sc.ndim != 3 or sc.shape[2] < 2:
        return np.full(sc.shape[0] if sc.ndim >= 1 else 0, np.nan, dtype=np.float64)
    # Partition is O(K log 2); avoids a full sort along the K axis.
    top2 = np.partition(sc, -2, axis=-1)[..., -2:]   # [L, T, 2]
    gap  = top2[..., 1] - top2[..., 0]               # [L, T] (top1 - top2)
    return np.nanmean(gap.astype(np.float64), axis=1)  # [L]


def _top_act_gap_per_layer_topk(topk_values: np.ndarray) -> np.ndarray:
    """Same gap from topk-compressed scores.

    topk_values has shape [L, T, k] with values pre-sorted descending along
    the last axis.  Top1 is column 0, top2 column 1.
    """
    v = topk_values.astype(np.float32)
    if v.ndim != 3 or v.shape[2] < 2:
        return np.full(v.shape[0] if v.ndim >= 1 else 0, np.nan, dtype=np.float64)
    gap = v[..., 0] - v[..., 1]                       # [L, T]
    return np.nanmean(gap.astype(np.float64), axis=1)  # [L]


def _compute_sample_divergence(
    arrays: dict[str, np.ndarray],
    scores: dict[str, np.ndarray] | None = None,
    logit_lens: dict[str, np.ndarray] | None = None,
    null_control: dict[str, np.ndarray] | None = None,
) -> SampleDivergenceResult:
    """Wrap metrics.divergences.compute_sample_divergences for one .npz dict."""
    div = compute_sample_divergences(
        prompt_energy=arrays.get("prefill_energy_mean", arrays["prefill_energy"]),
        gen_energy=arrays["gen_energy"],
        prompt_entropy=arrays.get("prefill_entropy_mean"),
        prompt_norm_entropy=arrays.get("prefill_norm_entropy_mean"),
        prompt_top_act=arrays.get("prefill_top_act_mean"),
        prompt_lse=arrays.get("prefill_lse_mean"),
        gen_entropy=arrays.get("gen_entropy"),
        gen_norm_entropy=arrays.get("gen_norm_entropy"),
        gen_top_act=arrays.get("gen_top_act"),
        gen_lse=arrays.get("gen_lse"),
        gen_js=arrays.get("gen_js"),
        gen_hellinger=arrays.get("gen_hellinger"),
    )
    if scores is not None:
        if "gen_scores" in scores:
            div.top_act_gap_mean = _top_act_gap_per_layer(scores["gen_scores"])
        elif "gen_scores_topk" in scores:
            div.top_act_gap_mean = _top_act_gap_per_layer_topk(scores["gen_scores_topk"])

    # Logit-lens sidecar
    if logit_lens is not None:
        gen_ent = logit_lens.get("gen_logit_entropy")        # [L, T]
        pre_ent = logit_lens.get("prefill_logit_entropy")    # [L]
        gen_top = logit_lens.get("gen_logit_top_prob")       # [L, T]
        if gen_ent is not None and pre_ent is not None and gen_ent.ndim == 2:
            gen_ent_mean = np.nanmean(gen_ent.astype(np.float64), axis=1)
            div.delta_logit_entropy    = gen_ent_mean - pre_ent.astype(np.float64)
            div.gen_logit_entropy_mean = gen_ent_mean
        if gen_top is not None and gen_top.ndim == 2 and gen_top.shape[1] > 0:
            div.gen_logit_top_prob_min = np.nanmin(gen_top.astype(np.float64), axis=1)

    # Null-control sidecar: compute the same delta_energy on the shuffled bank
    if null_control is not None:
        pref_n = null_control.get("prefill_energy_shuffled")
        gen_n  = null_control.get("gen_energy_shuffled")
        if pref_n is not None and gen_n is not None and gen_n.ndim == 2:
            div.delta_energy_null = (
                np.nanmean(gen_n.astype(np.float64), axis=1) - pref_n.astype(np.float64)
            )

    return div


def analyze_trajectories(
    traj_dir: str | Path,
    output: str | Path,
    score_metric: str = "delta_energy",
    score_aggregation: str = "mean",
    signal_zone: tuple[int, int] | None = None,
    pool_over_answer: bool = False,
    tokenizer_id: str | None = None,
) -> Path:
    """Aggregate trajectory .npz/.json files into an analysis JSON.

    Writes a single ``analysis.json`` compatible with ``visualize_analysis``.

    Args:
        traj_dir:          Directory containing .npz/.json trajectory pairs.
        output:            Path for the output analysis JSON file.
        score_metric:      Metric for the scalar hallucination score.
        score_aggregation: Aggregation method (mean / max / weighted).
        signal_zone:       Optional (start, end) layer slice for scoring.
        pool_over_answer:  When True, pool generation metrics only over the
                           tokens that span the gold answer (or the content
                           tokens as fallback) instead of all generation tokens.
                           Loads a tokenizer matching the model recorded in the
                           sample metadata; pass ``tokenizer_id`` to override.
        tokenizer_id:      HuggingFace model id or alias used to load a
                           tokenizer when ``pool_over_answer=True``.  Defaults
                           to the model alias in the first sample's metadata.

    Returns:
        Path to the written analysis JSON file.
    """
    traj_dir = Path(traj_dir)
    output   = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    sample_ids = [
        jf.stem
        for jf in sorted(traj_dir.glob("*.json"))
        if not jf.stem.endswith("_hpre")
        and (traj_dir / f"{jf.stem}.npz").exists()
    ]

    if not sample_ids:
        raise ValueError(f"No trajectory samples found in {traj_dir}")

    log.info("Analyzing %d samples from %s", len(sample_ids), traj_dir)

    tokenizer = None
    if pool_over_answer:
        # Load a tokenizer once (no model weights).  Resolve via model profile
        # so callers can pass an alias like "qwen25_3b".
        from hopfield_llm.models.profiles import resolve_model_id
        from transformers import AutoTokenizer  # noqa: WPS433
        first_meta = json.loads(
            (traj_dir / f"{sample_ids[0]}.json").read_text(encoding="utf-8")
        )
        alias = tokenizer_id or first_meta.get("model") or "qwen25_3b"
        try:
            hf_id = resolve_model_id(alias)
        except Exception:
            hf_id = alias
        log.info("pool_over_answer=True: loading tokenizer for %s (%s)", alias, hf_id)
        tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)

    all_divs:   list[SampleDivergenceResult] = []
    all_scores: list[float]                  = []
    all_metas:  list[dict[str, Any]]         = []
    category_scores: dict[str, list[float]]  = defaultdict(list)
    n_layers = None

    for sample_id in sample_ids:
        try:
            meta, arrays, scores, logit_lens, null_control = _load_sample(
                traj_dir, sample_id
            )
            all_metas.append(meta)
            if n_layers is None:
                n_layers = arrays["prefill_energy"].shape[0]

            if pool_over_answer and tokenizer is not None:
                T_gen = arrays["gen_energy"].shape[1]
                mask = _answer_token_mask(meta, T_gen, tokenizer)
                if mask is not None and mask.any():
                    arrays = _apply_answer_mask(arrays, mask)

            div = _compute_sample_divergence(
                arrays,
                scores=scores,
                logit_lens=logit_lens,
                null_control=null_control,
            )
            all_divs.append(div)

            score = hallucination_score(
                div,
                signal_zone=signal_zone,
                metric=score_metric,
                aggregation=score_aggregation,
            )
            all_scores.append(score)
            # Prefer dataset-specific category (e.g. TruthfulQA category field);
            # fall back to source name (dataset alias) when no category is set.
            cat_key = meta.get("category") or meta.get("source") or "unknown"
            category_scores[cat_key].append(score)

        except Exception as exc:
            log.warning("Skipping %s: %s", sample_id, exc)

    if not all_divs:
        raise ValueError("No samples could be analyzed successfully")

    n_scored    = len(all_scores)
    scores_arr  = np.array(all_scores)

    # -- Per-layer delta metrics: stack [N, L] → mean/std --
    LAYER_METRIC_NAMES = [
        "delta_energy", "delta_entropy", "delta_norm_entropy",
        "delta_top_activation", "delta_lse",
        # Per-layer temporal metrics (always populated from gen_energy [L, T])
        "energy_slope", "energy_var", "energy_spread",
        # Per-layer top1-top2 gap (NaN when scores not saved for this sample)
        "top_act_gap_mean",
        # Logit-lens metrics (NaN when enable_logit_lens=False at capture)
        "delta_logit_entropy", "gen_logit_entropy_mean", "gen_logit_top_prob_min",
        # Shuffled-bank null control (NaN when enable_null_control=False)
        "delta_energy_null",
        # Inline JS and Hellinger divergences (NaN when p_ref not computed)
        "js_mean_per_layer", "js_max_per_layer",
        "hellinger_mean_per_layer", "hellinger_max_per_layer",
    ]
    layer_metrics: dict[str, dict[str, Any]] = {}
    for mn in LAYER_METRIC_NAMES:
        cols = []
        for d in all_divs:
            v = getattr(d, mn, None)
            cols.append(v if v is not None else np.full(n_layers, np.nan, dtype=np.float64))
        stacked = np.stack(cols, axis=0)  # [N, L]
        layer_metrics[mn] = {
            "mean": np.nanmean(stacked, axis=0).tolist(),
            "std":  np.nanstd(stacked,  axis=0).tolist(),
            "n_samples": int(np.sum(np.any(np.isfinite(stacked), axis=1))),
        }

    # -- Interpretable scalar summaries: one value per sample → distribution --
    SCALAR_METRIC_NAMES = [
        "energy_shift_l1", "energy_shift_l2", "peak_delta_layer",
    ]
    global_divergences: dict[str, dict[str, Any]] = {}
    for mn in SCALAR_METRIC_NAMES:
        vals = np.array([getattr(d, mn) for d in all_divs], dtype=np.float64)
        global_divergences[mn] = {
            "mean": float(np.nanmean(vals)),
            "std":  float(np.nanstd(vals)),
            "min":  float(np.nanmin(vals)),
            "max":  float(np.nanmax(vals)),
            "n_samples": n_scored,
        }

    score_summary: dict[str, Any] = {
        "count": n_scored,
        "mean":  float(np.mean(scores_arr)),
        "std":   float(np.std(scores_arr)),
        "min":   float(np.min(scores_arr)),
        "max":   float(np.max(scores_arr)),
    }

    score_by_category: dict[str, dict[str, Any]] = {}
    for cat, cat_scores in sorted(category_scores.items()):
        arr = np.array(cat_scores)
        score_by_category[cat] = {
            "count": len(cat_scores),
            "mean":  float(np.mean(arr)),
            "std":   float(np.std(arr)),
            "min":   float(np.min(arr)),
            "max":   float(np.max(arr)),
        }

    analysis: dict[str, Any] = {
        "artifact_type":      "analysis",
        "n_samples":          len(sample_ids),
        "n_scored":           n_scored,
        "n_layers":           n_layers,
        "score_metric":       score_metric,
        "score_aggregation":  score_aggregation,
        "score_summary":      score_summary,
        "score_by_category":  score_by_category,
        "layer_metrics":      layer_metrics,
        "global_divergences": global_divergences,
    }

    output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    log.info(
        "Analysis complete: %d samples, mean score=%.6f → %s",
        n_scored, score_summary["mean"], output,
    )
    return output
