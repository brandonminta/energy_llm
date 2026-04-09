"""Experiment 1: Factual vs hallucinated answer discrimination.

Compares Hopfield retrieval distributions between factual and hallucinated
answers for the same question, producing per-sample divergence metrics and
a scalar hallucination score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from hopfield_llm.extraction.hidden_states import (
    SegmentHiddenStates,
    align_banks_to_layers,
    extract_hidden_states_batch,
    extract_segment_hidden_states,
)
from hopfield_llm.metrics.divergences import LayerDivergenceResult, compute_layer_divergences
from hopfield_llm.metrics.scoring import hallucination_score
from hopfield_llm.utils.logging import get_logger

log = get_logger("experiments.experiment1")


@dataclass
class Experiment1Result:
    idx: int
    category: str
    question: str
    answer_factual: str
    answer_hallucinated: str
    label: int
    div: LayerDivergenceResult | None
    score: float | None
    layer_indices: list[int]
    pooling: str
    error: str | None = None


def run_experiment_1(
    llm,
    banks: dict[int, torch.Tensor],
    samples: list,
    pooling: Literal["mean", "last", "max", "first"] = "mean",
    layers: list[int] | None = None,
    beta: float = 15.0,
    energy_mode: Literal["cosine", "dot"] = "cosine",
    active_threshold: float = 0.1,
    separator: str = "\n",
    signal_zone: tuple[int, int] | None = None,
    score_metric: str = "js",
    score_aggregation: Literal["mean", "max", "weighted"] = "mean",
    score_use_absolute: bool = False,
    skip_on_error: bool = True,
    verbose: bool = True,
    batch_size: int = 1,
) -> tuple[list[Experiment1Result], pd.DataFrame]:
    """Run experiment 1 on a list of paired samples.

    Each sample must have .question, .answer_factual, .answer_hallucinated,
    .idx, .category, .label attributes (e.g. TruthfulQASample-like objects
    or any object with those fields).

    Args:
        llm: HFLLM instance.
        banks: Memory banks keyed by 0-indexed layer number.
        samples: List of sample objects with paired answers.
        batch_size: Forward-pass batch size (>1 uses batched extraction).
        (other args): See individual functions for details.

    Returns:
        Tuple of (results list, summary DataFrame).
    """
    results: list[Experiment1Result] = []

    if batch_size > 1:
        results = _run_batched(
            llm, banks, samples,
            pooling=pooling, layers=layers, beta=beta,
            energy_mode=energy_mode, active_threshold=active_threshold,
            separator=separator, signal_zone=signal_zone,
            score_metric=score_metric, score_aggregation=score_aggregation,
            score_use_absolute=score_use_absolute,
            skip_on_error=skip_on_error, verbose=verbose,
            batch_size=batch_size,
        )
    else:
        iterator = tqdm(samples, desc="Exp1") if verbose else samples
        for s in iterator:
            results.append(
                _process_single(
                    llm, banks, s,
                    pooling=pooling, layers=layers, beta=beta,
                    energy_mode=energy_mode, active_threshold=active_threshold,
                    separator=separator, signal_zone=signal_zone,
                    score_metric=score_metric, score_aggregation=score_aggregation,
                    score_use_absolute=score_use_absolute,
                    skip_on_error=skip_on_error,
                )
            )

    df = _build_dataframe(results)

    if verbose:
        ok = sum(1 for r in results if r.error is None)
        failed = len(results) - ok
        scores = [r.score for r in results if r.score is not None]
        log.info("Completed: %d/%d", ok, len(results))
        if failed:
            log.info("Errors: %d", failed)
        if scores:
            log.info(
                "Score: %.4f +/- %.4f  [%.4f, %.4f]",
                np.mean(scores), np.std(scores), min(scores), max(scores),
            )

    return results, df


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _process_single(llm, banks, s, *, pooling, layers, beta, energy_mode,
                     active_threshold, separator, signal_zone, score_metric,
                     score_aggregation, score_use_absolute, skip_on_error):
    """Process one sample (two forward passes)."""
    try:
        seg_f = extract_segment_hidden_states(
            llm, question=s.question, answer=s.answer_factual,
            separator=separator, pooling=pooling, layers=layers,
            normalize=False, to_cpu=True,
        )
        seg_h = extract_segment_hidden_states(
            llm, question=s.question, answer=s.answer_hallucinated,
            separator=separator, pooling=pooling, layers=layers,
            normalize=False, to_cpu=True,
        )
        aligned = align_banks_to_layers(banks, seg_f.layer_indices)
        if not aligned:
            raise RuntimeError(
                f"No bank available for layer_indices={seg_f.layer_indices}"
            )
        div = compute_layer_divergences(
            h_ref=seg_f.h_answer, h_cmp=seg_h.h_answer,
            mlp_keys=aligned, beta=beta, energy_mode=energy_mode,
            active_threshold=active_threshold,
            reference_label="factual", target_label="hallucinated",
        )
        score = hallucination_score(
            div, signal_zone=signal_zone, metric=score_metric,
            aggregation=score_aggregation, use_absolute=score_use_absolute,
        )
        return Experiment1Result(
            idx=s.idx, category=s.category, question=s.question,
            answer_factual=s.answer_factual,
            answer_hallucinated=s.answer_hallucinated,
            label=s.label, div=div, score=score,
            layer_indices=seg_f.layer_indices, pooling=pooling,
        )
    except Exception as e:
        if not skip_on_error:
            raise
        return Experiment1Result(
            idx=s.idx, category=s.category, question=s.question,
            answer_factual=s.answer_factual,
            answer_hallucinated=s.answer_hallucinated,
            label=s.label, div=None, score=None,
            layer_indices=layers or [], pooling=pooling, error=str(e),
        )


def _run_batched(llm, banks, samples, *, pooling, layers, beta, energy_mode,
                  active_threshold, separator, signal_zone, score_metric,
                  score_aggregation, score_use_absolute, skip_on_error,
                  verbose, batch_size):
    """Process samples using batched forward passes."""
    # Build (question, answer) pairs: factual then hallucinated per sample
    factual_pairs = [(s.question, s.answer_factual) for s in samples]
    hallucinated_pairs = [(s.question, s.answer_hallucinated) for s in samples]

    log.info("Batched extraction: %d samples, batch_size=%d", len(samples), batch_size)

    segs_f = extract_hidden_states_batch(
        llm, factual_pairs,
        separator=separator, pooling=pooling, layers=layers,
        normalize=False, batch_size=batch_size,
    )
    segs_h = extract_hidden_states_batch(
        llm, hallucinated_pairs,
        separator=separator, pooling=pooling, layers=layers,
        normalize=False, batch_size=batch_size,
    )

    results = []
    iterator = (
        tqdm(zip(samples, segs_f, segs_h), total=len(samples), desc="Exp1-score")
        if verbose
        else zip(samples, segs_f, segs_h)
    )

    for s, sf, sh in iterator:
        try:
            aligned = align_banks_to_layers(banks, sf.layer_indices)
            if not aligned:
                raise RuntimeError(
                    f"No bank available for layer_indices={sf.layer_indices}"
                )
            div = compute_layer_divergences(
                h_ref=sf.h_answer, h_cmp=sh.h_answer,
                mlp_keys=aligned, beta=beta, energy_mode=energy_mode,
                active_threshold=active_threshold,
                reference_label="factual", target_label="hallucinated",
            )
            score = hallucination_score(
                div, signal_zone=signal_zone, metric=score_metric,
                aggregation=score_aggregation, use_absolute=score_use_absolute,
            )
            results.append(Experiment1Result(
                idx=s.idx, category=s.category, question=s.question,
                answer_factual=s.answer_factual,
                answer_hallucinated=s.answer_hallucinated,
                label=s.label, div=div, score=score,
                layer_indices=sf.layer_indices, pooling=pooling,
            ))
        except Exception as e:
            if not skip_on_error:
                raise
            results.append(Experiment1Result(
                idx=s.idx, category=s.category, question=s.question,
                answer_factual=s.answer_factual,
                answer_hallucinated=s.answer_hallucinated,
                label=s.label, div=None, score=None,
                layer_indices=layers or [], pooling=pooling, error=str(e),
            ))

    return results


def _build_dataframe(results: list[Experiment1Result]) -> pd.DataFrame:
    rows = []
    for r in results:
        row: dict = {
            "idx": r.idx, "category": r.category, "label": r.label,
            "score": r.score, "pooling": r.pooling, "error": r.error,
        }
        if r.div is not None:
            metrics = {
                "js": r.div.js,
                "kl_fwd": r.div.kl_fwd,
                "kl_rev": r.div.kl_rev,
                "hellinger": r.div.hellinger,
                "delta_entropy": r.div.delta_entropy,
                "delta_norm_entropy": r.div.delta_norm_entropy,
                "delta_energy": r.div.delta_energy,
                "delta_top_act": r.div.delta_top_activation,
                "delta_mean_act": r.div.delta_mean_activation,
            }
            for metric_name, arr in metrics.items():
                for layer_pos, val in enumerate(arr):
                    row[f"{metric_name}_l{layer_pos}"] = (
                        float(val) if not np.isnan(val) else np.nan
                    )
        rows.append(row)
    return pd.DataFrame(rows)
