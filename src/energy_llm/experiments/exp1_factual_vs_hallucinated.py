from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from energy_llm.metrics.divergences import (
    LayerDivergenceResult,
    compute_layer_divergences,
)
from energy_llm.metrics.scores import hallucination_score
from energy_llm.representations.segment_states import extract_segment_hidden_states


@dataclass
class Experiment1Result:
    """
    Resultado de una muestra en el Experimento 1.

    Comparacion: h_answer_factual vs h_answer_hallucinated
    contra el mismo banco MLP.
    """

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
    score_metric: Literal[
        "kl_fwd", "kl_rev", "js", "hellinger",
        "delta_entropy", "delta_norm_entropy", "delta_energy",
        "delta_top_activation", "delta_mean_activation",
    ] = "js",
    score_aggregation: Literal["mean", "max", "weighted"] = "mean",
    score_use_absolute: bool = False,
    skip_on_error: bool = True,
    verbose: bool = True,
) -> tuple[list[Experiment1Result], pd.DataFrame]:
    """
    Experimento 1: discriminacion supervisada factual vs alucinado.
    """
    results: list[Experiment1Result] = []

    iterator = tqdm(samples, desc="Exp1") if verbose else samples

    for s in iterator:
        try:
            seg_f = extract_segment_hidden_states(
                llm=llm,
                question=s.question,
                answer=s.answer_factual,
                separator=separator,
                pooling=pooling,
                layers=layers,
                normalize=False,
                to_cpu=True,
            )

            seg_h = extract_segment_hidden_states(
                llm=llm,
                question=s.question,
                answer=s.answer_hallucinated,
                separator=separator,
                pooling=pooling,
                layers=layers,
                normalize=False,
                to_cpu=True,
            )

            aligned_banks = {
                i: banks[li - 1]
                for i, li in enumerate(seg_f.layer_indices)
                if (li - 1) in banks
            }

            if not aligned_banks:
                raise RuntimeError(
                    f"Ninguna capa de layer_indices={seg_f.layer_indices} "
                    "tiene banco MLP disponible."
                )

            div = compute_layer_divergences(
                h_ref=seg_f.h_answer,
                h_cmp=seg_h.h_answer,
                mlp_keys=aligned_banks,
                beta=beta,
                energy_mode=energy_mode,
                active_threshold=active_threshold,
                reference_label="factual",
                target_label="hallucinated",
            )

            score = hallucination_score(
                div=div,
                signal_zone=signal_zone,
                metric=score_metric,
                aggregation=score_aggregation,
                use_absolute=score_use_absolute,
            )

            results.append(Experiment1Result(
                idx=s.idx,
                category=s.category,
                question=s.question,
                answer_factual=s.answer_factual,
                answer_hallucinated=s.answer_hallucinated,
                label=s.label,
                div=div,
                score=score,
                layer_indices=seg_f.layer_indices,
                pooling=pooling,
                error=None,
            ))

        except Exception as e:
            if not skip_on_error:
                raise
            results.append(Experiment1Result(
                idx=s.idx,
                category=s.category,
                question=s.question,
                answer_factual=s.answer_factual,
                answer_hallucinated=s.answer_hallucinated,
                label=s.label,
                div=None,
                score=None,
                layer_indices=layers or [],
                pooling=pooling,
                error=str(e),
            ))

    df = _build_dataframe_exp1(results)

    if verbose:
        ok = sum(1 for r in results if r.error is None)
        failed = len(results) - ok
        scores = [r.score for r in results if r.score is not None]
        print(f"\n[Exp1] Completadas : {ok}/{len(results)}")
        if failed:
            print(f"[Exp1] Con error   : {failed}")
        if scores:
            print(f"[Exp1] Score medio : {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
            print(f"[Exp1] Score rango : [{min(scores):.4f}, {max(scores):.4f}]")

    return results, df


def _build_dataframe_exp1(results: list[Experiment1Result]) -> pd.DataFrame:
    """
    Aplana la lista de resultados a un DataFrame.
    """
    rows = []

    for r in results:
        row: dict = {
            "idx": r.idx,
            "category": r.category,
            "label": r.label,
            "score": r.score,
            "pooling": r.pooling,
            "error": r.error,
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
                    col = f"{metric_name}_l{layer_pos}"
                    row[col] = float(val) if not np.isnan(val) else np.nan

        rows.append(row)

    return pd.DataFrame(rows)
