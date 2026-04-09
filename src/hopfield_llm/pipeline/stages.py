"""Five decoupled pipeline stages.

Each stage can run independently as long as its upstream artifacts exist.

    Stage 1: build_memory_bank   — model weights → .pt bank artifact
    Stage 2: extract_hidden      — forward passes → .pt hidden-state artifact
    Stage 3: score               — energy + divergences → .pt score artifact
    Stage 4: analyze             — aggregate statistics → .json analysis
    Stage 5: visualize           — plots from analysis JSON
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from hopfield_llm.datasets.prompts import load_prompt_records
from hopfield_llm.extraction.hidden_states import (
    align_banks_to_layers,
    extract_segment_hidden_states,
    segment_hidden_states_to_dict,
)
from hopfield_llm.extraction.memory_bank import extract_mlp_memory_bank
from hopfield_llm.metrics.divergences import (
    compute_layer_divergences,
    compute_layer_energy_profile,
)
from hopfield_llm.metrics.scoring import hallucination_score
from hopfield_llm.models.loader import HFLLM
from hopfield_llm.utils.io import (
    load_json_artifact,
    load_torch_artifact,
    save_json_artifact,
    save_torch_artifact,
)
from hopfield_llm.utils.logging import get_logger
from hopfield_llm.visualization.plots import visualize_analysis

log = get_logger("pipeline.stages")


# ------------------------------------------------------------------
# Stage 1 — Memory bank extraction
# ------------------------------------------------------------------


def build_memory_bank(
    model: str,
    output: str | Path,
    bank: str = "gate",
    normalize: bool = False,
    strict: bool = True,
    device: str | None = None,
    load_in_4bit: bool = True,
) -> Path:
    """Extract MLP memory banks and save as a torch artifact."""
    llm = HFLLM(
        model_id=model, device=device, load_in_4bit=load_in_4bit,
        output_hidden_states=False,  # not needed for weight extraction
    )
    banks, metadata = extract_mlp_memory_bank(
        llm, bank=bank, normalize=normalize, strict=strict, return_metadata=True,
    )
    artifact = {
        "artifact_type": "memory_bank",
        "model": llm.summary(),
        "bank_name": bank,
        "normalize": normalize,
        "banks": banks,
        "metadata": metadata,
    }
    path = save_torch_artifact(artifact, output)
    log.info("Stage 1 complete: saved bank to %s", path)
    return path


# ------------------------------------------------------------------
# Stage 2 — Hidden-state extraction
# ------------------------------------------------------------------


def extract_hidden(
    memory: str | Path,
    prompts: str | Path,
    output: str | Path,
    pooling: Literal["mean", "last", "max", "first"] = "mean",
    separator: str = "\n",
    layers: list[int] | None = None,
    device: str | None = None,
) -> Path:
    """Extract hidden states from prompts and save as a torch artifact."""
    memory_artifact = load_torch_artifact(memory)
    model_summary = memory_artifact["model"]
    model_ref = model_summary.get("alias") or model_summary["model_id"]

    llm = HFLLM(
        model_id=model_ref, device=device,
        load_in_4bit=model_summary.get("load_in_4bit_requested", True),
        output_hidden_states=True,
    )

    samples = []
    records = load_prompt_records(prompts)
    for i, record in enumerate(records):
        sample: dict[str, Any] = {
            "id": record["id"],
            "mode": record["mode"],
            "question": record["question"],
            "category": record.get("category", "unknown"),
            "label": record.get("label"),
        }
        if record["mode"] == "paired":
            factual = extract_segment_hidden_states(
                llm, question=record["question"], answer=record["answer_factual"],
                separator=separator, pooling=pooling, layers=layers,
                normalize=False, to_cpu=True,
            )
            hallucinated = extract_segment_hidden_states(
                llm, question=record["question"], answer=record["answer_hallucinated"],
                separator=separator, pooling=pooling, layers=layers,
                normalize=False, to_cpu=True,
            )
            sample["answer_factual"] = record["answer_factual"]
            sample["answer_hallucinated"] = record["answer_hallucinated"]
            sample["factual"] = segment_hidden_states_to_dict(factual)
            sample["hallucinated"] = segment_hidden_states_to_dict(hallucinated)
        else:
            hidden = extract_segment_hidden_states(
                llm, question=record["question"], answer=record["answer"],
                separator=separator, pooling=pooling, layers=layers,
                normalize=False, to_cpu=True,
            )
            sample["answer"] = record["answer"]
            sample["hidden"] = segment_hidden_states_to_dict(hidden)
        samples.append(sample)
        if (i + 1) % 50 == 0:
            log.info("Extracted %d/%d samples", i + 1, len(records))

    artifact = {
        "artifact_type": "hidden_states",
        "model": llm.summary(),
        "memory_summary": {
            "bank_name": memory_artifact.get("bank_name"),
            "normalize": memory_artifact.get("normalize"),
        },
        "pooling": pooling,
        "separator": separator,
        "layers": layers,
        "samples": samples,
    }
    path = save_torch_artifact(artifact, output)
    log.info("Stage 2 complete: saved %d samples to %s", len(samples), path)
    return path


# ------------------------------------------------------------------
# Stage 3 — Scoring
# ------------------------------------------------------------------


def score_hidden_states(
    memory: str | Path,
    hidden: str | Path,
    output: str | Path,
    beta: float = 15.0,
    energy_mode: Literal["cosine", "dot"] = "cosine",
    active_threshold: float = 0.1,
    score_metric: str = "js",
    score_aggregation: Literal["mean", "max", "weighted"] = "mean",
    score_use_absolute: bool = False,
) -> Path:
    """Compute divergence scores from cached hidden states and banks."""
    memory_artifact = load_torch_artifact(memory)
    hidden_artifact = load_torch_artifact(hidden)
    banks = memory_artifact["banks"]

    scored_samples = []
    for sample in hidden_artifact["samples"]:
        if sample["mode"] == "paired":
            layer_indices = sample["factual"]["layer_indices"]
            aligned = align_banks_to_layers(banks, layer_indices)
            div = compute_layer_divergences(
                h_ref=sample["factual"]["h_answer"],
                h_cmp=sample["hallucinated"]["h_answer"],
                mlp_keys=aligned, beta=beta, energy_mode=energy_mode,
                active_threshold=active_threshold,
                reference_label="factual", target_label="hallucinated",
            )
            metrics = {
                "kl_fwd": div.kl_fwd.tolist(),
                "kl_rev": div.kl_rev.tolist(),
                "js": div.js.tolist(),
                "hellinger": div.hellinger.tolist(),
                "delta_entropy": div.delta_entropy.tolist(),
                "delta_norm_entropy": div.delta_norm_entropy.tolist(),
                "delta_energy": div.delta_energy.tolist(),
                "delta_top_activation": div.delta_top_activation.tolist(),
                "delta_mean_activation": div.delta_mean_activation.tolist(),
            }
            score = hallucination_score(
                div, metric=score_metric, aggregation=score_aggregation,
                use_absolute=score_use_absolute,
            )
        else:
            layer_indices = sample["hidden"]["layer_indices"]
            aligned = align_banks_to_layers(banks, layer_indices)
            profile = compute_layer_energy_profile(
                h_state=sample["hidden"]["h_answer"],
                mlp_keys=aligned, beta=beta, energy_mode=energy_mode,
                active_threshold=active_threshold,
            )
            metrics = {k: v.tolist() for k, v in profile.items()}
            score = None

        scored_samples.append({
            "id": sample["id"],
            "mode": sample["mode"],
            "question": sample["question"],
            "category": sample.get("category", "unknown"),
            "label": sample.get("label"),
            "layer_indices": layer_indices,
            "metrics": metrics,
            "score": score,
        })

    score_values = [s["score"] for s in scored_samples if s["score"] is not None]
    artifact = {
        "artifact_type": "scores",
        "memory_summary": {
            "bank_name": memory_artifact.get("bank_name"),
            "model": memory_artifact.get("model", {}),
        },
        "hidden_summary": {
            "pooling": hidden_artifact.get("pooling"),
            "layers": hidden_artifact.get("layers"),
        },
        "params": {
            "beta": beta, "energy_mode": energy_mode,
            "active_threshold": active_threshold,
            "score_metric": score_metric,
            "score_aggregation": score_aggregation,
            "score_use_absolute": score_use_absolute,
        },
        "samples": scored_samples,
        "summary": {
            "n_samples": len(scored_samples),
            "n_scored": len(score_values),
            "score_mean": (
                None if not score_values
                else float(sum(score_values) / len(score_values))
            ),
        },
    }
    path = save_torch_artifact(artifact, output)
    log.info("Stage 3 complete: scored %d samples → %s", len(scored_samples), path)
    return path


# ------------------------------------------------------------------
# Stage 4 — Analysis
# ------------------------------------------------------------------


def _summarize_scalar(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def analyze_scores(scores: str | Path, output: str | Path) -> Path:
    """Aggregate score artifact into dataset-level statistics."""
    scores_artifact = load_torch_artifact(scores)
    samples = scores_artifact.get("samples", [])

    scalar_scores = [
        float(s["score"]) for s in samples if s.get("score") is not None
    ]

    by_category: dict[str, list[float]] = defaultdict(list)
    by_label: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        score = sample.get("score")
        if score is None:
            continue
        by_category[str(sample.get("category", "unknown"))].append(float(score))
        by_label[str(sample.get("label"))].append(float(score))

    metric_buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    for sample in samples:
        for name, values in sample.get("metrics", {}).items():
            if isinstance(values, list):
                metric_buckets[name].append(np.asarray(values, dtype=np.float64))

    layer_metrics = {}
    for name, arrays in metric_buckets.items():
        if not arrays:
            continue
        stacked = np.vstack(arrays)
        layer_metrics[name] = {
            "mean": np.nanmean(stacked, axis=0).tolist(),
            "std": np.nanstd(stacked, axis=0).tolist(),
            "n_samples": int(stacked.shape[0]),
        }

    analysis = {
        "artifact_type": "analysis",
        "n_samples": len(samples),
        "score_summary": _summarize_scalar(scalar_scores),
        "score_by_category": {
            k: _summarize_scalar(v) for k, v in sorted(by_category.items())
        },
        "score_by_label": {
            k: _summarize_scalar(v) for k, v in sorted(by_label.items())
        },
        "layer_metrics": layer_metrics,
        "source_summary": scores_artifact.get("summary", {}),
    }
    path = save_json_artifact(analysis, output)
    log.info("Stage 4 complete: analysis → %s", path)
    return path


# ------------------------------------------------------------------
# Stage 5 — Visualization
# ------------------------------------------------------------------


def visualize(analysis: str | Path, output: str | Path) -> Path:
    """Render plots from analysis JSON."""
    out = visualize_analysis(analysis, output)
    log.info("Stage 5 complete: plots → %s", out)
    return out
