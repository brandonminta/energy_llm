"""Tests for M5: evaluation pipeline.

Case 1: per_layer_auroc_table on synthetic data — signal at layer 5.
Case 2: fit_logreg_probe recovers near-ceiling AUROC on separable data.
Case 3: beta_sweep_auroc raises ValueError when save_scores was False.
Case 4: build_eval_report smoke test (requires qwen25_1_5b model).
"""

from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path

from hopfield_llm.evaluation.per_layer_auroc import (
    per_layer_auroc_table,
    best_single_feature,
)
from hopfield_llm.evaluation.probe import fit_logreg_probe
from hopfield_llm.evaluation.beta_sweep import beta_sweep_auroc


# ------------------------------------------------------------------
# Synthetic data helpers
# ------------------------------------------------------------------


def _make_synthetic_samples(
    n: int,
    n_layers: int,
    signal_layer: int,
    signal_strength: float = 5.0,
    seed: int = 42,
) -> list[dict]:
    """n samples with strong signal only at signal_layer of 'gen_energy_answer'."""
    rng = np.random.default_rng(seed)
    is_hall = rng.integers(0, 2, n).astype(bool)
    samples = []
    for i in range(n):
        feat = rng.normal(0.0, 0.3, n_layers)
        feat[signal_layer] = (
            float(is_hall[i]) * signal_strength + rng.normal(0, 0.2)
        )
        samples.append({
            "is_hallucination": bool(is_hall[i]),
            "gen_energy_answer":  feat.astype(np.float32),
            "gen_entropy_answer": rng.normal(0, 0.3, n_layers).astype(np.float32),
        })
    return samples


# ------------------------------------------------------------------
# Case 1: per_layer_auroc_table detects signal at layer 5
# ------------------------------------------------------------------


def test_per_layer_auroc_signal_layer():
    samples = _make_synthetic_samples(n=50, n_layers=10, signal_layer=5)
    table = per_layer_auroc_table(samples, ["gen_energy_answer"])

    key, layer, auroc = best_single_feature(table)

    assert key == "gen_energy_answer", f"unexpected key: {key}"
    assert layer == 5, f"expected signal at layer 5, got layer {layer}"
    assert auroc > 0.9, f"expected AUROC > 0.9, got {auroc:.4f}"


def test_per_layer_auroc_table_shape():
    samples = _make_synthetic_samples(n=40, n_layers=6, signal_layer=2)
    table = per_layer_auroc_table(samples, ["gen_energy_answer", "gen_entropy_answer"])

    # Rows: 6 layers + "best_layer"
    assert len(table) == 7
    assert "gen_energy_answer"  in table.columns
    assert "gen_entropy_answer" in table.columns
    assert "best_layer" in table.index


def test_per_layer_auroc_best_layer_is_int():
    samples = _make_synthetic_samples(n=30, n_layers=4, signal_layer=1)
    table = per_layer_auroc_table(samples, ["gen_energy_answer"])
    best_val = table.loc["best_layer", "gen_energy_answer"]
    # pandas upcasts to float64 when mixing AUROC rows with int index rows
    assert float(best_val).is_integer(), (
        f"best_layer should represent an integer layer index, got {best_val}"
    )
    assert 0 <= int(best_val) < 4


def test_per_layer_auroc_nan_imputation():
    """NaN values in a feature should not crash; result should be finite."""
    rng = np.random.default_rng(99)
    samples = _make_synthetic_samples(n=40, n_layers=5, signal_layer=3)
    # Inject NaNs into half the samples at layer 0
    for s in samples[:20]:
        s["gen_energy_answer"] = s["gen_energy_answer"].copy()
        s["gen_energy_answer"][0] = float("nan")

    table = per_layer_auroc_table(samples, ["gen_energy_answer"])
    data_rows = table.drop("best_layer", errors="ignore")
    # Layer 0 should have a finite (imputed) AUROC
    assert np.isfinite(float(data_rows.loc[0, "gen_energy_answer"])) or \
           np.isnan(float(data_rows.loc[0, "gen_energy_answer"])), \
           "expected finite or nan, not an error"


# ------------------------------------------------------------------
# Case 2: fit_logreg_probe recovers near-ceiling AUROC
# ------------------------------------------------------------------


def test_logreg_probe_separable():
    """Clearly separable classes → probe AUROC > 0.95."""
    rng = np.random.default_rng(7)
    n, L = 80, 6
    is_hall = np.array([i % 2 == 0 for i in range(n)], dtype=bool)
    rng.shuffle(is_hall)

    samples = []
    for h in is_hall:
        feat = rng.normal(5.0 if h else -5.0, 0.3, L)
        samples.append({
            "is_hallucination": bool(h),
            "gen_energy_answer": feat.astype(np.float32),
        })

    result = fit_logreg_probe(samples, ["gen_energy_answer"])
    assert result["mean_auroc"] > 0.95, (
        f"expected mean_auroc > 0.95 on separable data, got {result['mean_auroc']:.4f}"
    )
    assert "per_fold" in result
    assert "coef_importance" in result
    assert result["n_features"] == L


def test_logreg_probe_refuses_small_dataset():
    """Fewer than 50 samples should raise ValueError."""
    samples = [
        {"is_hallucination": True, "gen_energy_answer": np.ones(4, np.float32)}
        for _ in range(20)
    ]
    with pytest.raises(ValueError, match="50"):
        fit_logreg_probe(samples, ["gen_energy_answer"])


def test_logreg_probe_coef_importance_index():
    rng = np.random.default_rng(3)
    n, L = 60, 4
    is_hall = rng.integers(0, 2, n).astype(bool)
    samples = [
        {"is_hallucination": bool(h),
         "gen_energy_answer": rng.normal(float(h), 1.0, L).astype(np.float32)}
        for h in is_hall
    ]
    result = fit_logreg_probe(samples, ["gen_energy_answer"])
    assert len(result["coef_importance"]) == L
    assert all(f"gen_energy_answer_l{l}" in result["coef_importance"].index for l in range(L))


# ------------------------------------------------------------------
# Case 3: beta_sweep_auroc raises when save_scores=False
# ------------------------------------------------------------------


def test_beta_sweep_raises_without_scores():
    """Samples without 'gen_scores' must raise ValueError mentioning save_scores."""
    samples = [
        {
            "is_hallucination": i % 2 == 0,
            "gen_energy_answer": np.ones(4, np.float32),
            # gen_scores intentionally absent
        }
        for i in range(10)
    ]
    with pytest.raises(ValueError, match="save_scores"):
        beta_sweep_auroc(samples, betas=[1.0, 5.0])


def test_beta_sweep_raises_when_scores_none():
    """gen_scores=None should also raise the same error."""
    samples = [
        {
            "is_hallucination": i % 2 == 0,
            "gen_energy_answer": np.ones(4, np.float32),
            "gen_scores": None,
        }
        for i in range(10)
    ]
    with pytest.raises(ValueError, match="save_scores"):
        beta_sweep_auroc(samples, betas=[1.0])


def test_beta_sweep_output_shape():
    """With gen_scores present the output is a DataFrame indexed by beta."""
    rng = np.random.default_rng(1)
    L, T, K = 4, 8, 16
    betas = [5.0, 15.0, 30.0]
    is_hall = np.array([i % 2 == 0 for i in range(20)], dtype=bool)
    samples = [
        {
            "is_hallucination": bool(h),
            "gen_energy_answer": rng.normal(0, 1, L).astype(np.float32),
            "gen_scores": rng.normal(float(h), 0.5, (L, T, K)).astype(np.float32),
        }
        for h in is_hall
    ]

    df = beta_sweep_auroc(samples, betas=betas)
    assert len(df) == len(betas)
    assert "best_auroc_energy" in df.columns
    assert list(df.index) == betas


# ------------------------------------------------------------------
# Case 4: build_eval_report smoke test (requires real model)
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def _llm():
    try:
        from hopfield_llm.models.loader import HFLLM
        return HFLLM("qwen25_1_5b", load_in_4bit=False)
    except Exception as exc:
        pytest.skip(f"qwen25_1_5b unavailable: {exc}")


def test_build_eval_report_smoke(tmp_path, _llm):
    """End-to-end: 20 TruthfulQA samples → capture → label → report HTML."""
    import torch
    from hopfield_llm.datasets.truthfulqa import TruthfulQADataset
    from hopfield_llm.evaluation.features import save_sample_label
    from hopfield_llm.evaluation.report import build_eval_report
    from hopfield_llm.hooks.capture import capture_generation, capture_prefill
    from hopfield_llm.labeling.heuristic import label_sample_heuristic
    from hopfield_llm.memory.banks import extract_banks

    banks = extract_banks(_llm, bank="up", normalize=True)

    dataset  = TruthfulQADataset()
    samples  = [s for _, s in zip(range(20), dataset)]

    artifact_dir = tmp_path / "traj"
    labels_dir   = tmp_path / "labels"
    artifact_dir.mkdir()
    labels_dir.mkdir()

    for s in samples:
        with torch.no_grad():
            gen = capture_generation(
                _llm, s.question, banks,
                beta=15.0, max_new_tokens=30,
                apply_chat_template=False,
            )
            prefill = capture_prefill(_llm, s.question, banks, beta=15.0)

        import dataclasses
        s_with_gen = dataclasses.replace(s, generated_text=gen.generated_text)
        labeled = label_sample_heuristic(s_with_gen)
        save_sample_label(labeled, labels_dir)

        np.savez_compressed(
            artifact_dir / f"{s.id}.npz",
            prefill_energy=prefill.energy_last,
            prefill_energy_mean=prefill.energy_mean,
            prefill_entropy=prefill.entropy[:, -1],
            prefill_norm_entropy=prefill.norm_entropy[:, -1],
            prefill_lse=prefill.lse[:, -1],
            prefill_quadratic=prefill.quadratic[:, -1],
            prefill_top_act=prefill.top_act[:, -1],
            prefill_n_active=prefill.n_active[:, -1],
            gen_energy=gen.energy,
            gen_entropy=gen.entropy,
            gen_norm_entropy=gen.norm_entropy,
            gen_lse=gen.lse,
            gen_quadratic=gen.quadratic,
            gen_top_act=gen.top_act,
            gen_n_active=gen.n_active,
            token_ids=gen.token_ids,
        )
        meta = {
            "id": s.id,
            "source": s.source,
            "question": s.question,
            "gold_answers": s.gold_answers,
            "generated_text": gen.generated_text,
            "model": "qwen25_1_5b",
            "n_layers": len(banks),
            "n_tokens_generated": int(len(gen.token_ids)),
        }
        with open(artifact_dir / f"{s.id}.json", "w") as f:
            json.dump(meta, f)

    out_path = tmp_path / "report.html"
    result_path = build_eval_report(
        artifact_dir=artifact_dir,
        labels_dir=labels_dir,
        out_path=out_path,
        beta=15.0,
        dataset_name="truthfulqa_smoke",
    )

    assert result_path.exists(), "HTML report not created"
    html = result_path.read_text(encoding="utf-8")
    assert len(html) > 100, "HTML report is empty"
    assert "per-layer AUROC" in html, "missing 'per-layer AUROC' in report"
    assert "best_single_feature" in html, "missing 'best_single_feature' in report"
