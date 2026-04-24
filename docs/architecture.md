# Architecture

## Overview

`hopfield_llm` is a 6-stage artifact-based pipeline for energy-based hallucination detection in LLMs.

```
Dataset + Model
      │
      ▼
 [Stage 1] build_banks
      │   Extract per-layer memory banks (MLP weight matrices)
      │   Options: W_gate, W_up, W_down, W_fc1, W_fc2, W_w1, W_w2, W_w3, ...
      │   Output: banks.pt {layer_idx → Tensor[K, D]}
      │
      ▼
 [Stage 2] run_trajectory
      │   Hook into model; run prefill + autoregressive generation
      │   Capture h_pre per layer per token; compute energy/entropy
      │   Output: {sample_id}.npz (energy [L] and [L,T]) + {sample_id}.json (metadata)
      │
      ▼
 [Stage 3] label_sample
      │   Label each sample: is_hallucination (boolean)
      │   Options: heuristic (F1-based) or llm_judge (external LLM)
      │   Output: {sample_id}_label.json
      │
      ▼
 [Stage 4] evaluate_features
      │   Load trajectory artifacts + labels; bridge M2/M3 boundary
      │   Extract baseline metrics (log-probability, entropy)
      │   Output: feature dicts [{sample_id, is_hallucination, per_layer_features[L], ...}]
      │
      ▼
 [Stage 5] evaluate_probes
      │   Compute per-layer AUROC table (best feature per layer)
      │   Fit logistic regression probe with stratified nested CV
      │   Perform beta sweep over cached score tensors
      │   Output: metrics dict with AUROC, probe results, beta sweep curves
      │
      ▼
 [Stage 6] report
          Build self-contained HTML report with plots and metrics
          Output: report.html (AUROC table, probe AUROC, beta curves, baselines)
```

Each stage writes artifacts to disk and can be run independently, making it
easy to re-run later stages without re-running expensive GPU stages.

---

## Package layout (`src/hopfield_llm/`)

### Core Inference Pipeline

| Module | Exports | Responsibility |
|--------|---------|-----------------|
| `models/` | `HFLLM`, `ModelProfile` | HuggingFace model loader with 4-bit quantization, architecture layer resolution, model alias registry |
| `hooks/` | `capture_prefill`, `capture_generation`, `PrefillResult`, `GenerationResult` | Forward-pre-hook registration; h_pre capture at last question token (prefill) and per-token (generation) |
| `memory/` | `extract_banks` | Extract MLP weight matrices (W_gate, W_up, W_down, W_fc1, W_fc2, etc.) per layer as memory banks |
| `queries/` | `last_token`, `mean_tokens`, `make_positional` | Query extraction strategies: produce [d_m] from [T, d_m] activations |
| `metrics/` | `compute_energy`, `compute_entropy`, divergence functions, `hallucination_score` | Modern Hopfield energy/entropy primitives, KL/JS/Hellinger divergences, per-layer aggregation |
| `datasets/` | `BaseDataset`, `DataSample` | Dataset abstraction + TruthfulQA/TriviaQA/NQ adapters; yields question, gold_answers, ID |
| `pipeline/` | `build_banks`, `run_trajectory`, `analyze_trajectories`, `ExperimentConfig` | Stage orchestration, YAML config loader, per-sample divergence aggregation |
| `storage/` | `save_torch_artifact`, `load_torch_artifact`, `save_json_artifact` | Persistent I/O for banks, trajectories, analysis JSON |
| `utils/` | `ExperimentTracker`, `setup_logging`, `load_env_config` | Structured logging, experiment run tracking with git/environment metadata, auto-detect local vs HPC |
| `cli/` | `build_parser`, `main` | Argparse CLI entry point (5 core subcommands) |
| `visualization/` | `visualize_analysis` | Matplotlib plots from analysis JSON (layer metrics, divergences) |

### Labeling & Evaluation

| Module | Exports | Responsibility |
|--------|---------|-----------------|
| `labeling/` | `label_sample_heuristic`, `LLMJudge`, `label_sample_with_judge` | **Heuristic**: SQuAD F1-based binary classification (fast, no API). **LLMJudge**: External LLM or local HF model for binary judgment. Supports OpenAI-compatible APIs and HuggingFace models for air-gapped HPC. |
| `analysis/` | `find_answer_span`, `content_token_mask`, `extract_answer_features` | Token-level alignment utilities. Match gold answers in generated text; extract per-layer features ([L] arrays) over answer spans via mean-pooling. |
| `evaluation/` | Per-layer AUROC, logistic regression probe, beta sweep, report building | **features**: load trajectory artifacts + labels → unified feature dicts. **baselines**: seq_logprob, token entropy. **per_layer_auroc**: per-layer AUROC per feature (best layer per column). **probe**: logistic regression with stratified nested CV. **beta_sweep**: post-hoc recomputation from cached scores. **report**: HTML report with embedded plots. |

---

## Module Dependencies

The architecture uses clean separation of concerns: each module is independent and swappable.

```
Dataset ─────┐
             ├─► models/HFLLM ──► hooks/ (capture activations)
Memory Bank ─┤                        ↓
             └─► queries/ (extract query)
                 ↓
            metrics/ (compute energy/entropy)
                 ↓
            pipeline/stages (aggregate per sample)
                 ├─► labeling/ (mark is_hallucination)
                 └─► analysis/ (find answer spans, extract features)
                        ↓
                   evaluation/ (AUROC, probe, report)
```

The pipeline supports **post-hoc modification without recomputation**:
- Change the labeling method without re-running GPU stages (labels are separate artifacts)
- Swap evaluation metrics without re-running inference (features are cached)
- Re-sweep beta values without re-running trajectory (raw scores cached)

---

## Separation of concerns

| Concern | Location | Change impact |
|---------|----------|--------------|
| Which weights form the memory | `memory/banks.py` | Zero — hooks and metrics unaffected; re-run `build-banks` only |
| Which activation is the query | `queries/extractors.py` | Zero — pass `query_fn` to `capture_prefill`; re-run `run-trajectory` only |
| How energy is computed | `metrics/energy.py` | Zero — banks and hooks unaffected; re-run `run-trajectory` and `analyze` only |
| Adding a new dataset | `datasets/` | Zero — subclass `BaseDataset`, register in `run_trajectory` |
| Changing labeling method | `labeling/` | Zero — label artifacts separate; re-label only |
| Adding new evaluation metrics | `evaluation/` | Zero — features are cached; re-run evaluation pipeline only |

---

## Current scientific hypothesis

> The memory bank is the W_down matrix of the SwiGLU MLP.
> The query is h_pre = SiLU(gate) ⊙ up at the last question token.
> Energy = Modern Hopfield retrieval energy of this query against W_down.

**This hypothesis is not hardcoded.** To test alternatives:
- Change the bank: pass `bank="gate"` or `bank="gate_plus_up"` to `build_banks`
- Change the query: pass a different `query_fn` to `capture_prefill`
- Change energy formula: replace `compute_energy` in `metrics/energy.py`

---

## Key data structures

```
LayerEnergyResult       — scalar metrics from one compute_energy() call
PrefillResult           — [L] arrays from one forward pass on the question
GenerationResult        — [L, T] arrays from one generation pass
SampleDivergenceResult  — global divergence scalars + per-layer delta arrays
ExperimentConfig        — flat config loaded from hierarchical YAML
ExperimentTracker       — manages run directory, config snapshot, git info
```

---

## Artifact schema

**banks.pt** (torch.save)
```
{
  "artifact_type": "banks",
  "model": {summary dict},
  "bank": "down",
  "banks": {0: Tensor[K, D], 1: Tensor[K, D], ...}
}
```

**{sample_id}.npz** (numpy compressed)
```
prefill_energy, prefill_entropy, prefill_norm_entropy,
prefill_lse, prefill_quadratic, prefill_top_act, prefill_n_active  — [L]
gen_energy, gen_entropy, gen_norm_entropy,
gen_lse, gen_quadratic, gen_top_act, gen_n_active                  — [L, T]
token_ids                                                           — [T]
```

**{sample_id}.json**
```json
{
  "id": "truthfulqa_0",
  "source": "truthfulqa",
  "question": "...",
  "gold_answers": ["..."],
  "generated_text": "...",
  "model": "qwen25_3b",
  "n_layers": 36,
  "n_tokens_generated": 12
}
```

**analysis.json**
```json
{
  "artifact_type": "analysis",
  "n_samples": 817, "n_scored": 817, "n_layers": 36,
  "score_summary": {"mean": ..., "std": ..., "min": ..., "max": ...},
  "score_by_category": {"truthfulqa": {"mean": ..., "count": ...}},
  "layer_metrics": {
    "delta_energy":          {"mean": [...L], "std": [...L]},
    "delta_entropy":         {"mean": [...L], "std": [...L]},
    "delta_norm_entropy":    {"mean": [...L], "std": [...L]},
    "delta_top_activation":  {"mean": [...L], "std": [...L]},
    "delta_mean_activation": {"mean": [...L], "std": [...L]}
  },
  "global_divergences": {
    "kl_fwd": {"mean": ..., "std": ..., "min": ..., "max": ...},
    "js":     {"mean": ..., "std": ...},
    "hellinger": {"mean": ..., "std": ...}
  }
}
```

Note: `layer_metrics` contains genuine per-layer signals (generation − prefill).
`global_divergences` contains distributional comparisons across the layer axis
(one scalar per sample, summarised across N samples). Do not interpret
`global_divergences` as per-layer values.

---

## HPC / SLURM

Stage 2 supports stride-based sharding: `samples[shard_id::num_shards]`.
All shards write to the same output directory (per-sample filenames are unique).
Stage 3 (CPU-only) runs once after all shards complete.

SLURM templates live in `scripts/hpc/templates/`. To submit the 3-stage pipeline:
```bash
./scripts/hpc/submit_pipeline.sh
```

The project does not require SLURM for local use. All stages run as plain CLI
commands on a single machine.
