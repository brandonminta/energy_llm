# Architecture

## Overview

`hopfield_llm` is a 4-stage artifact-based pipeline for studying hallucination
emergence in LLMs through a thermodynamic lens.

```
Dataset + Model
      │
      ▼
 [Stage 1] build_banks
      │   Extract per-layer memory banks (W_down weights)
      │   Output: banks.pt
      │
      ▼
 [Stage 2] run_trajectory
      │   Hook into model; run prefill + autoregressive generation
      │   Capture h_pre per layer; compute energy/entropy metrics
      │   Output: {sample_id}.npz + {sample_id}.json per sample
      │
      ▼
 [Stage 3] analyze_trajectories
      │   CPU-only. Read .npz files; compute divergences and scores
      │   Output: analysis.json
      │
      ▼
 [Stage 4] visualize_analysis
          Generate matplotlib plots from analysis JSON
          Output: plots/*.png
```

Each stage writes artifacts to disk and can be run independently, making it
easy to re-run later stages without re-running expensive GPU stages.

---

## Package layout (`src/hopfield_llm/`)

| Module | Responsibility |
|--------|---------------|
| `models/` | `HFLLM` wrapper, architecture resolution, model profile registry |
| `hooks/` | Forward-pre-hook registration and h_pre capture during prefill and generation |
| `memory/` | Memory bank construction from MLP weight matrices |
| `queries/` | Query extraction strategies (which activation to use as the retrieval query) |
| `metrics/` | Energy computation, KL/JS/Hellinger divergences, hallucination score aggregation |
| `datasets/` | `BaseDataset` ABC, `DataSample` dataclass, TruthfulQA/TriviaQA/NQ adapters |
| `pipeline/` | Stage functions, `ExperimentConfig`, YAML loader, analysis |
| `storage/` | Artifact save/load (torch + JSON) |
| `utils/` | Logging, environment config, experiment tracker |
| `visualization/` | Matplotlib plots from analysis JSON |
| `cli/` | Argparse CLI with 5 subcommands |

---

## Separation of concerns

The architecture is designed so that changing one concern does not require
touching unrelated modules:

| Concern | Location | Change impact |
|---------|----------|--------------|
| Which weights form the memory | `memory/banks.py` | Zero — hooks and metrics are unaffected |
| Which activation is the query | `queries/extractors.py` | Zero — pass a different `query_fn` to `capture_prefill` |
| How energy is computed | `metrics/energy.py` | Zero — banks and hooks don't know the formula |
| Adding a new dataset | `datasets/` | Zero — implement `BaseDataset`, plug into `run_trajectory` |
| Adding new metrics | `metrics/` | Zero — only `pipeline/analysis.py` needs updating |

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
