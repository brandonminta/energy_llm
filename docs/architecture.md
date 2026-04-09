# Architecture

## Overview

hopfield_llm implements a five-stage pipeline for energy-based hallucination detection in LLMs. Each stage is independently runnable and persists its output to disk so downstream stages can reuse cached results.

## Data Flow

```
                    +---------+
                    | Dataset |
                    +----+----+
                         |
                         v
+--------+    +---------------------+
| Model  |--->| Stage 1: Bank       |---> banks/{model}__{bank}.pt
+--------+    | (MLP weights only)  |
              +---------------------+
                         |
              +----------v----------+
              | Stage 2: Hidden     |---> hidden_states/{model}__{pooling}.pt
              | (forward passes)    |
              +---------------------+
                         |
              +----------v----------+
              | Stage 3: Score      |---> scores/{params}.pt
              | (Hopfield energy +  |
              |  divergences)       |
              +---------------------+
                         |
              +----------v----------+
              | Stage 4: Analyze    |---> analysis.json
              | (statistics)        |
              +---------------------+
                         |
              +----------v----------+
              | Stage 5: Visualize  |---> plots/
              | (matplotlib)        |
              +---------------------+
```

## Module Dependency Graph

```
cli/run.py
  └─> pipeline/stages.py
        ├─> models/loader.py          (HFLLM)
        │     └─> models/profiles.py  (ModelProfile)
        ├─> extraction/memory_bank.py
        ├─> extraction/hidden_states.py
        ├─> metrics/hopfield.py       (energy primitive)
        ├─> metrics/divergences.py    (KL, JS, Hellinger + layer comparisons)
        ├─> metrics/scoring.py        (hallucination_score aggregation)
        ├─> datasets/prompts.py
        └─> visualization/plots.py

experiments/experiment1.py
  ├─> extraction/hidden_states.py
  ├─> metrics/divergences.py
  └─> metrics/scoring.py

datasets/truthfulqa.py
  └─> datasets/base.py  (BaseHallucinationDataset)

pipeline/cache.py            (independent — no model dependencies)
utils/env.py                 (independent — reads YAML configs)
utils/tracking.py            (independent — filesystem + git)
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `models/loader.py` | Loads HuggingFace causal LMs with 4-bit quantization, resolves architecture-specific backbone and layers |
| `models/profiles.py` | Predefined profiles (Qwen, Phi, Mistral, Llama) with layer count, hidden dim, and 4-bit safety flags |
| `extraction/memory_bank.py` | Extracts MLP weight matrices as per-layer memory banks (gate, up, down, composites) |
| `extraction/hidden_states.py` | Tokenises Q+sep+A, runs forward pass, pools hidden states per segment. Supports single and batched modes |
| `metrics/hopfield.py` | Modern Hopfield energy primitive: cosine/dot similarity, softmax retrieval, entropy metrics |
| `metrics/divergences.py` | KL, JS, Hellinger primitives + layer-wise divergence computation comparing two states against the same bank |
| `metrics/scoring.py` | Aggregates per-layer divergence arrays into a scalar hallucination score |
| `datasets/base.py` | Abstract base class and DataSample dataclass for uniform dataset interface |
| `datasets/truthfulqa.py` | TruthfulQA adapter producing paired factual/hallucinated DataSamples |
| `datasets/prompts.py` | Loads standalone JSON prompt files for decoupled hidden-state extraction |
| `experiments/experiment1.py` | Orchestrates the factual-vs-hallucinated comparison experiment |
| `pipeline/stages.py` | Five decoupled stages callable from CLI or programmatically |
| `pipeline/cache.py` | CacheManager with parameter-encoded filenames, supports .pt, .json, .parquet |
| `utils/env.py` | Auto-detects local vs HPC environment, loads YAML configs |
| `utils/tracking.py` | ExperimentTracker: config snapshots, git info, incremental metrics, replay |
| `utils/io.py` | Save/load for torch and JSON artifacts |
| `utils/logging.py` | Structured logging under the hopfield_llm namespace |
| `visualization/plots.py` | Bar charts (score by category), line plots (metrics by layer) |
| `cli/run.py` | Argparse CLI with subcommands for each pipeline stage |

## Key Design Decisions

**Layer indexing**: Memory banks use 0-indexed keys (matching `enumerate(llm.layers)`). Hidden states use 1-indexed layer numbers (matching HuggingFace's `output_hidden_states` convention where index 0 is the embedding layer). The `align_banks_to_layers()` function in `extraction/hidden_states.py` bridges this gap.

**Batched extraction**: `extract_hidden_states_batch()` pads variable-length sequences and processes them in a single forward pass. This is the primary parallelisation mechanism — the forward pass accounts for 99%+ of total runtime.

**Cache-first pipeline**: Each stage checks for existing artifacts before recomputing. This is critical for the HPC workflow where forward passes run in separate SLURM jobs.

**Environment abstraction**: All environment-specific configuration (VRAM limits, batch sizes, paths) lives in YAML files under `configs/environments/`. Business logic has no `if local / if hpc` branches.

## Artifact Formats

| Artifact | Format | Contents |
|----------|--------|----------|
| Memory bank | `.pt` (torch) | `{artifact_type, model, bank_name, normalize, banks: {layer: Tensor[K,D]}, metadata}` |
| Hidden states | `.pt` (torch) | `{artifact_type, model, pooling, samples: [{id, mode, h_question, h_answer, ...}]}` |
| Scores | `.pt` (torch) | `{artifact_type, params, samples: [{id, metrics: {name: [L]}, score}], summary}` |
| Analysis | `.json` | `{score_summary, score_by_category, score_by_label, layer_metrics}` |
| Experiment run | directory | `config.yaml + environment.yaml + git_info.json + metrics.parquet + log.txt` |
