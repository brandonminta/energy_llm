# hopfield-llm

Energy-based hallucination detection for LLMs using Modern Hopfield networks. Extracts MLP memory banks, computes Hopfield retrieval energy and entropy over hidden states during prefill and generation, and produces layer-wise divergence metrics and scalar hallucination scores.

## Quick Start (Local)

```bash
micromamba create -n hopfield-llm python=3.12 -y && micromamba activate hopfield-llm
pip install -r requirements.txt && pip install -e .

# Run stages individually
hopfield-llm build-banks --model qwen25_3b --output banks.pt
hopfield-llm run-trajectory --model qwen25_3b --dataset truthfulqa --banks banks.pt --output traj/
hopfield-llm analyze --trajectories traj/ --output analysis.json
hopfield-llm visualize --analysis analysis.json --output plots/

# Or run everything from a config
hopfield-llm run-experiment --config configs/experiments/exp01_truthfulqa_baseline.yaml
```

## Quick Start (HPC)

```bash
bash setup_hpc.sh
./jobs/submit_pipeline.sh                              # defaults
MODEL=qwen25_7b DATASET=nq NUM_SHARDS=16 \
  ./jobs/submit_pipeline.sh                            # larger run
```

## Pipeline Stages

| Stage | Command | GPU | Input | Output |
|-------|---------|-----|-------|--------|
| 1 | `build-banks` | Yes | model alias | `banks.pt` (MLP W_down weights) |
| 2 | `run-trajectory` | Yes | banks + dataset | `.npz` + `.json` per sample |
| 3 | `analyze` | No | trajectory dir | `analysis.json` (divergences, scores) |
| 4 | `visualize` | No | analysis JSON | `plots/` (PNG layer-metric plots) |

Each stage runs independently. `run-experiment` orchestrates all 4 from a YAML config with full experiment tracking (config snapshot, git info, timing, GPU metadata).

## Supported Models

| Alias | Model | Params | VRAM (4-bit) | Local (RTX 2050) | HPC (A100) |
|-------|-------|--------|-------------|------------------|------------|
| `qwen25_1_5b` | Qwen2.5-1.5B-Instruct | 1.5B | ~1.2 GB | Yes | Yes |
| `qwen25_3b` | Qwen2.5-3B-Instruct | 3.0B | ~2.0 GB | Yes | Yes |
| `llama32_3b` | Llama-3.2-3B-Instruct | 3.2B | ~2.1 GB | Tight | Yes |
| `phi3_mini` | Phi-3-mini-4k-instruct | 3.8B | ~2.4 GB | Marginal | Yes |
| `gemma2_2b` | gemma-2-2b-it | 2.5B | ~1.8 GB | Yes | Yes |
| `qwen25_7b` | Qwen2.5-7B-Instruct | 7.6B | ~4.5 GB | No | Yes |
| `llama31_8b` | Llama-3.1-8B-Instruct | 8.0B | ~5.0 GB | No | Yes |
| `gemma2_9b` | gemma-2-9b-it | 9.2B | ~5.5 GB | No | Yes |
| `mistral7b` | Mistral-7B-Instruct-v0.3 | 7.2B | fp16 only | No | A100-20GB+ |

## Supported Datasets

- **TruthfulQA** (`truthfulqa`) — 817 questions testing factual accuracy
- **TriviaQA** (`triviaqa`) — Trivia questions with multiple answer aliases
- **Natural Questions** (`nq`) — Google search questions with short answers

## Configuration

Experiment parameters are defined in YAML configs (`configs/experiments/`):

```yaml
experiment:
  name: exp01_truthfulqa_baseline
model:
  alias: qwen25_3b
  load_in_4bit: true
dataset:
  name: truthfulqa
  max_samples: null
  seed: 42
trajectory:
  beta: 15.0
  energy_mode: dot    # dot | cosine
analysis:
  score_metric: js    # kl_fwd | js | hellinger | delta_energy | ...
  score_aggregation: mean
output:
  base_dir: runs
```

Environment configs (`configs/environments/`) auto-detect local vs HPC via `HOPFIELD_ENV` or `SLURM_JOB_ID`.

## Repository Layout

```
src/hopfield_llm/          Python package
  cli/                      CLI entry point (5 subcommands)
  models/                   HFLLM loader + model profiles
  extraction/               MLP memory bank + h_pre extraction (prefill + generation)
  metrics/                  Hopfield energy, divergences, hallucination scoring
  datasets/                 Base class + TruthfulQA, TriviaQA, NQ adapters
  pipeline/                 Stage functions, analysis, experiment config
  utils/                    I/O, logging, environment config, experiment tracking
  visualization/            Matplotlib plots from analysis JSON
configs/
  environments/             local.yaml (RTX 2050) + hpc.yaml (A100)
  experiments/              Experiment YAML configs
jobs/                       SLURM templates + submission script (3-stage pipeline)
scripts/                    Local convenience scripts + result inspection
runs/                       Experiment outputs (tracked run directories)
tests/                      Unit tests
```

## Documentation

- [Architecture](docs/architecture.md) — Module responsibilities, data flow, design decisions
- [Setup](docs/setup.md) — Installation for local and HPC environments
- [Usage](docs/usage.md) — CLI, Python API, adding datasets, experiment tracking
- [HPC Workflow](docs/hpc_workflow.md) — SLURM submission, monitoring, troubleshooting
