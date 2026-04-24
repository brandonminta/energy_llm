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

| Stage | Command | GPU | Input | Output | Purpose |
|-------|---------|-----|-------|--------|---------|
| 1 | `build-banks` | Yes | model alias | `banks.pt` | Extract MLP weight matrices as memory banks |
| 2 | `run-trajectory` | Yes | banks + dataset | `.npz` + `.json` per sample | Capture h_pre activations; compute energy/entropy during prefill and generation |
| 3 | `analyze` | No | trajectory dir | `analysis.json` | Aggregate trajectories; compute per-sample divergences and scores |
| 4 | `visualize` | No | analysis JSON | `plots/` | Generate matplotlib plots of layer metrics |

Each stage runs independently and writes artifacts to disk, allowing re-running of downstream stages without GPU cost. `run-experiment` orchestrates all 4 from a YAML config with full experiment tracking (config snapshot, git info, timing, GPU metadata).

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
  bank: down          # gate | up | down | gate_plus_up | gate_up_concat
  beta: 15.0
  energy_mode: dot    # dot | cosine
analysis:
  score_metric: js    # js | kl_fwd | hellinger | delta_energy | ...
  score_aggregation: mean
output:
  base_dir: outputs
```

Environment configs (`configs/environments/`) auto-detect local vs HPC via `HOPFIELD_ENV` or `SLURM_JOB_ID`.

## Repository Layout

```
src/hopfield_llm/          Python package
  cli/                      CLI entry point (5 subcommands)
  models/                   HFLLM loader + architecture resolution + model profiles
  hooks/                    Forward-hook activation capture (prefill + generation)
  memory/                   Memory bank construction from MLP weights
  queries/                  Query extraction strategies (swappable)
  metrics/                  Energy computation, divergences, hallucination scoring
  datasets/                 BaseDataset + TruthfulQA, TriviaQA, NQ adapters
  pipeline/                 Stage functions, analysis, ExperimentConfig
  storage/                  Artifact save/load (torch + JSON)
  utils/                    Logging, environment config, experiment tracking
  visualization/            Matplotlib plots from analysis JSON
configs/
  environments/             local.yaml (RTX 2050) + hpc.yaml (A100)
  experiments/              Experiment YAML configs
jobs/                       SLURM templates + submission script
scripts/                    Local convenience scripts + result inspection
outputs/                    Experiment outputs (tracked run directories, gitignored)
tests/
  unit/                     Pure function tests
  integration/              Pipeline integration tests
docs/                       Architecture, setup, and usage documentation
```

## Documentation

- [Architecture](docs/architecture.md) — Module responsibilities, data flow, design decisions
- [Setup](docs/setup.md) — Installation for local and HPC environments
- [Usage](docs/usage.md) — CLI, Python API, adding datasets, query extractors
