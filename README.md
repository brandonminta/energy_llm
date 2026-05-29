# hopfield-llm

Energy-based hallucination detection for LLMs using Modern Hopfield networks. Extracts MLP memory banks, computes Hopfield retrieval energy and entropy over hidden states during prefill and generation, and produces layer-wise divergence metrics and scalar hallucination scores.

## Quick Start (Local)

```bash
micromamba create -n hopfield-llm python=3.12 -y && micromamba activate hopfield-llm
pip install -r requirements.txt && pip install -e .

# Core pipeline (GPU stages 1-2, CPU stages 3-4)
hopfield-llm build-banks --model qwen25_3b --output banks.pt
hopfield-llm run-trajectory --model qwen25_3b --dataset truthfulqa --banks banks.pt --output traj/
hopfield-llm analyze --trajectories traj/ --output analysis.json
hopfield-llm visualize --analysis analysis.json --output plots/

# Optional labeling + evaluation (CPU)
export OPENAI_API_KEY=sk-...
hopfield-llm label    --trajectories traj/ --output labels/
hopfield-llm evaluate --trajectories traj/ --labels labels/ --output report.html

# Or run stages 1-4 from a config (the definitive headline run)
hopfield-llm run-experiment --config configs/experiments/final_qwen_truthfulqa.yaml

# Baselines for the evaluate report (token uncertainty, P(True), semantic
# entropy, HalluField, INTRA) — each also runnable via `python -m`:
python scripts/local/run_baselines.py --trajectories traj/ --model qwen25_3b
```

See `configs/experiments/README.md` for the full config matrix and
`RUNBOOK.md` for the ordered execution plan.

## Quick Start (HPC)

```bash
bash scripts/hpc/setup_hpc.sh
# Single GPU job per config (auto-resumes; add --skip-banks on resubmit):
sbatch --job-name=final_qwen_tqa \
  scripts/hpc/jobs/run_experiment.sh configs/experiments/final_qwen_truthfulqa.yaml
# Or sharded array submission for large sweeps:
MODEL=qwen25_7b DATASET=nq NUM_SHARDS=16 scripts/hpc/submit_pipeline.sh
```

## Pipeline Stages

| Stage | Command | GPU | Input | Output | Purpose |
|-------|---------|-----|-------|--------|---------|
| 1 | `build-banks` | Yes | model alias | `banks.pt` | Extract MLP weight matrices as memory banks |
| 2 | `run-trajectory` | Yes | banks + dataset | `.npz` + `.json` per sample | Capture activations and Hopfield energy/entropy during prefill and generation; both observe the chat-templated prompt |
| 3 | `analyze` | No | trajectory dir | `analysis.json` | Aggregate trajectories; compute per-layer deltas, score summary, score-by-category |
| 4 | `visualize` | No | analysis JSON | `plots/` | Generate matplotlib plots of layer metrics |
| 5 | `label` | No | trajectory dir | `{id}_labeled.json` + `calibration.json` | Heuristic F1 + LLM-judge labeling (optional) |
| 6 | `evaluate` | No | trajectory + labels | `report.html` | Per-layer AUROC, logreg probe, baselines, β-sweep (optional) |

Each stage runs independently and writes artifacts to disk, allowing re-running of downstream stages without GPU cost. `run-experiment` orchestrates stages 1–4 from a YAML config with full experiment tracking (config snapshot, git info, timing, GPU metadata). Stages 5 and 6 are explicitly opt-in.

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
  name: final_qwen_truthfulqa
model:
  alias: qwen25_3b
  load_in_4bit: true
dataset:
  name: truthfulqa
  max_samples: null
  seed: 42
trajectory:
  bank: up            # legacy field; primary_probe.bank wins below
  beta: 15.0          # starting point; overridden when calibrate_beta=true
  calibrate_beta: true
  beta_target: 0.55
  energy_mode: dot    # dot | cosine
primary_probe:
  bank: gated_key     # gated_key | up | down_values | gate | gate_proj | up_proj | ...
  normalize: true     # L2-normalise bank rows (M=1)
  normalize_query: false
  hook_target: mlp_input
  label: gated_key
analysis:
  score_metric: delta_energy
  score_aggregation: mean
  pool_over_answer: true   # pool generation metrics over the gold-answer span
output:
  base_dir: outputs
```

Environment configs (`configs/environments/`) auto-detect local vs HPC via `HOPFIELD_ENV` or `SLURM_JOB_ID`.

## Repository Layout

```
src/hopfield_llm/          Python package
  cli/                      CLI entry point (run.py — 7 subcommands)
  models/                   HFLLM loader + architecture resolution + model profiles
  hooks/                    Forward-hook activation capture (prefill + generation)
  memory/                   Memory bank construction from MLP weights
  queries/                  Query reduction strategies (last/mean/positional)
  metrics/                  Energy, divergences, scoring, β-calibration, logit-lens, null-control
  analysis/                 Answer-span alignment + chat-prompt construction
  datasets/                 BaseDataset + TruthfulQA, TriviaQA, NQ adapters
  evaluation/               Features, per-layer AUROC, probe, baselines, report
  labeling/                 F1 heuristic + LLM judge
  pipeline/                 Stage functions, analysis, ExperimentConfig
  storage/                  Artifact save/load (torch + JSON)
  utils/                    Logging, environment config, experiment tracking
  visualization/            Matplotlib plots from analysis JSON (plots.py)
configs/
  environments/             local.yaml (RTX 2050) + hpc.yaml (A100)
  experiments/              Experiment YAML configs + README (config matrix)
  labeling/                 Judge + heuristic defaults
scripts/
  hpc/                      setup, submit_pipeline (sharded array), jobs/run_experiment.sh, templates/
  local/                    test_pipeline.sh (smoke), run_baselines.py, chat_model.py
outputs/                    Experiment outputs (run directories, gitignored)
tests/
  unit/                     Pure function tests
  integration/              Pipeline integration tests
docs/                       Theory (.tex), architecture, usage, findings, pre-registration
notebooks/                  Analysis notebooks (01–06)
```

## Documentation

- [Architecture](docs/architecture.md) — Module responsibilities, data flow, design decisions
- [Usage](docs/usage.md) / [Quick reference](docs/quickref.md) — CLI + Python API
- [Setup](docs/setup.md) — Installation for local and HPC environments
- [Findings](docs/findings.md) — Exp-04 empirical results and design rationale
- [Config matrix](configs/experiments/README.md) — every experiment config and its role
