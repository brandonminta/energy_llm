# hopfield-llm

Energy-based hallucination detection for LLMs using Modern Hopfield networks. Compares Hopfield retrieval distributions between factual and hallucinated answers to produce layer-wise divergence metrics and scalar hallucination scores.

## Quick Start (Local)

```bash
micromamba create -n hopfield-llm python=3.12 -y && micromamba activate hopfield-llm
pip install -r requirements.txt && pip install -e .
hopfield-llm build-memory --model qwen25_3b --output bank.pt
hopfield-llm extract-hidden --memory bank.pt --prompts prompts.json --output hidden.pt
hopfield-llm score --memory bank.pt --hidden hidden.pt --output scores.pt
```

## Quick Start (HPC)

```bash
bash setup_hpc.sh
source .env.hpc
./jobs/submit_pipeline.sh
```

## Pipeline Stages

| Stage | Command | Input | Output | Cost |
|-------|---------|-------|--------|------|
| 1 | `build-memory` | model id | bank .pt | Low (weight read) |
| 2 | `extract-hidden` | bank + prompts | hidden states .pt | **High** (forward passes) |
| 3 | `score` | bank + hidden | scores .pt | Medium (matrix ops) |
| 4 | `analyze` | scores | analysis .json | Low |
| 5 | `visualize` | analysis | plots/ | Low |

Each stage runs independently. Cached artifacts are reused across experiments.

## Supported Models

| Model | Params | VRAM (4-bit) | RTX 2050 | A100-10GB |
|-------|--------|-------------|----------|-----------|
| Qwen2.5-1.5B | 1.5B | ~1.2 GB | Yes | Yes |
| Qwen2.5-3B | 3.0B | ~2.0 GB | Yes | Yes |
| Llama-3.2-3B | 3.2B | ~2.1 GB | Tight | Yes |
| Phi-3-mini | 3.8B | ~2.4 GB | Marginal | Yes |
| Mistral-7B | 7.2B | ~14 GB (fp16) | No | A100-20GB+ |

## Documentation

- [Architecture](docs/architecture.md) — Module responsibilities, data flow, design decisions
- [Setup](docs/setup.md) — Installation for local and HPC environments
- [Usage](docs/usage.md) — CLI, Python API, adding datasets, experiment tracking
- [HPC Workflow](docs/hpc_workflow.md) — SLURM submission, monitoring, troubleshooting
- [Audit Report](docs/audit_report.md) — Code audit findings and recommendations

## Repository Layout

```
src/hopfield_llm/          Python package
  models/                   HFLLM loader + model profiles
  extraction/               Memory bank + hidden-state extraction (with batching)
  metrics/                  Hopfield energy, divergences, hallucination scoring
  datasets/                 Base class + TruthfulQA adapter
  experiments/              Experiment runners
  pipeline/                 Stage orchestration + cache manager
  utils/                    I/O, logging, environment config, experiment tracking
  cli/                      CLI entry point
  visualization/            Matplotlib plots
configs/                    YAML configs (models, datasets, experiments, environments)
jobs/                       SLURM job templates + submission script
data/                       Raw data, cache, results (not in git)
experiments/runs/           Tracked experiment outputs
docs/                       Documentation
```
