# energy_llm

`energy_llm` is a research framework for energy-based analysis of LLM internal states. The current codebase focuses on MLP memory-bank extraction, Hopfield-style retrieval energy, hidden-state segmentation, layer-wise divergence metrics, and hallucination-oriented comparison experiments.

The repository is intentionally lightweight. It is meant to support:

- multiple LLM backends under `src/energy_llm/models`
- multiple datasets under `src/energy_llm/datasets`
- bank extraction under `src/energy_llm/memory`
- hidden-state extraction under `src/energy_llm/representations`
- divergence and scoring logic under `src/energy_llm/metrics`
- experiment runners under `src/energy_llm/experiments`
- local and Slurm-based execution through `scripts/`

## Status

The repository currently has one implemented end-to-end baseline:

- TruthfulQA loading
- Hugging Face causal LLM loading
- MLP memory-bank extraction
- segment-level hidden-state extraction
- layer-wise divergence computation
- experiment 1: factual vs hallucinated answer comparison

Several other packages and config files are present as placeholders for the planned multi-dataset and multi-experiment expansion. Those placeholders are documented in `docs/architecture.md`.

## Repository Layout

- `src/energy_llm/`: Python package
- `configs/`: YAML configs for models, datasets, experiments, and sweeps
- `scripts/`: local launchers and Slurm entrypoints
- `data/`: local datasets and intermediate processed files
- `artifacts/`: reusable intermediate outputs such as banks or hidden states
- `runs/`: per-run outputs and result tables
- `docs/`: contributor-facing documentation
- `tests/`: test suite, currently minimal

## Setup

### Local WSL / micromamba

```bash
micromamba create -n energy-llm python=3.12 -y
micromamba activate energy-llm
pip install -r requirements.txt
pip install -e .
```

Notes:

- `bitsandbytes` is included for Linux GPU workflows. If 4-bit loading is not available in your environment, set `load_in_4bit: false` in the model config.
- The recommended workflow is editable install plus `src/` layout. The package metadata is configured for `pip install -e .`.

### HPC / Slurm

The Slurm scripts assume:

- a Linux GPU environment
- `micromamba` or an equivalent activated Python environment
- the repository is available on shared storage or copied into the job working directory

The provided Slurm entrypoints use environment variables rather than hardcoded cluster-specific paths:

- `REPO_ROOT`
- `MAMBA_ENV_NAME`
- `PYTHONPATH`

See `scripts/slurm/exp01.slurm`.

## Minimal Working Path

The minimal baseline experiment is Experiment 01 on TruthfulQA.

After installing dependencies:

```bash
./scripts/run_exp01.sh
```

Equivalent direct command:

```bash
python -m energy_llm.cli.main run-exp1 --config configs/experiments/exp01_truthfulqa_baseline.yaml
```

The default experiment config points to:

- `configs/experiments/exp01_truthfulqa_baseline.yaml`
- `configs/models/qwen25_3b.yaml`
- `configs/datasets/truthfulqa.yaml`

## High-Level Architecture

The current baseline pipeline is:

1. Load a model profile and instantiate `HFLLM`.
2. Extract one MLP bank per layer from the model.
3. Load normalized dataset samples.
4. Extract pooled hidden states for question and answer segments.
5. Compare factual and hallucinated answer states against the same bank.
6. Aggregate layer-wise divergences into scalar scores.
7. Save run outputs under `runs/`.

## Artifacts vs Runs

- `artifacts/` is for reusable intermediate material such as cached banks, hidden states, score tables, and figures.
- `runs/` is for experiment-specific outputs, config snapshots, and summary tables produced by a single invocation.

## Documentation

- Architecture overview: `docs/architecture.md`
- Baseline runner: `src/energy_llm/cli/main.py`
