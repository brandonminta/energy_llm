# Architecture Overview

## Purpose

`energy_llm` studies whether energy-like signals over LLM internal representations can separate factual and hallucinated behavior. The working baseline combines:

1. model loading
2. MLP bank extraction
3. hidden-state segmentation
4. layer-wise divergence computation
5. scalar score aggregation

The repository is intentionally organized as a research framework, not a large application. Some modules are implemented now, while others are placeholders kept in place to preserve the planned package layout.

## Top-Level Folders

- `src/energy_llm/`: installable Python package
- `configs/`: YAML configs for models, datasets, experiments, and sweeps
- `scripts/`: local shell launchers and Slurm wrappers
- `data/`: raw, interim, and processed dataset storage
- `artifacts/`: reusable intermediate outputs such as banks, hidden states, retrievals, and plots
- `runs/`: per-execution outputs and summaries
- `docs/`: contributor-facing documentation
- `tests/`: test suite, currently minimal
- `notebooks/`: exploratory analysis notebooks

## Package Layout

### `src/energy_llm/models/`

- `model_profiles.py`: canonical model aliases and dimensions
- `hf_llm.py`: Hugging Face model loader and hidden-state-capable wrapper
- `model_factory.py`: placeholder for a future backend selection layer

Current status:

- implemented: `model_profiles.py`, `hf_llm.py`
- placeholder: `model_factory.py`

### `src/energy_llm/memory/`

- `mlp_bank.py`: extraction of per-layer MLP banks from model layers
- `bank_cache.py`, `bank_extractors.py`, `bank_metadata.py`: planned support modules

Current status:

- implemented: `mlp_bank.py`
- placeholders: `bank_cache.py`, `bank_extractors.py`, `bank_metadata.py`

### `src/energy_llm/representations/`

- `segment_states.py`: pooled hidden-state extraction for question/answer segments
- `state_cache.py`, `token_pooling.py`, `trajectory_states.py`: planned extensions

Current status:

- implemented: `segment_states.py`
- placeholders: `state_cache.py`, `token_pooling.py`, `trajectory_states.py`

### `src/energy_llm/metrics/`

- `divergences.py`: Hopfield-style energy and layer-wise divergence metrics
- `scores.py`: scalar aggregation from layer-wise metrics
- `classification.py`, `ranking.py`: future evaluation helpers

Current status:

- implemented: `divergences.py`, `scores.py`
- placeholders: `classification.py`, `ranking.py`

### `src/energy_llm/datasets/`

- `truthfulqa.py`: normalized TruthfulQA loader for the current baseline
- `base.py`, `halu_eval.py`, `hotpotqa.py`, `local_jsonl.py`, `splits.py`, `triviaqa.py`: reserved for multi-dataset support

Current status:

- implemented: `truthfulqa.py`
- placeholders: all other modules in this package

### `src/energy_llm/experiments/`

- `exp1_factual_vs_hallucinated.py`: implemented baseline runner
- `base.py`, `exp2_context_vs_answer.py`, `exp3_cross_model.py`, `exp4_cross_dataset.py`, `exp5_generation_trajectory.py`: reserved for future experiments

Current status:

- implemented: `exp1_factual_vs_hallucinated.py`
- placeholders: all other experiment modules

### `src/energy_llm/cli/`

- `main.py`: thin command-line entrypoint that wires configs to the implemented baseline runner

### `src/energy_llm/analysis/`, `evaluation/`, `pipelines/`, `core/`, `utils/`, `energies/`

These packages currently exist to preserve the intended architecture, but most files are placeholders. They should remain lightweight until the corresponding functionality is actually needed.

## Data Flow

The working baseline data flow is:

1. `HFLLM` loads a causal LM with hidden-state output enabled.
2. `extract_mlp_memory_bank` extracts a layer-indexed bank from each MLP.
3. `load_truthfulqa` returns normalized question/factual/hallucinated samples.
4. `extract_segment_hidden_states` produces one `[L, D]` representation for each segment.
5. `compute_layer_divergences` compares factual and hallucinated answer states against the same bank.
6. `hallucination_score` reduces layer-wise metrics to a scalar.
7. `energy_llm.cli.main` writes CSV and JSON outputs under `runs/`.

## Config Layout

- `configs/models/`: model aliases and load settings
- `configs/datasets/`: dataset selection and sampling settings
- `configs/experiments/`: experiment wiring, bank parameters, scoring parameters, and output locations
- `configs/sweeps/`: documented sweep definitions

Implemented now:

- `configs/models/*.yaml`
- `configs/datasets/truthfulqa.yaml`
- `configs/experiments/exp01_truthfulqa_baseline.yaml`

Documented placeholders:

- non-TruthfulQA dataset configs
- experiments `exp02` through `exp05`

## Local vs HPC Workflow

Local workflow:

- activate the environment
- install editable package
- run `./scripts/run_exp01.sh`

HPC workflow:

- activate micromamba inside the job
- export `PYTHONPATH=$REPO_ROOT/src`
- launch `scripts/slurm/exp01.slurm`

The Slurm wrappers avoid hardcoded cluster-specific paths and expect `REPO_ROOT` plus an existing environment.
