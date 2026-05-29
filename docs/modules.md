# Module Reference

Complete reference for all modules in the `hopfield_llm` package.

## Table of Contents

1. [Core Inference Pipeline](#core-inference-pipeline)
2. [Labeling & Evaluation](#labeling--evaluation)
3. [Utilities & Infrastructure](#utilities--infrastructure)

---

## Core Inference Pipeline

### `models/` — Model Loading & Architecture Resolution

**Purpose**: Wrap HuggingFace models with 4-bit quantization and multi-architecture support.

**Key exports**:
- `HFLLM(model_alias, load_in_4bit=True, device=None)` — Load a model with auto-architecture resolution
- `ModelProfile` — Registry entry (HuggingFace ID, parameter count, supported layer types)
- `profiles.py` — Alias → HF model ID mapping (qwen25_3b → Qwen/Qwen2.5-3B-Instruct, etc.)

**Example**:
```python
from hopfield_llm.models.loader import HFLLM

llm = HFLLM("qwen25_3b", load_in_4bit=True)
print(llm.model)  # AutoModelForCausalLM instance
print(llm.config.num_hidden_layers)  # 36 (resolved automatically)
```

**Architecture resolution**: `HFLLM` auto-detects which field contains layer objects (e.g., `transformer.h` for Qwen, `model.layers` for Llama, `transformer.layer` for T5).

---

### `hooks/` — Activation Capture During Inference

**Purpose**: Register forward-pre-hooks to capture hidden states (h_pre) at specific layers.

**Key exports**:
- `capture_prefill(llm, question, banks, query_fn=last_token, beta=15.0, energy_mode="dot")` — Single-pass prefill; returns `PrefillResult`
- `capture_generation(llm, question, banks, max_new_tokens=50, ...)` — Autoregressive generation; returns `GenerationResult`
- `PrefillResult(energy[L], entropy[L], ...)` — Per-layer metrics from prefill
- `GenerationResult(energy[L,T], entropy[L,T], generated_text, ...)` — Per-layer per-token metrics

**How it works**:
1. Registers forward-pre-hooks on target layers (e.g., SwiGLU gate_proj or up_proj)
2. Runs forward pass(es) on input text
3. Intercepts h_pre activations; applies query extraction
4. Computes energy/entropy using banks; accumulates per-layer results
5. Removes hooks and returns results

**Example**:
```python
from hopfield_llm.hooks.capture import capture_prefill

prefill = capture_prefill(
    llm,
    "Who wrote Hamlet?",
    banks,
    beta=15.0,
    energy_mode="dot",
)
print(prefill.energy.shape)  # (36,) for qwen25_3b
```

---

### `memory/` — Memory Bank Construction

**Purpose**: Extract MLP weight matrices from a model to serve as memory banks.

**Key exports**:
- `extract_banks(llm, bank="down")` — Extract and return `{layer_idx → Tensor[K, D]}`
- Supported banks: `down`, `gate`, `up`, `down_values`, `gate_proj`, `up_proj`, `down_proj`, `fc1`, `fc2`, `w1`, `w2`, `w3`

**How it works**:
1. Iterates over model layers
2. Locates target projection matrix (e.g., W_down for SwiGLU)
3. Extracts weights; normalizes (L2 norm per row)
4. Returns as dict mapping layer index to Tensor

**Example**:
```python
from hopfield_llm.memory.banks import extract_banks

banks = extract_banks(llm, bank="down")
print(len(banks))  # 36 (num_layers)
print(banks[0].shape)  # torch.Size([3072, 1024]) for Qwen
```

---

### `queries/` — Query Extraction Strategies

**Purpose**: Select which activation to use as the retrieval query in Hopfield computation.

**Key exports**:
- `last_token(activations[T, d])` → `[d]` — Last token activation (default)
- `mean_tokens(activations[T, d])` → `[d]` — Mean over all tokens
- `make_positional(idx)(activations[T, d])` → `[d]` — Token at specific index

**Example**:
```python
from hopfield_llm.queries.extractors import last_token, mean_tokens
from hopfield_llm.hooks.capture import capture_prefill

# Use mean-tokens as query
result = capture_prefill(
    llm,
    "Who wrote Hamlet?",
    banks,
    query_fn=mean_tokens,
)
```

---

### `metrics/` — Energy, Entropy, Divergences

**Purpose**: Compute Modern Hopfield energy, entropy, and distributional divergences.

**Key exports**:
- `compute_energy(query[d], memory[K, d], beta=15.0, mode="dot")` → `float` — Energy value
- `compute_entropy(energy[L], ...)` → `float` — Shannon entropy of Hopfield distribution
- `compute_kl_divergence(p, q)` → `float` — KL(p||q)
- `compute_js_divergence(p, q)` → `float` — Jensen-Shannon divergence
- `compute_hellinger_distance(p, q)` → `float` — Hellinger distance
- `hallucination_score(deltas[L])` → `float` — Aggregate per-layer signal

**Example**:
```python
from hopfield_llm.metrics.energy import compute_energy
import torch

query = torch.randn(1024)
memory = torch.randn(3072, 1024)
energy = compute_energy(query, memory, beta=15.0, mode="dot")
print(energy)  # scalar tensor
```

---

### `datasets/` — Data Loaders

**Purpose**: Abstract interface for Q&A datasets; adapters for TruthfulQA, TriviaQA, Natural Questions.

**Key exports**:
- `BaseDataset` (ABC) — Implement `__len__`, `__iter__`, `name` property
- `DataSample(id, source, question, gold_answers, generated_text, ...)` — Data record
- `TruthfulQA`, `TriviaQA`, `NaturalQuestions` — Concrete adapters

**DataSample fields**:
- Required: `id` (str), `question` (str), `gold_answers` (list[str])
- Generated: `generated_text` (str, optional)
- Labeling fields: `is_hallucination` (bool, optional), `correctness_score` (float, optional)

**Example**:
```python
from hopfield_llm.datasets.truthfulqa import TruthfulQA

dataset = TruthfulQA(max_samples=100, seed=42)
for sample in dataset:
    print(f"{sample.id}: {sample.question}")
```

---

### `pipeline/` — Stage Orchestration

**Purpose**: Implement CLI stages (build-banks, run-trajectory, analyze) and configuration management.

**Key exports**:
- `build_banks(model_alias, bank, output, load_in_4bit)` — Stage 1
- `run_trajectory(model_alias, dataset_name, banks_path, output_dir, ...)` — Stage 2
- `analyze_trajectories(traj_dir, output, score_metric, score_aggregation)` — Stage 3
- `ExperimentConfig` — Flat dataclass from hierarchical YAML
- `load_experiment_config(yaml_path)` → `ExperimentConfig`

**Example**:
```python
from hopfield_llm.pipeline.config import load_experiment_config
from hopfield_llm.pipeline.stages import build_banks, run_trajectory

cfg = load_experiment_config("configs/experiments/final_qwen_truthfulqa.yaml")
build_banks(cfg.model.alias, cfg.trajectory.bank, "banks.pt", cfg.model.load_in_4bit)
```

---

### `storage/` — Artifact I/O

**Purpose**: Persist and load banks, trajectories, and analysis results.

**Key exports**:
- `save_torch_artifact(artifact_dict, path)` — Save via `torch.save`
- `load_torch_artifact(path)` → dict
- `save_json_artifact(data, path)` — Save via `json.dump`
- `load_json_artifact(path)` → dict

**Artifact schemas**:
- `banks.pt`: `{"artifact_type": "banks", "model": {...}, "banks": {layer → Tensor}}`
- `{sample_id}.npz`: numpy compressed array with energy/entropy/token_ids
- `{sample_id}.json`: metadata (question, gold_answers, n_layers, n_tokens_generated)
- `analysis.json`: global divergences + per-layer metrics

---

### `visualization/` — Plotting

**Purpose**: Generate matplotlib plots from analysis JSON.

**Key exports**:
- `visualize_analysis(analysis_json_path, output_dir)` — Generate PNG plots

**Outputs**:
- Layer metrics plots (delta_energy, delta_entropy per layer)
- Divergence histograms (KL, JS, Hellinger distributions)
- Global score distributions

**Example**:
```python
from hopfield_llm.visualization.plots import visualize_analysis

visualize_analysis("outputs/final_qwen_truthfulqa/analysis.json", "plots/")
```

---

## Labeling & Evaluation

### `labeling/` — Hallucination Classification

**Purpose**: Two strategies for labeling samples as hallucinated or correct.

#### Heuristic Labeling (SQuAD F1)

**Key exports**:
- `compute_f1_score(prediction, gold_answers[])` → float — SQuAD-style token F1
- `label_sample_heuristic(sample, threshold=0.5)` → DataSample (with is_hallucination flag)
- `compute_cohen_kappa(labels1, labels2)` → float — Agreement metric for calibration

**How it works**:
1. Normalize prediction and gold answers (lowercase, strip articles, remove punctuation)
2. Tokenize and compute token-level precision, recall, F1
3. Return max F1 across all gold answers
4. Label as hallucination if F1 < threshold

**Threshold guidance**:
- 0.3: high recall (more samples marked hallucination)
- 0.5: balanced (default)
- 0.7: high precision (fewer hallucinations)

**Example**:
```python
from hopfield_llm.labeling.heuristic import label_sample_heuristic

labeled = label_sample_heuristic(sample, threshold=0.5)
print(f"Hallucination: {labeled.is_hallucination}")
print(f"F1 score: {labeled.correctness_score:.3f}")
```

#### LLM-Judge Labeling

**Key exports**:
- `LLMJudge(api_base=None, model_id, load_in_4bit=False)` — Judge instance
- `build_judge_prompt(question, generated_text, gold_answers, template_path)` → str
- `label_sample_with_judge(sample, judge)` → DataSample (with is_hallucination flag)

**Two backends**:
1. **OpenAI-compatible API** (vLLM, OpenAI): `api_base="http://localhost:8000/v1"`
2. **Local HuggingFace model**: `api_base=None` (air-gapped HPC)

**Example**:
```python
from hopfield_llm.labeling.llm_judge import LLMJudge, label_sample_with_judge

judge = LLMJudge(
    api_base="http://localhost:8000/v1",
    model_id="qwen25_7b",
)

labeled = label_sample_with_judge(sample, judge)
print(f"Judge decision: {labeled.is_hallucination}")  # 0 or 1
```

---

### `analysis/` — Token-Level Feature Extraction

**Purpose**: Align gold answers in generated text; extract per-layer metrics over answer spans.

**Key exports**:
- `find_answer_span(generated_text, gold_answers, tokenizer)` → `(tok_start, tok_end) | None`
- `content_token_mask(tokens)` → bool[T] — Mask excluding leading/trailing whitespace
- `extract_answer_features(metric_array[L, T], span, tokenizer, generated_text)` → `[L]` — Mean-pool over span

**How it works**:
1. Search for first gold answer substring in generated text (case-insensitive)
2. Convert character offsets to token indices
3. Extract [L, T] metrics; compute mean over answer span
4. Fall back to content tokens if span not found

**Example**:
```python
from hopfield_llm.analysis.alignment import find_answer_span, extract_answer_features

span = find_answer_span("Shakespeare wrote Hamlet.", ["Shakespeare"], tokenizer)
if span:
    start, end = span
    # energy shape: [L, T]
    answer_energy = extract_answer_features(energy, span, tokenizer, generated_text)
    print(answer_energy.shape)  # (L,) — per-layer mean energy over answer
```

---

### `evaluation/` — AUROC, Probes, Reports

**Purpose**: Comprehensive evaluation pipeline combining per-layer AUROC, multi-layer probe, and HTML reporting.

#### Features Loading

**Key exports**:
- `load_all_features(artifact_dir, labels_dir)` → list[dict] — Load trajectory + label artifacts

**Feature dict keys**:
- `is_hallucination` (bool) — Label from M3 stage
- `delta_energy[L]` — Difference between generation and prefill energy
- `delta_entropy[L]` — Per-layer entropy difference
- `seq_logprob_mean` (float) — Baseline: mean log-probability
- `token_entropy_mean` (float) — Baseline: mean token entropy
- ... (many more per-layer features)

#### Per-Layer AUROC

**Key exports**:
- `per_layer_auroc_table(samples, per_layer_feature_keys)` → DataFrame — AUROC per (layer, feature)
- `best_single_feature(auroc_df)` → (feature_key, layer, auroc) — Global best cell

**Output**: DataFrame with layers as rows, features as columns, AUROC values, plus 'best_layer' row.

**Example**:
```python
from hopfield_llm.evaluation.per_layer_auroc import per_layer_auroc_table

auroc_df = per_layer_auroc_table(
    samples,
    ["delta_energy", "delta_entropy", "gen_entropy_answer"],
)
print(auroc_df.loc[[0, 10, 20], :])  # Layers 0, 10, 20
```

#### Logistic Regression Probe

**Key exports**:
- `fit_logreg_probe(samples, feature_keys, n_splits=5)` → dict — Probe results

**How it works**:
1. Build feature matrix: concatenate [L] arrays from all feature_keys → [N, n_keys*L]
2. Standardize and impute NaNs
3. Fit logistic regression with stratified K-fold CV
4. Inner CV loop tunes C parameter
5. Return mean/std AUROC and per-fold details

**Result dict**:
```python
{
  "mean_auroc": 0.85,
  "std_auroc": 0.03,
  "per_fold": [0.84, 0.86, 0.85, 0.84, 0.86],
  "n_features": 3 * 36,  # 3 features * 36 layers
  "best_c": 1.0,
}
```

#### Beta Sweep

**Key exports**:
- `beta_sweep_auroc(samples_with_scores, betas=[1.0, 5.0, 10.0, ...])` → DataFrame — Best-layer AUROC per beta

**How it works**:
1. Loads cached raw dot-product scores [L, T, K] from samples
2. Recomputes energy/entropy at multiple beta values
3. Computes per-layer AUROC for each beta
4. Returns best-layer AUROC per beta

**Example**:
```python
from hopfield_llm.evaluation.beta_sweep import beta_sweep_auroc

auroc_by_beta = beta_sweep_auroc(
    samples,
    betas=[1.0, 5.0, 10.0, 15.0, 25.0, 50.0],
)
print(auroc_by_beta)  # {beta: auroc}
```

#### HTML Report Builder

**Key exports**:
- `build_eval_report(artifact_dir, labels_dir, output_path)` → None — Write HTML report

**Report contents**:
- Per-layer AUROC table (best feature per layer)
- Best single-feature cell highlighted
- Logistic regression probe AUROC (mean ± std, per-fold)
- Beta sweep curve (AUROC vs beta)
- Baseline comparison (seq_logprob, token entropy)
- Sample statistics

**Example**:
```python
from hopfield_llm.evaluation.report import build_eval_report

build_eval_report(
    artifact_dir="outputs/final_qwen_truthfulqa/trajectories",
    labels_dir="outputs/final_qwen_truthfulqa/trajectories",
    output_path="outputs/final_qwen_truthfulqa/report.html",
)
```

---

## Utilities & Infrastructure

### `utils/` — Logging, Configuration, Experiment Tracking

**Key exports**:
- `setup_logging(name, level="INFO")` — Configure structured logger
- `load_env_config()` → EnvConfig — Auto-detect local vs HPC
- `ExperimentTracker(experiment_name)` → Manager — Create timestamped run directory

**ExperimentTracker**:
- Creates: `outputs/{name}/{timestamp}_{model}_{dataset}_{bank}_{beta}_{hash}/`
- Saves: config.yaml, git_info.json, environment.yaml, metrics_log.json
- Provides: run_dir property, timing context manager

**Example**:
```python
from hopfield_llm.utils.tracking import ExperimentTracker

tracker = ExperimentTracker("final_qwen_truthfulqa")
print(tracker.run_dir)  # outputs/final_qwen_truthfulqa/20260101_1345_qwen25_3b_truthfulqa_gated_key_b15.0_0a1b2c3d/

with tracker.stage_timing("build-banks"):
    # run stage
    pass
```

---

### `cli/` — CLI Entry Point

**Key exports**:
- `build_parser()` → ArgumentParser
- `main(argv=None)` → int — CLI entry point

**5 core subcommands**:
1. `build-banks` — Extract memory banks
2. `run-trajectory` — Capture activations
3. `analyze` — Aggregate divergences
4. `visualize` — Generate plots
5. `run-experiment` — Orchestrate all stages

**Distributed support**:
- `--shard-id N` and `--num-shards M` for stage 2 (run-trajectory)

**Example**:
```bash
hopfield-llm run-experiment \
    --config configs/experiments/final_qwen_truthfulqa.yaml \
    --override beta=20.0 \
    --skip-banks
```

---

## Data Flow Recap

```
┌─ models.HFLLM
│  └─ Load AutoModelForCausalLM with 4-bit quantization
│
├─ memory.extract_banks
│  └─ Extract MLP weight matrices → {layer → Tensor[K, D]}
│
├─ hooks.capture_prefill / capture_generation
│  ├─ Register forward-pre-hooks
│  ├─ Run forward pass(es)
│  ├─ Intercept h_pre; apply query extraction
│  ├─ Compute energy/entropy via metrics
│  └─ Return PrefillResult or GenerationResult
│
├─ datasets + pipeline.run_trajectory
│  ├─ Load dataset; iterate samples
│  ├─ Run prefill + generation for each
│  ├─ Save .npz (energy/entropy) + .json (metadata)
│  └─ Accumulate per-sample artifacts
│
├─ labeling (heuristic or llm_judge)
│  ├─ Read generated_text + gold_answers
│  ├─ Compute F1 or obtain LLM judgment
│  └─ Save {sample_id}_label.json with is_hallucination flag
│
├─ pipeline.analyze_trajectories
│  ├─ Read .npz files; compute divergences
│  ├─ Aggregate per-sample metrics
│  └─ Save analysis.json
│
├─ evaluation.features.load_all_features
│  ├─ Load trajectory .npz + label .json
│  ├─ Extract baseline metrics
│  └─ Bridge to unified feature dicts
│
├─ evaluation.per_layer_auroc_table + probe + beta_sweep
│  ├─ Compute per-layer AUROC per feature
│  ├─ Fit multi-layer logistic regression
│  ├─ Sweep beta parameter on cached scores
│  └─ Return metrics
│
└─ evaluation.report.build_eval_report
   └─ Generate HTML with tables and plots
```
