# Usage

## CLI — Core Pipeline Stages (GPU)

```bash
# Stage 1: extract MLP memory banks
hopfield-llm build-banks \
    --model qwen25_3b \
    --bank down \                    # gate | up | down | down_values | gate_proj | up_proj | down_proj | fc1 | fc2 | w1 | w2 | w3
    --output banks.pt

# Stage 2: capture activations and compute energy/entropy
hopfield-llm run-trajectory \
    --model qwen25_3b \
    --dataset truthfulqa \
    --banks banks.pt \
    --output traj/ \
    --beta 15.0 \                    # Hopfield inverse-temperature
    --energy-mode dot \              # or: cosine
    --max-new-tokens 50
```

## CLI — Analysis & Visualization (CPU)

```bash
# Stage 3: aggregate trajectories into analysis JSON
hopfield-llm analyze \
    --trajectories traj/ \
    --output analysis.json \
    --score-metric delta_energy      # or: delta_entropy, delta_norm_entropy, delta_top_activation, delta_mean_activation, kl_gen_to_prompt_per_layer, js_gen_to_prompt_per_layer, hellinger_gen_to_prompt_per_layer, energy_shift_l1, energy_shift_l2, peak_delta_layer, gen_drift_mean, gen_drift_max

# Stage 4: generate plots from analysis JSON
hopfield-llm visualize \
    --analysis analysis.json \
    --output plots/
```

## CLI — Orchestration (Full Pipeline)

```bash
# Run all stages from a YAML config with full experiment tracking
hopfield-llm run-experiment \
    --config configs/experiments/exp01_truthfulqa_baseline.yaml \
    --override beta=20.0 dataset_name=triviaqa \
    --skip-banks                     # Reuse existing banks.pt in run directory
```

## Distributed Execution (HPC/SLURM)

Stage 2 supports stride-based sharding for multi-node execution:

```bash
# Node 0
hopfield-llm run-trajectory \
    --model qwen25_3b \
    --dataset truthfulqa \
    --banks banks.pt \
    --output traj/ \
    --shard-id 0 \
    --num-shards 8

# Node 1
hopfield-llm run-trajectory \
    ... --shard-id 1 --num-shards 8

# After all shards complete, run analysis (CPU-only, runs once)
hopfield-llm analyze --trajectories traj/ --output analysis.json
```

Submit via: `./scripts/hpc/submit_pipeline.sh`

---

## Labeling Samples for Hallucination Detection

After running `run-trajectory`, label each sample with a binary `is_hallucination` flag:

### Heuristic: F1-based (fast, no API calls)

```python
from hopfield_llm.labeling.heuristic import label_sample_heuristic
from hopfield_llm.datasets.base import DataSample

sample = DataSample(
    id="truthfulqa_0",
    question="Who wrote Hamlet?",
    gold_answers=["William Shakespeare"],
    generated_text="Shakespeare penned Hamlet.",
)

labeled = label_sample_heuristic(sample, threshold=0.5)
print(labeled.is_hallucination)  # False (F1 > threshold)
print(labeled.correctness_score)  # F1 score
```

**Threshold guidance:**
- `threshold=0.5` (default): moderate sensitivity
- `threshold=0.3`: higher recall, more hallucinations labeled
- `threshold=0.7`: higher precision, fewer hallucinations labeled

### LLM-Judge: External LLM (higher accuracy, requires API or local model)

```python
from hopfield_llm.labeling.llm_judge import LLMJudge, label_sample_with_judge

# OpenAI-compatible API (e.g., vLLM, OpenAI)
judge = LLMJudge(
    api_base="http://localhost:8000/v1",
    model_id="qwen25_7b",
)

# Or local HuggingFace model (air-gapped HPC)
judge = LLMJudge(
    model_id="meta-llama/Llama-2-7b",
    load_in_4bit=True,
)

sample = DataSample(
    id="truthfulqa_0",
    question="Who wrote Hamlet?",
    gold_answers=["William Shakespeare"],
    generated_text="Shakespeare penned Hamlet.",
)

labeled = label_sample_with_judge(sample, judge)
print(labeled.is_hallucination)  # 0 (correct) or 1 (hallucination)
print(labeled.llm_judge_raw)     # Full judge response for custom parsing
```

---

## Evaluation Pipeline: AUROC, Probe, Beta Sweep

After labeling, compute feature matrices and evaluation metrics:

### Step 1: Load features (bridge trajectory + label artifacts)

```python
from hopfield_llm.evaluation.features import load_all_features

samples = load_all_features(
    artifact_dir="outputs/exp01/trajectories",  # .npz + .json files
    labels_dir="outputs/exp01/trajectories",    # {sample_id}_label.json files
)

print(f"Loaded {len(samples)} samples")
print(f"Keys: {samples[0].keys()}")  # is_hallucination, per-layer features, baselines
```

### Step 2: Compute per-layer AUROC table

```python
from hopfield_llm.evaluation.per_layer_auroc import per_layer_auroc_table, best_single_feature

auroc_df = per_layer_auroc_table(
    samples=samples,
    per_layer_feature_keys=["delta_energy", "delta_entropy", "gen_entropy_answer"],
)

print(auroc_df)
feature, layer, auroc = best_single_feature(auroc_df)
print(f"Best single feature: {feature} at layer {layer}, AUROC={auroc:.3f}")
```

### Step 3: Fit logistic regression probe (multi-layer ensemble)

```python
from hopfield_llm.evaluation.probe import fit_logreg_probe

probe_result = fit_logreg_probe(
    samples=samples,
    feature_keys=["delta_energy", "delta_entropy", "delta_top_activation"],
    n_splits=5,
)

print(f"Probe AUROC: {probe_result['mean_auroc']:.3f} ± {probe_result['std_auroc']:.3f}")
print(f"Per-fold: {probe_result['per_fold']}")
```

### Step 4: Beta sweep (post-hoc parameter tuning)

```python
from hopfield_llm.evaluation.beta_sweep import beta_sweep_auroc

# Requires cached score tensors from run-trajectory
auroc_by_beta = beta_sweep_auroc(
    samples_with_scores=samples,
    betas=[1.0, 5.0, 10.0, 15.0, 25.0, 50.0],
)

print(auroc_by_beta)  # {beta: {feature: auroc}}
```

### Step 5: HTML evaluation report

```python
from hopfield_llm.evaluation.report import build_eval_report

build_eval_report(
    artifact_dir="outputs/exp01/trajectories",
    labels_dir="outputs/exp01/trajectories",
    output_path="outputs/exp01/report.html",
)

print("Report written to outputs/exp01/report.html")
```

---

## Customizing Memory Banks

The `--bank` flag controls which MLP projection forms the memory:

```bash
# W_down (down-projection, default)
hopfield-llm build-banks --model qwen25_3b --bank down --output banks_down.pt

# W_gate (gate projection)
hopfield-llm build-banks --model qwen25_3b --bank gate --output banks_gate.pt

# W_up (up projection)
hopfield-llm build-banks --model qwen25_3b --bank up --output banks_up.pt
```

Banks are stored as a dict `{layer_idx → Tensor[K, D]}`.  All downstream modules (hooks, metrics, analysis) are unaffected by this choice.

---

## Customizing Query Extraction

The query extractor selects which activation position becomes the retrieval query. Pass a callable `[T, d_m] → [d_m]` as `query_fn`:

```python
from hopfield_llm.hooks.capture import capture_prefill
from hopfield_llm.queries.extractors import mean_tokens, last_token, make_positional

# Default: last question token
result = capture_prefill(llm, question, banks, query_fn=last_token)

# Alternative: mean over all question tokens
result = capture_prefill(llm, question, banks, query_fn=mean_tokens)

# Alternative: a specific token index (e.g., first token)
result = capture_prefill(llm, question, banks, query_fn=make_positional(0))
```

---

## Python API — Core Inference

```python
from hopfield_llm.models.loader import HFLLM
from hopfield_llm.memory.banks import extract_banks
from hopfield_llm.hooks.capture import capture_prefill, capture_generation

# Load model
llm = HFLLM("qwen25_3b", load_in_4bit=True)

# Build memory banks
banks = extract_banks(llm, bank="down")

# Prefill pass (single query)
prefill = capture_prefill(
    llm,
    "Who wrote Hamlet?",
    banks,
    beta=15.0,
    energy_mode="dot",  # or "cosine"
)
print(prefill.energy)  # [L] float32
print(prefill.entropy) # [L] float32

# Generation pass (full trajectory)
gen = capture_generation(
    llm,
    "Who wrote Hamlet?",
    banks,
    beta=15.0,
    max_new_tokens=30,
)
print(gen.generated_text)
print(gen.energy)      # [L, T] float32
print(gen.entropy)     # [L, T] float32
```

## Python API — Pipeline Orchestration

```python
from hopfield_llm.pipeline.stages import build_banks, run_trajectory
from hopfield_llm.pipeline.analysis import analyze_trajectories
from hopfield_llm.pipeline.config import load_experiment_config
from hopfield_llm.utils.tracking import ExperimentTracker

# Load configuration
cfg = load_experiment_config("configs/experiments/exp01_truthfulqa_baseline.yaml")

# Initialize experiment tracker
tracker = ExperimentTracker(cfg.experiment.name)

# Stage 1: Build banks
banks_path = tracker.run_dir / "banks.pt"
build_banks(
    model_alias=cfg.model.alias,
    bank=cfg.trajectory.bank,
    output=banks_path,
    load_in_4bit=cfg.model.load_in_4bit,
)

# Stage 2: Run trajectory
traj_dir = tracker.run_dir / "trajectories"
run_trajectory(
    model_alias=cfg.model.alias,
    dataset_name=cfg.dataset.name,
    banks_path=banks_path,
    output_dir=traj_dir,
    beta=cfg.trajectory.beta,
    max_samples=cfg.dataset.max_samples,
)

# Stage 3: Analyze
analysis_path = tracker.run_dir / "analysis.json"
analyze_trajectories(
    traj_dir=traj_dir,
    output=analysis_path,
    score_metric=cfg.analysis.score_metric,
)
```

---

## Adding a New Dataset

Subclass `BaseDataset` and yield `DataSample` objects:

```python
from hopfield_llm.datasets.base import BaseDataset, DataSample
from typing import Iterator

class MyDataset(BaseDataset):
    def __init__(self, max_samples=None, seed=42):
        super().__init__(max_samples=max_samples, seed=seed)
        self._samples = []  # Load your data here

    @property
    def name(self) -> str:
        return "mydataset"

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[DataSample]:
        return iter(self._samples)
```

Then register it in `src/hopfield_llm/pipeline/stages.py` inside `run_trajectory`:

```python
from hopfield_llm.datasets.mydataset import MyDataset

dataset_map = {
    "truthfulqa": TruthfulQA,
    "triviaqa": TriviaQA,
    "nq": NaturalQuestions,
    "mydataset": MyDataset,  # Add here
}
```

---

## Experiment Tracking

`ExperimentTracker` creates timestamped run directories:

```
outputs/{name}/{timestamp}_{model}_{dataset}_{bank}_{beta}_{hash}/
    config.yaml              — full config snapshot
    git_info.json            — commit, branch, dirty flag
    environment.yaml         — GPU name, VRAM, start/end time
    metrics_log.json         — stage completion timestamps
    trajectories/            — per-sample .npz + .json
    labels/                  — {sample_id}_label.json
    analysis.json            — divergences, per-layer metrics
    plots/                   — PNG visualizations
    report.html              — evaluation report (if generated)
```

To inspect a previous run:

```python
from hopfield_llm.utils.tracking import ExperimentTracker
import json

# Load metadata
with open("outputs/exp01/.../git_info.json") as f:
    git_info = json.load(f)
print(f"Commit: {git_info['commit']}")

# Load config
with open("outputs/exp01/.../config.yaml") as f:
    config = f.read()
```

---

## Security: Hugging Face Tokens

Models are downloaded from the HuggingFace Hub on first use. For private models (e.g., gated Llama 3), provide your access token:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Never hardcode tokens in configs or scripts. The `.env` file is gitignored by default. See `.env.example` for format.

---

## Common Workflows

### Baseline: TruthfulQA with Qwen 2.5-3B

```bash
# Create experiment config
cat > configs/experiments/truthfulqa_baseline.yaml <<'EOF'
experiment:
  name: baseline_tqa
model:
  alias: qwen25_3b
  load_in_4bit: true
dataset:
  name: truthfulqa
  max_samples: null
  seed: 42
trajectory:
  bank: down
  beta: 15.0
  energy_mode: dot
analysis:
  score_metric: delta_energy
  score_aggregation: mean
output:
  base_dir: outputs
EOF

# Run full pipeline
hopfield-llm run-experiment --config configs/experiments/truthfulqa_baseline.yaml
```

### Reproduction: Sweep Beta Parameter

```bash
# Run multiple beta values
for beta in 1.0 5.0 10.0 15.0 25.0 50.0; do
    hopfield-llm run-experiment \
        --config configs/experiments/truthfulqa_baseline.yaml \
        --override beta=$beta
done

# Then evaluate with beta sweep (from cached scores)
python -c "
from hopfield_llm.evaluation.features import load_all_features
from hopfield_llm.evaluation.beta_sweep import beta_sweep_auroc

samples = load_all_features('outputs/baseline_tqa/.../trajectories')
auroc_by_beta = beta_sweep_auroc(samples, betas=[1.0, 5.0, 10.0, 15.0, 25.0, 50.0])
print(auroc_by_beta)
"
```
