# Quick Reference

One-page reference for common CLI commands and workflows.

## Installation

```bash
# Local (Linux with GPU)
micromamba create -n hopfield-llm python=3.12 -y
micromamba activate hopfield-llm
pip install -r requirements.txt && pip install -e .

# HPC (SLURM cluster)
bash setup_hpc.sh
```

## Verify Installation

```bash
hopfield-llm --help
python -c "from hopfield_llm.models.loader import HFLLM; print('✓ OK')"
```

---

## Core Commands

### Stage 1: Extract Memory Banks

```bash
hopfield-llm build-banks \
    --model qwen25_3b \
    --bank down \
    --output banks.pt
```

**Bank options**: `gate`, `up`, `down`, `down_values`, `gate_proj`, `up_proj`, `down_proj`, `fc1`, `fc2`, `w1`, `w2`, `w3`

### Stage 2: Capture Activations

```bash
hopfield-llm run-trajectory \
    --model qwen25_3b \
    --dataset truthfulqa \
    --banks banks.pt \
    --output traj/ \
    --beta 15.0
```

**Dataset options**: `truthfulqa`, `triviaqa`, `nq`

**Energy mode**: `--energy-mode dot` (default) or `cosine`

**Distributed (HPC)**: Add `--shard-id 0 --num-shards 8` (run on 8 nodes, each gets 1/8 of samples)

### Stage 3: Aggregate Analysis

```bash
hopfield-llm analyze \
    --trajectories traj/ \
    --output analysis.json \
    --score-metric delta_energy
```

**Score metric options**: `delta_energy`, `delta_entropy`, `delta_norm_entropy`, `delta_top_activation`, `delta_mean_activation`, `kl_gen_to_prompt_per_layer`, `js_gen_to_prompt_per_layer`, `hellinger_gen_to_prompt_per_layer`, `energy_shift_l1`, `energy_shift_l2`, `peak_delta_layer`, `gen_drift_mean`, `gen_drift_max`

### Stage 4: Generate Plots

```bash
hopfield-llm visualize \
    --analysis analysis.json \
    --output plots/
```

---

## Full Pipeline (Single Command)

### From YAML Config

```bash
hopfield-llm run-experiment \
    --config configs/experiments/exp01_truthfulqa_baseline.yaml
```

**With overrides**:
```bash
hopfield-llm run-experiment \
    --config configs/experiments/exp01_truthfulqa_baseline.yaml \
    --override beta=20.0 dataset_name=triviaqa max_samples=100
```

**Skip bank recomputation**:
```bash
hopfield-llm run-experiment \
    --config configs/experiments/exp01_truthfulqa_baseline.yaml \
    --skip-banks
```

---

## Python API

### Quick Inference

```python
from hopfield_llm.models.loader import HFLLM
from hopfield_llm.memory.banks import extract_banks
from hopfield_llm.hooks.capture import capture_prefill, capture_generation

# Load
llm = HFLLM("qwen25_3b", load_in_4bit=True)
banks = extract_banks(llm, bank="down")

# Prefill
prefill = capture_prefill(llm, "Who wrote Hamlet?", banks, beta=15.0)
print(prefill.energy)  # [L]

# Generation
gen = capture_generation(llm, "Who wrote Hamlet?", banks, max_new_tokens=30)
print(gen.generated_text)
print(gen.energy)  # [L, T]
```

### Label Samples

```python
# Heuristic (F1-based)
from hopfield_llm.labeling.heuristic import label_sample_heuristic
labeled = label_sample_heuristic(sample, threshold=0.5)
print(labeled.is_hallucination)

# LLM Judge
from hopfield_llm.labeling.llm_judge import LLMJudge, label_sample_with_judge
judge = LLMJudge(api_base="http://localhost:8000/v1", model_id="qwen25_7b")
labeled = label_sample_with_judge(sample, judge)
```

### Evaluate

```python
from hopfield_llm.evaluation.features import load_all_features
from hopfield_llm.evaluation.per_layer_auroc import per_layer_auroc_table
from hopfield_llm.evaluation.probe import fit_logreg_probe
from hopfield_llm.evaluation.report import build_eval_report

# Load features
samples = load_all_features("outputs/exp01/trajectories")

# Per-layer AUROC
auroc_df = per_layer_auroc_table(samples, ["delta_energy", "delta_entropy"])
print(auroc_df)

# Probe
probe = fit_logreg_probe(samples, ["delta_energy", "delta_entropy"], n_splits=5)
print(f"AUROC: {probe['mean_auroc']:.3f} ± {probe['std_auroc']:.3f}")

# HTML Report
build_eval_report("outputs/exp01/trajectories", "outputs/exp01/trajectories", "report.html")
```

---

## Common Workflows

### Quick Test (small model, few samples)

```bash
hopfield-llm run-experiment \
    --config configs/experiments/exp01_truthfulqa_baseline.yaml \
    --override model.alias=qwen25_1_5b dataset.max_samples=50
```

### Full TruthfulQA Experiment

```bash
# Create config
cat > config_tqa.yaml <<'EOF'
experiment:
  name: tqa_full
model:
  alias: qwen25_3b
  load_in_4bit: true
dataset:
  name: truthfulqa
  max_samples: null
trajectory:
  bank: down
  beta: 15.0
analysis:
  score_metric: delta_energy
output:
  base_dir: outputs
EOF

# Run
hopfield-llm run-experiment --config config_tqa.yaml
```

### HPC Submission (distributed)

```bash
# Edit environment variables (optional)
export MODEL=qwen25_3b
export DATASET=truthfulqa
export NUM_SHARDS=8

# Submit
./scripts/hpc/submit_pipeline.sh
```

### Beta Parameter Sweep

```bash
# Run multiple betas (reuses banks)
for beta in 1.0 5.0 10.0 15.0 25.0 50.0; do
    hopfield-llm run-experiment \
        --config config_tqa.yaml \
        --override trajectory.beta=$beta \
        --skip-banks
done

# Evaluate with sweep
python -c "
from hopfield_llm.evaluation.features import load_all_features
from hopfield_llm.evaluation.beta_sweep import beta_sweep_auroc

samples = load_all_features('outputs/tqa_full/.../trajectories')
aurocs = beta_sweep_auroc(samples, betas=[1.0, 5.0, 10.0, 15.0, 25.0, 50.0])
print(aurocs)
"
```

### Memory Bank Comparison

```bash
# Build multiple banks
for bank in gate up down; do
    hopfield-llm build-banks --model qwen25_3b --bank $bank --output banks_$bank.pt
    hopfield-llm run-trajectory --model qwen25_3b --dataset truthfulqa --banks banks_$bank.pt --output traj_$bank/
    hopfield-llm analyze --trajectories traj_$bank/ --output analysis_$bank.json
done

# Compare analysis results manually or in Python
```

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `HOPFIELD_ENV` | auto-detect | Force `local` or `hpc` environment |
| `CUDA_VISIBLE_DEVICES` | all | Restrict visible GPUs (e.g., "0,1") |
| `HF_TOKEN` | none | HuggingFace API token (for gated models) |
| `HOPFIELD_SCRATCH` | $SCRATCH (HPC) | Cache directory for large files |

**Example**: 
```bash
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
hopfield-llm build-banks --model qwen25_3b --output banks.pt
```

---

## File Organization

```
outputs/
  {experiment_name}/
    {timestamp}_{model}_{dataset}_{bank}_{beta}_{hash}/
      config.yaml              ← Full configuration
      git_info.json            ← Git commit/branch
      environment.yaml         ← GPU, VRAM, timing
      metrics_log.json         ← Stage completion times
      trajectories/
        {sample_id}.npz        ← Energy/entropy arrays
        {sample_id}.json       ← Metadata (question, gold answers, generated text)
        {sample_id}_label.json ← Labeling result (is_hallucination)
      analysis.json            ← Aggregated divergences and per-layer metrics
      plots/
        *.png                  ← Layer metric plots
      report.html              ← Evaluation report
```

---

## Supported Models

| Alias | Model | Params | VRAM (4-bit) | Local | HPC |
|-------|-------|--------|-------------|-------|-----|
| `qwen25_1_5b` | Qwen2.5-1.5B | 1.5B | 1.2 GB | ✓ | ✓ |
| `qwen25_3b` | Qwen2.5-3B | 3.0B | 2.0 GB | ✓ | ✓ |
| `llama32_3b` | Llama-3.2-3B | 3.2B | 2.1 GB | ~ | ✓ |
| `gemma2_2b` | Gemma-2-2B | 2.5B | 1.8 GB | ✓ | ✓ |
| `qwen25_7b` | Qwen2.5-7B | 7.6B | 4.5 GB | ✗ | ✓ |
| `llama31_8b` | Llama-3.1-8B | 8.0B | 5.0 GB | ✗ | ✓ |
| `gemma2_9b` | Gemma-2-9B | 9.2B | 5.5 GB | ✗ | ✓ |

Legend: ✓ = Confirmed working, ~ = Tight fit, ✗ = OOM expected

---

## Troubleshooting

### CUDA Out of Memory

- Reduce model size: `--model qwen25_1_5b`
- Disable 4-bit: `--no-4bit` (will use fp16, need more VRAM)
- Reduce max_samples: `--override dataset.max_samples=10`
- Check GPU: `python -c "import torch; print(torch.cuda.get_device_name(0))"`

### Model Not Found

- Check alias: `python -c "from hopfield_llm.models.profiles import PROFILES; print(PROFILES.keys())"`
- Set HF token: `export HF_TOKEN=hf_xxx` (for gated models)
- Manual download: `huggingface-cli download Qwen/Qwen2.5-3B-Instruct`

### HPC Environment Not Detected

- Force HPC mode: `export HOPFIELD_ENV=hpc`
- Check SLURM: `sinfo` (should show cluster)
- Verify Python: `python -c "from hopfield_llm.utils.env import load_env_config; print(load_env_config().environment)"`

---

## Documentation

- **[Architecture](docs/architecture.md)** — System design, module dependencies, data structures
- **[Usage](docs/usage.md)** — CLI commands, Python API, common patterns
- **[Setup](docs/setup.md)** — Installation for local and HPC
- **[Modules](docs/modules.md)** — Complete module reference
- **[CLAUDE.md](CLAUDE.md)** — Developer notes for this repo
