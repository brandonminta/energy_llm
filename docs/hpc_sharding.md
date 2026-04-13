# HPC Parallel Extraction with Sharding

The Stage 2 (hidden state extraction) is the bottleneck — it's the only stage that runs
forward passes on GPU. This document explains how to parallelize it using SLURM array jobs.

## Quick Start

```bash
# One command does everything: shard, submit array job, merge, score, analyze, visualize
./scripts/hpc_parallel_pipeline.sh \
    --model qwen25_3b \
    --dataset data/interim/prompts/truthfulqa_paired.json \
    --num-shards 10
```

**Timeline comparison:**

| Approach | Time | Notes |
|----------|------|-------|
| Sequential (1 GPU) | ~120 min | `./jobs/submit_pipeline.sh` |
| Sharded (10 GPUs) | ~15 min | Stage 2 parallelized, stages 3–5 sequential |
| Speedup | **8×** | Limited by merge + stages 3–5 |

## How It Works

### 1. Split Prompts into Shards

```bash
python scripts/shard_prompts.py \
    data/interim/prompts/truthfulqa_paired.json \
    --output-dir data/interim/prompts/shards \
    --num-shards 10
```

Produces:
```
data/interim/prompts/shards/
  shard_000.json  (82 prompts)
  shard_001.json  (82 prompts)
  ...
  shard_009.json  (82 prompts)
```

### 2. Submit Array Job

```bash
export BANK_PATH="data/cache/banks/qwen25_3b__gate.pt"
export SHARD_DIR="data/interim/prompts/shards"
export OUTPUT_DIR="data/cache/hidden_states"
export BATCH_SIZE=8

sbatch --array=0-9%4 scripts/slurm/extract_hidden_array.slurm
```

This submits 10 tasks, allowing up to 4 to run in parallel:

```
SLURM Queue:
  Task 0: shard_000.json → hidden_states/shard_000.pt  (GPU 0) ████
  Task 1: shard_001.json → hidden_states/shard_001.pt  (GPU 1) ████
  Task 2: shard_002.json → hidden_states/shard_002.pt  (GPU 2) ████
  Task 3: shard_003.json → hidden_states/shard_003.pt  (GPU 3) ████
  Task 4: waiting...
  Task 5: waiting...
  ...
```

The `%4` means "max 4 tasks running simultaneously". Adjust based on GPU availability.

### 3. Monitor Progress

```bash
# Watch the array job
squeue -j <JOB_ID> --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Or just watch the output directory
watch 'ls -lh data/cache/hidden_states/shard_*.pt | wc -l'
```

### 4. Merge Shards (after array job completes)

```bash
python scripts/merge_hidden_states.py \
    data/cache/hidden_states/shard_*.pt \
    --output data/cache/hidden_states/truthfulqa_merged.pt
```

Combines all `shard_*.pt` files into a single artifact with all samples.

### 5. Continue Pipeline

```bash
# Stages 3–5 run normally on the merged artifact
hopfield-llm score \
    --memory data/cache/banks/qwen25_3b__gate.pt \
    --hidden data/cache/hidden_states/truthfulqa_merged.pt \
    --output runs/exp01/scores.pt

hopfield-llm analyze --scores runs/exp01/scores.pt --output runs/exp01/analysis.json
hopfield-llm visualize --analysis runs/exp01/analysis.json --output runs/exp01/plots
```

## Choosing `--num-shards`

| Shards | Shard size (817 total) | Forward passes per shard | GPU time | Network overhead |
|--------|------------------------|-------------------------|----------|------------------|
| 4 | ~200 | ~200 | ~30 min | Low |
| 8 | ~100 | ~100 | ~15 min | Low |
| 10 | ~82 | ~82 | ~12 min | Low |
| 20 | ~41 | ~41 | ~6 min | Medium |
| 40 | ~20 | ~20 | ~3 min | High |

**Recommendation:** `--num-shards 10` for A100 (20 GB VRAM, batch_size=8).

Increasing shards reduces per-shard time, but overhead grows:
- Shard creation (negligible)
- Array job submission/scheduling (~1–2 min)
- File I/O for merging (~2–3 min)

## Advanced: Manual Control

If you prefer to manage each step separately:

```bash
# Step 1: Build bank (on local machine if you want, or on HPC)
hopfield-llm build-memory --model qwen25_3b --output banks/qwen25_3b__gate.pt

# Step 2: Shard locally
python scripts/shard_prompts.py data/interim/prompts/truthfulqa_paired.json \
    --output-dir data/interim/prompts/shards --num-shards 10

# Step 3: Submit array job
BANK_PATH="banks/qwen25_3b__gate.pt" \
SHARD_DIR="data/interim/prompts/shards" \
sbatch --array=0-9%4 scripts/slurm/extract_hidden_array.slurm

# Step 4: Wait for completion
JOB_ID=12345  # from sbatch output
scontrol wait_job "$JOB_ID"

# Step 5: Merge locally (after job completes)
python scripts/merge_hidden_states.py \
    data/cache/hidden_states/shard_*.pt \
    --output data/cache/hidden_states/truthfulqa_merged.pt

# Steps 3–5: Run locally
hopfield-llm score ...
hopfield-llm analyze ...
hopfield-llm visualize ...
```

## Troubleshooting

**Array job fails partway through:**

Check the failing task's log:
```bash
tail slurm-extract-<SHARD_ID>-<JOB_ID>.out
```

Common issues:
- **CUDA out of memory:** reduce `--batch-size` (e.g., `BATCH_SIZE=4`)
- **Shard file not found:** verify `SHARD_DIR` path is correct
- **Model download fails:** ensure HF token is set and internet access works

**Merge fails with shape mismatch:**

Verify all shards used the same bank, pooling, and model:
```bash
python3 << 'EOF'
import torch
files = ["data/cache/hidden_states/shard_000.pt", "data/cache/hidden_states/shard_001.pt"]
for f in files:
    a = torch.load(f, weights_only=False)
    print(f"{f}: model={a['model']['alias']}, pooling={a['pooling']}, samples={len(a['samples'])}")
EOF
```

## Local Testing

Test sharding locally before submitting to HPC:

```bash
# Create 2 shards from your test data
python scripts/shard_prompts.py \
    data/interim/prompts/test_paired.json \
    --output-dir data/interim/prompts/test_shards \
    --num-shards 2

# Extract from both shards sequentially (no SLURM)
hopfield-llm extract-hidden \
    --memory data/cache/banks/qwen25_3b__gate.pt \
    --prompts data/interim/prompts/test_shards/shard_000.json \
    --output data/cache/hidden_states/test_shard_000.pt

hopfield-llm extract-hidden \
    --memory data/cache/banks/qwen25_3b__gate.pt \
    --prompts data/interim/prompts/test_shards/shard_001.json \
    --output data/cache/hidden_states/test_shard_001.pt

# Merge
python scripts/merge_hidden_states.py \
    data/cache/hidden_states/test_shard_*.pt \
    --output data/cache/hidden_states/test_merged.pt

# Score merged result
hopfield-llm score \
    --memory data/cache/banks/qwen25_3b__gate.pt \
    --hidden data/cache/hidden_states/test_merged.pt \
    --output runs/test_merged/scores.pt
```

This validates the entire workflow before going to HPC.
