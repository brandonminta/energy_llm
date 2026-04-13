# Storage Scaling Reference

Measured on a test run of 3 paired samples using `qwen25_3b` (36 layers, D=2048, 4-bit NF4).

## Artifact Sizes (Measured)

| Artifact | 3 samples | Formula |
|----------|-----------|---------|
| Memory bank (`gate`) | **3.1 GB** | Fixed per model |
| Hidden states | **3.4 MB** | ~1.1 MB × N samples |
| Scores | **12 KB** | ~4 KB × N samples |
| Analysis JSON | **24 KB** | Roughly fixed (per-layer stats) |

## Memory Bank

The bank is a **fixed cost** — it does not scale with dataset size. It is extracted once and reused for every scoring run.

```
bank[layer_i].shape = [K, D]   where K = intermediate_size, D = hidden_size
```

| Model | Layers | D | K | Bank size (fp32) |
|-------|--------|---|---|-----------------|
| qwen25_1_5b | 28 | 1536 | 8960 | ~1.6 GB |
| gemma2_2b | 26 | 2304 | 9216 | ~2.1 GB |
| qwen25_3b | 36 | 2048 | 11008 | **~3.1 GB** |
| phi3_mini | 32 | 3072 | 8192 | ~3.0 GB |
| llama32_3b | 28 | 3072 | 8192 | ~2.6 GB |
| qwen25_7b | 28 | 3584 | 18944 | ~7.5 GB |
| mistral7b | 32 | 4096 | 14336 | ~7.1 GB |
| llama31_8b | 32 | 4096 | 14336 | ~7.1 GB |
| gemma2_9b | 42 | 3584 | 14336 | ~8.6 GB |

> Bank is stored as `float32` on CPU regardless of model quantization.
> One bank per `(model, bank_type, normalize)` combination.

## Hidden States

Hidden states **scale linearly** with number of samples. Each paired sample stores
two segment pairs (factual + hallucinated), each consisting of two tensors `[L, D]`.

```
Per paired sample = 4 tensors × L × D × 4 bytes (float32)
```

| Model | L | D | Per paired sample | Per single sample |
|-------|---|---|-------------------|-------------------|
| qwen25_1_5b | 28 | 1536 | ~0.66 MB | ~0.33 MB |
| gemma2_2b | 26 | 2304 | ~0.72 MB | ~0.36 MB |
| qwen25_3b | 36 | 2048 | **~1.13 MB** | ~0.56 MB |
| phi3_mini | 32 | 3072 | ~1.19 MB | ~0.60 MB |
| llama32_3b | 28 | 3072 | ~1.05 MB | ~0.53 MB |
| qwen25_7b | 28 | 3584 | ~1.22 MB | ~0.61 MB |
| mistral7b | 32 | 4096 | ~1.59 MB | ~0.79 MB |
| llama31_8b | 32 | 4096 | ~1.59 MB | ~0.79 MB |
| gemma2_9b | 42 | 3584 | ~1.80 MB | ~0.90 MB |

### Dataset Projections (qwen25_3b, paired)

| Samples | Hidden states | Scores | Total artifacts |
|---------|--------------|--------|-----------------|
| 10 | 11 MB | 40 KB | ~11 MB |
| 100 | 113 MB | 400 KB | ~114 MB |
| 817 (TruthfulQA full) | ~924 MB | ~3.3 MB | ~927 MB |
| 1,000 | 1.1 GB | 4.0 MB | ~1.1 GB |
| 5,000 | 5.7 GB | 20 MB | ~5.7 GB |
| 10,000 | 11.3 GB | 40 MB | ~11.3 GB |

## Scores

Scores are tiny. Each sample stores 9 metric arrays of length L (36 float64 values each):

```
Per sample = 9 metrics × L × 8 bytes ≈ 2.6 KB  (+ sample metadata)
```

Scores artifact is the safest to commit to git for datasets up to ~50k samples.

## Analysis JSON

The analysis file aggregates layer statistics across all samples:

```
size ≈ 9 metrics × L × (mean + std arrays) + category/label breakdowns
     ≈ fixed ~20–30 KB for most datasets
```

## Batch Size and VRAM Headroom

During extraction, peak VRAM is:

```
VRAM ≈ model_base + batch_size × seq_len × L × D × 2 bytes (fp16 activations)
```

For `qwen25_3b` 4-bit (model_base ≈ 2.0 GB), `seq_len ≈ 200`, `L=36`, `D=2048`:

| batch_size | Activation VRAM | Total (model + activations) | Safe on |
|------------|----------------|----------------------------|---------|
| 1 | ~58 MB | ~2.1 GB | RTX 2050 (4 GB) ✓ |
| 2 | ~117 MB | ~2.2 GB | RTX 2050 ✓ |
| 4 | ~234 MB | ~2.3 GB | RTX 2050 ✓ |
| 8 | ~467 MB | ~2.5 GB | RTX 2050 (tight) ⚠ |
| 16 | ~935 MB | ~2.9 GB | RTX 2050 (risky) ✗ |
| 32 | ~1.9 GB | ~3.9 GB | A100 20 GB ✓ |

> **RTX 2050 recommendation**: `--batch-size 4` (safe headroom, ~4x speedup)
> **A100 20 GB recommendation**: `--batch-size 16` (~16x speedup vs batch_size=1)

The env config (`configs/environments/local.yaml` / `hpc.yaml`) sets `default_batch_size`
which is used automatically when `--batch-size` is omitted.

## Selective Layer Extraction

Use `--layers` to reduce hidden state size when you only need specific layers:

```bash
# Extract only layers 12, 18, 24, 30 (4 instead of 36)
hopfield-llm extract-hidden --layers 12 18 24 30 ...
# Reduces hidden state size by 89% (4/36)
```

| Layers selected | Size reduction | Use case |
|-----------------|---------------|----------|
| All 36 | 0% | Full analysis |
| Every 6th (6 layers) | 83% | Quick sweep |
| Last quarter (9 layers) | 75% | Late-layer focus |
| 4 key layers | 89% | Minimal footprint |

## .gitignore Coverage

The following artifact directories are git-ignored and will not be committed:

```
data/cache/          # banks, hidden_states (large)
data/results/
data/interim/
artifacts/
runs/
experiments/runs/
```

**Safe to commit**: `data/interim/prompts/*.json` (prompt files, small), analysis JSON files.
