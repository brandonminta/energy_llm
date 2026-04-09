# Audit Report — energy_llm

**Date**: 2026-04-09
**Scope**: Full codebase audit — `src/energy_llm/`, `configs/`, `scripts/`
**Commit**: `5f45f6b` (Structure reajustment in src)

---

## 1. Flow Coherence Analysis

### Complete Pipeline Trace

```
HFLLM(model_id)                         → model + tokenizer + layers
  ↓
extract_mlp_memory_bank(llm, bank)       → banks: dict[int, Tensor[K, D]]  (0-indexed keys)
  ↓
extract_segment_hidden_states(llm, Q, A) → SegmentHiddenStates: h_q[L,D], h_a[L,D]  (1-indexed layer_indices)
  ↓
compute_layer_divergences(h_ref, h_cmp, mlp_keys)  → LayerDivergenceResult: arrays[L]
  ↓
hallucination_score(div, metric, aggregation)       → float
```

### Inconsistencies Found

**BUG-01 (Critical): Layer indexing mismatch between banks and hidden states**

- `extract_mlp_memory_bank` stores banks with **0-indexed** keys: `banks[0], banks[1], ..., banks[35]`
- `extract_segment_hidden_states` produces **1-indexed** `layer_indices`: `[1, 2, ..., 36]`
- The alignment code in `scoring/truthfulqa.py:40` and `pipelines/compute_divergences.py:21` bridges this with:
  ```python
  aligned_banks = {i: banks[li - 1] for i, li in enumerate(layer_indices) if (li - 1) in banks}
  ```
- This works correctly but is **fragile, non-obvious, and duplicated** in two locations. If a third consumer is added without this exact mapping, it will silently use wrong layers or fail.
- **Recommendation**: Standardize on 0-indexed everywhere, or encapsulate the alignment in a single utility function.

**BUG-02 (Critical): CLI entry point `run-exp1` does not exist**

- `README.md:85` and `scripts/run_exp01.sh:8` invoke:
  ```bash
  python -m energy_llm.cli.main run-exp1 --config configs/experiments/exp01_truthfulqa_baseline.yaml
  ```
- The CLI parser in `cli/run.py` defines only 5 subcommands: `build-memory`, `extract-hidden`, `score`, `analyze`, `visualize`. There is no `run-exp1` command.
- The function `run_experiment_1()` in `scoring/truthfulqa.py` exists but is **never wired to the CLI**. It is the monolithic experiment runner from the pre-refactor design.
- **Result**: The documented "Minimal Working Path" in the README is broken. Running the script produces an argparse error.

**BUG-03 (Medium): `build_memory_bank` loads model with `output_hidden_states=True`**

- `pipelines/build_memory_bank.py:12` passes `output_hidden_states=True` to HFLLM.
- Bank extraction only reads model weights — no forward pass is needed.
- This is harmless for memory (the flag only affects forward pass output), but misleading and prevents a future optimization where we could skip loading the model entirely and just load the state_dict.

**BUG-04 (Low): `torch.load` without `weights_only` parameter**

- `utils/serialization.py:26`: `torch.load(Path(path), map_location="cpu")`
- Since PyTorch 2.6+, `torch.load` defaults to `weights_only=True` and will warn or fail on non-tensor payloads. Our artifacts contain dicts with mixed types (strings, lists, tensors).
- **Fix**: Add `weights_only=False` explicitly, or migrate to a safer serialization format.

---

## 2. Bugs and Inconsistencies

### 2.1 Tensor Shapes

| Location | Issue | Severity |
|----------|-------|----------|
| `hopfield.py:25` | `xi` validated as 1D `[D]`, correct | OK |
| `hopfield.py:29` | `W_keys` validated as 2D `[K,D]` with shape check vs xi | OK |
| `layer_scores.py:30-33` | `h_ref`, `h_cmp` validated as 2D `[L,D]` with mutual shape check | OK |
| `segment_states.py:77` | `stacked` is `[n_layers, T, D]`, pooled to `[n_layers, D]` | OK |
| `mlp_bank.py:188-191` | Weight tensor validated as 2D | OK |
| `mlp_bank.py:193-208` | Shape validated against `hidden_size` and `intermediate_size` from config | OK |

**No shape bugs found.** All critical tensor operations have explicit shape validation.

### 2.2 Device Handling

| Location | Issue | Severity |
|----------|-------|----------|
| `hf_llm.py:64-71` | 4-bit silently disabled on CPU with no log/warning | Low |
| `hf_llm.py:197-198` | Warning printed when `safe_4bit=False`, but not when `device != cuda` | Low |
| `hopfield.py:33-34` | Converts to float32 — correct for CPU computation | OK |
| `segment_states.py:56-57` | Tensor moved to `llm.device` for forward pass | OK |
| `segment_states.py:78-79` | `to_cpu=True` moves stacked states to CPU — correct | OK |
| `mlp_bank.py:29,35` | Weight extracted to CPU via `.cpu()` — correct | OK |

**No device bugs**, but the silent 4-bit→fp32 fallback on CPU can surprise users who expect quantized inference.

### 2.3 Memory Management

| Location | Issue | Severity |
|----------|-------|----------|
| `segment_states.py:65-66` | Forward pass output (`outputs`) is never explicitly freed. On CUDA with long sequences, the full hidden state tensor for all layers lives in VRAM until garbage collected | Medium |
| `scoring/truthfulqa.py:36-49` | Per-sample loop creates two `SegmentHiddenStates` objects and one `LayerDivergenceResult` per iteration. The `div.softmax_w` tensor (size K per layer) accumulates across results | Low |
| `hopfield.py:65` | `softmax_w` and `similarities` detached and moved to CPU — correct | OK |
| `mlp_bank.py:25-35` | Weight dequantized to float32 CPU — correct | OK |

**Most critical**: After the forward pass in `extract_segment_hidden_states`, the full `outputs` object (containing hidden states for ALL layers, not just selected ones) remains in scope until the function returns. With Qwen-3B (36 layers × ~2048D × T tokens × 2 bytes fp16), this can be significant for long sequences on 4GB VRAM.

**Recommendation**: Add explicit cleanup:
```python
stacked = torch.stack([all_hs[l][0] for l in layer_indices], dim=0).float()
del outputs, all_hs  # free VRAM before pooling
torch.cuda.empty_cache()
```

### 2.4 Hardcoded Values That Should Be Configurable

| Value | Location | Current | Notes |
|-------|----------|---------|-------|
| `beta` | Multiple | 15.0 | Already configurable via CLI/config — OK |
| `active_threshold` | Multiple | 0.1 | Already configurable — OK |
| Weighted agg. weights | `hallucination.py:38` | `linspace(0.5, 1.0)` | The slope (0.5→1.0) is hardcoded. Should be configurable for signal zone experiments |
| Entropy clamp | `hopfield.py:58` | `1e-9` | Fine — numerical stability constant |
| BnB quant type | `hf_llm.py:89` | `"nf4"` | Could be configurable but NF4 is the standard choice |

### 2.5 Language Inconsistency

Error messages and docstrings are **mixed Spanish and English**:

| File | Line | Language |
|------|------|----------|
| `hf_llm.py` | 135, 149 | Spanish: "No se pudo resolver..." |
| `mlp_bank.py` | 13-26, 38, 44, 51-61, 103, 136, 183-186, 228 | Spanish: docstrings, error messages |
| `segment_states.py` | 29, 38, 44, 46-48, 74 | Spanish: error messages |
| `hopfield.py` | 27-31, 51 | Spanish: error messages |
| `truthfulqa.py` | 58, 65, 78-82 | Spanish: error messages, print statements |
| `layer_scores.py` | 31, 33, 70 | Spanish: error messages |
| `hallucination.py` | 27, 37, 40 | Spanish: error messages |

**Recommendation**: Standardize all user-facing messages to English for international readability. Keep Spanish comments if preferred for internal notes.

---

## 3. Code Quality Assessment

### 3.1 Single Responsibility

| Module | Assessment |
|--------|-----------|
| `core/hf_llm.py` | **Good**. Single class, clear responsibility. |
| `memory/mlp_bank.py` | **Good**. Single function with helpers. |
| `hidden_states/segment_states.py` | **Good**. Extraction + pooling, well-scoped. |
| `energy/hopfield.py` | **Good**. Pure computation, no side effects. |
| `analysis/divergences.py` | **Good**. Pure divergence primitives. |
| `scoring/layer_scores.py` | **Good**. Bridges energy and divergences. |
| `scoring/hallucination.py` | **Good**. Pure aggregation. |
| `scoring/truthfulqa.py` | **Mixed**. `run_experiment_1` mixes orchestration (hidden state extraction, bank alignment, scoring) with experiment-specific logic. It also directly calls `extract_segment_hidden_states`, coupling it to the extraction layer. |
| `pipelines/*.py` | **Good**. Thin orchestration wrappers. |
| `data/truthfulqa.py` | **Good**. Clean dataset normalization. |

### 3.2 Docstrings and Type Annotations

- **Docstrings**: Present in `mlp_bank.py` (good, includes bank types). Missing or minimal in `segment_states.py`, `layer_scores.py`, `hallucination.py`, `truthfulqa.py`, all pipeline files.
- **Type annotations**: Good on function signatures (`hf_llm.py`, `mlp_bank.py`, `hopfield.py`, `divergences.py`). Missing on some return types in pipeline functions.
- **Dataclass fields**: Documented via field names but no field-level docstrings. Shape expectations (`[L, D]`, `[K, D]`) are in comments only, not enforced.

### 3.3 Error Handling

- `mlp_bank.py`: Good — strict/non-strict modes with structured error metadata.
- `hf_llm.py`: Good — backbone/layer resolution with helpful error messages listing available attributes.
- `scoring/truthfulqa.py`: Good — skip_on_error with error capture.
- **Missing**: No error handling in pipeline files (`build_memory_bank.py`, `extract_hidden_states.py`, etc.). A model download failure or disk-full condition would produce an unstructured traceback.

### 3.4 Logging

- **All output is via `print()`** — no structured logging anywhere.
- `HFLLM._print_load_summary()`: prints to stdout.
- `load_truthfulqa()`: prints to stdout.
- `run_experiment_1()`: prints summary stats to stdout.
- **Impact**: No log levels, no file output, no timestamps, no ability to silence output in batch jobs.

### 3.5 Dead Code and Stubs

**46 empty Python files** exist as placeholders:
- `core/`: constants.py, device.py, hashing.py, logging_utils.py, paths.py, registry.py, seeds.py, types.py
- `memory/`: bank_cache.py, bank_extractors.py, bank_metadata.py
- `datasets/`: base.py, halu_eval.py, hotpotqa.py, local_jsonl.py, splits.py
- `representations/`: state_cache.py, token_pooling.py, trajectory_states.py
- `metrics/`: classification.py, ranking.py
- `energies/`: entropy.py, retrieval.py, sparse_hopfield.py
- `experiments/`: base.py, exp2 through exp5
- `evaluation/`: all files
- `analysis/`: loaders.py, report_tables.py, significance.py, summaries.py
- `utils/`: cache.py, io.py, json_utils.py, pandas_utils.py, torch_utils.py

**Compatibility wrappers** (`models/`, `datasets/`, `representations/`, `metrics/`, `energies/`, `experiments/`): 12 files that only re-export from canonical modules.

**Total bloat**: ~58 files with no functional code. These add cognitive load for navigation and maintenance.

---

## 4. Performance Bottlenecks

### 4.1 Time Costs by Stage

| Stage | Operation | Cost | Notes |
|-------|-----------|------|-------|
| 1. `build_memory_bank` | Read MLP weights from loaded model | **Low** (~10s) | One-time per model. No forward pass. |
| 2. `extract_segment_hidden_states` | Forward pass per sample | **Very High** (~0.5-2s/sample) | 818 TruthfulQA samples × 2 passes = 1636 forward passes. **13-55 min total.** |
| 3. `hopfield_energy_mlp` | Matmul [K,D]@[D] per layer | **Low** (~0.5ms/layer) | K=11008, D=2048 for Qwen-3B. 36 layers × 2 modes = ~36ms/sample |
| 4. `compute_layer_divergences` | KL/JS/Hellinger on softmax distributions | **Negligible** (~0.1ms/layer) | Pure scalar ops on distributions of size K |
| 5. `hallucination_score` | Aggregation over L values | **Negligible** | Single nanmean/nanmax over 36 values |

**The forward pass is 99%+ of total runtime.** All optimization effort should focus on Stage 2.

### 4.2 Batching Opportunity (High Impact)

Currently, `extract_segment_hidden_states` processes one sample at a time. Each call:
1. Tokenizes Q + separator + A
2. Creates input_ids tensor
3. Runs one forward pass
4. Extracts hidden states

**Batching strategy**: Group N samples into padded batches, run one forward pass per batch.

- **RTX 2050 (4 GB)**: With Qwen-3B 4-bit (~2 GB), only ~1.5 GB free for activations. Batch size = 1-2 (short sequences) or 1 (long sequences). Marginal benefit.
- **A100 10 GB**: ~7 GB free. Batch size = 4-8 depending on sequence length. **~3-6x speedup**.
- **A100 40 GB**: ~37 GB free. Batch size = 16-32. **~10-20x speedup**.

**Challenge**: Variable sequence lengths require padding and careful segment boundary tracking per sample.

### 4.3 Vectorization of Divergences (Low Impact)

`compute_layer_divergences` iterates over L layers in Python. Could be vectorized:
```python
# Current: Python loop over L layers
for i in range(L):
    W = mlp_keys[i]        # [K, D]
    v_ref = W @ h_ref[i]   # [K]
    ...

# Possible: batched matmul
W_stacked = torch.stack(...)  # [L, K, D]
v_ref = torch.bmm(W_stacked, h_ref.unsqueeze(-1)).squeeze(-1)  # [L, K]
```

**However**: The total computation (~36ms/sample on CPU) is negligible versus forward pass time. Vectorizing would save ~5-10ms from Python loop overhead. **Not worth the complexity** unless running massive sweep experiments offline.

### 4.4 CPU Parallelism for Post-Processing (Medium Impact)

Stages 3-5 (scoring, analysis, visualization) are CPU-bound once hidden states are cached. For large experiments (multiple models × datasets × hyperparameters):
- Score computation across cached hidden states: embarrassingly parallel per sample.
- `multiprocessing.Pool` or `joblib` over sample shards would provide near-linear speedup.
- **Worth implementing** only when the number of scoring configurations (beta × metric × aggregation) grows beyond ~10.

### 4.5 Lazy Layer Selection (Low Impact)

Currently, `extract_segment_hidden_states` extracts ALL transformer layers and then selects. With `output_hidden_states=True`, the model computes and returns all hidden states regardless.

There is no way to avoid this without modifying the model's forward pass (e.g., registering hooks on specific layers). The current approach is the standard HuggingFace pattern. **No optimization possible here** without significant complexity.

### 4.6 Redundant Forward Passes in Experiment 1

`run_experiment_1` calls `extract_segment_hidden_states` twice per sample (factual and hallucinated) with the same question. Both forward passes process the shared question tokens redundantly.

**Optimization**: A custom forward pass that processes `[Q + sep + A_factual]` and `[Q + sep + A_hallucinated]` as a batch of 2 would halve the overhead of processing question tokens. More impactful for long questions with short answers.

---

## 5. Viability by Environment

### 5.1 VRAM Estimates (4-bit NF4 Quantized)

Model parameters × 0.5 bytes (4-bit) + ~0.5 GB overhead for activations/buffers:

| Model | Params | VRAM (4-bit) | VRAM (fp16) | `safe_4bit` |
|-------|--------|-------------|-------------|-------------|
| Qwen2.5-1.5B | 1.5B | ~1.2 GB | ~3.0 GB | True |
| Qwen2.5-3B | 3.0B | ~2.0 GB | ~6.0 GB | True |
| Llama-3.2-3B | 3.2B | ~2.1 GB | ~6.4 GB | True |
| Phi-3-mini (3.8B) | 3.8B | ~2.4 GB | ~7.6 GB | True |
| Mistral-7B | 7.2B | ~4.1 GB | ~14.4 GB | **False** |

### 5.2 Environment Compatibility Matrix

| Model | RTX 2050 (4 GB) | A100 5 GB | A100 10 GB | A100 20 GB | A100 40 GB |
|-------|-----------------|-----------|------------|------------|------------|
| Qwen2.5-1.5B (4-bit) | **Yes** | **Yes** | Yes | Yes | Yes |
| Qwen2.5-3B (4-bit) | **Yes** (tight) | **Yes** (tight) | Yes | Yes | Yes |
| Llama-3.2-3B (4-bit) | **Yes** (tight) | **Yes** (tight) | Yes | Yes | Yes |
| Phi-3-mini (4-bit) | **Marginal** | **Marginal** | Yes | Yes | Yes |
| Mistral-7B (4-bit) | **No** (`safe_4bit=False`) | No | No | No | No |
| Mistral-7B (fp16) | **No** (14 GB) | No | No | **Yes** (tight) | Yes |

### 5.3 Recommendations per Environment

**RTX 2050 (4 GB VRAM, 16 GB RAM)**:
- Safe models: Qwen2.5-1.5B, Qwen2.5-3B (batch_size=1)
- Marginal: Phi-3-mini (may OOM on long sequences)
- Not viable: Mistral-7B, Llama-3.2-3B is tight with long sequences
- Batch size: always 1
- Use `torch.cuda.empty_cache()` between samples

**HPC gpu-dev (a100_1g.5gb, 1 GPU, 96h)**:
- Sufficient for: bank extraction for all models
- Forward passes: only small models (Qwen-1.5B, Qwen-3B)
- Best use: development, debugging, bank caching

**HPC gpu (a100_2g.10gb, 2 GPUs, 48h)**:
- All models in 4-bit fit comfortably
- Batch size: 4-8
- Best use: full TruthfulQA experiments on single models

**HPC gpu (a100_3g.20gb)**:
- All models including Mistral-7B in fp16
- Batch size: 8-16
- Best use: multi-model comparisons, sweeps

**HPC gpu-max (a100-sxm4-40gb, 4 GPUs, 24h)**:
- All models with large batches
- Best use: massive sweeps, cross-dataset experiments
- **Caution**: 4 GPUs only useful with data parallelism over samples, not model parallelism (models fit in 1 GPU)

### 5.4 Estimated Runtime for Full TruthfulQA (818 samples, 1636 forward passes)

| Environment | Model | Estimated Time |
|-------------|-------|---------------|
| RTX 2050 (4-bit, batch=1) | Qwen-3B | ~40-60 min |
| RTX 2050 (4-bit, batch=1) | Qwen-1.5B | ~20-30 min |
| A100-10GB (4-bit, batch=4) | Qwen-3B | ~10-15 min |
| A100-10GB (4-bit, batch=8) | Qwen-1.5B | ~5-8 min |
| A100-20GB (fp16, batch=8) | Mistral-7B | ~15-25 min |
| A100-40GB (fp16, batch=16) | Qwen-3B | ~3-5 min |

---

## 6. Summary of Findings

### Critical Issues (must fix)

1. **`run-exp1` CLI command does not exist** — documented entry point is broken
2. **Layer indexing mismatch (0-indexed banks vs 1-indexed hidden states)** — works via manual alignment but fragile and duplicated
3. **No explicit memory cleanup after forward pass** — risks OOM on 4 GB VRAM

### Medium Issues (should fix)

4. **`torch.load` without `weights_only=False`** — will break on PyTorch 2.6+
5. **No structured logging** — all output via print(), unusable in batch jobs
6. **Mixed Spanish/English messages** — confusing for international use
7. **58 empty stub/wrapper files** — cognitive overhead, no functional value

### Low Issues (nice to have)

8. **Silent 4-bit fallback on CPU** — no warning to user
9. **Weighted aggregation slope hardcoded** — minor configurability gap
10. **Missing docstrings** in pipeline and scoring modules
11. **No input validation** in pipeline functions (file existence, artifact type checks)

### Architecture Strengths

- Clean separation between energy computation and experiment logic
- Flexible MLP bank extraction supporting multiple projection types
- Good tensor shape validation in core computation modules
- Artifact-based pipeline design enables independent stage execution
- Model profile system abstracts architecture differences cleanly

### Optimization Priority

1. **Forward pass batching** — 99% of runtime, highest impact
2. **VRAM management** — critical for RTX 2050 viability
3. **CPU parallelism for scoring sweeps** — medium impact for hyperparameter exploration
4. **Layer divergence vectorization** — low impact, not recommended
