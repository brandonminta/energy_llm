# Runbook

How to run this repo from a fresh checkout, both locally and on the HPC. This is the operational complement to `CLAUDE.md` (architecture) and `README.md` (overview).

## 1. The pipeline at a glance

| Stage | Command | Where it runs | Reads | Writes |
|---|---|---|---|---|
| 1 | `build-banks` | GPU | model | `banks.pt` + `banks_metadata.json` |
| 2 | `run-trajectory` | GPU | banks + dataset | `{id}.npz` + `{id}.json` per sample |
| 3 | `analyze` | CPU | trajectory dir | `analysis.json` |
| 4 | `visualize` | CPU | `analysis.json` | `plots/*.png` |
| 5 | `label` | CPU (API) | trajectory dir | `{id}_labeled.json` + `calibration.json` |
| 6 | `evaluate` | CPU | trajectory + labels | `report.html` |

`run-experiment` chains 1 → 4 from a YAML config. 5 and 6 are run separately.

## 2. Local quick test (RTX 2050)

```bash
micromamba activate hopfield-llm

# 20 samples, 1.5B model, completes in a few minutes
hopfield-llm run-experiment \
    --config configs/experiments/exp01_truthfulqa_baseline.yaml \
    --override max_samples=20 model_alias=qwen25_1_5b
```

Output lands under `outputs/exp01_truthfulqa_baseline/<timestamp>_qwen25_1_5b_truthfulqa_up_b15.0_<hash>/`.

## 3. HPC — single-job experiments (recommended)

`scripts/hpc/jobs/run_exp02.sh` and `run_exp03.sh` are the canonical templates:

```bash
sbatch scripts/hpc/jobs/run_exp02.sh   # Qwen2.5-3B, 500 TruthfulQA
sbatch scripts/hpc/jobs/run_exp03.sh   # Llama 3.2-3B, 500 TruthfulQA
```

Each job writes to `$HOME/hopfield_results/<exp>/<timestamp>.../` containing `banks.pt`, `trajectories/`, `analysis.json`, `plots/`, plus tracker metadata.

To run any other config, copy `run_exp02.sh` and change the `--config` path.

## 4. HPC — array-job sharding (for very large datasets)

`scripts/hpc/submit_pipeline.sh` chains three SLURM jobs: bank extraction → array of trajectory shards → CPU-only analysis. Useful when you want N shards in parallel.

```bash
MODEL=qwen25_3b DATASET=truthfulqa NUM_SHARDS=8 \
  ./scripts/hpc/submit_pipeline.sh
```

## 5. After a trajectory run lands — labeling and evaluation

The capture is done; everything else is CPU.

```bash
# 5a. Label samples with GPT (or local HF if you pass --api-base '')
export OPENAI_API_KEY=sk-...
hopfield-llm label \
    --trajectories  ~/hopfield_results/exp04_qwen_chat_template_fix/<run-dir>/trajectories \
    --output        ~/hopfield_results/exp04_qwen_chat_template_fix/<run-dir>/labels \
    --judge-model   gpt-4o-mini \
    --heuristic-threshold 0.3

# 5b. Build the eval report
hopfield-llm evaluate \
    --trajectories  ~/hopfield_results/exp04_qwen_chat_template_fix/<run-dir>/trajectories \
    --labels        ~/hopfield_results/exp04_qwen_chat_template_fix/<run-dir>/labels \
    --output        ~/hopfield_results/exp04_qwen_chat_template_fix/<run-dir>/report.html \
    --beta          15.0 \
    --dataset-name  truthfulqa
```

Open `report.html` to see per-layer AUROC, the logreg probe results, scalar baselines (token logprob / entropy), and a β-sweep curve when score tensors were captured.

## 6. Re-analyzing existing trajectories with a different score metric

The CPU analyze stage is cheap. Don't re-capture if you only want to change the metric:

```bash
hopfield-llm analyze \
    --trajectories  <run-dir>/trajectories \
    --output        <run-dir>/analysis_energy_l1.json \
    --score-metric  energy_shift_l1 \
    --score-aggregation mean

hopfield-llm visualize \
    --analysis  <run-dir>/analysis_energy_l1.json \
    --output    <run-dir>/plots_energy_l1
```

## 7. Recommended next experiments (post-fix)

After the chat-template fix in `capture_prefill`, the previous exp02/exp03 trajectories are theoretically incomparable (their prefill came from the raw question, not the chat-templated prompt). Re-run with:

| Config | Hypothesis | GPU cost vs exp02 |
|---|---|---|
| `exp04_qwen_chat_template_fix.yaml` | The prompt asymmetry alone explained the null result | Same |
| `exp05_llama_chat_template_fix.yaml` | Same fix on Llama 3.2-3B | Same |
| `exp06_qwen_value_space.yaml` | Value-space (W_down.T) probe carries more semantic signal than key-space (W_up) | +~30% (one extra teacher-forcing pass per sample) |
| `exp07_qwen_layer_scan.yaml` | Signal localizes in a layer band rather than averaging out | Zero — reuses exp04 trajectories with `--skip-banks` and different `signal_zone` overrides |
| `exp08_qwen_full_metrics.yaml` | Logit-lens entropy / temporal slope / top1-top2 gap pick up signal that delta_energy doesn't | +~80% (logit-lens + null control = +2 forward passes per sample, but disk drops ~95% via topk score capture) |

A reasonable order:

1. exp04 + exp05 first — confirm the chat-template fix changes the picture.
2. Run `label` + `evaluate` on both to get AUROC numbers.
3. exp08 — turn on every new metric in parallel, see which (if any) gives signal.
4. exp06 if value-space-vs-key-space comparison is what you want.
5. exp07 — no-cost scan to find which layers carry the signal once one is found.

## 8. Storage budget reminder

Per Qwen2.5-3B sample (36 layers, 50 gen tokens, 11008 intermediate dim):

- `{id}.npz` (scalars): ~30–60 KB — always saved
- `{id}.json`: ~2 KB — always saved
- `{id}_scores.npz` (`[L, T, K]`): ~38 MB compressed — only first `scores_subset_size` samples
- `{id}_hpre.npz` (`[L, T_prompt, d]`): ~3 MB — only first `diagnostic_subset` samples

For 500 samples at default `scores_subset_size=50` → ~1.9 GB of `_scores.npz`. Set `scores_subset_size=0` to disable score capture entirely if you don't plan to run a β-sweep.

Hidden states are never persisted — they live only in GPU memory during the forward pass.

## 9. Common operations

- **Skip bank rebuild**: `--skip-banks` on `run-experiment` (looks for `banks.pt` in the run dir).
- **Override any config field**: `--override key1=val1 key2=val2` (`max_samples=null` to disable cap).
- **Multi-shard analyze**: when `num_shards > 1`, `run-experiment` skips the CPU stages; run `hopfield-llm analyze` once after all shards land.
- **Local debugging**: `scripts/local/test_pipeline.sh` runs 20 samples with the 1.5B model end-to-end.
- **Inspect a single trajectory file**: `python scripts/local/inspect_results.py <traj-dir>`.
- **Chat with a model interactively** (no banks needed): `python scripts/local/chat_model.py --model qwen25_3b`.
