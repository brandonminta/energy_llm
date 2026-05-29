# Runbook

How to run this repo from a fresh checkout, both locally and on the HPC. This
is the operational complement to `CLAUDE.md` (architecture) and `README.md`
(overview). The definitive configuration is specified in
`docs/methodology.tex §experimental_config` and realised by the `final_*`
configs; `configs/experiments/README.md` is the config matrix.

## 1. The pipeline at a glance

| Stage | Command | Where it runs | Reads | Writes |
|---|---|---|---|---|
| 1 | `build-banks` | GPU | model | `banks.pt` + `banks_metadata.json` |
| 2 | `run-trajectory` | GPU | banks + dataset | `{id}.npz` + `{id}.json` per sample |
| 3 | `analyze` | CPU | trajectory dir | `analysis.json` |
| 4 | `visualize` | CPU | `analysis.json` | `plots/*.png` |
| 5 | `label` | CPU (API) | trajectory dir | `{id}_labeled.json` + `calibration.json` |
| 6 | `evaluate` | CPU | trajectory + labels (+ baseline sidecars) | `report.html` |

`run-experiment` chains 1 → 4 from a YAML config. Baselines (for stage 6) and
stages 5–6 are run separately.

## 2. Local smoke test (RTX 2050)

```bash
micromamba activate hopfield-llm

# Stages 1–4 + baselines on 20 samples with the 1.5B model (a few minutes).
# Exercises the headline path: gated_key banks, normalize_query=false,
# answer-span pooling.
bash scripts/local/test_pipeline.sh                 # MODEL=qwen25_1_5b N=20

# Or run a config directly with small overrides:
hopfield-llm run-experiment \
    --config configs/experiments/final_qwen_truthfulqa.yaml \
    --override max_samples=20 model_alias=qwen25_1_5b
```

Output lands under `outputs/<config-name>/<timestamp>_..._<hash>/`.

## 3. HPC — single-job experiments (recommended)

`scripts/hpc/jobs/run_experiment.sh` is the one parametrized wrapper around
`run-experiment` (it auto-resumes the trajectory stage):

```bash
mkdir -p logs
sbatch --job-name=final_qwen_tqa \
    scripts/hpc/jobs/run_experiment.sh configs/experiments/final_qwen_truthfulqa.yaml
sbatch --job-name=final_llama_tqa \
    scripts/hpc/jobs/run_experiment.sh configs/experiments/final_llama_truthfulqa.yaml

# Resume after an OOM / time-out (banks already on disk):
sbatch scripts/hpc/jobs/run_experiment.sh configs/experiments/final_qwen_truthfulqa.yaml --skip-banks
```

Each job writes to `$HOME/hopfield_results/<config-name>/<timestamp>.../`
containing `banks.pt`, `trajectories/`, `analysis.json`, `plots/`, plus tracker
metadata.

## 4. HPC — array-job sharding (for very large datasets)

`scripts/hpc/submit_pipeline.sh` chains three SLURM jobs: bank extraction →
array of trajectory shards → CPU-only analysis.

```bash
MODEL=qwen25_3b DATASET=truthfulqa NUM_SHARDS=8 \
  scripts/hpc/submit_pipeline.sh
```

## 5. After a trajectory run lands — baselines, labeling, evaluation

The capture is done; everything else is CPU (except baselines, which need the
GPU model). Let `RUN=$HOME/hopfield_results/final_qwen_truthfulqa/<run-dir>`.

```bash
# 5a. Baseline sidecars (token uncertainty + P(True) + semantic entropy + HalluField + INTRA)
python scripts/local/run_baselines.py --trajectories "$RUN/trajectories" --model qwen25_3b
#   (or a single one:  python -m hopfield_llm.evaluation.baselines --trajectories "$RUN/trajectories")

# 5b. Label samples with GPT (or local HF if you pass --api-base '')
export OPENAI_API_KEY=sk-...
hopfield-llm label \
    --trajectories "$RUN/trajectories" --output "$RUN/labels" \
    --judge-model gpt-4o-mini --heuristic-threshold 0.3

# 5c. Build the eval report
hopfield-llm evaluate \
    --trajectories "$RUN/trajectories" --labels "$RUN/labels" \
    --output "$RUN/report.html" --beta 15.0 --dataset-name truthfulqa
```

`report.html` contains per-layer AUROC, the logreg probe, the baseline AUROC
table, the **nested-model paired-bootstrap test** (does [Hopfield+baselines]
beat [baselines-only]?), and a β-sweep curve when score tensors were captured.

## 6. Re-analyzing existing trajectories with a different score metric

The CPU analyze stage is cheap. Don't re-capture if you only want to change the
metric:

```bash
hopfield-llm analyze \
    --trajectories "$RUN/trajectories" \
    --output       "$RUN/analysis_energy_l1.json" \
    --score-metric energy_shift_l1 --score-aggregation mean --pool-over-answer
hopfield-llm visualize --analysis "$RUN/analysis_energy_l1.json" --output "$RUN/plots_energy_l1"
```

## 7. The final campaign and pre-registered ablations

The headline result is the `final_*` matrix; the `exp0*` configs are the
pre-registered ablations (see `docs/preregistration.md` for H1–H4 and
`docs/findings.md` for why each exists).

| Config | Tests | GPU cost |
|---|---|---|
| `final_qwen_truthfulqa.yaml` | Headline (gated_key, normalize_query=false, β→0.55) | baseline |
| `final_llama_truthfulqa.yaml` | Cross-architecture replication (H4 transfer) | baseline |
| `final_qwen_triviaqa.yaml` / `final_llama_triviaqa.yaml` | Cross-dataset transfer | baseline |
| `exp04b_qwen_normalize_query_false.yaml` | H1/T2: falsifies the E_quad tautology (W_up bank) | baseline |
| `exp06_qwen_value_space.yaml` | Key-space vs value-space paired probe (Geva 2021) | +~30% (one extra teacher-forcing pass) |
| `exp07b_qwen_gated_key.yaml` | H3: gating vs raw W_up | baseline |
| `diagnostic_logit_lens_null.yaml` | Logit-lens + shuffled-bank null (reported as ablations) | +~80% (2 extra passes; disk −95% via topk) |

Suggested order: run the four `final_*` first → baselines + label + evaluate on
each → then `exp04b`/`exp07b`/`exp06` ablations → `diagnostic_*` last.

## 8. Storage budget reminder

Per Qwen2.5-3B sample (36 layers, 50 gen tokens, 11008 intermediate dim):

- `{id}.npz` (scalars): ~30–60 KB — always saved
- `{id}.json`: ~2 KB — always saved
- `{id}_scores.npz` (`[L, T, K]`): ~38 MB compressed — only first `scores_subset_size` samples
- `{id}_hpre.npz` (`[L, T_prompt, d]`): ~3 MB — only first `diagnostic_subset` samples

For 500 samples at default `scores_subset_size=50` → ~1.9 GB of `_scores.npz`.
Set `scores_subset_size=0` to disable score capture entirely if you don't plan
to run a β-sweep. Hidden states are never persisted.

## 9. Common operations

- **Skip bank rebuild**: `--skip-banks` on `run-experiment` (looks for `banks.pt` in the run dir).
- **Override any config field**: `--override key1=val1 key2=val2` (`max_samples=null` to disable cap).
- **Multi-shard analyze**: when `num_shards > 1`, `run-experiment` skips the CPU stages; run `hopfield-llm analyze` once after all shards land.
- **Local debugging**: `scripts/local/test_pipeline.sh` runs stages 1–4 + baselines on 20 samples with the 1.5B model.
- **Chat with a model interactively** (no banks needed): `python scripts/local/chat_model.py --model qwen25_3b`.
