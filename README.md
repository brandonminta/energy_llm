# energy_llm

Hallucination detection in LLMs via Modern Hopfield retrieval over SwiGLU
memory banks — the exact implementation of the thesis methodology chapter.

**Ground truth:** `docs/fundamentals-FINAL.tex` (theory) and
`docs/metodology-FINAL.tex` (methodology). `docs/MAPPING.md` maps every
operational statement (equations, stages, `tab:hyperparameters`) to the
module, function, and config key that implements it.

## Pipeline

```
Stage 1  banks        GPU/CPU  model weights        -> results/<exp>/banks.pt
Calibr.  beta*        GPU      30 prefill passes    -> calibration.json + config_snapshot.yaml
Stage 2  trajectories GPU      banks + dataset      -> trajectories/{id}.npz/.json (+ scores cache, E1)
Stage 3  labeling     CPU+API  trajectories         -> labels/{id}_label.json
Stage 4  features     CPU      trajectories+labels  -> features/features.csv
         probe+stats  CPU      features             -> probe_results.json / notebook 02 figures
```

Every stage writes typed artifacts and reads only earlier-stage artifacts;
every stage is independently re-runnable. The per-sample artifact is the
checkpoint unit: **all stages resume by default** (completed samples are
skipped; all writes are atomic tmp+fsync+rename, so a kill leaves no corrupt
partials, and rerunning a killed array task is a no-op for finished samples).

## Setup

```bash
micromamba create -n hopfield-llm python=3.12 -y && micromamba activate hopfield-llm
pip install -r requirements.txt && pip install -e .
pytest                       # full suite runs on CPU with a tiny model
```

`bitsandbytes` (4-bit NF4) is a **local-dev path only** — never used for a
reported run (reported runs: BF16 on a single A100). Set
`OPENAI_API_KEY` for Stage 3. Never hardcode tokens; use `HF_TOKEN` env.

**Existing cluster env:** the `hopfield-llm` micromamba env predates this
rebuild and is missing three packages the new pipeline needs — update it
once with `pip install -r requirements-hpc.txt` (adds `scipy`,
`scikit-learn`, `openai`; everything else is already pinned there).

## E1 (primary): exact command sequence

```bash
# 0. on the cluster, from the repo root
mkdir -p logs

# 1. banks (CPU-bound; reads gate_proj to float32, row-normalises, stores norms)
sbatch hpc/banks.sbatch configs/e1.yaml

# 2. beta* calibration (writes beta_star into results/e1/config_snapshot.yaml;
#    every later stage reads the snapshot)
sbatch --dependency=afterok:<JOB1> hpc/calibrate.sbatch configs/e1.yaml

# 3. Stage 2 as an 8-shard array (resume-by-default; resubmit freely)
sbatch --dependency=afterok:<JOB2> hpc/trajectories.sbatch configs/e1.yaml

#    ... or submit 1-3 with dependencies in one go:
./hpc/submit_pipeline.sh configs/e1.yaml

# 4. labeling (CPU + OpenAI API; run where you keep the key)
#    -> notebooks/01_labeling_openai.ipynb   (judge + label distribution + kappa)
#    or headless:
python scripts/label_samples.py --config configs/e1.yaml
python scripts/label_samples.py --config configs/e1.yaml --make-human-csv
#    fill results/e1/human_labels.csv by hand, then:
python scripts/label_samples.py --config configs/e1.yaml --kappa results/e1/human_labels.csv

# 5. Stage 4: features + probe + statistics
sbatch hpc/features.sbatch configs/e1.yaml          # or locally:
python scripts/build_features.py --config configs/e1.yaml
python scripts/run_probe.py --config configs/e1.yaml

# 6. analysis + thesis figures
#    -> notebooks/02_results_analysis.ipynb  (per-layer folded AUROC profiles +
#       permutation threshold, test AUROC CIs, paired-bootstrap dAUROC,
#       beta sensitivity from cached scores, per-category table;
#       figures saved to results/e1/figures/)
```

## E2 / E3

- **E2 (replication, Llama-3.2-3B):** identical sequence with
  `configs/e2.yaml`. beta* is **recalibrated** for the model (the calibrate
  step does this automatically; per-model calibration is the rule).
- **E3 (contingent, Qwen on TriviaQA rc.nocontext, 500 @ seed 42):** beta*
  is **inherited from E1** (same model — do not recalibrate). After E1
  calibration, copy `beta_star` from `results/e1/config_snapshot.yaml` into
  `configs/e3.yaml` (`calibration.beta_star`); `bank.path` already points at
  the E1 banks. Then run Stage 2 onward with `configs/e3.yaml`.

## Dry run (`--limit 5`, exercises every stage locally)

```bash
python scripts/build_banks.py          --config configs/e1.yaml
python scripts/calibrate_beta.py       --config configs/e1.yaml
python scripts/capture_trajectories.py --config configs/e1.yaml --limit 5
python scripts/label_samples.py        --config configs/e1.yaml --limit 5
python scripts/build_features.py       --config configs/e1.yaml
python scripts/run_probe.py            --config configs/e1.yaml   # <30 labelled -> smoke mode
```

On a small local GPU set `model.load_in_4bit: true` (dev only). With <30
labelled samples `run_probe.py` enters an explicit smoke mode (exercises the
code path, prints that the numbers are meaningless).

## Resume semantics

- A sample is *complete* iff its `{id}.json` exists and parses; the JSON is
  always written **last** (after the npz files), so it is the commit marker.
- Stage 2/3 scan the manifest at startup and process only missing samples.
- SIGUSR1/SIGTERM (sent by SLURM 120 s before walltime via
  `--signal=B:USR1@120`) are trapped: the current sample is finished,
  flushed, and the process exits cleanly in a resumable state.
- A failing sample is recorded in `trajectories/failures.jsonl` (with
  traceback) and gets NaN rows — median imputation in the probe handles it —
  then the run continues. One bad sample never kills a shard.
- `--max-samples-per-job M` + `SELF_RESUBMIT=1` in `hpc/trajectories.sbatch`
  chunk work across jobs on short-walltime partitions.
- OOM defenses baked in: `torch.inference_mode()` everywhere, batch size 1,
  scores moved to CPU inside the hook, per-sample artifact flush + release
  (nothing accumulates across the run), `del` + `torch.cuda.empty_cache()`
  per sample, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the
  sbatch templates, GPU/RSS logged every `trajectory.log_memory_every`
  samples.

## Disk budget (E1: Qwen2.5-3B, L=36, K=11008, 817 samples, T_g <= 50)

| Artifact | Per sample | E1 total | Notes |
|---|---|---|---|
| `{id}.npz` (scalars: [L] ref + [L,T_g] per-token) | ~100 KB | ~80 MB | always written |
| `{id}.json` (metadata) | ~2 KB | ~2 MB | always written, commit marker |
| `{id}_retrieval_scores.npz` (float16 [L,K] + [L,T_g,K]) | ~40 MB | **~33 GB** | E1 only (`cache_score_tensors: true`); feeds the beta sweep with zero forward passes |
| `{id}_label.json` | ~1 KB | ~1 MB | Stage 3 |
| `banks.pt` (float32 m_hat + norms) | — | ~3.3 GB | once per model |

E2/E3 (no score cache): ~100 MB per run plus banks. Hidden states are never
persisted.

## Layout

```
energy_llm/        banks.py trajectory.py calibration.py labeling.py
                   features.py probe.py stats.py metrics.py
                   io_artifacts.py config.py models.py data.py
configs/           e1.yaml e2.yaml e3.yaml
scripts/           one CLI per stage (build_banks, calibrate_beta,
                   capture_trajectories, label_samples, build_features, run_probe)
hpc/               banks/calibrate/trajectories/features.sbatch + submit_pipeline.sh
                   (resource directives preserved from the proven working jobs)
notebooks/         01_labeling_openai.ipynb  02_results_analysis.ipynb
tests/             pytest, CPU-only, tiny random Qwen2-style model
docs/              fundamentals-FINAL.tex  metodology-FINAL.tex  MAPPING.md
results/           per-experiment artifacts (gitignored)
```

## Naming convention (thesis-aligned, everywhere)

`probe_point` (r), `gate_preactivations` (g), `retrieval_scores` (s),
`memory_bank` / `m_hat` (M, row-normalised), `row_norms`, `beta_star`,
`phi`, `delta_energy`, `delta_entropy` (diagnostic), `delta_norm_entropy`,
`mean_js`, `mean_hellinger_sq`, `lse_term` + `quadratic_term` (energy stored
decomposed). Layers are 0-indexed `l = 0..L-1` in code, artifact keys
(`*_l{l}`), configs, and notebooks.
