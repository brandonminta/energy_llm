# Run Manifest — `energy_llm` (E1 / E2 / E3)

Read-only audit. Generated from the artefacts under `results/` and the
checked-in code at repo HEAD. Every value is marked **[read]** when taken
verbatim from a stored artefact, or **[inferred]** when it is not recorded in
any artefact and had to be deduced.

## Provenance note (applies to all three runs)

- **No `config_snapshot.yaml` is present** in any `results/<exp>/` directory,
  and **no `banks.pt`** survives on disk. `beta_star` is therefore read from
  `results/<exp>/calibration/calibration.json` and `probe_results.json`, not
  from the snapshot the methodology says all stages consume.
- **The per-sample artefacts are gone**: `results/<exp>/trajectories/` and
  `results/<exp>/labels/` exist but contain **0** `*.json` files, **0**
  `*_retrieval_scores.npz`, and **0** `*_label.json`. Only the *aggregated*
  artefacts (`features/features.csv`, `labels.json`, `generated_texts.jsonl`,
  `human_labels.csv`, `split.json`, `probe_results.json`,
  `analysis_summary.json`, and for E1 `beta_sensitivity.csv`) remain.
  Reproduction below is built on those aggregates.
- **No git commit hash is stored in any run artefact**, and **no HuggingFace
  model revision is pinned anywhere** in `energy_llm/`, `scripts/`, or
  `configs/` (grep for `revision` returns nothing; `models.py:73` calls
  `from_pretrained(model_id, ...)` with no `revision=`).

## Inferred run commit

Artefact mtimes vs. `git log --date=short`:

| Artefact | mtime | Nearest commit(s) on that day |
|---|---|---|
| `results/e1/probe_results.json`, `features.csv` | 2026-06-13 16:43 | `e50ef35` / `91dd264` (both 2026-06-13) |
| `results/e1/beta_sensitivity.csv` | 2026-06-13 22:59 | `529b88f` (HEAD, 2026-06-13) |

**[inferred]** All three runs' GPU stages and probe results were produced on
2026-06-13 at or before HEAD `529b88f`; the beta-sensitivity CSV was produced
under `529b88f`. The exact per-run commit is **not recorded** — treat
`529b88f` as the upper bound.

---

## E1 — primary

| Field | Value | Source |
|---|---|---|
| Experiment name | `e1_qwen_truthfulqa` | [read] `probe_results.json`, `configs/e1.yaml` |
| Run commit | `529b88f` (upper bound) | [inferred] |
| Model id | `Qwen/Qwen2.5-3B-Instruct` | [read] `calibration.json` |
| Model alias / dims | `qwen25_3b`, L=36, D=2048, D_ff=11008 | [read] `models.py:28`; L=36 confirmed by `n_layers` in `probe_results.json` |
| HF revision | not pinned | [inferred] NOT-FOUND |
| dtype | bfloat16 (A100) | [read] `configs/e1.yaml` |
| Dataset / split | TruthfulQA `generation` / `validation`, all questions | [read] `data.py:29`, `configs/e1.yaml` |
| Sample count (labelled) | 817 | [read] `probe_results.json` `n_labelled`; `features.csv` = 817 rows |
| Hallucination rate | 0.4835 (395 of 817 positive) | [read] `analysis_summary.json`; `features.csv` label counts {0:422, 1:395} |
| **beta_star** | **1.9578453524806456** (clamped=false) | [read] `calibration.json`, `probe_results.json` |
| beta grid | [1,2,5,10,15,20,30,50,100] | [read] `calibration.json` |
| target H̄* | 0.55 | [read] `calibration.json` |
| N_c (calibration) | 30 (ids listed; TruthfulQA) | [read] `calibration.json` |
| Seed | 42 (data, calib, split, CV, bootstrap, permutation) | [read] `configs/e1.yaml`; single `seed` field threaded everywhere |
| max_new_tokens | 50, greedy | [read] `configs/e1.yaml`, `trajectory.py:183` |
| cache_score_tensors | true (E1 only) | [read] `configs/e1.yaml` |
| Split sizes | train 571 / val 123 / test 123 | [read] `split.json`, `probe_results.json` |
| Test split hash | `ceb27bae…1bfae0` | [read] `split.json`; reverified (matches) |
| Probe C grid | [0.01, 0.1, 1, 10]; inner 3-fold, outer 5-fold | [read] `configs/e1.yaml` |
| Selected C (base / full) | 10.0 / 0.01 | [read] `probe_results.json` |
| Bootstrap / permutation B | 10000 / 1000 | [read] `configs/e1.yaml`, `probe_results.json` |
| Judge | gpt-4o-mini (OpenAI chat-completions) | [read] `configs/e1.yaml`, `labels.json` populated |
| Human subset | 60 samples (30/30), κ=0.833, ag=91.7% | [read] `human_labels.csv` (reproduced) |

**SLURM (from `hpc/*.sbatch`, [read]):** banks — `gpu`, `a100-sxm4-40gb:1`,
8 cpus, 32G, 02:00:00. calibrate — `gpu`, a100-40gb, 8 cpus, 80G, 06:00:00.
trajectories — `gpu`, a100-40gb, 8 cpus, 64G, 12:00:00, `--array=0-7%4`,
`--signal=B:USR1@120 --requeue`. features (+probe) — `cpu`, 4 cpus, 16G,
01:00:00. beta_sensitivity — `cpu`, 8 cpus, 32G, 02:00:00.
*These directives are checked-in defaults; actual `sbatch` submission
parameters for the real run are not separately logged.*

**Artefact paths produced (present on disk):**
`results/e1/calibration/calibration.json`, `.../features/features.csv`,
`.../labels.json`, `.../generated_texts.jsonl`, `.../human_labels.csv`,
`.../split.json`, `.../probe_results.json`, `.../analysis_summary.json`,
`.../beta_sensitivity.csv`, `.../figures/*`.
**Absent:** `banks.pt`, `config_snapshot.yaml`, `trajectories/*.npz|json`,
`trajectories/*_retrieval_scores.npz`, `labels/*_label.json`.

---

## E2 — replication

| Field | Value | Source |
|---|---|---|
| Experiment name | `e2_llama_truthfulqa` | [read] `probe_results.json` |
| Run commit | `529b88f` (upper bound) | [inferred] |
| Model id | `meta-llama/Llama-3.2-3B-Instruct` | [read] `calibration.json` |
| Model alias / dims | `llama32_3b`, L=28, D=3072, D_ff=8192 | [read] `models.py:29`; L=28 confirmed by `probe_results.json` |
| HF revision | not pinned | [inferred] NOT-FOUND |
| dtype | bfloat16 | [read] `configs/e2.yaml` |
| Dataset / split | TruthfulQA `generation` / `validation`, all | [read] `configs/e2.yaml` |
| Sample count (labelled) | 817 | [read] `probe_results.json`; `features.csv` = 817 |
| **beta_star** | **5.058341935910842** (recalibrated, clamped=false) | [read] `calibration.json`, `probe_results.json` |
| N_c | 30 (TruthfulQA ids — same id set as E1) | [read] `calibration.json` |
| Seed | 42 | [read] `configs/e2.yaml` |
| cache_score_tensors | false | [read] `configs/e2.yaml` |
| Split sizes | 571 / 123 / 123 | [read] `split.json`, `probe_results.json` |
| Test split hash | `a7d40917…ae0198` | [read] `split.json`; reverified |
| Selected C (base / full) | 10.0 / 10.0 | [read] `probe_results.json` |
| Human subset | 60 (30/30), κ=0.900, ag =95.0% | [read] `human_labels.csv` (reproduced) |

**SLURM:** same `hpc/*.sbatch` templates as E1. README/`submit_pipeline.sh`
note `NUM_SHARDS=16` is suggested for E2 (`--array=0-15%4`) — [read] from the
usage comment; not separately logged per run.

**Artefacts present / absent:** identical pattern to E1 except **no
`beta_sensitivity.csv`** (sensitivity is defined on E1 only) and no cached
score tensors.

---

## E3 — exploratory (Qwen + TriviaQA)

| Field | Value | Source |
|---|---|---|
| Experiment name | `e3_qwen_triviaqa` | [read] `probe_results.json` |
| Run commit | `529b88f` (upper bound) | [inferred] |
| Model id | `Qwen/Qwen2.5-3B-Instruct` (banks reused from E1) | [read] `calibration.json`; `configs/e3.yaml` `bank.path: results/e1/banks.pt` |
| Model alias / dims | `qwen25_3b`, L=36 | [read]; L=36 confirmed by `probe_results.json` |
| HF revision | not pinned | [inferred] NOT-FOUND |
| Dataset / split | TriviaQA `rc.nocontext` / `validation`, 500 @ seed 42 | [read] `data.py:50`, `configs/e3.yaml` |
| Sample count (labelled) | 500 | [read] `probe_results.json`; `features.csv` = 500 |
| **beta_star** | **1.9881622367733318** (clamped=false) | [read] `calibration.json`, `probe_results.json` |
| beta_star provenance | **DEVIATION from config comment** — see note | [read] |
| N_c | 30 (TriviaQA ids) | [read] `calibration.json` |
| Seed | 42 | [read] `configs/e3.yaml` |
| Split sizes | train 350 / val 75 / test 75 | [read] `split.json`, `probe_results.json` |
| Test split hash | `0e21296b…d865b0` | [read] `split.json`; reverified |
| Selected C (base / full) | 0.01 / 0.01 | [read] `probe_results.json` |
| **Test AUROC vs outer-CV (full)** | **0.8874 (test) vs 0.7715 (outer-CV mean ±0.039)** | [read] `probe_results.json` |
| Human subset | 60 (30/30), κ=0.900, agr=95.0% | [read] `human_labels.csv` (reproduced) |

> **E3 beta_star note (DEVIATION, factual):** `configs/e3.yaml` instructs
> "E3 inherits the E1 value … do NOT recalibrate on TriviaQA." But the stored
> `results/e3/calibration/calibration.json` shows beta_star **1.98816** derived
> from a **TriviaQA** calibration curve over **TriviaQA** sample ids
> (`triviaqa_*`), i.e. E3 *was* recalibrated on TriviaQA. The inherited E1
> value is 1.95785. The two happen to be close (1.988 vs 1.958), so the
> numerical impact is small, but the operational instruction "inherit, do not
> recalibrate" was not followed. This is the *only* place E3's actual run
> diverges from its own config's written intent.

**SLURM:** same templates; E3 reuses E1's banks so the banks job is skippable.

**Artefacts present / absent:** same pattern as E2 (no sensitivity CSV, no
score tensors, no per-sample trajectories/labels, no snapshot/banks).
