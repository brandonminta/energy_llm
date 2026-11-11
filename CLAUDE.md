# CLAUDE.md

Guidance for Claude Code in this repository. Everything here describes the
code **as it is**; `docs/overview.md` is the science it serves. When they
disagree, the code is the fact; Brandon decides what changes.

## What this is

`energy_llm` is an **extraction module**: per question and per layer it
computes Modern-Hopfield retrieval metrics from the SwiGLU gate of
Qwen2.5-3B-Instruct / Llama-3.2-3B-Instruct and writes them to disk indexed
by `id`. It ends at the metrics — no labelling, judges, probes, CV or
statistics (those live in Brandon's analysis repository; do not add such
code to the package).

## Documents and their authority

| document | role |
|---|---|
| `README.md` | how to run: setup, SLURM workflow, outputs, verification on the cluster. |
| `docs/ARCHITECTURE.md` | **what the code does today**, module by module, file formats, verification tolerances, and §13 the audit of inconsistencies. Keep it true when you change behaviour. |
| `docs/overview.md` | why: the scientific specification. |

`docs/CHANGES_v0.2.md` is the change log of the v0.2 plan (record, not a
specification). `dev/README.md` describes the verification data.

Open points listed in ARCHITECTURE.md §13 are **not** to be "fixed" on your
own initiative: they are the list Brandon uses to decide what to correct
according to his research. Mention them; change them only when asked.

## Setup and tests

```bash
micromamba activate hopfield-llm   # HPC env name; local dev env: energy-lab
pip install -r requirements.txt && pip install -e ".[dev]"
pytest                             # CPU only, ~55 s; tiny random Qwen2 + Llama models
# locally, if python is not on PATH:
micromamba run -n energy-lab python -m pytest -q
```

Run the tests after every change; 130 pass + 1 skipped (real tokenizers,
opt-in) is the baseline (v2.1.0).

## CLI and layout

```
energy-llm calibrate --config C --input warmup.jsonl          -> <out>/calibration/{beta_star.json,bank.h5}
energy-llm extract   --config C --input eval.jsonl --name N --shard-index K
energy-llm status    --config C --input eval.jsonl --name N [--missing|--num-shards|--json|--extraction-dir]
energy-llm merge     --config C --input eval.jsonl --name N   -> <out>/extract/N/{manifest.json,*.parquet,*.h5}
energy-llm smoke     --config C --input eval.jsonl [--n 20]
energy-llm verify    --config C --input dev/dev_questions.jsonl [--n 4]      (verification only)
energy-llm inspect   --run <extraction_dir> --id ID                          (verification only)

energy_llm/   cli.py config.py storage.py manifest.py
  core/         model prompting bank metrics capture      what is computed; never touches output_dir
  pipeline/     signals calibrate extract merge smoke     everything under output_dir
  verification/ verify inspection                         never production outputs
hpc/          env.sh (all resource values + run_requeueable), submit.sh, *.sbatch,
              missing_shards.sh, prefetch_models.sh, pilot_bundle.sh (login node)
dev/          verification-only data + reference.py (imports nothing from energy_llm)
```

Dependency direction: `verification → pipeline → core → {config, storage,
manifest}`. Keep it that way. `cli.py` imports commands lazily.

Configs: `configs/qwen25_3b.yaml`, `configs/llama32_3b.yaml` (hub revisions
pinned). `output_dir` uses `${SCRATCH_ROOT}`, which is mandatory.

## Invariants as implemented (tests enforce most)

- Input JSONL `{id, question}`; every other field is dropped at read time and
  never reaches an output. Everything is indexed by `id`.
- Hook: forward pre-hook on each layer's `post_attention_layernorm` — the
  residual x at its INPUT. `x_hat = x/sqrt(mean(x^2)+eps)` with the model's
  eps, float32.
- Bank: `W_tilde = gate_proj.weight * post_attention_layernorm.weight`
  (column-wise), conceptually unit rows — NEVER materialised: only gamma and
  the row norms are kept (bank.h5); `s = (W_g.float() @ (gamma*x_hat)) /
  (norms*sqrt(D))`, one layer's W_g in float32 at a time. No CPU-bank
  fallback. `g_i == ||w_tilde_i|| * sqrt(D) * s_i`.
- `E = -(1/beta) logsumexp(beta s)` (no quadratic term); `H_bar = H/log(D_ff)`;
  margin = top1-top2. All bank, score, softmax and metric computation is
  float32.
- Divergences (JS, KL) and topk_mass are computed from LOG-probabilities.
  Never reintroduce p-domain versions (float32 underflow at large beta; old
  bug). Hellinger and Jaccard were removed (Jaccard of topk_ids is analysis).
- No features table: aggregation over tokens is analysis, not extraction.
  Per-token traces per layer (never mix layers) over every position at
  every beta; `role_mask` separates prompt and response. Response =
  generated tokens excluding EVERY stop token (generation_config eos list +
  tokenizer eos). Divergences (js, signed dkl) to three references:
  question_last, prompt_last, response_first (NaN for an empty response).
  Computed in float32, stored float16 (topk_ids int32).
- Prompt: official chat template, explicit single system instruction
  ("Answer the question briefly and directly."), no few-shot; Llama
  `date_string` fixed. Greedy with a clean generation config (Qwen's hub
  repetition_penalty=1.05 must never leak in); teacher-forced argmax is
  checked against greedy and mismatches recorded.
- dtype: model must load as bfloat16 (fail otherwise); actual dtype recorded.
  transformers >= 4.56.
- Extraction betas: absolute log grid `extraction.beta_grid_log10`
  (10^0..10^4, 9 points, same for every model since s is a cosine) with
  beta* appended last; `merge` asserts identical (betas, topk_mass_k,
  topk_ids_k) across items.
- Gamma diagnostic (top-5 |gamma| dims per layer and their share of
  sum ||w_tilde_i||^2) is report only: bank.h5 attrs + manifest; the
  ablation lives in the smoke report. No production ablation flag.
- beta*: one per model, from `calibrate` (warm-up JSONL disjoint from
  evaluation; last prompt token + response tokens; log grid; fail if not
  bracketed — never clamp). `extract` never recalibrates and refuses a beta*
  from another context or overlapping questions.
- Robustness: one shard dir per array task; per-id atomic item file, then
  `completed.jsonl`; failures never marked complete — `failed.jsonl`
  (shared schema via `storage.append_failure`), retried up to
  `max_attempts`. USR1/SIGTERM -> finish current id, exit 75 ->
  `scontrol requeue` (the ONLY retry mechanism; no bash retry loops). Shards
  refuse to resume under a changed input/config/beta*.
- Empty responses are VALID samples: `empty_response=true`, completed,
  traces computed for every position, response-only quantities NaN by
  design.
- Orphaned merges: every submission is logged in
  `<extraction_dir>/submissions.log`; `missing_shards.sh --resubmit` cancels
  pending merges, never relaunches a shard with an active task, and chains a
  new merge on the relaunched array + still-active tasks.
- Verification only (never production outputs): `energy-llm verify`,
  `energy-llm inspect`, `hpc/verify.sbatch`, `dev/`. Any beta* calibrated on
  `dev/dev_warmup.jsonl` never goes in the paper.
- SLURM resource values live in `hpc/env.sh` and come from the previously
  working jobs (gpu partition, gpu:a100-sxm4-40gb:1, 8 CPUs, 64G, 3 h for
  extract — 3 h approved 2026-10-26; calibrate 80G/6 h; merge on cpu
  4 CPU/16G/1 h). Do NOT change partition/GRES/mem/time without asking
  Brandon. `SCRATCH_ROOT` stays a required variable.
- Batch size 1. Seed 42. Never hardcode HF/OpenAI tokens (`HF_TOKEN` env).

## Working rules

- Prefer simple, reusable code: DRY/KISS/YAGNI. Reuse or parameterise before
  adding a function; no new abstraction, class or layer without a concrete
  reason; no trivial one-line wrappers. Preserve behaviour when refactoring.
- When behaviour changes, update `docs/ARCHITECTURE.md` (and §13 if an
  open point is resolved), `README.md` if commands or outputs move, and the
  tests.
- Generated renders of the docs (`docs/*.html`, `docs/*_files/`) are
  git-ignored; do not commit them.
