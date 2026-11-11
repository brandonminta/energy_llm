# CLAUDE.md

Guidance for Claude Code in this repository. Everything here describes the
code **as it is**; `docs/overview.md` is the science it serves. When they
disagree, the code is the fact; Brandon decides what changes.

## What this is

`energy_llm` is an **extraction module**: per question and per layer it
computes Modern-Hopfield retrieval metrics from the SwiGLU gate of
Qwen2.5-3B-Instruct / Llama-3.2-3B-Instruct, under two score variants
(`cos_gamma_query` primary, `cos_gamma_key` control), and writes them to disk
indexed by `id`. It ends at the metrics — no labelling, judges, probes, CV or
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
pytest                             # ~80-115 s; tiny random Qwen2 + Llama models; GPU files skip without CUDA
# locally, if python is not on PATH:
micromamba run -n energy-lab python -m pytest -q
```

Run the tests after every change; with a CUDA device 192 pass + 1 skipped
(real tokenizers, opt-in) is the baseline (v2.1.0 + step 10); without CUDA,
183 pass + 10 skipped (`test_gpu.py`, `test_gpu_verify.py`).

## CLI and layout

```
energy-llm calibrate --config C --input warmup.jsonl          -> <out>/calibration/{beta_star.json,bank.h5}
energy-llm extract   --config C --input eval.jsonl --name N --shard-index K
energy-llm status    --config C --input eval.jsonl --name N [--missing|--num-shards|--json|--extraction-dir]
energy-llm merge     --config C --input eval.jsonl --name N   -> <out>/extract/N/{manifest.json,*.parquet,*.h5}
energy-llm smoke     --config C --input eval.jsonl [--n 20]
energy-llm verify    --config C --input dev/dev_questions.jsonl [--n 4]      (verification only)
energy-llm inspect   --run <extraction_dir> --id ID [--variant V]            (verification only)

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
- Architecture adapter (`core.model.ffn_norm` / `effective_gain`, by
  `model_type`): qwen2/llama only (read point `post_attention_layernorm`,
  gain = its weight gamma); gemma* raises NotImplementedError (gain 1+gamma,
  `pre_feedforward_layernorm`); others rejected. No fixed layer or dimension
  index in code or config (smoke bands are thirds of l/L).
- Hook: forward pre-hook on each layer's FFN norm of `bundle.trace_model` —
  the residual x at its INPUT. `x_hat = x/sqrt(mean(x^2)+eps)` with the
  model's eps, float32.
- Bank: rows w_i of `gate_proj.weight`, gamma, `W_tilde = W_g * gamma`
  (column-wise) — NEVER materialised: only gamma, `key_norms = ||w_i||` and
  `row_norms = ||w_tilde_i||` are kept (bank.h5, all in the fingerprint). ONE
  pre-activation per layer `z = W_g.float() @ (gamma*x_hat)` (one layer's W_g
  in float32 at a time) serves both variants: `cos_gamma_query` (PRIMARY)
  `c = z / (||w|| ||gamma*x_hat||)`; `cos_gamma_key` (CONTROL)
  `s = z / (||w_tilde|| sqrt(D))`. `query_gain g_t = ||gamma*x_hat||/sqrt(D)`;
  `s = g_t (||w||/||w_tilde||) c`. No CPU-bank fallback.
  `g_i == ||w_i|| ||gamma*x_hat|| c_i == ||w_tilde_i|| sqrt(D) s_i`.
- `E = -(1/beta) logsumexp(beta s)` (no quadratic term); `H_bar = H/log(D_ff)`;
  margin = top1-top2. All bank, score, softmax and metric computation is
  float32. Every trace exists per variant (`<variant>/<name>`);
  `query_gain` and `xhat_top1_frac` once; `topk_ids` only for the primary
  unless `extraction.control_topk_ids_k > 0`.
- Divergences (JS, KL) and topk_mass are computed from LOG-probabilities.
  Never reintroduce p-domain versions (float32 underflow at large beta; old
  bug). JS is pointwise in d = log p - log q (ln2 = softplus(0), xlogy
  masking) so JS(p,p) and KL(p||p) are exactly 0. Hellinger and Jaccard were removed (Jaccard of topk_ids is analysis).
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
  checked against greedy and mismatches recorded (with top1-top2 logit
  margins; smoke criterion: every disagreement has margin < delta).
- dtype: model must load as bfloat16 (fail otherwise); actual dtype recorded.
  `model.trace_dtype` (bfloat16 | float32): generation always in the model
  dtype; the traced TF pass on `bundle.trace_model` (a second float32
  instance with exactly the same bank weights when float32; the bank is
  built from it). Part of the calibration context. transformers >= 4.56.
- Extraction betas, per variant: grid RELATIVE to beta*_v,
  `beta*_v * 10^o` for `extraction.beta_grid_log10_offset`
  ({min: -0.8, max: 0.8, num: 9}: offset 0 included), beta*_v appended last; the absolute
  `beta_grid_log10` is rejected; `merge` asserts identical per-variant
  (betas, topk_mass_k, topk_ids_k) across items.
- Gamma diagnostic (top-5 |gamma| dims per layer and their share of
  sum ||w_tilde_i||^2) is report only: bank.h5 attrs + manifest; the
  ablations and the layer exclusion rule (first layer + CI-backed gamma
  ablation sign flips at the re-matched beta, proposed from cos_gamma_query)
  live in the smoke report and are NEVER applied. No production ablation flag.
- verify: every bank check for both variants; `tf_vs_cache` PASS only on the
  analysis positions (response + question_last, prompt_last, response_first),
  template / position 0 reported apart; precision rule in `fp32_reference`
  (p99/sigma > 0.1 -> `RECOMMEND trace_dtype=float32`). Comparisons between
  paths (tf, cache, fp32) on CPU float64 inside the helper (`verify.cpu64`).
- beta*: one per model AND score variant, from `calibrate` (warm-up JSONL
  disjoint from evaluation; last prompt token + response tokens; shared log
  grid; fail if either is not bracketed — never clamp; grid position printed,
  WARNING within 0.5 decades of an edge). `extract` never recalibrates and
  refuses a record missing a variant, a beta* from another context or
  overlapping questions.
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
