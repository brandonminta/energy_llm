# energy_llm — architecture and state of the code

> **What this document is.** A description of what the code in this
> repository *actually does*, as of 2026-10-27, written from the source.
> It is not the methodology of the thesis: where the code departs from
> `docs/fundamentals-FINAL.tex` / `docs/metodology-FINAL.tex` the departure
> is stated here as a fact about the code (section 13 collects them all)
> and the detailed item-by-item mapping lives in [`MAPPING.md`](MAPPING.md).
> Use it to decide what must change; it does not say what *should* be.

Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Repository map](#2-repository-map)
3. [Package layers](#3-package-layers)
4. [One question end to end](#4-one-question-end-to-end)
5. [The computation, as coded](#5-the-computation-as-coded)
6. [β\* calibration](#6-β-calibration)
7. [Extraction runs: shards, resume, failures, signals](#7-extraction-runs-shards-resume-failures-signals)
8. [Merge and the output files](#8-merge-and-the-output-files)
9. [Smoke, verify, inspect](#9-smoke-verify-inspect)
10. [Configuration reference](#10-configuration-reference)
11. [HPC layer](#11-hpc-layer)
12. [Tests](#12-tests)
13. [Audit: inconsistencies and open points](#13-audit-inconsistencies-and-open-points)
14. [Glossary](#14-glossary)

---

## 1. Purpose and scope

`energy_llm` is an **extraction module**. For every question of a JSONL
input and every decoder layer of a Qwen2/Llama-style model it computes
Modern-Hopfield retrieval metrics (energy, normalised entropy, margin,
divergences) from the residual stream that feeds the SwiGLU gate, using the
gate matrix (with the RMSNorm scale folded in) as the memory bank. It writes
everything to disk, indexed by question `id`.

It deliberately **ends at the metrics**. Labelling (judge), probes, cross
validation and statistics are *not* in the package: the old versions live
in `downstream/`, are not installed, and target a previous artefact format.

Targets: `Qwen/Qwen2.5-3B-Instruct` and `meta-llama/Llama-3.2-3B-Instruct`
at pinned hub revisions, bfloat16, batch size 1, one A100 per job, SLURM
array jobs that survive walltime kills and preemption without supervision.

## 2. Repository map

```
energy_llm/                 the installed package (CLI: energy-llm, python -m energy_llm)
  cli.py                    argparse entry point; one cmd_* per sub-command
  config.py                 RunConfig: the YAML schema, validation, hashes
  storage.py                atomic writes, JSONL logs, failure log, hashing, input items
  manifest.py               provenance: git commit, library versions, hardware, seeds
  core/                     WHAT is computed (no output directory involved)
    model.py                loading at a pinned revision, dims from the checkpoint, clean greedy config
    prompting.py            chat template, single system instruction, per-token role mask
    bank.py                 the unmaterialised bank: gamma + row norms, on-the-fly cosine scores
    metrics.py              float32 primitives: x_hat, E, log p, H_bar, margin, JS, Hellinger^2, Jaccard
    capture.py              per question: greedy + teacher forcing + hooks + every metric
  pipeline/                 HOW it runs (everything that touches <output_dir>)
    signals.py              SIGUSR1/SIGTERM -> finish the current id, exit 75
    calibrate.py            beta* from a warm-up JSONL -> calibration/{beta_star.json, bank.h5}
    extract.py              one shard: identity, resume, per-id HDF5, completed/failed logs
    merge.py                shard status; final parquet/HDF5 files + manifest.json
    smoke.py                extract + merge on N questions in one process, summary
  verification/             never a production output
    verify.py               checks vs the model and vs dev/reference.py, PASS/FAIL report
    inspection.py           position/phase table of one id from a finished run
configs/                    qwen25_3b.yaml, llama32_3b.yaml (revisions pinned)
hpc/                        env.sh (resources + helpers), submit.sh, *.sbatch,
                            missing_shards.sh, prefetch_models.sh, RUNBOOK.md
tests/                      CPU-only; tiny random Qwen2 + Llama models built in conftest.py
dev/                        verification only: 24 dev questions, 12 warm-up questions,
                            reference.py (naive float64, imports nothing from energy_llm)
docs/                       this file, MAPPING.md, the thesis .tex (never edited by tooling)
downstream/                 NOT part of the module: legacy labelling/probe/stats, notebooks
archive/, results/          old runs, git-ignored
```

Entry points: `pyproject.toml` installs `energy-llm = energy_llm.cli:main`;
`python -m energy_llm` is the same. Dependencies are pinned once in
`requirements.txt` (transformers ≥ 4.56 is required because older releases
ignore `dtype=` and load float32).

## 3. Package layers

```
          cli.py
            │  (lazy imports inside each cmd_*)
            ▼
   verification/  ──►  pipeline/  ──►  core/
   (verify, inspect)   (calibrate,      (model, prompting, bank,
                        extract, merge,  metrics, capture)
                        smoke, signals)
            │               │               │
            └───────────────┴───────────────┘
                            ▼
              config.py   storage.py   manifest.py
```

Rules the code follows today:

- `core/` never reads or writes the output directory and never imports
  `pipeline/` or `verification/`. It imports `storage.atomic_path` (bank.h5
  writer) and nothing else from outside `core/`.
- `pipeline/` owns every file under `<output_dir>`. `merge` imports
  `extract` (shard layout); `smoke` imports both; `calibrate` and `extract`
  do not import each other.
- `verification/` imports `pipeline` (it runs the production extract +
  merge through `smoke.run_single_process`) and `core`, but nothing in the
  production path imports `verification/`.
- `cli.py` imports every command lazily so that `status`/`merge`/`inspect`
  never import torch-heavy modules they do not need.
- `dev/reference.py` imports nothing from `energy_llm` (by design: it is
  the independent oracle); `verification/verify.py` loads it by path.

## 4. One question end to end

For one item `{id, question}` inside `extract` (the same sequence runs in
`calibrate`, `smoke` and `verify`):

| step | module : function | what happens |
|---|---|---|
| 1 | `storage.read_items` | JSONL line → `Item(id, question.strip())`; every other field is dropped |
| 2 | `core.prompting.build_prompt` | official chat template on `[system, user]`, `add_generation_prompt=True`; tokenised with `add_special_tokens=False`; role per token from the offset mapping |
| 3 | `core.capture.forward_item` | `model.generate` with the clean greedy config (hooks disabled) → `gen_ids`, per-token log-prob and entropy from the decoding logits (float32), `truncated`, `stop_token_id`, `response_positions` |
| 4 | same | one teacher-forced forward pass over `prompt ++ gen_ids` with `use_cache=False`, hooks **enabled** → residual `x` per layer `[T, D]` (model dtype); argmax of its logits compared with `gen_ids` → `tf_mismatch_positions` |
| 5 | `core.capture.compute_item` | per layer: `x̂`, scores `s` `[T, D_ff]`, all features at every β and variant, per-token traces at β\*, hidden states, `answer` and `baselines` dicts, finiteness checks |
| 6 | `pipeline.extract.write_item` | one HDF5 file per id (tmp + fsync + rename), then one line in `completed.jsonl` |
| 7 | `pipeline.merge.run_merge` | later, on CPU: every item file in input order → parquet + HDF5 + `manifest.json` |

## 5. The computation, as coded

### 5.1 Model loading and validation (`core/model.py`)

- `load_model`: `AutoModelForCausalLM.from_pretrained(id, revision, dtype=torch.bfloat16)`
  then `.to(device)`; `device: auto` → cuda if available else cpu. The
  tokenizer must be a *fast* tokenizer (offset mapping is needed for the
  role mask). With `HF_HUB_OFFLINE=1` only the local cache is read.
- `prepare_bundle`: `model.eval()`, gradients off, **fails unless the
  parameters are bfloat16** (`require_bf16=True` in the CLI; tests pass
  `False`). Reads `ModelDims(L, D, D_ff, rms_norm_eps)` from `model.config`
  and checks every layer's `gate_proj.weight` is `[D_ff, D]`, every
  `post_attention_layernorm.weight` is `[D]` and every norm eps equals the
  config eps. Collects the stop tokens and installs the clean generation
  config on the model.
- `stop_token_ids`: `generation_config.eos_token_id` (int or list, e.g.
  `<|im_end|>` / `<|eot_id|>`) ∪ `tokenizer.eos_token_id`, sorted. Read
  **before** the generation config is replaced.
- `clean_generation_config`: `do_sample=False, num_beams=1,
  max_new_tokens, eos_token_id=stops, pad_token_id, repetition_penalty=1.0,
  temperature=1.0, top_k=None, top_p=1.0, no_repeat_ngram_size=0,
  return_dict_in_generate=True, output_logits=True`. It *replaces* the hub
  generation config (Qwen2.5 ships `repetition_penalty=1.05`, which would
  otherwise be merged in). Pad id = tokenizer pad or, if none, the first
  stop token.
- `release_cuda`: `gc.collect()` + `torch.cuda.empty_cache()`; called after
  every item in `extract` and after a failure in `calibrate`.

### 5.2 Prompt and role mask (`core/prompting.py`)

- Exactly two messages: `{"role": "system", "content": prompt.system}` and
  `{"role": "user", "content": question}`; `date_string` is passed as a
  template kwarg when configured. `check_template` fails when the template
  mentions `date_string` and none is configured (Llama 3.x would insert
  today's date).
- The rendered text is tokenised with `add_special_tokens=False` (the
  template already contains BOS where the model expects it).
- Role mask, one code per prompt token, from character offsets: a token
  overlapping the question span → `2 question`; overlapping the system span
  → `1 instruction`; otherwise `0 template`. Generated tokens get
  `3 response`, except the final stop token which is coded `0 template`
  (`ForwardTrace.roles`). The system sentence and the question must be found
  verbatim in the rendered text, else `PromptError`. Questions are
  `strip()`ed at read time because Llama's template trims message content.

### 5.3 Generation and teacher forcing (`core/capture.py: forward_item`)

- Greedy decoding, batch 1, attention mask of ones, `max_new_tokens` cap.
  Hooks are disabled during generation.
- From `out.logits` (one `[1, V]` per step, cast to float32): per-token
  `log p(chosen)` and the predictive entropy `-Σ p log p` (nats).
- `stop_token_id = gen[-1] if gen[-1] in stops else None`;
  `truncated = len(gen) >= max_new_tokens and stop_token_id is None`;
  `response_positions = [n_prompt + j for j, t in enumerate(gen) if t not in stops]`
  (absolute positions in the full sequence). With greedy decoding only the
  last generated token can be a stop token, so "excluding every stop token"
  and "excluding the final stop token" coincide.
- Teacher forcing: one forward pass over `prompt ++ gen` with
  `use_cache=False` and `logits_to_keep=len(gen)+1`; the argmax at positions
  `n_prompt-1 … n_prompt+len(gen)-2` is compared with `gen`;
  `tf_mismatch_positions` lists the indices `j` that differ. Mismatches are
  **recorded, not fatal** (`answers.parquet: tf_mismatch_count`,
  `manifest.json: checks.tf_mismatch_ids`).

### 5.4 Read point (`core/capture.py: ResidualHooks`)

A forward **pre**-hook on every `layer.post_attention_layernorm` copies its
input, i.e. the residual stream after the attention block and before the
FFN RMSNorm: `x ∈ R^{T×D}` in the model dtype. The normalised query is
computed later in float32 by the extractor itself:

    x̂ = x / sqrt(mean_j(x_j²) + eps)          (metrics.rms_normalise, eps = config.rms_norm_eps)

so `‖x̂‖² = D·ms/(ms+eps)` ≈ D. The maximum relative deviation of
`‖x̂‖/√D` from 1 over all layers and positions of an item is stored
(`answers.parquet: xhat_norm_max_rel_dev`).

### 5.5 Bank and scores (`core/bank.py`)

Definition per layer `l`, with `W_g = gate_proj.weight ∈ R^{D_ff×D}` and
`γ = post_attention_layernorm.weight ∈ R^D`:

    W̃ = W_g · diag(γ)          (column j scaled by γ_j)
    bank = rows of W̃ normalised to unit length     — conceptually

The bank is **never materialised**. `build_bank` keeps, per layer, a
reference to the live `gate_proj.weight` (no copy), `γ` in float32 and the
row norms `‖W̃_i‖` in float32 (computed from a transient float32 `W̃` that
is freed immediately; an all-zero row raises). Scores for float32 `x̂ [T, D]`:

    s = ((x̂ ⊙ γ) @ W_g.float().T) / (‖W̃‖ · √D)        [T, D_ff]

converting only the current layer's `W_g` to float32 and deleting it after
the matmul. Algebraically `s_i = (W̃_i · x̂) / (‖W̃_i‖ √D)` = cosine(W̃_i, x̂)
up to the ‖x̂‖/√D ≈ 1 factor, so `s ∈ [-1, 1]`. The model's own gate
pre-activation satisfies `g_i = W_g,i · (γ ⊙ x̂) = ‖W̃_i‖ · √D · s_i`
(`verify` and `tests/test_bank.py` check this against a real forward pass).

`bank.h5` (written once by `calibrate`, copied by `merge`) stores only
`layer_l/gamma [D]`, `layer_l/row_norms [D_ff]` and attrs (model id,
revision, dims, eps, definition string, `fingerprint` = sha256 of all γ and
norms). `extract` refuses to run if the live bank's fingerprint differs from
`beta_star.json` or `bank.h5`.

### 5.6 Per-position quantities (`core/metrics.py`, all float32 torch)

| quantity | code |
|---|---|
| log-distribution | `log p = log_softmax(β s)` over the D_ff memories |
| energy | `E = -(1/β) · logsumexp(β s)` — **no** quadratic term |
| normalised entropy | `H̄ = -Σ p log p / log(D_ff)`, with `p = exp(log p)` and `0·log 0 := 0` |
| margin | `top1(s) - top2(s)` on the raw scores (β-independent) |
| log of a mean distribution | `logsumexp_rows(log p) - log n` |
| Jensen–Shannon (nats) | `log m = logaddexp(log p, log q) - log 2`; `JS = ½Σ p (log p - log m) + ½Σ q (log q - log m)`, clamped to `[0, log 2]` |
| squared Hellinger | `½ Σ (exp(½ log p) - exp(½ log q))²` |
| Jaccard top-k | top-k indices of `log p` and `log q`; `|∩| / (2k - |∩|)` with `k = min(k, D_ff)` |

Every divergence takes **log-probabilities**: at the β\* values that cosine
scores produce (hundreds or more) float32 probabilities underflow, which
made the earlier p-domain JS stick at its log 2 clamp and top-k rank
arbitrary zeros (fixed in commit `652fccc`; regression test
`tests/test_metrics.py::test_divergences_survive_float32_underflow`).

### 5.7 Phases, variants, β values, features (`core/capture.py: compute_item`)

Per layer (layers are never mixed):

- **prompt phase** = the single position `n_prompt - 1` (the last prompt
  token, i.e. the state that emits the first generated token).
- **response phase** = `response_positions` (generated tokens, stop tokens
  excluded).
- Selected rows `s_sel = [prompt] + response` are reused for every β.

β values: `betas = [β* · m for m in extraction.beta_multipliers]`, default
`{0.5, 1, 2}·β*`.

Variants: `mean` (response quantities averaged over response positions; the
divergences compare the **mean of the response distributions**
`log_mean_distribution(log p[1:])` with the prompt distribution) and `first`
(the first response position stands in for the whole response, including
for the divergences).

The 11 features, in stored order (`FEATURES`):

| feature | prompt/response | definition |
|---|---|---|
| `energy_prompt` | prompt | `E` at the prompt position |
| `energy_resp` | response | mean of `E` over response positions (`first`: first position) |
| `delta_energy` | response | `energy_resp - energy_prompt` |
| `norm_entropy_prompt` | prompt | `H̄` at the prompt position |
| `norm_entropy_resp` | response | mean `H̄` over response positions |
| `delta_norm_entropy` | response | difference |
| `js` | response | `JS(mean response distribution, prompt distribution)` |
| `hellinger_sq` | response | same pair, squared Hellinger |
| `jaccard_topk` | response | same pair, top-`extraction.jaccard_top_k` Jaccard |
| `margin_prompt` | prompt | margin at the prompt position |
| `margin_resp` | response | mean margin over response positions |

Stored as `features [L, n_beta, n_variant, 11]` float32 per item; `merge`
unpivots it to a long table.

Not computed anywhere: the un-normalised entropy `H`, per-token
divergences, pooled prompt-span references, any aggregate across layers.

### 5.8 Traces, hidden states, baselines

- **Traces** at β\* only, over *all* `T = n_prompt + n_generated` positions
  (template, instruction, question, response and stop tokens alike):
  `energy [L, T]`, `norm_entropy [L, T]`, `margin [L, T]`, plus
  `role_mask [T] int8`.
- **Hidden states**: for the layers in `extraction.hidden_layers` (`all` →
  every layer), the residual `x` (same tensor the metrics read, *not* the
  layer output) at the last prompt position (`prompt_last`) and at the last
  response position (`response_last`), cast to float16 `[L_h, D]`. A value
  beyond the float16 range raises `NonFiniteError`.
- **Baselines** over response tokens, from the decoding logits:
  `mean_logprob`, `min_logprob`, `mean_token_entropy`, `n_response_tokens`,
  `empty_response`.
- **Answer record**: prompt text, decoded response (response tokens only,
  special tokens skipped), prompt and generated token ids, counts,
  `empty_response`, `truncated`, `stop_token_id`, per-token log-probs and
  entropies, TF mismatch count and positions, `xhat_norm_max_rel_dev`.

### 5.9 Empty responses and failures

- **Empty response** = the first generated token is a stop token
  (`response_positions == []`). It is a **valid, completed sample**:
  prompt-phase features are computed; every response-phase feature
  (`RESPONSE_FEATURES`), the response baselines and `hidden:response_last`
  are NaN by design; `empty_response=True` everywhere. The finiteness check
  exempts exactly those fields.
- **Failure** = any exception in prompt/forward/compute/write (OOM, a
  `PromptError`, a `NonFiniteError` for an unexpected NaN/inf, a residual
  outside float16, `RuntimeError("generation returned no tokens")`). It is
  logged to `failed.jsonl` (`id, attempt, max_attempts, error, traceback,
  time, host`), never marked complete, and retried on the next run until
  `extraction.max_attempts`; after that the id is *exhausted* and reported
  in `manifest.json: exhausted_ids`.

### 5.10 Numerics

- Model forward passes: model dtype (bfloat16 in production; float32 in
  tests). Everything after the hook — `x̂`, scores, softmax, metrics,
  decoding log-probs — is float32 on the model's device.
- `manifest.set_determinism(seed)`: seeds `random`, `numpy`, `torch`;
  disables TF32 for matmul and cuDNN; `set_float32_matmul_precision("highest")`.
- `verify` tolerances are derived in `verification/verify.py`
  (`score_tolerance`, `reference_tolerances`, `check_gate`,
  `tf_cache_tolerance`) and documented in [`MAPPING.md`](MAPPING.md).

## 6. β\* calibration (`pipeline/calibrate.py`)

Inputs: a warm-up JSONL (must be disjoint from every evaluation set) and the
`calibration` block of the config. Output: `<output_dir>/calibration/`:

```
calibration/
  state.json         {input_sha256, context, ids}   refuses a different warm-up/config in the same dir
  bank.h5            gamma + row norms (section 5.5), written once; later runs must match its fingerprint
  items/<sha1(id)>.json   one H_bar curve per warm-up question (the unit of resumption)
  failed.jsonl       failed attempts (same schema as extract)
  beta_star.json     the record below, written last
```

Per warm-up question: full `forward_item` (greedy + teacher forcing), then
`norm_entropy_curve`: for every β on the grid, `H̄` averaged over **all
layers** and over the calibration positions = last prompt token + response
tokens (stop tokens excluded). Questions are then weighted equally
(`mean_curve = mean over questions`).

Grid: `num_grid` points, log-uniform between `10^log10_beta_min` and
`10^log10_beta_max` (defaults 10⁰…10⁴, 81 points = 20 per decade).

β\*: `interpolate_log_beta` finds the first consecutive grid pair with
`curve[i] ≥ target ≥ curve[i+1]` and interpolates linearly in `log β`. If no
pair brackets the target it raises `CalibrationError` — **never clamps**.

`beta_star.json` holds `beta_star`, `target_norm_entropy`, `beta_grid`,
`mean_norm_entropy_curve`, `n_questions`, `failed_ids`, `warmup_ids`,
`warmup_question_sha256` (sha256 of the whitespace-normalised, lower-cased
question), `input_path`, `input_sha256`, `context`
(`RunConfig.calibration_context()`), `bank_fingerprint`, `code`, `created`.

`run_calibrate` returns `None` when work remains (interrupted by a signal, or
some question failed fewer than `max_attempts` times) → CLI exit 75 →
requeue. If `beta_star.json` already exists it returns it without touching
the model (the CLI even skips loading the model).

`extract` **never** recalibrates. `load_beta_star` refuses the record if
`context` differs from the current config's calibration context, or if any
evaluation `id` is in `warmup_ids` or any evaluation question hash is in
`warmup_question_sha256`.

## 7. Extraction runs: shards, resume, failures, signals (`pipeline/extract.py`)

Layout of `<output_dir>/extract/<name>/` while running:

```
shards/shard_KKKKK/
  shard.json        identity: shard_index, num_shards, ids, input_sha256, config_sha256,
                    beta_star, bank_fingerprint  — a resume under a different value refuses
  items/<sha1(id)>.h5   every output of one id (attrs id, betas, beta_multipliers;
                    datasets answer_json, baselines_json, features, energy, norm_entropy,
                    margin, role_mask, hidden_prompt_last, hidden_response_last)
  completed.jsonl   {id, key, time} appended only AFTER the item file is fsynced and renamed
  failed.jsonl      one record per failed attempt
  runs.jsonl        one record per process run: counts, timings, model dtype,
                    peak GPU memory, hardware, SLURM ids, code commit
submissions.log     written by hpc/ (array id, merge id, shard list per submission)
```

- **Shard plan**: contiguous blocks of `extraction.samples_per_shard` ids in
  input order; shard `k` ↔ SLURM array task `k`. One process owns one shard
  directory; nothing is written concurrently.
- **State** (`shard_state`): `completed` = ids in `completed.jsonl` whose
  item file exists; `attempts`/`last_error` from `failed.jsonl`;
  `exhausted` = not completed and `attempts ≥ max_attempts`;
  `pending` = the rest; `done` = no pending.
- **Resume**: a rerun processes only `pending` ids; it never recomputes a
  completed id. Orphan item files without a `completed.jsonl` line are
  recomputed (and overwritten atomically). A torn trailing JSONL line is
  skipped by `read_jsonl`.
- **Per id**: `build_prompt → forward_item → compute → write_item`, wrapped
  in `try/except Exception` (failure log) and `finally: release_cuda()`.
- **Signals** (`pipeline/signals.py`): `GracefulStop` installs handlers for
  SIGUSR1 and SIGTERM that only set a flag; the loop checks it before every
  id, so the in-flight id always finishes and is indexed. The run report
  carries `stopped_by_signal` and the signal name.
- **Exit codes** (`cli.py`): 0 when the shard is `done` (every id completed
  or exhausted); 75 (`EX_TEMPFAIL`) when pending work remains (signal or
  retryable failures) — the batch script requeues on 75; anything else is
  a crash. `cmd_extract` first checks `shard.json` + state on disk and
  returns 0 **without loading the model** when the shard is already done.
- **Guards before any work**: β\* context + overlap check, shard identity,
  chat template check, bank fingerprint equal to `beta_star.json` and to
  `bank.h5`.

## 8. Merge and the output files (`pipeline/merge.py`)

`run_merge` refuses while any shard is not `done` unless
`--allow-incomplete`. It verifies each shard's `input_sha256`/`config_sha256`
against the current input and config, reads every completed item file in
input order and writes, each atomically, into `<output_dir>/extract/<name>/`:

| file | content |
|---|---|
| `answers.parquet` | one row per id: the answer record of 5.8 |
| `features.parquet` | long: `id, layer (int16), beta, beta_multiplier, variant, empty_response` + the 11 features; `L × n_beta × 2` rows per id |
| `baselines.parquet` | one row per id |
| `traces.h5` | `/<id>/{energy, norm_entropy, margin} [L, T]` (gzip), `/<id>/role_mask [T]`; attrs `role_codes`, `role_names`, `beta_star`, `layout` |
| `hidden.h5` | `/<id>/{prompt_last, response_last} [L_h, D] float16` (gzip); attrs `layers`, `layout` |
| `bank.h5` | byte copy of `calibration/bank.h5` |
| `manifest.json` | written last: name, created, full config + `config_sha256`, β\* and the calibration summary, input path/sha/count, counts (completed/exhausted/pending), `exhausted_ids` with last error, `incomplete_shards`, checks (`xhat_norm_max_rel_dev` max, `tf_mismatch_ids`, `n_truncated`, `n_empty_response`), actual model dtypes seen, feature/variant/role names, code commit, library versions, merge host, every shard run record |

`status_report` (CLI `status`) uses the same shard states and `runs.jsonl`
records: per-shard counts, `missing_shards`, `exhausted_ids`, and a
throughput suggestion `samples_per_shard = (7200 - median model load) /
median s per item` (a 2 h target inside the 3 h walltime).

## 9. Smoke, verify, inspect

- **smoke** (`pipeline/smoke.py`): `run_single_process` deep-copies the
  config, sets `samples_per_shard = len(sel)` (one shard), writes the
  selected items to `<ext_dir>/input.jsonl`, runs `run_extract` for shard 0
  and `run_merge(allow_incomplete=True)` in the same process; `run_smoke`
  then summarises: counts, dtype, β\*, TF==greedy % of tokens and of
  answers, max `‖x̂‖` deviation, peak GPU memory, s/question, empty and
  truncated counts, mean `H̄` per layer at β\* (prompt | response). Output
  name `smoke-<UTC time>`, report `smoke_report.json`. CLI exit 1 if any id
  failed or a signal stopped it.
- **verify** (`verification/verify.py`): runs the same single-process path
  under `verify-<time>`, then on the first `n` questions runs twelve checks
  (reference, gate, tf_vs_cache, alignment, hidden, files_vs_memory,
  permutation, limits, monotone, self_divergence, determinism,
  calibration). Each check reports max observed error and tolerance;
  `verify_report.json`; exit 1 on any FAIL. Tolerances and what each check
  proves: [`MAPPING.md`](MAPPING.md) "Verification → code". Keeps its own
  minimal hooks (`_Capture`) and its own stop-token derivation on purpose.
- **inspect** (`verification/inspection.py`): no model; reads
  `manifest.json`, `answers.parquet`, `traces.h5`, `features.parquet` of a
  finished run and prints every position (token id, decoded token if the
  tokenizer is in the local cache, role, phase) and per layer `E | H̄ |
  margin` for prompt / response-mean / response-first at β\*, flagging
  disagreements between `traces.h5` and `features.parquet` above
  `1e-4·max(1,|value|)`.

## 10. Configuration reference (`config.py`)

```yaml
name: qwen25_3b                      # required
seed: 42
output_dir: ${SCRATCH_ROOT}/energy_llm/runs/qwen25_3b   # required; env vars expanded; unset var -> error
model:
  id: Qwen/Qwen2.5-3B-Instruct       # required
  revision: <40-hex hub commit>      # required, validated by regex
  dtype: bfloat16                    # the only accepted value
  device: auto                       # auto | cuda | cpu
prompt:
  system: "Answer the question briefly and directly."   # non-empty
  date_string: null                  # required when the chat template reads it (Llama)
generation:
  max_new_tokens: 50                 # >= 1
calibration:
  target_norm_entropy: 0.55          # in (0, 1)
  log10_beta_min: 0.0
  log10_beta_max: 4.0                # > min
  num_grid: 81                       # >= 2
extraction:
  beta_multipliers: [0.5, 1.0, 2.0]  # positive, must include 1.0
  jaccard_top_k: 64                  # >= 1
  hidden_layers: all                 # 'all' or a list of layer indices (validated against L at run time)
  samples_per_shard: 1000            # >= 1
  max_attempts: 3                    # >= 1 (also used by calibrate)
```

Unknown keys anywhere are errors. Derived paths: `calibration_dir`,
`beta_star_path`, `bank_h5_path`, `extraction_dir(name)` (`name` must match
`[A-Za-z0-9._-]+`).

Two hashes/contexts drive the guards:

- `RunConfig.sha256()` — everything **except `output_dir`** (so `name`,
  `seed`, `model.*` including `device`, `prompt`, `generation`,
  `calibration`, `extraction` including `samples_per_shard` and
  `max_attempts`). Stored in `shard.json` and `manifest.json`; a shard
  refuses to resume and `merge` refuses to read under a different value.
- `RunConfig.calibration_context()` — `model` (with `device` nulled),
  `prompt`, `generation`, `calibration`. Stored in `beta_star.json` and
  `calibration/state.json`; `extract` refuses a β\* with another context.

## 11. HPC layer (`hpc/`)

- `env.sh` (sourced by everything): requires `SCRATCH_ROOT`; sets `HF_HOME`,
  the micromamba env, and every SLURM resource value (gpu partition,
  `gpu:a100-sxm4-40gb:1`, 8 CPUs, 64G / 3 h extract, 80G / 6 h calibrate,
  30 min smoke, 45 min verify, cpu partition 4 CPU / 16G / 1 h merge),
  `SIGNAL_LEAD_S=120`, `MAX_CONCURRENT=4`, `MAX_REQUEUES=5`. Helpers:
  `submit_extract_array`, `submit_merge`, `record_submission`,
  `run_requeueable` (runs Python in the background, forwards USR1/TERM,
  and on exit 75 after USR1 calls `scontrol requeue` while
  `SLURM_RESTART_COUNT < MAX_REQUEUES` — the **only** retry mechanism; TERM
  = preemption/scancel is left to SLURM's `--requeue`).
- `submit.sh calibrate|smoke|verify|extract`: passes the env.sh values on
  the sbatch command line (overriding the `#SBATCH` fallbacks in the
  `.sbatch` files); `extract` asks the CLI for the shard count and the
  extraction dir, submits the array `0-(N-1)%MAX_CONCURRENT` plus a merge
  job with `afterok`, and logs both ids in `<ext_dir>/submissions.log`.
- `missing_shards.sh … [--resubmit]`: prints `status`; with `--resubmit`
  cancels still-pending merges of earlier submissions, relaunches only the
  unfinished shards that have no active array task, and submits a new merge
  depending on the relaunched array **and** the still-active tasks.
- `prefetch_models.sh` (login node): `snapshot_download` at the pinned
  revision into `HF_HOME`, then an offline load check.
- `RUNBOOK.md`: the verification sequence on the cluster (env check, real
  tokenizers, calibrate on `dev/dev_warmup.jsonl`, verify, smoke, inspect,
  requeue test). β\* values from `dev/` are verification-only.

## 12. Tests (`tests/`, CPU only, ~45 s)

`conftest.py` builds tiny random Qwen2 and Llama models (3 layers, D=32,
D_ff=48, non-trivial γ), word-level tokenizers with templates that mimic the
real ones (Qwen injects a default system prompt when none is given; Llama
reads `date_string`), and hub-style sampling settings that the extractor
must override.

| file | proves |
|---|---|
| `test_storage.py` | atomic writes leave no temp file and keep the old file on error; torn JSONL lines skipped; input validation (ids, duplicates, extra fields dropped) |
| `test_bank.py` | `g = ‖W̃‖·√D·s` against the real forward; bank keeps no copy of `W_g`; scores equal explicit unit rows and are cosines; dims from config; `bank.h5` round trip; bf16 enforcement |
| `test_metrics.py` | `‖x̂‖` formula; distributions sum to 1; entropy bounds/extremes; energy vs scipy; JS symmetry/bounds/scipy convention/log 2 on disjoint supports; log-domain underflow regression; Hellinger; margin; Jaccard; log-mean |
| `test_capture.py` | explicit system prompt and exact role mask; Llama `date_string`; hub sampling overridden; teacher forcing == greedy == explicit argmax loop; hook reads the norm input; truncation; every stop token excluded; empty response as a valid sample; shapes/bounds; phases/variants match traces; JS of the mean distribution; NaN raises |
| `test_calibrate.py` | log-β interpolation; unbracketed target fails; curve decreasing and β\* hits the target; idempotence; extract refuses without β\*, with overlap, with another context |
| `test_resume.py` | interrupted + resumed run is byte-identical to a full run; outputs indexed by id without input extras; failures retried and never completed; exhausted ids end the shard and are reported; changed config refuses to resume; status/missing shards and CLI; SIGUSR1 finishes the current id; CLI skips a finished shard without loading the model; empty responses completed and flagged; smoke summary |
| `test_invariance.py` | permuting memories changes nothing; β→0 and β→∞ limits; `H̄` monotone in β; self-divergence zero; two runs bit-identical on CPU |
| `test_reference.py` | production == `dev/reference.py` within the derived tolerances at three β\*; empty response; negative control (no γ) fails |
| `test_verify.py` | all twelve checks PASS on the tiny models; a wrong read point makes `gate` and `reference` FAIL; CLI exit code; inspect output |
| `test_gpu.py` | (skipped without CUDA) GPU == CPU within float32, peak memory recorded |
| `test_real_tokenizer.py` | (opt-in via `ENERGY_LLM_TOKENIZER_DIRS`) prompt ids equal the official template, role mask exact |

## 13. Audit: inconsistencies and open points

Facts about the current code that a reader comparing it with the
methodology, or deciding what to change next, should know. None of them
is fixed by the 2026-10-27 reorganisation (which was behaviour-preserving).

### 13.1 Code vs the thesis `.tex` (summary; details and the approval in `MAPPING.md`)

| topic | `.tex` | code |
|---|---|---|
| query / read point | `r = RMSNorm(x) = γ ⊙ x̂` (output of the second RMSNorm, `hook_target = mlp_input`) | residual `x` at the **input** of `post_attention_layernorm`; `x̂` recomputed in float32; γ folded into the bank |
| bank rows | `A = W_gᵀ` rows, unit-normalised by `‖w_i‖` | `W̃ = W_g·diag(γ)`, normalised by `‖w̃_i‖` (never materialised) |
| scores | `s_i = m̂_i · r = ‖r‖ cos θ_i` (query norm retained, no `1/√D`) | `s_i = (W̃_i·x̂)/(‖W̃_i‖√D)` ≈ cos θ_i ∈ [-1, 1] (query norm removed) |
| energy | `-(1/β) lse(β s) + ½‖q‖² (+ c₀)` | `-(1/β) lse(β s)` only |
| JS, Hellinger | mean over generated tokens of the per-token divergence to the reference | divergence between the **mean response distribution** and the prompt distribution (`first` variant: first response token only) |
| response positions | all `T_g` generated tokens | generated tokens **excluding stop tokens** |
| calibration positions | prefill reference position only, warm-up from the training split | last prompt token **+ response tokens**, warm-up JSONL disjoint from evaluation |
| β grid / no bracket | approximately log grid; **clamp** to the closest endpoint and log it | `10^0…10^4`, 81 points; **fail** |
| feature set | `φ_q = [ΔE ‖ ΔH̄ ‖ JS̄ ‖ Hel̄] ∈ R^{4L}`; `ΔH` (un-normalised) stored; `E_gen`, `H̄_gen` stored as diagnostics | 11 features × 3 β × 2 variants per layer, including margin and top-k Jaccard; un-normalised `H` **not computed**; no `φ_q` assembled (that is downstream) |
| prompt | fixed chat template with system turn | explicit single system instruction (`prompt.system`), Llama `date_string` fixed |
| decoding | greedy | greedy with a **clean** generation config (Qwen's hub `repetition_penalty=1.05` removed); TF argmax checked |
| precision / loading | bf16 on A100; 4-bit NF4 for local dev; bank cast to float32 on CPU; `banks.pt` | bf16 enforced; no quantisation path; bank never materialised, γ + row norms in `bank.h5` |
| scope | banks → features → probe → stats in one package | extraction only; probe/stats in `downstream/` (legacy, incompatible with the new artefacts) |
| extras not in the `.tex` | — | β multipliers, `first` variant, margin, Jaccard, per-token traces, `hidden.h5`, token-level baselines, empty-response convention |

### 13.2 Code vs code (internal)

1. **`RunConfig.sha256()` includes operational keys.** `model.device`,
   `extraction.max_attempts` and `name` are part of the hash that
   `shard.json` and `merge` check. Changing `device: auto → cuda` or raising
   `max_attempts` on a resume makes the shard refuse ("use a new extraction
   name"), while `calibration_context()` deliberately nulls `device`. The
   two notions of "same configuration" differ.
2. **`extraction.max_attempts` is reused by `calibrate`** (there is no
   `calibration.max_attempts`), although the key lives under `extraction`.
3. **Deterministic failures are retried.** A `PromptError`, a residual
   outside float16, or an unexpected NaN is deterministic under greedy
   decoding, yet it is retried `max_attempts` times across runs before the
   id is marked exhausted. Only OOM-type failures benefit from retries.
4. **`smoke` counts `n_failed` as ids with *any* failed attempt**, so an id
   that failed once and then succeeded is counted both as completed and as
   failed, and the CLI exits 1.
5. **Throughput suggestion hard-codes 7200 s** (`merge.status_report`)
   while the walltime is a variable in `hpc/env.sh` (`TIME_EXTRACT`).
6. **`merge` takes `betas`/`beta_multipliers` from the last item read**
   (guaranteed equal by the shard identity, but not asserted) and reads
   `n_layers` for `hidden.h5` attrs from `bank.h5` rather than from the
   items.
7. **`verify` re-derives the stop-token set** from `generation_config` +
   tokenizer instead of using `bundle.stop_token_ids` (intentional
   independence, but it duplicates `core.model.stop_token_ids`).
8. **`inspection._tokenizer` swallows every exception** and silently prints
   no token text when the tokenizer is not in the local cache.
9. **`load_model` loads on CPU first then `.to(device)`**, so the full bf16
   model transits CPU RAM (part of why calibrate asks 80G).
10. **The role mask codes the stop token as `template`**, which is
    consistent with "stop tokens are not response" but means `traces.h5`
    rows at those positions are indistinguishable from chat-template
    tokens without `answers.parquet`.
11. **Un-normalised entropy `H` is recoverable** as `H̄ · log(D_ff)` but not
    stored; `D_ff` is in `bank.h5` attrs and `manifest.json`.
12. **`hidden_layers`** affects only `hidden.h5`; features and traces are
    always computed for every layer.

### 13.3 Code vs README / MAPPING (fixed by this reorganisation)

- Module paths in `MAPPING.md` and the README layout block now point to
  `core/`, `pipeline/`, `verification/`.
- The README's "tests ~30 s" is ~45 s on this machine.

### 13.4 Deliberately duplicated (left as is)

- `dev/reference.py` re-implements every quantity with explicit loops and
  its own `FEATURES`/`VARIANTS` tuples (a test checks they match).
- `verification/verify.py: _Capture` is a second hook implementation; the
  alignment check recomputes response positions from scratch.

## 14. Glossary

| term | meaning in this code |
|---|---|
| `L`, `D`, `D_ff` | number of decoder layers, hidden size, SwiGLU intermediate size (from the checkpoint config) |
| `x`, `x̂` | residual at the input of `post_attention_layernorm`; its RMS-normalised version (no γ) |
| `W_g`, `γ`, `W̃` | `gate_proj.weight`, `post_attention_layernorm.weight`, `W_g·diag(γ)` |
| bank | unit rows of `W̃`, conceptually; only γ and `‖W̃_i‖` exist on disk/in memory |
| `s` | cosine scores `[T, D_ff]`; `g = ‖W̃‖√D·s` is the model's gate pre-activation |
| `β*` | inverse temperature calibrated so the mean `H̄` on the warm-up set equals `target_norm_entropy` |
| prompt phase / response phase | position `n_prompt-1` / generated non-stop positions |
| variant `mean` / `first` | response reduction: mean over response positions / first response position |
| empty response | first generated token is a stop token; valid sample with NaN response features |
| shard | contiguous block of `samples_per_shard` ids = one SLURM array task = one directory |
| exhausted | id that failed `max_attempts` times; reported, never completed |
| exit 75 | work remains (signal or retryable failures); the batch script requeues |
