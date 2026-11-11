# energy_llm — architecture and state of the code

> **What this document is.** A description of what the code in this
> repository *actually does*, written from the source. The science it
> serves is [`overview.md`](overview.md); where the code departs from it
> (or from the older thesis chapters, in Overleaf; last in this repository
> at tag `thesis-e1e3`) the departure is stated here as a fact about the
> code (section 13 collects them all). Use it to decide what must change; it
> does not say what *should* be.

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
validation and statistics are *not* in the package: they live in Brandon's
analysis repository (the legacy E1–E3 versions are in git at tag
`thesis-e1e3`). No aggregation over tokens is done here either.

Targets: `Qwen/Qwen2.5-3B-Instruct` and `meta-llama/Llama-3.2-3B-Instruct`
at pinned hub revisions, bfloat16, batch size 1, one A100 per job, SLURM
array jobs that survive walltime kills and preemption without supervision.

## 2. Repository map

```
energy_llm/                 the installed package (CLI: energy-llm, python -m energy_llm)
  cli.py                    argparse entry point; one cmd_* per sub-command
  config.py                 RunConfig: the YAML schema, validation, hashes
  storage.py                atomic writes, HDF5 array layout, JSONL logs, failure log, hashing, input items
  manifest.py               provenance: git commit, library versions, hardware, seeds
  core/                     WHAT is computed (no output directory involved)
    model.py                loading at a pinned revision, dims from the checkpoint, clean greedy config
    prompting.py            chat template, single system instruction, per-token role mask
    bank.py                 the unmaterialised bank: gamma + row norms, on-the-fly cosine scores
    metrics.py              float32 primitives: x_hat, E, log p, H_bar, margin, mean/var score,
                            top-k mass/indices, x_hat top-1 fraction, KL, JS
    capture.py              per question: greedy + teacher forcing + hooks + every metric
  pipeline/                 HOW it runs (everything that touches <output_dir>)
    signals.py              SIGUSR1/SIGTERM -> finish the current id, exit 75
    calibrate.py            beta* from a warm-up JSONL -> calibration/{beta_star.json, bank.h5}
    extract.py              one shard: identity, resume, per-id HDF5, completed/failed logs
    merge.py                shard status; final parquet/HDF5 files + manifest.json
    smoke.py                extract + merge on N questions in one process, summary
  verification/             never a production output
    verify.py               checks vs the model and vs dev/reference.py, PASS/FAIL report
    inspection.py           position table and per-layer reference tables of one id
configs/                    qwen25_3b.yaml, llama32_3b.yaml (revisions pinned)
hpc/                        env.sh (resources + helpers), submit.sh, *.sbatch,
                            missing_shards.sh, prefetch_models.sh, pilot_bundle.sh
tests/                      CPU-only; tiny random Qwen2 + Llama models built in conftest.py
dev/                        verification only: 24 dev questions, 12 warm-up questions,
                            reference.py (naive float64, imports nothing from energy_llm)
docs/                       this file, overview.md (the science), CHANGES_v0.2.md (change log)
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
| 5 | `core.capture.compute_item` | per layer: `x̂`, scores `s` `[T, D_ff]`, per-token traces at every β of the grid (β\* last) with divergences to three reference positions, hidden states, `answer` and `baselines` dicts, finiteness checks |
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
  `tf_exact`, `manifest.json: checks.tf_mismatch_ids`). For every generated
  step `j` the TF/decoding discrepancy `max_v |logit_gen − logit_tf|` (float32)
  is kept (`tf_logit_diff`, its max `tf_logit_diff_max`, and
  `tf_logit_diff_step0` = the alignment discrepancy at the last prompt
  position); at each mismatch the top1 − top2 logit margin in TF
  (`tf_mismatch_margin_tf`) and in generation (`tf_mismatch_margin_gen`), the
  TF argmax (`tf_mismatch_tf_tokens`) and `max(|top1|)` of the two passes
  (`tf_mismatch_top_logit`, the scale of a bfloat16 ulp).

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
norms, and the γ diagnostic below).

**γ diagnostic** (`gamma_diagnostic(bank, k=5)`, report only): column `j`
of `W̃` carries `γ_j²·‖W_g[:, j]‖²` of `Σ_i ‖w̃_i‖²`. Per layer it records
the indices of the `k` largest `|γ_j|` (descending) and the fraction of
`Σ_i ‖w̃_i‖²` those `k` columns carry, one layer's `W_g` in float32 at a
time. Written by `calibrate` as `bank.h5` attrs `gamma_top_k`,
`gamma_top_indices [L, k]`, `gamma_top_fraction [L]`; copied by `merge`
into `manifest.json: gamma_diagnostic` (`null` for a `bank.h5` calibrated
before the diagnostic existed). It changes nothing else: not `scores`, not
the fingerprint, not any hash or the calibration context; there is no
config key and no ablation path in production. `extract` refuses to run if the live bank's fingerprint differs from
`beta_star.json` or `bank.h5`.

### 5.6 Per-position quantities (`core/metrics.py`, all float32 torch)

| quantity | code |
|---|---|
| log-distribution | `log p = log_softmax(β s)` over the D_ff memories |
| energy | `E = -(1/β) · logsumexp(β s)` — **no** quadratic term |
| normalised entropy | `H̄ = -Σ p log p / log(D_ff)`, with `p = exp(log p)` and `0·log 0 := 0` |
| margin | `top1(s) - top2(s)` on the raw scores (β-independent) |
| mean score | `mean_score = Σ p s` |
| score variance | `var_score = Σ p (s - mean_score)²` |
| top-k mass | `topk_mass = exp(logsumexp of the k largest log p)` (sum of the k largest `p`, in the log domain) |
| top-k indices | `topk_indices`: int32 indices of the k largest `log p`, descending |
| x̂ top-1 fraction | `xhat_top1_frac = max_j x̂_j² / D` per row (1 for a one-hot residual, 1/D for a constant one) |
| Kullback–Leibler (nats) | `KL(p‖q) = Σ p (log p - log q)`, clamped at 0; both rows come from one `log_softmax(β s)`, so `KL(p‖p) = 0` exactly |
| Jensen–Shannon (nats) | pointwise in `d = log p - log q`: `log(p/m) = ln2 - softplus(-d)`, `log(q/m) = ln2 - softplus(d)`, `JS = ½Σ p log(p/m) + ½Σ q log(q/m)` with `ln2 := softplus(0)` in float32, so `JS(p,p) = 0` exactly; terms with `p_i = 0` (resp. `q_i = 0`) are 0 (xlogy convention, masked: no `0·inf`); clamped to `[0, log 2]` |

Divergences take **log-probabilities**: at the β\* values that cosine
scores produce (hundreds or more) float32 probabilities underflow, which
made the earlier p-domain JS stick at its log 2 clamp (fixed in commit
`652fccc`; regression test
`tests/test_metrics.py::test_divergences_survive_float32_underflow`).
Every primitive has a naive float64 counterpart in `dev/reference.py`
(compared in `tests/test_metrics.py` and, through the traces, by `verify`).
Squared Hellinger, top-k Jaccard and the mean-distribution helper were
removed with the features table (v0.2, step 0).

### 5.7 Positions, references and β (`core/capture.py: compute_item`)

Per layer (layers are never mixed), the scores `s [T, D_ff]` of **every**
position are computed once and every quantity is stored per position: no
reduction over tokens is done in the module (aggregation is analysis).

- **response positions** = `response_positions` (generated tokens, stop
  tokens excluded); the role mask (`3 response`) marks them.
- **reference positions** (`ForwardTrace.reference_positions`, names in
  `capture.REFERENCES`):
  `question_last` = last prompt position with role `question`
  (`PromptError` if there is none); `prompt_last` = `n_prompt - 1` (the
  state that emits the first generated token); `response_first` = the first
  response position, `None` for an empty response. Stop tokens never carry
  the question role.
- β: `betas = ExtractionConfig.betas(β*)` = the absolute log grid
  `10^min … 10^max` with `num` points (`extraction.beta_grid_log10`,
  default `{min: 0, max: 4, num: 9}`: half-decade steps, identical across
  models because `s` is a cosine) **plus β\* appended last**, so
  `n_beta = num + 1` and `betas[-1] == β*`. The grid is looped inside the
  layer (one `log_softmax` of `[T, D_ff]` per β).

### 5.8 Traces, hidden states, baselines

- **Traces** (`ItemResult.traces`, float32 in memory; float16 in the files
  except `topk_ids`), over *all* `T = n_prompt + n_generated` positions
  (template, instruction, question, response and stop tokens alike), with
  `p_t = softmax(β s_t)`:

  | dataset | shape | definition |
  |---|---|---|
  | `energy` | `[L, T, n_beta]` | `E` |
  | `norm_entropy` | `[L, T, n_beta]` | `H̄` |
  | `topk_mass` | `[L, T, n_beta]` | mass of the `extraction.topk_mass_k` largest `p` |
  | `mean_score`, `var_score` | `[L, T, n_beta]` | `Σ p s`, `Σ p (s - mean)²` |
  | `margin` | `[L, T]` | top1 − top2 of `s` |
  | `xhat_top1_frac` | `[L, T]` | `max_j x̂_j² / D` |
  | `topk_ids` | `[L, T, k_ids]` int32 | indices of the `extraction.topk_ids_k` largest `s` (= largest `p` at any β), descending |
  | `js/<ref>` | `[L, T, n_beta]` | `JS(p_t, p_ref)` |
  | `dkl/<ref>` | `[L, T, n_beta]` | `KL(p_t‖p_ref) − KL(p_ref‖p_t)` |

  `<ref>` ∈ {`question_last`, `prompt_last`, `response_first`}. At a
  reference position its own `dkl` and `js` rows are exactly 0 (pointwise
  forms, one `log_softmax`). Plus
  `role_mask [T] int8`.
- Not computed anywhere: the un-normalised entropy `H`, pooled
  prompt-span references, any aggregate across layers or tokens.
- **Hidden states** (`ItemResult.hidden`): for the layers in
  `extraction.hidden_layers` (`all` → every layer), the residual `x` (same
  tensor the metrics read, *not* the layer output) at the three reference
  positions and at the last response position (`response_last`), float16
  `[L_h, D]`; `response_first` and `response_last` are NaN for an empty
  response. A value beyond the float16 range raises `NonFiniteError`.
- **Baselines** over response tokens, from the decoding logits:
  `mean_logprob`, `min_logprob`, `mean_token_entropy`, `mean_neg_logit`
  (minus the mean logit of the chosen tokens: one-pass Semantic-Energy
  baseline), `n_response_tokens`, `empty_response`; NaN when empty.
- **Answer record**: prompt text, decoded response (response tokens only,
  special tokens skipped), prompt and generated token ids, counts,
  `empty_response`, `truncated`, `stop_token_id`, per-token log-probs,
  logits of the chosen token (`token_logits`, float32, the same decoding
  logits as the log-probs) and entropies, TF mismatch count and positions,
  `tf_exact`, the per-mismatch margins, TF tokens and top logits, the per-step
  `tf_logit_diff` with `tf_logit_diff_max` and `tf_logit_diff_step0` (5.3),
  `xhat_norm_max_rel_dev`.

### 5.9 Empty responses and failures

- **Empty response** = the first generated token is a stop token
  (`response_positions == []`). It is a **valid, completed sample**, not a
  failure: greedy retries would reproduce it, and dropping it would bias the
  comparison between datasets. Traces are computed for every position (the
  stop token included); `js/response_first`, `dkl/response_first`, the
  response baselines and `hidden:response_first/response_last` are NaN by
  design; `empty_response=True` everywhere. The finiteness check covers
  every trace except exactly those `*/response_first` datasets of an empty
  response, and also rejects a trace beyond the float16 range; any other
  non-finite value is a real failure.
- **Failure** = any exception in prompt/forward/compute/write (OOM, a
  `PromptError`, a `NonFiniteError` for an unexpected NaN/inf, a residual
  outside float16, `RuntimeError("generation returned no tokens")`). It is
  logged to `failed.jsonl` (`id, attempt, max_attempts, error, traceback,
  time, host`), never marked complete, and retried on the next run until
  `extraction.max_attempts`; after that the id is *exhausted* and reported
  in `manifest.json: exhausted_ids`.

### 5.10 Numerics

- Storage: traces are written as float16 (relative precision 2⁻¹¹ ≈ 5e-4;
  subnormals below 6e-5), `topk_ids` as int32, residuals as float16.
- Model forward passes: model dtype (bfloat16 in production; float32 in
  tests). Everything after the hook — `x̂`, scores, softmax, metrics,
  decoding log-probs — is float32 on the model's device.
- `manifest.set_determinism(seed)`: seeds `random`, `numpy`, `torch`;
  disables TF32 for matmul and cuDNN; `set_float32_matmul_precision("highest")`.
- `verify` tolerances are derived in `verification/verify.py`
  (`score_tolerance`, `reference_tolerances`, `check_gate`,
  `tf_cache_tolerance`) and tabulated in section 9.

## 6. β\* calibration (`pipeline/calibrate.py`)

Inputs: a warm-up JSONL (must be disjoint from every evaluation set) and the
`calibration` block of the config. Output: `<output_dir>/calibration/`:

```
calibration/
  state.json         {input_sha256, context, ids}   refuses a different warm-up/config in the same dir
  bank.h5            gamma + row norms + gamma diagnostic (section 5.5), written once; later runs must match its fingerprint
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
  items/<sha1(id)>.h5   every output of one id (attrs id, betas = grid + [beta*],
                    topk_mass_k, topk_ids_k, reference_positions [3] in REFERENCES
                    order, -1 for None; datasets answer_json, baselines_json, role_mask,
                    traces/<name> (float16; topk_ids int32), hidden/<position> float16;
                    arrays gzip 4, chunked one layer at a time: storage.write_array)
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
| `baselines.parquet` | one row per id |
| `traces.h5` | `/<id>/<dataset>` for every trace of 5.8, `/<id>/role_mask [T]`, gzip 4, chunks `[1, T, …]` (one layer reads alone); group attr `reference_positions [3]` (−1 = none); file attrs `role_codes`, `role_names`, `reference_names`, `beta_star`, `betas` (float64, last = β\*), `topk_mass_k`, `topk_ids_k`, `layout`. `merge` raises if two items disagree on `(betas, topk_mass_k, topk_ids_k)` |
| `hidden.h5` | `/<id>/{question_last, prompt_last, response_first, response_last} [L_h, D] float16` (gzip, chunks `[1, D]`); attrs `layers`, `layout` |
| `bank.h5` | byte copy of `calibration/bank.h5` |
| `manifest.json` | written last: name, created, full config + `config_sha256`, β\* and the calibration summary, input path/sha/count, counts (completed/exhausted/pending), `exhausted_ids` with last error, `incomplete_shards`, checks (`xhat_norm_max_rel_dev` max, `tf_mismatch_ids`, `n_truncated`, `n_empty_response`, `max_xhat_top1_frac` per reference position over ids and layers, `gamma_top_fraction_max` = largest top-k γ fraction and its layer), `gamma_diagnostic` (from `bank.h5`), actual model dtypes seen, role names, code commit, library versions, merge host, every shard run record |

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
  answers, the **list of TF/greedy disagreements** (id, step, generated and TF
  token, margins in both passes) against `δ = max(p99 over the non-divergent
  generated steps of the smoke of tf_logit_diff, 3 bf16 ulp(|top1 logit|))`
  with the criterion "every disagreement has max(margin_tf, margin_gen) < δ"
  (`smoke.tf_disagreements`; the step-0 maximum is reported as information), max `‖x̂‖` deviation, peak GPU memory, s/question, empty and
  truncated counts, per layer the mean over ids of `H̄` at β\* (last
  β column) read from `traces.h5` (prompt = last prompt token | response = each id's mean over
  its response positions, empty responses skipped), and the report-only
  diagnostics of `smoke.retrieval_diagnostics`, from residuals captured
  afresh for the smoke questions (a second `forward_item` per question, kept
  on CPU; no production output involved):
  - per layer and per band — thirds, and the same thirds **without layer 0**
    (`bands_excluding_layer0`; None when a third is layer 0 alone): median
    and max `xhat_top1_frac` at the three references; mean `var_score` at β\*
    over prompt vs response tokens; the γ top-5 fraction; the pilot
    **criteria 3** (median `xhat_top1_frac` at `prompt_last` > 0.5) and
    **4** (γ top-5 fraction > 0.5, or `ΔH̄` changes sign under the γ
    ablation);
  - **d\***: per layer the most frequent `argmax_j |x̂_j|` over every position
    of every smoke question, and its share of positions;
  - **ablations** (`zeroed_gamma_bank`: γ zeroed on the given dims, row norms
    of the ablated `W̃` recomputed) of the top-5 `|γ|` dims and of d\*: mean
    `ΔH̄ = H̄(response mean) − H̄(prompt_last)` and the response mean of JS to
    `question_last` / `prompt_last`, at β\* and at the **re-matched β** — the
    β at which the ablated bank's mean `H̄(prompt_last)` over ids and layers
    returns to its intact value at β\*, interpolated in log β on the
    calibration grid (None if not bracketed);
  - **β_ℓ** per layer: `H̄_ℓ(β_ℓ) = target` (mean over ids of the mean over
    the calibration positions), interpolated on the calibration grid; min,
    max, their layers, unbracketed count;
  - mean `H̄(prompt_last)` at β\* against `target ± 0.05` and against the
    calibration bracket (`beta_star.json` curve at the grid points around
    β\*).

  Output name
  `smoke-<UTC time>`, report `smoke_report.json`. CLI exit 1 if any id
  failed or a signal stopped it.
- **verify** (`verification/verify.py`): runs the same single-process path
  under `verify-<time>`, then on the first `n` questions runs thirteen checks
  (table below). Each check reports max observed error and tolerance (status PASS, FAIL, SKIP or INFO); the CLI options (`n`, `reference_n`, `reference_layers`, `tf_max_noise_ratio`, `fp32_reference`) are recorded under `options`;
  `verify_report.json`; exit 1 on any FAIL. Keeps its own minimal hooks
  (`_Capture`) and its own stop-token derivation on purpose.
- **inspect** (`verification/inspection.py`): no model; reads
  `manifest.json`, `answers.parquet` and `traces.h5` of a finished run and
  prints every position (token id, decoded token if the tokenizer is in the
  local cache, role, reference position or `response`) and one table per
  quantity — `E`, `H̄`, `margin`, `var_score`, `topk_mass` at β\* (last β
  column) — with one row per layer and columns `question_last`,
  `prompt_last`, `response_first`, `response_mean`.

Verification checks (none is part of the production path):

| check | implementation | tolerance and justification |
|---|---|---|
| `reference`: every per-token trace == independent float64 reference, at every β and reference position | `dev/reference.py` (explicit loops, no `energy_llm` imports; `reference_item` returns the traces of 5.8 and the reference positions, `question_last` passed in from the role mask); `verify.compare_with_reference`; `tests/test_reference.py` | scores ≤ D·u₃₂ (float32 dot of D terms, Cauchy–Schwarz); every trace by a first-order bound on the observed score error δ at each β (`reference_tolerances`): E δ; margin 2δ; H̄ 2βδ; `mean_score` 2βδ·range(s) + δ; `var_score` 2βδ·range(s) + 2δ·range(s); `topk_mass` max(k, 2)·βδ; JS 4βδ; `dkl` 2βδ·A + 8βδ with A = Σp\|log p − log q\| + Σq\|log q − log p\| computed per element from the reference scores; `xhat_top1_frac` within the score tolerance; plus a float32 floor 1e-5·max(1, \|value\|) per element; `topk_ids` as sets, row Jaccard ≥ 0.98 (near-ties at the k-th score); NaN patterns and reference positions equal |
| `gate`: read point + γ integration | `verify.check_gate` | `g` vs `‖w̃‖√D·s`: 4·u of the model dtype (three roundings in the model path), floor 1e-5; the no-γ control must exceed 10× |
| `tf_vs_cache`: TF residuals == cached generation, judged on the traces | `verify.check_tf_vs_cache`, `propagate`, `noise_to_signal` | tokens must agree, and per layer ℓ and trace m ∈ {`H̄`, `E`, JS}: p99 over (questions × compared positions) of `|m(x_cache) − m(x_tf)|` (JS: `JS(p_cache, p_tf)`) at β\* ≤ `--tf-max-noise-ratio` (0.1) × σ_ℓ(m), σ_ℓ = std over (response positions × questions) of `norm_entropy`, `energy`, `js/question_last` at β\*. Reported apart: max/σ ratio; `‖x_cache − x_tf‖/‖x_tf‖` per layer (median, p99, max) and the location of its max (id, layer, position, role), against the former thresholds fp32 1e-4 / bf16 5e-2 (informative since step 9) |
| `fp32_reference` (INFO) | `verify.fp32_reference`, `load_fp32_model` | no PASS/FAIL. A separate float32 instance from the local cache (never an in-place cast), one TF pass per question on the same tokens, freed afterwards; `set_float32_matmul_precision("highest")` and TF32 off in `cuda.matmul` and `cudnn` during the check, restored after (recorded in the report). Per layer: `d(x_tf, x_fp32)`, `d(x_cache, x_fp32)` (median/p99/max) and both propagated to the traces with the same σ ratios. A float32 bundle (tests) is its own reference. `--no-fp32-reference` → SKIP |
| `alignment`: last prompt position emits the first token; response = generated non-stop tokens | `verify.run_verify` | exact (booleans) |
| `hidden`: residuals at the four positions, derived independently (roles, stop tokens) | `verify.run_verify` | exact |
| `files_vs_memory`: `hidden.h5` / `traces.h5` hold what was computed | `verify.run_verify` | in float16 ulps: 0 on CPU, ≤ 1 on GPU (a float32 difference of a recomputation can flip the last float16 bit); `topk_ids` exact on CPU, row Jaccard ≥ 0.98 on GPU |
| `permutation`: permuting memories changes nothing | `verify.PermutedBank`, `permutation_tolerances`, `trace_ratios`, `gamma_shuffled_bank`; `tests/test_invariance.py` | per trace and element, tolerance from float32 reduction bounds over D_ff (tree reduction, `g_n = ⌈log₂ D_ff⌉·u₃₂`, safety factor c = 2; derivation in §13.5); PASS if every `|Δ|/tol ≤ 1`, the **negative control** (γ permuted over D without permuting `W_g`) exceeds 1, and `topk_ids` mapped through the permutation have row Jaccard ≥ 0.98. Reported: per trace max error, max ratio and where (layer, position, β); the overall worst; the control's ratios |
| `limits`, `monotone`, `self_divergence` | `verify.run_verify`; `tests/test_invariance.py` | 1e-5; β∞ = 50/margin so runner-ups weigh ≤ e⁻⁵⁰: `H̄` → 0, `E` → −max s, `var_score` → 0, `topk_mass` → 1; β → 0: `H̄` → 1; JS(p,p) and \|KL(p‖p)\| expected exactly 0 (pointwise forms); tolerance `4·u₃₂·log D_ff` (`self_divergence_tolerance`, 2.2e-6 at D_ff = 11008) |
| `calibration`: H̄(β\*) on the warm-up set | `verify._calibration_check` | H̄ monotone ⇒ it must lie between the two bracketing grid values |
| `determinism` | `verify.run_verify`; `tests/test_invariance.py` | bit-exact on CPU, ≤ 1 float16 ulp on GPU |

The cluster sequence (environment, real tokenizers, dev calibration,
verify, smoke, inspect, requeue test) is in the README, "Verification on
the cluster".

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
  beta_grid_log10: {min: 0.0, max: 4.0, num: 9}   # exactly these keys; num >= 2, max > min; beta* appended
  topk_mass_k: 10                    # >= 1
  topk_ids_k: 100                    # >= 1
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
- `pilot_bundle.sh <verify_jobid> <smoke_jobid>` (login node: bash ≥ 4.2,
  no Python packages): writes `~/pilot_bundle_<date>.txt` with `sacct` of
  both jobs, head/tail of their `.out`, tail of their `.err`, and the
  `verify_report.json` / `smoke_report.json` and `manifest.json` of the run
  directories **those jobs** printed (`report: <dir>/<x>_report.json` line of
  each log; never a `find`). JSON is pretty-printed with the stdlib's
  `python3 -m json.tool` when available, else copied. `LOG_DIR`, `TAIL_N`
  override the defaults (`logs/`, 150). Tested in `tests/test_hpc_scripts.py`.
- The verification sequence on the cluster is in the README ("Verification
  on the cluster"). β\* values from `dev/` are verification-only.

## 12. Tests (`tests/`, CPU only, ~55 s)

`conftest.py` builds tiny random Qwen2 and Llama models (3 layers, D=32,
D_ff=48, non-trivial γ), word-level tokenizers with templates that mimic the
real ones (Qwen injects a default system prompt when none is given; Llama
reads `date_string`), and hub-style sampling settings that the extractor
must override.

| file | proves |
|---|---|
| `test_storage.py` | atomic writes leave no temp file and keep the old file on error; torn JSONL lines skipped; input validation (ids, duplicates, extra fields dropped); item file round trip (float16 / int32, attrs, gzip, chunks `[1, T, n_beta]`) |
| `test_bank.py` | `g = ‖W̃‖·√D·s` against the real forward; bank keeps no copy of `W_g`; scores equal explicit unit rows and are cosines; dims from config; `bank.h5` round trip; γ diagnostic (top-`|γ|` indices, fraction vs explicit `W̃`, in [0, 1], all dims sum to 1, deterministic); bf16 enforcement |
| `test_hpc_scripts.py` | `pilot_bundle.sh` bundles the run directories printed by the given jobs (not another run), works without `sacct`, reports an unfinished job; usage error |
| `test_metrics.py` | `JS(p,p) == 0` and `KL(p‖p) == 0` exactly at three β; JS with `log p = −inf` (xlogy masking); `var_score` is the centred form (raw-moment form fails on scores packed near 0.9); `‖x̂‖` formula; distributions sum to 1; entropy bounds/extremes; energy vs scipy; JS symmetry/bounds/scipy convention/log 2 on disjoint supports; log-domain underflow regression; margin; KL zero on identical inputs and the KL difference antisymmetric; `var_score` → variance of `s` (β→0) and → 0 (β→∞); `topk_mass` → 1 (β→∞); `topk_indices` sorted, unique, int32; `xhat_top1_frac` extremes; the six primitives vs `dev/reference.py` |
| `test_capture.py` | TF/decoding logit discrepancy recorded per step; a forced TF flip records the mismatch, TF token and both margins; `tf_exact` and the `tf_logit_diff*` answer fields; explicit system prompt and exact role mask; Llama `date_string`; hub sampling overridden; teacher forcing == greedy == explicit argmax loop; `token_logits` == chosen-token logits of an explicit forward; hook reads the norm input; truncation; every stop token excluded; empty response as a valid sample (NaN exactly in `*/response_first` and the response residuals); shapes/dtypes/bounds of every trace; reference positions (and `PromptError` without a question token); own-reference rows zero; `js`/`dkl` vs explicit float64 formulas; `topk_ids` sorted and unique; `energy[..., -1]` == the single-β trace at β\* (and within float16 after the cast); `betas` recorded; NaN raises |
| `test_config.py` | YAML round trip keeps values and hash; `betas` = absolute grid + β\* last; invalid grid / top-k rejected; removed keys (`beta_multipliers`, `jaccard_top_k`) rejected |
| `test_calibrate.py` | log-β interpolation; unbracketed target fails; curve decreasing and β\* hits the target; idempotence; extract refuses without β\*, with overlap, with another context |
| `test_resume.py` | interrupted + resumed run is byte-identical to a full run; outputs indexed by id without input extras; failures retried and never completed; exhausted ids end the shard and are reported; changed config (top-k or β grid) refuses to resume; merge of two shards (attrs, per-id reference positions, chunks, `max_xhat_top1_frac`, `mean_neg_logit`, γ summary); `merge` refuses items with different `betas`; status/missing shards and CLI; SIGUSR1 finishes the current id; CLI skips a finished shard without loading the model; empty responses completed and flagged (finite traces, NaN baselines and `response_last`); smoke summary (three bands and the bands without layer 0; the intact branch of the ablation equals the production traces; d\*, criteria, re-matched β, β_ℓ, `prompt_last` band/bracket; disagreement list; report path line); the δ rule of `tf_disagreements` |
| `test_invariance.py` | permuting memories changes nothing (`topk_ids` map through the permutation); the derived per-trace tolerances hold (worst ≤ 1, `dkl` ratio > 1e-3: not loose by orders) and the γ-shuffled control exceeds them by > 10³; β→0 and β→∞ limits; `H̄` monotone in β; self-divergence zero; two runs bit-identical on CPU |
| `test_reference.py` | every production trace == `dev/reference.py` within the derived tolerances on the grid {1, 10², 10⁴, β\*} at three β\*; reference positions equal; empty response (NaN `*/response_first`); negative control (no γ) fails |
| `test_verify.py` | all checks PASS on the tiny models (`fp32_reference` INFO); options recorded; `noise_to_signal` ratios; `fp32_reference` loads a separate instance and restores the matmul precision; a wrong read point makes `gate` and `reference` FAIL; CLI exit code; inspect output (each reference position marked once; five per-layer tables with the four columns) |
| `test_gpu.py` | (skipped without CUDA) GPU == CPU within float32, peak memory recorded |
| `test_real_tokenizer.py` | (opt-in via `ENERGY_LLM_TOKENIZER_DIRS`) prompt ids equal the official template, role mask exact |

## 13. Audit: inconsistencies and open points

Facts about the current code that a reader comparing it with the
methodology, or deciding what to change next, should know.

### 13.1 Code vs `docs/overview.md`

The scientific specification is [`overview.md`](overview.md) (v0.2; its §3,
§6 and §7 were aligned with the code in v0.2 step 7). Where the code still
departs from it:

| topic | `overview.md` | code |
|---|---|---|
| hooks (§5, Etapa 2) | output of `gate_proj` (`g`) and input of `mlp` (`r`) | one pre-hook on `post_attention_layernorm` (residual `x`); `x̂` and `s` recomputed in float32, γ folded into the bank; `g = ‖w̃‖√D·s` checked against the real `gate_proj` by `verify` (`gate`) |
| bank storage (§5, Etapa 0.1) | `‖m_i‖` and `M̂` in float32 with a hash | `M̂` never materialised: γ and `‖w̃_i‖` in `bank.h5` with a fingerprint |
| stored residuals (§7) | `r` and `h` at the three references | `x` (from which `r = γ ⊙ x̂`) at the three references **and** `response_last` |
| TF vs generation (§8.1, revised step 9) | noise/signal of the traces ≤ 0.1 | `verify: tf_vs_cache`: as specified, propagated from `x` (not `r`) to `H̄`, `E`, JS at β\*; `‖Δx‖/‖x‖` informative; `fp32_reference` (INFO) |
| TF/greedy disagreements (§8.6, step 9) | every disagreement with margin < δ | recorded per answer (`answers.parquet`); evaluated in `smoke` only, on its own questions |
| extras (§6.2), `intervene` (§5) | optional | not implemented |
| ablation (§6.3) | flag `drop_outlier_dims` | no flag; report-only ablation line in `smoke` (top-5 `|γ|` dims zeroed vs intact) |
| response positions | generated tokens | generated tokens **excluding every stop token**; an empty response is a valid sample (NaN `*/response_first`) |
| decoding | greedy | greedy with a **clean** generation config (Qwen's hub `repetition_penalty=1.05` removed); TF argmax checked |
| extras not in `overview.md` | — | `margin`, `xhat_top1_frac`, `mean_score` stored per token; `hidden.h5:response_last`; `answers.parquet: xhat_norm_max_rel_dev`; token-level baselines (`mean/min_logprob`, `mean_token_entropy`, `mean_neg_logit`) |

The thesis chapters (`fundamentals-FINAL.tex`, `metodology-FINAL.tex`, in
Overleaf; last in this repository at tag `thesis-e1e3`) predate both: query
`r = γ⊙x̂` with its norm kept in the score, `E = lse + ½‖q‖²`, mean of
per-token JS/Hellinger, β clamped when not bracketed, `banks.pt`, 4-bit
loading — all superseded by `overview.md` and the code above.

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
6. ~~`merge` takes `betas` from the last item read without asserting
   they are equal~~ — **closed in v0.2 step 3**: `merge` raises if two
   items carry different `betas`. Still a fact: `n_layers` for `hidden.h5`
   attrs comes from `bank.h5` rather than from the items.
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
11. ~~Un-normalised entropy `H` is not stored~~ — **closed in v0.2**: `H̄`
    is the specified quantity (`overview.md` §6.1); `H = H̄ · log(D_ff)` if
    ever needed, with `D_ff` in `bank.h5` attrs and `manifest.json`.
12. **`hidden_layers`** affects only `hidden.h5`; traces are always
    computed for every layer.

### 13.3 Documentation (v0.2)

- The former specification→code mapping document was folded: its
  departures table into 13.1, its spec→code table into section 5, its
  verification tolerances into section 9 and the `verify.py` docstrings. `hpc/RUNBOOK.md` was folded into
  the README ("Verification on the cluster").

### 13.4 Deliberately duplicated (left as is)

- `dev/reference.py` re-implements every quantity with explicit loops,
  including its own `REFERENCES` tuple. At Qwen2.5-3B size it costs about
  75 s per layer and question for the scores plus a few minutes for the β
  grid and the three references in pure Python; `verify` runs it on
  `--reference-n` questions × `--reference-layers` only.
- `pipeline/smoke.py` builds its own ablated `Bank` for the γ ablation line
  (report only) instead of a production flag.
- `verification/verify.py: _Capture` is a second hook implementation; the
  alignment check recomputes response positions from scratch.

### 13.5 Permutation tolerances (v0.2 step 9)

Pilot (Qwen2.5-3B, bf16 model, float32 metrics): the former global criterion
`|Δ|/max(1,|value|) ≤ 1e-5` failed at exactly 2⁻¹⁵ = 3.05e-5, on `dkl`:
`dkl = KL(p‖q) − KL(q‖p)` (overview §6.1, unchanged) subtracts two KLs of up
to hundreds of nats at large β; one float32 ulp of a value in [256, 512) is
2⁻¹⁵, so its absolute error scales with `A = Σp|log p − log q| + Σq|log q −
log p|`, not with `|dkl|`. **`dkl` is ill-conditioned at high β**; the
definition stays, the tolerance follows its conditioning. `verify` reports
the β at which the permutation maximum falls.

Derivation (`verify.permutation_tolerances`). Permuting the memories gives
the same scores in another order (observed `ds = max|s_perm − s[perm]|` = 0
on CPU; if not, the Lipschitz terms of `reference_tolerances` in `ds` are
added). What changes is the order of every reduction over D_ff. Assumed:
**tree (pairwise) reduction** — torch's CPU cascade sum and CUDA tree
reductions — so a sum of n terms is off by ≤ `g_n Σ|terms|`,
`g_n = ⌈log₂ D_ff⌉·u₃₂` (14·u at D_ff = 11008), first order. Error sources:

- `log_softmax` subtracts the max, then `log S`, `S = Σ exp(z − max) ∈ [1, D_ff]`:
  a shift of every `log p_i` by at most `δ = g_n + u(1 + log D_ff)`,
  independent of `|lse|`;
- each `log p_i` is then rounded on its own, `ε_i ≤ u|log p_i|`; their
  weights are computed exactly from the data: `H = Σp|log p|`,
  `A₂ = Σp|log p||d| + Σq|log q||d|` (`d = log p − log q`),
  `J = Σp|log(p/m)| + Σq|log(q/m)|`, and `Λ = β·range(s) + log D_ff ≥ max|log p_i|`
  where no exact weight is cheap.

Tolerances (× c = 2: each term's own roundings — exp, product, difference,
≤ 3u — and second-order terms; plus a floor u₃₂, the float16 storage
resolution near 0):

| trace | tolerance / c |
|---|---|
| `energy` | `δ/β + 2u\|E\|` |
| `norm_entropy` | `(δ(1+H) + uH(1+Λ) + g_n H) / log D_ff` |
| `mean_score` | `δ + uH + g_n` |
| `var_score` | `(δ + uΛ + g_n)·var` (centred form, non-negative terms) |
| `topk_mass` | `(δ + uΛ + g_n)·mass` |
| `js/<ref>` | `2δ·2JS + u(H_t + H_r)(1 + ln2) + uA + uA₂ + g_n J` — a uniform shift of `d` cancels at first order (`pσ(−d) = qσ(d)`); a uniform rescaling of p multiplies the signed `KL(p‖m) ≤ 2JS` |
| `dkl/<ref>` | `2δ(2 + A) + u(2H_t + 2H_r + A) + uA₂ + g_n A` (+ `2β·ds·A`) |
| `margin` | `2·ds` (0 when exact) |
| `xhat_top1_frac` | 0 (does not involve the bank) |

Tightness on the tiny test models (D_ff = 48; c = 2): observed / tolerance
peaks at 0.1–0.3 for `energy`, `H̄`, `mean_score`, `var_score`, `topk_mass`
and `dkl`, 0.02–0.2 for `js` (worst-case bounds on terms of mixed sign);
the γ-shuffled control exceeds the tolerances by ≥ 10⁸. `verify_report.json`
gives the same ratios per trace on the real model.

### 13.6 Other tolerances revised after the pilot (v0.2 step 9)

Pilot, Qwen2.5-3B bf16 on A100, β\* = 72.776: `verify` failed `tf_vs_cache`,
`permutation` (13.5) and `self_divergence`; `smoke` gave TF == greedy on
99.64 % of tokens.

| check | before | now | justification |
|---|---|---|---|
| `self_divergence` | `≤ 1e-7` (pilot 3.27e-7: JS via `logaddexp(a, a) − ln2`, one float32 rounding) | expected exactly 0; tolerance `4·u₃₂·log D_ff` (2.2e-6 at D_ff = 11008) | JS is computed pointwise in `d = log p − log q` with `ln2 = softplus(0)` in float32, so every term vanishes at `d = 0`; KL comes from one `log_softmax`. The bound is what a non-exact form would have to meet: D_ff terms, a rounding `u` on log-ratios ≤ `log D_ff`, c = 4 (two roundings per term, the ½ weights, one of margin). Tests require exact 0 |
| `tf_vs_cache` | `‖x_cache − x_tf‖/‖x_tf‖ ≤ 5e-2` bf16 (provisional; pilot 0.0907, tokens equal) | per layer, p99 of `|ΔH̄|`, `|ΔE|`, `JS(p_cache, p_tf)` at β\* ≤ 0.1·σ_ℓ (`--tf-max-noise-ratio`), σ_ℓ = std over response positions × questions; tokens equal | bf16 rounding in KV-cached vs full-sequence attention compounds with depth; what matters for the analysis is whether it moves the stored traces relative to their spread between questions. A noise of 10 % of σ shifts an AUROC-type comparison negligibly. Max/σ, the per-layer x distribution and the location of its max are reported; `fp32_reference` shows whether TF and cached residuals are equally far from float32 (bf16 rounding) or not (pipeline error) |
| TF/greedy (`smoke`) | 100 % of tokens | every disagreement has max(margin_TF, margin_gen) < δ, δ = max(p99 over non-divergent steps of `max_v\|logit_gen − logit_tf\|`, 3 bf16 ulp of `\|top1 logit\|`) | a disagreement is a near-tie flipped by the TF/decoding discrepancy if its margin is below the typical discrepancy of the two passes (measured on the steps that did NOT flip, so the criterion is not tautological: at the flipped step the margin is ≤ 2·\|Δlogit\| by construction) or below the bf16 resolution of the logit. `tf_logit_diff_step0` (the alignment discrepancy, pilot 0.375) is kept as information |
| `alignment` | booleans; logit difference informative | unchanged | the step-0 logit difference (0.375 in the pilot) now also enters the smoke δ statistics via `tf_logit_diff` |

Open points this step does not decide (for Brandon): mean `H̄(prompt_last)`
at β\* was 0.598 in the pilot, outside both the calibration bracket
[0.474, 0.574] and 0.55 ± 0.05 — β\* is calibrated on prompt_last **and**
response tokens, so the two need not agree; `smoke` now reports β_ℓ per
layer and the band/bracket comparison. Layer 0 (`H̄` ≈ 0.95) dominates the
first band's `ΔH̄`; criteria are now also reported without it.

## 14. Glossary

| term | meaning in this code |
|---|---|
| `L`, `D`, `D_ff` | number of decoder layers, hidden size, SwiGLU intermediate size (from the checkpoint config) |
| `x`, `x̂` | residual at the input of `post_attention_layernorm`; its RMS-normalised version (no γ) |
| `W_g`, `γ`, `W̃` | `gate_proj.weight`, `post_attention_layernorm.weight`, `W_g·diag(γ)` |
| bank | unit rows of `W̃`, conceptually; only γ and `‖W̃_i‖` exist on disk/in memory |
| `s` | cosine scores `[T, D_ff]`; `g = ‖W̃‖√D·s` is the model's gate pre-activation |
| `β*` | inverse temperature calibrated so the mean `H̄` on the warm-up set equals `target_norm_entropy` |
| last prompt position / response positions | position `n_prompt-1` / generated non-stop positions |
| empty response | first generated token is a stop token; valid sample with NaN response-only values |
| shard | contiguous block of `samples_per_shard` ids = one SLURM array task = one directory |
| exhausted | id that failed `max_attempts` times; reported, never completed |
| exit 75 | work remains (signal or retryable failures); the batch script requeues |
