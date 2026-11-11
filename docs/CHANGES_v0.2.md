# energy_llm: change plan v0.2 (final)

Base: `restructure-extraction` at `aef34c4` (2026-10-27). Science:
`docs/overview.md` (project overview v0.2). The module ends at the
metrics; nothing here touches sharding, resume, atomic writes,
calibration mechanics, generation, signals or the manifest guards.

These changes are Brandon's decisions on the open points of
`docs/ARCHITECTURE.md` §13 and on the specification. Read `CLAUDE.md`,
`docs/ARCHITECTURE.md` §5, §9, §13 and `hpc/RUNBOOK.md` before starting.

Working rules for every step: `pytest` green before moving on (baseline
106 passed + 1 skipped; the count changes as tests are removed and
added); no new abstraction without a concrete second use; a function
that is called once lives inline; behaviour changes update
`docs/ARCHITECTURE.md` and get a `dev/reference.py` counterpart checked
by `energy-llm verify`. One commit per step, message prefixed with the
step number.

---

## Step 0: remove what does not serve the module

Deletions (they remain in git history; nothing is lost):

| remove | why |
|---|---|
| `downstream/` (32 files, 7 020 lines) | legacy of the E1–E3 artefact format; not installed, not tested, incompatible with the new outputs. Labelling and analysis are written fresh in Brandon's analysis repository. Tag the last commit that contains it as `thesis-e1e3` before deleting. |
| `docs/fundamentals-FINAL.tex`, `docs/metodology-FINAL.tex` | thesis chapters; the literature lives in Overleaf. The scientific specification is now `docs/overview.md`. Remove the `!docs/*.pdf` gitignore exception and the "never edit the .tex" rule from `CLAUDE.md`. |
| `docs/MAPPING.md` | its departures table is already duplicated in `ARCHITECTURE.md` §13.1; its spec→code table is §5; its verification tolerances belong next to the code (`verify.py` docstrings) and in §9. Fold, then delete. |
| `hpc/RUNBOOK.md` | fold into `README.md` as the "Verification on the cluster" subsection (it is a sequence of the same commands the README already documents). |
| `features.parquet`, `FEATURES`, `VARIANTS`, `RESPONSE_FEATURES`, `_phase_features`, `metrics.log_mean_distribution`, `metrics.hellinger_sq`, `metrics.jaccard_topk`, `dev/reference.py: FEATURES, VARIANTS, phase_features, mean_distribution, hellinger_sq, jaccard`, the `features` branches of `verify.compare_with_reference`, `_read_outputs`, `inspection.py` and `tests/test_capture.py`, `tests/test_reference.py`, `tests/test_verify.py` | the per-layer aggregated table duplicates the per-token traces (Step 4). Aggregation over tokens is analysis, not extraction. |

After Step 0 the documentation is exactly three files with one
responsibility each: `README.md` (how to run, outputs, cluster),
`docs/ARCHITECTURE.md` (what the code does, audit §13),
`docs/overview.md` (why: the science). `CLAUDE.md` points to these
three and nothing else. `dev/README.md` stays (15 lines, verification
data).

Acceptance: `pytest` green; `energy-llm verify` on the tiny models
passes with the reduced check set; `git grep -n "features.parquet\|hellinger\|MAPPING"` returns nothing.

## Step 1: `core/metrics.py` — primitives

Add, float32, on `s [N, D_ff]` and `logp = log_softmax(beta*s)`:

- `mean_score(logp, s)`: `sum(p * s)`.
- `var_score(logp, s)`: `sum(p * (s - mean_score)^2)`.
- `topk_mass(logp, k)`: sum of the k largest `p`.
- `kl_divergence(logp, logq)`: `sum(p * (logp - logq))`, clamped at 0.
- `topk_indices(logp, k)`: int32 indices of the k largest, descending.
- `xhat_top1_frac(x_hat)`: `max_j x_hat_j^2 / D` per row.

Keep `rms_normalise`, `energy`, `log_distribution`, `norm_entropy`,
`margin`, `js_divergence`. Nothing else in the module.

Tests (`tests/test_metrics.py`): KL = 0 on identical inputs and
`KL(p||q) − KL(q||p)` antisymmetric; `var_score` → sample variance of
`s` as beta → 0 and → 0 as beta → inf; `topk_mass` → 1 as beta → inf;
`xhat_top1_frac` = 1 for a one-hot row, 1/D for a constant row;
`topk_indices` sorted, no duplicates.

`dev/reference.py`: naive float64 versions of the six (explicit loops).

## Step 2: `core/bank.py` — gamma diagnostic (report only)

- `gamma_diagnostic(bank, k=5) -> dict`: per layer, the indices of the k
  largest `|gamma_j|` and the fraction of `sum_i ||w_tilde_i||^2`
  carried by those k columns (one layer's `W_g` in float32 at a time, as
  `scores` does). Written to `bank.h5` attrs by `calibrate` and copied
  into `manifest.json` by `merge`.
- No ablation flag, no config key, no change to `scores`,
  `fingerprint`, hashes or calibration context. The ablation question
  ("does the sign change without those dims?") is answered by the smoke
  report of Step 6, which recomputes `H_bar` on the fly with those gamma
  entries zeroed, for the smoke questions only.

Tests (`tests/test_bank.py`): fractions in [0, 1]; the per-dimension
fractions sum to 1 over all dims; diagnostic is deterministic.

## Step 3: `config.py` — beta grid and k

- Replace `extraction.beta_multipliers` by
  `extraction.beta_grid_log10: {min: 0.0, max: 4.0, num: 9}` (absolute,
  log-spaced, identical across models because `s` is a cosine) plus
  `beta_star` always appended: `n_beta = num + 1`, `betas[-1] == beta*`.
- `extraction.topk_mass_k: 10`, `extraction.topk_ids_k: 100`; remove
  `jaccard_top_k`.
- `merge` asserts identical `betas` across items (closes §13.2.6).

Tests: config round trip; a shard written under the old hash is refused.

## Step 4: `core/capture.py` — three references, per-token traces

`ForwardTrace.reference_positions -> dict[str, int | None]`:
`question_last` (last position with role `ROLE_QUESTION`; `PromptError`
if none), `prompt_last` (`n_prompt − 1`), `response_first` (first
response position; `None` when empty). Stop tokens never carry the
question role, so §13.2.10 is unaffected.

`compute_item` writes, for every layer `l` and every position `t`
(prompt and response; `role_mask` separates them downstream):

| dataset | shape | dtype | beta |
|---|---|---|---|
| `energy` | `[L, T, n_beta]` | float16 | grid |
| `norm_entropy` | `[L, T, n_beta]` | float16 | grid |
| `topk_mass` | `[L, T, n_beta]` | float16 | grid |
| `mean_score` | `[L, T, n_beta]` | float16 | grid |
| `var_score` | `[L, T, n_beta]` | float16 | grid |
| `margin` | `[L, T]` | float16 | – |
| `xhat_top1_frac` | `[L, T]` | float16 | – |
| `topk_ids` | `[L, T, k_ids]` | int32 | – |
| `js/<ref>` | `[L, T, n_beta]` | float16 | grid |
| `dkl/<ref>` | `[L, T, n_beta]` | float16 | grid; `KL(p_t‖p_ref) − KL(p_ref‖p_t)` |

`<ref>` in `{question_last, prompt_last, response_first}`. The row at a
reference position is 0 for its own divergences; `response_first`
datasets are NaN for an empty response. One `s [T, D_ff]` per layer as
today; loop the grid inside the layer; finite checks as today.

Residuals for `hidden.h5` at the three reference positions and at
`response_last` (NaN when empty), float16, layers from `hidden_layers`.

Cost: `n_beta = 10`, `L = 36`, `T ≈ 90`: about 2 MB per question before
gzip. If smoke seconds/question exceeds 2× the previous smoke log, batch
the grid as one `log_softmax` over `[n_beta, T, D_ff]` per layer.

Tests (`tests/test_capture.py`): shapes; own-reference rows are zero;
empty-response NaN pattern; `topk_ids` sorted and duplicate-free;
`energy[..., -1]` equals the previous single-beta trace within float16
on the tiny models.

## Step 5: `storage.py`, `pipeline/merge.py` — layout, manifest, baselines

- Item HDF5 and `traces.h5`: the datasets of Step 4, gzip 4, chunks
  `[1, T, n_beta]`; attrs `betas` (float64, last = beta*),
  `reference_positions` (three ints per id, −1 for None), `topk_ids_k`,
  `topk_mass_k`, `layout`.
- `bank.h5`: gamma diagnostic attrs.
- `manifest.json:checks`: `max_xhat_top1_frac` per reference position
  over the run; gamma diagnostic summary.
- `answers.parquet`: per-token `token_logit` (logit of the chosen token,
  float32, same decoding logits as `token_logprobs`).
- `baselines.parquet`: `mean_neg_logit` over response tokens (one-pass
  Semantic-Energy baseline; NaN when empty).

Tests (`tests/test_storage.py`, `tests/test_resume.py`): round trip;
merge of two shards; `betas` mismatch raises.

## Step 6: `verification/`, `pipeline/smoke.py`

- `dev/reference.py:reference_item` returns the per-token traces of
  Step 4 for the layers under test. `verify.compare_with_reference`
  compares them with tolerances derived like `reference_tolerances`
  (Lipschitz in `delta_s` and beta): `mean_score`, `var_score` ≤
  `2·beta·delta_s·range(s)`; `topk_mass` ≤ `beta·k·delta_s`; `dkl` ≤
  twice the JS bound; `topk_ids` compared as sets, Jaccard ≥ 0.98
  (near-ties at the k-th score are expected); `xhat_top1_frac` within
  `score_tolerance`.
- `self`: `dkl(p, p) = 0`. `limits`: `var_score → 0`, `topk_mass → 1` as
  beta → inf. `permutation`: `topk_ids` compared after mapping through
  the permutation.
- `inspection.py`: per layer `E | H_bar | margin | var_score |
  topk_mass` at beta* for the three reference positions and the response
  mean.
- Smoke summary adds, per layer band (early / middle / late thirds):
  median and max `xhat_top1_frac` at the three references; mean
  `var_score` at beta* for prompt vs response; gamma top-5 fraction; and
  the ablation line: mean `H_bar(response) − H_bar(prompt_last)` with
  the top-5 gamma dims zeroed vs intact, computed in `smoke` from the
  captured residuals (no production path involved). Seconds/question.

## Step 7: documentation, packaging, release

- `README.md`: "What is computed" (the Step 4 table), "Outputs" (below),
  "Verification on the cluster" (former RUNBOOK). `docs/ARCHITECTURE.md`:
  §5, §8, §9 rewritten for the new layout; §13.1 rewritten as departures
  from `docs/overview.md` (the `.tex` is gone); §13.2.6 and §13.2.11
  closed. `docs/overview.md`: §3 score convention
  (`s = cos(w_tilde_i, x_hat)`, no `E_norm`), §6 and §7 aligned with the
  dataset names above. `CLAUDE.md`: document table reduced to the three
  files; invariants updated (grid, references, no features table, no
  Hellinger).
- `pyproject.toml`: version `2.1.0`; `requirements.txt` unchanged unless
  a step needed something new (it should not).
- After the pilot passes: merge `restructure-extraction` into `main`
  (fast-forward), delete the branch, tag `v2.1.0`. `manifest.json`
  already records the commit; every cell in the paper cites that tag.

### Outputs: the contract one cell delivers

`$SCRATCH_ROOT/energy_llm/runs/<model>/`

```
calibration/
  beta_star.json      beta*, calibration context, warm-up hash, bracket
  bank.h5             gamma, ||w_tilde|| per layer, gamma diagnostic, fingerprint
extract/<name>/
  manifest.json       config + hash, beta*, commit, library versions, hardware,
                      peak GPU memory per shard, dtype, checks, exhausted ids
  answers.parquet     id, prompt, response, token ids, lengths, empty_response,
                      truncated, stop_token_id, per-token logprob/entropy/logit,
                      TF mismatches, xhat_norm_max_rel_dev
  baselines.parquet   mean/min logprob, mean entropy, mean_neg_logit
  traces.h5           /<id>/{energy, norm_entropy, topk_mass, mean_score, var_score,
                             margin, xhat_top1_frac, topk_ids, js/<ref>, dkl/<ref>,
                             role_mask}; attrs betas, reference_positions
  hidden.h5           /<id>/{question_last, prompt_last, response_first, response_last}
  bank.h5             copy of calibration/bank.h5
```

Six files. The analysis repository reads `manifest.json` for provenance,
joins everything on `id`, and never needs the model.

## Pilot (HPC, after Step 7)

1. `energy-llm verify` on `dev/dev_questions.jsonl`: all checks PASS
   (`tf_vs_cache`, `permutation`, `self_divergence`: **revised after the
   pilot**, Step 9).
2. `hpc/submit.sh smoke configs/qwen25_3b.yaml data/<pilot>.jsonl 50`
   with the real warm-up beta*. Read, in order: TF == greedy 100 %
   (**revised after the pilot**, Step 9: every disagreement with margin < δ);
   `xhat_norm_max_rel_dev < 1e-3`; mean `H_bar` at beta* on
   `prompt_last` within 0.55 ± 0.05.
3. `xhat_top1_frac` at `prompt_last`: median above 0.5 in any band means
   the template token carries massive activations and `question_last`
   is the primary reference in the analysis.
4. Gamma: top-5 fraction above 0.5 in a band, or a sign flip in the
   ablation line, means the shallow-layer energy effect of the thesis is
   a magnitude artefact; it is then reported as such and the analysis
   uses `question_last` and middle/late layers.
5. Paste the smoke summary into `docs/pilot_<date>.md`.

## Step 9: numerics, diagnostics and reports after the pilot

Pilot (Qwen2.5-3B, bf16, A100, β\* = 72.776, jobs 32879/32880): `verify` FAIL on `tf_vs_cache` (max ‖x_cache − x_tf‖/‖x_tf‖ = 0.0907,
tol 0.05 provisional, tokens equal), `permutation` (3.05e-5 = 2⁻¹⁵, tol 1e-5,
Jaccard 1.0) and `self_divergence` (3.27e-7, tol 1e-7); `alignment` step-0
logit difference 0.375. `smoke` (24 questions): TF == greedy on 99.64 % of
tokens and 95.8 % of answers (one divergent token); mean H̄(prompt_last)
≈ 0.598, outside the calibration bracket [0.474, 0.574]; `xhat_top1_frac`
≈ 0.35 at question_last and prompt_last in layers 12–35; layer 0 H̄ = 0.95
dominates the ΔH̄ of band 0–11. No change to `beta_grid_log10`, to the
definitions of the metrics, or to SLURM resources. One commit per task.

| task | change | file(s) |
|---|---|---|
| T2 | JS pointwise in `d = log p − log q` (`ln2 = softplus(0)`): `JS(p,p) = 0` exactly; terms with `p_i = 0` / `q_i = 0` masked (xlogy). KL already from one `log_softmax`: `KL(p‖p) = 0` exactly. `self_divergence` tolerance `4·u₃₂·log D_ff` | `core/metrics.py`, `verify.py` |
| T1 | `tf_vs_cache`: per-layer x error distribution and location of its max; error propagated to H̄, E, JS at β\*; PASS = tokens equal and p99/σ_ℓ ≤ 0.1 (`--tf-max-noise-ratio`), max/σ apart. New `fp32_reference` (INFO, `--no-fp32-reference`): separate float32 instance, TF32 off, freed after. CLI options recorded in `verify_report.json` | `verify.py`, `cli.py`, `core/model.py` (`load_model(dtype=)`) |
| T3 | `permutation`: per-trace tolerances from float32 tree-reduction bounds (c = 2; ARCHITECTURE §13.5), location (trace, layer, position, β) of the max, γ-shuffled negative control; `dkl` (unchanged, = KL(p‖q) − KL(q‖p) as in overview §6.1) documented as ill-conditioned at high β; `var_score` confirmed centred (test) | `verify.py` |
| T4 | per generated step `max\|logit_gen − logit_tf\|`; at each mismatch top1−top2 margins in TF and generation; `answers.parquet`: `tf_exact`, `tf_mismatch_{tf_tokens,margin_tf,margin_gen,top_logit}`, `tf_logit_diff`, `tf_logit_diff_max`, `tf_logit_diff_step0`; smoke list of disagreements and criterion "all margins < δ" | `core/capture.py`, `smoke.py` |
| T5 | smoke (report only): criteria 3 and 4 per layer and per band, also without layer 0; outlier dim d\* per layer; ablations (top-5 γ, d\*) at β\* and at the re-matched β; β_ℓ per layer with H̄_ℓ = 0.55; H̄(prompt_last) vs 0.55 ± 0.05 and the calibration bracket | `smoke.py` |
| T6 | `hpc/pilot_bundle.sh <verify_jobid> <smoke_jobid>` (login node, bash 4.2, no Python packages): sacct, logs and the reports of the runs of THOSE jobs | `hpc/pilot_bundle.sh`, `smoke.py` (report path line) |
| T7 | this section; ARCHITECTURE §13.1, §13.5, §13.6; overview §8.1 and new §8.6 | docs |

Criteria **revised after the pilot** (overview §8, README):

- §8.1 TF vs generation: was `‖Δr‖/‖r‖ < 10⁻³` (and `‖Δx‖/‖x‖ ≤ 5e-2` in
  `verify`). Now noise/signal of the traces ≤ 0.1. Reason: in bf16 the
  residual differs by up to 9 % with identical tokens; the question is
  whether that moves the stored metrics relative to their spread between
  questions.
- Pilot step 2 / new §8.6 TF == greedy: was 100 % of tokens. Now every
  disagreement has margin < δ. Reason: 99.64 % with one flipped near-tie is
  the expected bf16 behaviour; a disagreement with a large margin would be
  an alignment error, which the criterion still catches.
- `permutation`: was a global `1e-5` relative to `max(1, |value|)`. Now
  per-trace bounds. Reason: `dkl` subtracts two KLs of up to hundreds of
  nats, so 1 ulp of a value in [256, 512) is 2⁻¹⁵ regardless of `|dkl|`.
- `self_divergence`: was `1e-7`. Now exactly 0 expected, tolerance
  `4·u₃₂·log D_ff`. Reason: the former JS form left one rounding of
  `logaddexp`; the pointwise form removes it.

Acceptance: `pytest` 148 passed + 1 skipped (CPU); on the cluster `verify`
and `smoke` re-run on the same config and dev data as jobs 32879/32880,
bundled with `hpc/pilot_bundle.sh`.

## Step 10: score variants, relative β grid, portability, trace dtype

Pilot evidence (Qwen2.5-3B, A100; figures as reported by Brandon, pending verification against the 33014/33019 reports):

- **Smoke 33014.** The dominant dimension of `x̂` is d\* = 1874, with γ_d\* ≈ 0 in layers 7–11 and 32–34. Under the former score `s_i = cos(γ⊙w_i, x̂)` this gives a position-dependent effective temperature `β·g_t`, `g_t = ‖γ⊙x̂_t‖/√D` (overview §6.0).
- **Verify 33019.**
  - TF and cached bf16 states are equally far from float32 in every layer (ratio ≈ 1).
  - The excess is in layers ≥ 30, with p99 ≈ max.
  - The `tf_vs_cache` maximum is at position 0, of role template (attention sink).
  - Large ratios in layers 26–27 come with a small σ_ℓ.

Decisions (Brandon, 2026-11-11):

- Control variant name: `cos_gamma_key`.
- Exclusion rule:
  - evaluated at the re-matched β;
  - bootstrap CI as in C6;
  - list proposed from `cos_gamma_query` only.
- `ΔH̄` reported against `prompt_last` and against `question_last`.
- `tf_vs_cache` must PASS for both variants.
- Overview §3 and §5 are updated as well.
- `topk_ids` only for the primary.
- β\* grid position printed, with a 0.5-decade warning.

| task | change | file(s) |
|---|---|---|
| C1 | architecture adapter (`ffn_norm`, `effective_gain`: Qwen2/Llama; Gemma `NotImplementedError`); one pre-activation per layer serves `cos_gamma_query` `c = z/(‖w‖‖γ⊙x̂‖)` and `cos_gamma_key` `s = z/(‖w̃‖√D)`; `query_gain`; `bank.h5` stores `key_norms`, fingerprint covers them | `core/model.py`, `core/bank.py` |
| C2 | `extraction.beta_grid_log10_offset {min: -0.8, max: 0.8, num: 9}` (β\*_v·10^o, offset 0 included; first draft −0.7…0.9 had no 0, corrected before the HPC run; β\* last); `beta_grid_log10` rejected; `extraction.control_topk_ids_k` (default 0) | `config.py`, `configs/*.yaml` |
| C3 | one β\* per variant (`beta_star.json: variants`), grid position and `near_edge` (< 0.5 decades), printed by calibrate; extract refuses a record missing a variant | `pipeline/calibrate.py`, `pipeline/extract.py`, `cli.py` |
| C4 | traces of both variants (`<variant>/<name>`), `query_gain` and `xhat_top1_frac` once; item files and `traces.h5` per variant group, file attrs per variant; `inspect --variant` | `core/capture.py`, `pipeline/extract.py`, `pipeline/merge.py`, `verification/inspection.py` |
| C5a/b | `dev/reference.py` computes both variants independently; verify: every bank check per variant (gate `‖w‖‖γ⊙x̂‖c`), `tf_vs_cache` / `fp32_reference` broken down by role and special position with σ_ℓ per layer, PASS on the analysis positions only; precision rule `RECOMMEND trace_dtype=float32` (p99/σ > 0.1) | `dev/reference.py`, `verification/verify.py` |
| C5c | `model.trace_dtype` (bfloat16 \| float32): bf16 generation, traced TF pass on the trace model; part of the calibration context | `core/model.py`, `core/capture.py`, `cli.py`, `config.py` |
| C6 | smoke per variant: `H̄`, `ΔH̄` vs `prompt_last` and `question_last`, JS, ablations at β\* and re-matched β; `g_t` distribution by role and band; relative-depth bands; `exclusion_layers` (reported, not applied); β\* grid position; topk_ids saving (MB/question) | `pipeline/smoke.py` |
| C7 | `pilot_bundle.sh`: the commit is recorded with any git (no `git -C`, `safe.directory`, error shown, `.git/HEAD` fallback) | `hpc/pilot_bundle.sh` |
| C8 | GPU tests for both variants and for `trace_dtype: float32`; found a missing check, so `prepare_bundle` now requires identical bank weights in the trace model, and the bank is built from it | `tests/test_gpu_verify.py`, `core/model.py` |
| C9 | this section; overview §3 (and §3.1 scope), §5, §6.0, §6.1, §6.3, §7, §8; ARCHITECTURE §1–§12, §13.1, §13.7, glossary; README; CLAUDE.md | docs |

Criteria **revised after the pilot** (overview §8):

- §8.1 TF vs generation: the PASS set is now the **analysis positions** (response tokens + `question_last`, `prompt_last`, `response_first`).
  - Before: every compared position.
  - Reason: the excess comes from position 0 and template tokens, which the analysis never reads.
  - Template tokens and position 0 are reported apart.
  - Pre-registered precision rule: p99/σ_ℓ of bf16 vs float32 > 0.1 on the analysis positions → `RECOMMEND trace_dtype=float32`.
- §8.2 gate: checked for both variants.
- §8.3 `H̄` band: checked per β\*_v.
- The extraction β grid is relative to each β\*_v.
  - Before: an absolute 10⁰…10⁴ grid shared by every model.
  - Reason: two variants with different β\*.

Acceptance: `pytest` 192 passed + 1 skipped with CUDA (183 + 10 skipped
without); on the cluster: calibrate in a clean `SCRATCH_ROOT`, then verify
and smoke, bundled with `hpc/pilot_bundle.sh`.

## Out of scope (do not implement)

Native energy, participation ratio of native coefficients, `FFN(r)`
norms and cosines, Spearman, template-only null distribution, production
ablation flag, `intervene` subcommand, §13.2 points 1–5 and 7–9.
