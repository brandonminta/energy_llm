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

1. `energy-llm verify` on `dev/dev_questions.jsonl`: all checks PASS.
2. `hpc/submit.sh smoke configs/qwen25_3b.yaml data/<pilot>.jsonl 50`
   with the real warm-up beta*. Read, in order: TF == greedy 100 %;
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

## Out of scope (do not implement)

Native energy, participation ratio of native coefficients, `FFN(r)`
norms and cosines, Spearman, template-only null distribution, production
ablation flag, `intervene` subcommand, §13.2 points 1–5 and 7–9.
