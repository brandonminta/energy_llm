# Methodology Conformance — `energy_llm`

Read-only audit against the locked methodology (`docs/metodology-FINAL.tex`,
`docs/fundamentals-FINAL.tex`, `CLAUDE.md`). Every row cites the implementing
`file:line`. Status ∈ {CONFORM, DEVIATION, NOT-FOUND}. PRIORITY items first.

## Conformance table

| # | Decision | Methodology criterion | Code location | Status | Note |
|---|---|---|---|---|---|
| **P1** | Retrieval scores `s`, not raw gate `g` | LSE term and softmax both from `s = g/‖mᵢ‖ = m̂ᵀr`; **division** by stored norm; query **not** normalized (`‖r‖` retained) | `banks.py:60-62` (`norms`, `m_hat = bank / norms…`), `trajectory.py:108` (`s = r @ m_hat.T`), `metrics.py:39,48` (softmax/lse on `s`), `metrics.py:53-54` quad uses full `r` | **CONFORM** | `m̂` built by **dividing** rows by their L2 norm at bank build (`banks.py:61`); the hook then forms `r @ m̂ᵀ`, the algebraic equal of `g/‖mᵢ‖` (eq:retrieval_scores). Query norm is kept: `quadratic_term = 0.5‖r‖²` uses the raw `r` (`trajectory.py:110-113`); no `normalize_query` path exists. Both `lse_term` and every divergence consume `s`/`p=softmax(βs)`. |
| **P2** | Energy stored decomposed | `lse_term` and `½‖r‖²` written as **separate** fields, summed only at feature time | store: `trajectory.py:229-247` + `253-254` (`ref_lse_term`,`ref_quadratic_term`,`gen_lse_term`,`gen_quadratic_term`); sum: `features.py:45-46` | **CONFORM** | Quadratic term captured independently in the hook (`trajectory.py:111-113`); never pre-summed. `metrics.energy()` (sum) exists but is a reference helper, unused by the pipeline. |
| **P3** | Headline = single locked test split | Reported AUROC = one eval on locked test split after refit on train∪val with C from inner CV; outer 5-fold descriptive only | `probe.py:171` (C on train only), `:173-181` (outer CV, C fixed, descriptive), `:184-187` (refit train∪val → single test eval); `run_probe.py:82-85` feeds `test_scores` to the headline bootstrap | **CONFORM** | Matches metodology lines 558-573 exactly. **E3 divergence quantified:** full-probe **test AUROC 0.8874** vs **outer-CV mean 0.7715 (±0.039)** — a +0.116 optimistic gap on n_test=75 (`probe_results.json`). Both are reported in the artefact; headline path is correctly the test path. |
| B1 | Bank = gate `A = W₁ᵀ`, row-indexed | gate matrix, no transpose | `models.py:35` (`gate→gate_proj`), `banks.py:59` (`bank = weight` for gate, no transpose) | CONFORM | `nn.Linear.weight` is already `[K,D]`; `w2` path transposes, gate does not. |
| B2 | gate/up via model-alias registry (Qwen+Llama) | alias registry | `models.py:27-38` (`qwen25_3b`,`llama32_3b`; `SWIGLU_PROJECTIONS`) | CONFORM | |
| B3 | float32 on CPU, dequantize first, then row-L2 normalize | precision rule | `banks.py:30-37` (`_dequantized_weight`, Params4bit branch), `:53` (`.float().cpu()`), `:60-61` (norm + normalize) | CONFORM | |
| B4 | Per-row norms stored | norms persisted | `banks.py:62,81-82` (`row_norms` in artifact) | CONFORM | |
| B5 | Raw `down_proj` rejected | ValueError on raw down orientation | `models.py:116-121` | CONFORM | `w2` allowed as transpose, capability-only; never evaluated (bank_id fixed to `gate`). |
| S2a | Pre-hook on each layer's `mlp`; r = post-attn RMSNorm input | `register_forward_pre_hook` on every `mlp` | `trajectory.py:97-100` (`register_forward_pre_hook` per layer), `:103-107` (`hidden = args[0]`) | CONFORM | |
| S2b | All passes under `inference_mode` | no grad | `trajectory.py:166`, `:290` (`@torch.inference_mode()`) | CONFORM | |
| S2c | Prefill reference = last prompt token only | position `n_prompt-1` | `trajectory.py:216` (`position_offset=n_prompt-1`), `:235` (`s_ref = s[0]`) | CONFORM | One teacher-forced pass over prompt++gen; ref = position `n_prompt-1` (causally identical, see docstring lines 11-19). |
| S2d | Greedy, max_new_tokens=50 | sampling off, cap | `trajectory.py:183-190` (`do_sample=False`), `config.py:52` / `configs/*` (`max_new_tokens: 50`) | CONFORM | |
| S2e | Chat template via checkpoint default | `apply_chat_template` | `trajectory.py:137-155` | CONFORM | Not hand-written; `add_generation_prompt=True`. |
| S2f | Hidden states never persisted; only scalar arrays | `[L]`, `[L,T_g]` to disk | `trajectory.py:221-224,251-258` (only scalar arrays + baselines) | CONFORM* | *E1 additionally caches `[L,K]`/`[L,T_g,K]` **retrieval-score** tensors in float16 (`trajectory.py:260-272`) for the sensitivity analysis — an explicitly documented E1-only extra (still not raw hidden states; the `r` itself is discarded). |
| D1 | JS in nats: `js(P,Q,base=2)²·log2` | both square **and** ·log2 | `metrics.py:103-107` (`root**2 * LOG2`, `LOG2=log(2)` at `:25`) | CONFORM | Adds an `EPS_P=1e-300` floor + renormalize (`:95-102`) to avoid `inf`; documented, O(K·EPS) negligible. |
| D2 | Squared Hellinger = `½Σ(√p−√q)²` | exact form | `metrics.py:117` | CONFORM | |
| D3 | logsumexp via scipy, max-shift | stable LSE | `metrics.py:48` (`scipy.special.logsumexp`) | CONFORM | |
| C1 | Warm-up N_c=30 training-split Qs, prefill-ref scores | calibration inputs | `calibration.py:82` (`calibration_sample_ids`), `:90-91` (`capture_prefill_reference_scores`), `config.py:44` (`num_samples=30`) | CONFORM† | †ids drawn from **all** samples (`data.py:88-97`) then **forced** into train (`probe.py:98-110`, `run_probe.py:64-71`), enforcing "train-split only" post-hoc by construction rather than sampling a pre-existing train split. Net membership identical. |
| C2 | Grid {1,2,5,10,15,20,30,50,100}; target H̄*=0.55 | grid + target | `config.py:19,46`, `configs/*.yaml`; `calibration.json` curves | CONFORM | |
| C3 | β* by log-β interpolation; clamp+log if unbracketed | interpolation rule | `calibration.py:48-68` (log-β interp), `:106-111` (clamp warning) | CONFORM | All three runs `clamped=false`. |
| C4 | β* in snapshot, read by all downstream; no per-stage recal | single β* flows | `calibration.py:127-128` (`save_snapshot`), `config.py:169-181` (`load_config` prefers snapshot), `trajectory.py:400` (`require_beta_star`) | CONFORM‡ | ‡Mechanism correct in code. **Artefact gap:** no `config_snapshot.yaml` survives on disk in any run (see manifest); β* is only recoverable from `calibration.json`. Also see **E3 recalibration deviation** in `run_manifest.md`. |
| F1 | Mean over all gen positions; no answer-span; gold never in features | pooling rule | `features.py:48-52` (`.mean(axis=1)`); gold only in metadata (`trajectory.py:280`), never read by `sample_features_from_arrays` | CONFORM | Labels joined only in `build_feature_table` (`features.py:106-107`) / at fit time. |
| F2 | Per-layer ΔE, ΔH̄, mean JS, mean Hel² | four per-layer scalars | `features.py:48-52` | CONFORM | |
| F3 | φ = [ΔE ‖ ΔH̄ ‖ JS ‖ Hel²] ∈ ℝ⁴ᴸ, group-major | ordering | `features.py:38` (`PHI_GROUPS`), `:66-68` (group-major `g`×`l`), `:115-117` (`phi_matrix`) | CONFORM | |
| F4 | ΔH (unnorm) + absolute gen-phase quantities excluded from φ | diagnostics only | `features.py:39` (`DIAGNOSTIC_GROUPS`), `:188` (`phi` uses only `PHI_GROUPS`) | CONFORM | `delta_entropy`, `gen_energy_mean`, `gen_norm_entropy_mean` stored but never in `phi_matrix`. |
| PR1 | Pipeline: Imputer(median)→StandardScaler→LogReg(lbfgs,L2) | exact pipeline | `probe.py:42-48` | CONFORM | |
| PR2 | Pipeline passed to CV → imputer/scaler refit per fold | no leakage | `probe.py:143` (`clone` + `fit` inside fold), `:178-179` | CONFORM | |
| PR3 | Stratified 70/15/15, fixed seed, test locked+hashed | split lock | `probe.py:68-126` (`make_locked_splits`, sha256, load-if-exists) | CONFORM | Test hashes reverified for all three runs (match). |
| PR4 | C by inner 3-fold stratified CV on train; grid {.01,.1,1,10}; val only in refit | selection protocol | `probe.py:129-149` (`select_c`), `config.py:72,75` | CONFORM | |
| ST1 | AUROC bootstrap B=10000, 95% CI, lower>0.5, single-class redrawn | discrimination | `stats.py:57-87` (`bootstrap_auroc_ci`), `:46-54` (redraw) | CONFORM | |
| ST2 | 2 baselines (len-norm seq logprob + mean token entropy); paired bootstrap Δ; CI lower>0 | contribution | baselines `features.py:59-63`; paired `stats.py:90-120` (same `idx` per draw) | CONFORM | `seq_logprob = mean(token_logprobs)` (length-normalized by construction, `features.py:61`). |
| ST3 | Folded max(AUROC,1−AUROC); permutation null of scan max; B_perm=1000; 95th pct | localisation | `stats.py:37-39` (`folded`), `:145-183` (`permutation_scan_threshold`, null max, `percentile=95`) | CONFORM | Exploratory; ranks cached once. |
| ST4 | Confirmatory family = {discrimination, contribution}, Bonferroni α=0.05 (→97.5% CI) | 2-test correction | `stats.py:85-86,118-119` (`*_bonferroni` use `ci_0.975`), `run_probe.py:117-122` | CONFORM | |
| L1 | Judge gpt-4o-mini, locked prompt (q, ŷ, gold), binary, raw archived | labelling | `labeling.py:32-46` (locked template + `JUDGE_PROMPT_SHA256`), `:86-92` (chat-completions, temp=0), `:142-149` (raw + hash archived) | CONFORM | |
| L2 | 3 human subsets of 60, archived; κ + agreement reproduce; κ≥0.6 | reliability | `results/{e1,e2,e3}/human_labels.csv` (60 each); recomputed below | CONFORM§ | §Subsets were produced by `scripts/label_human.py` (untracked, 30/30 stratified, embeds `judge_label`) rather than `labeling.make_human_subset_csv` (the 2-col template + `judge_human_kappa` path). κ still reproduces exactly from the stored CSVs. The in-repo `judge_human_kappa` joins `labels_dir/` (now empty), so it would not run as-is against current disk state. |
| DET1 | Global seed 42 | determinism | `config.py:88` (`seed=42`), threaded to data/calib/split/CV/bootstrap/perm | CONFORM | |
| DET2 | BF16 single A100; NF4 local-dev only, never reported | precision | `configs/*` `dtype: bfloat16`, `load_in_4bit: false`; NF4 gated at `models.py:62-69`; `banks.sbatch` a100-40gb | CONFORM | |
| M1 | Run commit recorded in artefacts | provenance | — | **NOT-FOUND** | No commit hash in any artefact; inferred `529b88f` (see manifest). |
| M2 | HF model revision pinned | provenance | `models.py:73` | **NOT-FOUND** | `from_pretrained` called with no `revision=`; not pinned in code or config. |

---

## Numbers reproduction

Recomputed **from the stored aggregates only** (`features/features.csv`,
`split.json`, `labels.json`, `human_labels.csv`) with the repo's own
`energy_llm.probe` / `energy_llm.stats`, seed 42, B=10000, B_perm=1000. No GPU
passes. "Reported" = the value in `probe_results.json` / notebook 01.

### Headline test AUROC + 95% CI

| Exp | Recomputed test AUROC (full) | Reported | 95% CI recomputed | Reported CI | Match |
|---|---|---|---|---|---|
| E1 | 0.666931 | 0.666931 | [0.56981, 0.75869] | [0.56981, 0.75869] | ✅ exact |
| E2 | 0.687003 | **0.687268** | [0.59057, 0.77871] | [0.59064, 0.77878] | ⚠ Δ≈2.6e-4 |
| E3 | 0.887374 | 0.887374 | [0.80927, 0.94950] | [0.80927, 0.94950] | ✅ exact |

### Contribution ΔAUROC + 95% CI (paired bootstrap)

| Exp | Recomputed Δ | Reported Δ | CI recomputed | Reported CI | Match |
|---|---|---|---|---|---|
| E1 | 0.195238 | 0.195238 | [0.05767, 0.33201] | [0.05767, 0.33201] | ✅ exact |
| E2 | 0.132361 | **0.132626** | [0.01376, 0.25067] | [0.01403, 0.25081] | ⚠ Δ≈2.7e-4 |
| E3 | 0.164275 | 0.164275 | [0.06258, 0.27287] | [0.06258, 0.27287] | ✅ exact |

Base-probe AUROC, selected C (base/full), and outer-CV means reproduced
**exactly** for E1 and E3; for E2 the **base** probe is exact while the
**full** probe differs at the 4th decimal (recomp full test 0.687003 /
outer-mean 0.714685 vs reported 0.687268 / 0.714810). The discrepancy is
confined to E2's full LogisticRegression solve (C=10 on D_ff=8192, L=28 → 114
features) and is at the level of lbfgs-convergence / BLAS non-determinism
across library builds. **Within numerical tolerance; flagged, not corrected.**
Note E1's **base** test AUROC is 0.4717 (below chance) — reproduced exactly;
it is a real artefact of the locked split, not an error.

### Cohen's κ + raw agreement (judge vs human, from `human_labels.csv`)

| Exp | n | Recomputed κ | Reported κ | Recomputed agr. | Reported agr. | κ≥0.6 |
|---|---|---|---|---|---|---|
| E1 | 60 | 0.8333 | 0.833 | 0.9167 | 91.7% | ✅ |
| E2 | 60 | 0.9000 | 0.900 | 0.9500 | 95.0% | ✅ |
| E3 | 60 | 0.9000 | 0.900 | 0.9500 | 95.0% | ✅ |

All three reproduce exactly. Each subset is exactly 60 (30 judged-positive /
30 judged-negative, seed 42).

### Folded per-layer AUROC profile & permutation scan

Recomputed `per_feature_folded_auroc` over the 4L φ-columns and the
permutation scan threshold (`permutation_scan_threshold`, B_perm=1000,
seed 42):

| Exp | Folded-AUROC vector max\|Δ\| vs stored | Scan threshold (recomp / stored) | n_significant (recomp / stored) |
|---|---|---|---|
| E1 | 0.0e+00 (144 cols) | 0.568777 / 0.568777 | 15 / 15 |
| E2 | 0.0e+00 (112 cols) | 0.569876 / 0.569876 | 5 / 5 |
| E3 | 0.0e+00 (144 cols) | 0.587950 / 0.587950 | 61 / 61 |

Exact match against `probe_results.json` for all three.

### Beta sensitivity curve (E1)

`results/e1/beta_sensitivity.csv` stores per-β test-split probe AUROC (β*
substituted into the ΔH̄ band, locked split):

| β | 1 | 2 | 5 | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| probe AUROC | 0.6656 | 0.6672 | 0.6751 | 0.6376 | 0.6354 | 0.6243 | 0.6148 | 0.6167 | 0.6175 |

Consistent with the headline: β*=1.9578 sits between grid points 1 and 2, and
the headline full-probe AUROC 0.6669 lies between the β=1 (0.6656) and β=2
(0.6672) sweep values. **This curve could NOT be independently re-derived**:
the cached `*_retrieval_scores.npz` tensors it is computed from are absent from
disk (see manifest). The stored CSV is internally self-consistent but not
reproducible from surviving artefacts.

### ⚠ Mismatch found — `results/e1/analysis_summary.json` localisation fields

| Field | E1 `analysis_summary.json` | E1 `probe_results.json` (canonical) | E3 `probe_results.json` |
|---|---|---|---|
| `scan_threshold` | **0.5879** | 0.5688 | **0.5879** |
| `n_layer_feature_pairs_significant` | **61** | 15 | **61** |
| `beta_sensitivity` | `[]` (empty) | — | — |

E1's `analysis_summary.json` carries **E3's** localisation numbers
(0.5879 / 61), not its own (0.5688 / 15). E2's and E3's summaries match their
own `probe_results.json`. This is a notebook aggregation/execution-order bug in
the derived summary file; the canonical `probe_results.json` is correct and
the per-layer profile reproduces exactly from `features.csv`. Additionally,
`beta_sensitivity` is empty `[]` in all three summaries despite
`results/e1/beta_sensitivity.csv` existing. **Reported as-is; not corrected.**

### Reproduction summary

- **Exact** (to stored precision): E1 and E3 headline AUROC, ΔAUROC, CIs,
  selected C, outer-CV, folded profile, scan; all three κ/agreement; all three
  split hashes and sizes.
- **Within tolerance (~3e-4):** E2 full-probe AUROC / ΔAUROC / outer-CV (lbfgs
  / BLAS-level), base probe exact.
- **Bug, flagged:** E1 `analysis_summary.json` localisation fields hold E3's
  values; `beta_sensitivity` empty in all summaries.
- **Not independently reproducible (artefacts deleted):** beta-sensitivity
  curve (score tensors gone); raw feature derivation (trajectories gone);
  β* snapshot (no `config_snapshot.yaml`).
- **Methodology deviations (factual):** E3 β* was recalibrated on TriviaQA
  rather than inherited from E1 as its config instructs (1.988 vs 1.958, low
  numerical impact); run commit and HF revision not recorded (NOT-FOUND).
