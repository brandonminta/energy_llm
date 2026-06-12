# MAPPING.md — thesis methodology → implementation

Ground truth: `docs/fundamentals-FINAL.tex` (`chap:theory`) and
`docs/metodology-FINAL.tex` (`chap:methods`). Every operational statement in
those chapters maps below to the module, function, and config key that
implements it. Implementation notes and any point where the code had to make
a choice the thesis leaves open are flagged at the end.

## Theory chapter (chap:theory)

| Statement | Label | Implementation |
|---|---|---|
| Probe point r_t^(l) = RMSNorm(h̃_t^(l)), the FFN input | `eq:probe_point` | forward pre-hook on each decoder layer's `mlp` module: `trajectory.ProbePointHooks` (a pre-hook receives the exact module input). Placement verified by `tests/test_trajectory.py::TestHookPlacement` against `post_attention_layernorm` output. |
| Gated FFN identification A=W1^T, B=V^T, C=W2 | `eq:gated_identification` | `models.SWIGLU_PROJECTIONS` (`gate_proj`/`up_proj`/`down_proj` for Qwen and Llama families); `nn.Linear.weight` is already `[out,in] = [K_l, D]`, i.e. row-indexed W1^T — no transpose for `gate`/`value`, transpose for `w2`. |
| Memory bank M^(l) := A^(l) (rows = gate patterns) | `eq:memory_bank_definition` | `banks.extract_memory_bank(bank_id="gate")`; config `bank.bank_id: gate`. |
| Modern Hopfield energy E(q; M, β), constant c₀ omitted | `eq:modern_hopfield_energy` | `metrics.lse_term` (`scipy.special.logsumexp`) + `metrics.quadratic_term` (0.5‖r‖²), stored as **separate** npz fields `*_lse_term`, `*_quadratic_term`; summed only at feature time (`features.sample_features_from_arrays`). c₀ is never computed. |
| Retrieval distribution p_i = softmax(β s) | `eq:retrieval_distribution` | `metrics.retrieval_distribution` (`scipy.special.softmax`). |
| Retrieval entropy H | `eq:retrieval_entropy` | `metrics.entropy` (nats). |
| Normalised entropy H̄ = H/log K | `eq:normalised_entropy` | `metrics.norm_entropy`. |
| KL is a building block only, never a feature | `subsec:kl_divergence` | no KL anywhere in the feature path; only JS and Hellinger² are computed. |
| JS divergence, symmetric, bounded by log 2 | `eq:js_divergence` | `metrics.js_nats`; bound and symmetry tested in `tests/test_metrics.py`. |
| Squared Hellinger distance | `eq:hellinger_squared` | `metrics.hellinger_sq`, direct NumPy (`0.5·Σ(√p−√q)²`). |

## Methodology chapter (chap:methods)

### Problem analysis → design decisions

| Statement | Label | Implementation |
|---|---|---|
| Last-token prefill reference; per-token generation states; identical prompt format across phases | `subsec:pa_probe_positions` | `trajectory.capture_sample`: one chat-templated prompt; reference = position `n_prompt−1`; generation states = positions `n_prompt..end` of the same token sequence (see note 1). |
| Energy decomposition kept separable (lse vs query-norm term) | `subsec:pa_scale_confounds` | separate `lse_term`/`quadratic_term` fields end-to-end; tested in `tests/test_metrics.py::TestEnergyDecomposition`. |
| Bank rows on a common scale; derived-view caveat | `subsec:pa_scale_confounds` | row-wise L2 normalisation at bank build; norms stored so g = s·‖m_i‖ is recoverable (`tests/test_banks.py::test_scores_reconstruct_gate_preactivations`). |
| Scores read **before** the nonlinearity (dot-product form) | `subsec:pa_scale_confounds` | the pre-hook fires before `mlp.forward`, so `s = m̂ @ r` uses the raw pre-Swish inner products. |

### Solution design

| Statement | Label | Implementation |
|---|---|---|
| Gate pre-activations g = A r | `eq:gate_preactivations` | never materialised in the pipeline (the hook computes s directly); reconstruction g = s·‖m_i‖ verified in tests. |
| Retrieval scores s_{t,i} = g_{t,i}/‖m_i‖ = m̂_iᵀ r; query NOT normalised | `eq:retrieval_scores` | `ProbePointHooks._make_hook`: `s = r @ m̂ᵀ` inside the hook; no query normalisation anywhere (`normalize_query=False` is structural, not a flag). **All energies/softmaxes/entropies/divergences operate on s, never on g** — resolves the thesis's open verification. |
| Five per-layer scalars ΔE, ΔH, ΔH̄, mean JS, mean Hel² | `eq:delta_energy`–`eq:mean_hellinger` | `features.sample_features_from_arrays`; pooling = mean over ALL generated positions 1..T_g (no answer-span logic; gold answers never touch features). |
| φ = [e ‖ h ‖ j ‖ d] ∈ R^{4L} | `eq:feature_vector` | `features.PHI_GROUPS` order `[delta_energy, delta_norm_entropy, mean_js, mean_hellinger_sq]`, group-major × layer-minor; `features.phi_matrix`. ΔH (unnormalised) and absolute gen-phase E/H̄ stored as diagnostics only (`features.DIAGNOSTIC_GROUPS`), excluded from φ. |
| β calibration: N_c warm-up, grid, target H̄*=0.55, log-β interpolation, clamp+log | `subsec:beta_calibration`, `subsec:impl_beta` | `calibration.calibrate_beta` / `interpolate_beta_star`; config `calibration.*`; β* written into `config_snapshot.yaml`, which all later stages read (`config.load_config` prefers the snapshot). Per model: E2 recalibrates; E3 inherits the E1 value (same model). |
| Probe ẑ = σ(φᵀw + b), L2-regularised BCE, sklearn C convention | `eq:logistic_probe`, `eq:logistic_objective` | `probe.make_probe_pipeline`: `Pipeline(SimpleImputer(median), StandardScaler, LogisticRegression(solver="lbfgs", C))`. |
| 70/15/15 stratified split, seed fixed, test locked before any fit | `subsec:probe` | `probe.make_locked_splits`: written once to `split.json` with a sha256 hash of the test ids; reloads verify the hash and never recompute. |
| C by stratified CV within train only; val enters only the final refit | `subsec:probe` | `probe.select_c` (3-fold stratified on train); refit on train∪val; **single** test evaluation (`probe.run_probe_protocol`). |
| Outer 5-fold CV over train∪val, C fixed, descriptive only | `subsec:probe` | `run_probe_protocol` → `outer_cv_aurocs`; reported as stability only. |
| Per-layer AUROC + single-feature unsupervised AUROC diagnostics | `subsec:probe` | `stats.per_feature_folded_auroc` (no fitted model); notebook 02 best-layer table. |
| Per-sample algorithm | `alg:per_sample` | `trajectory.capture_sample` line-for-line (prompt → prefill ref → greedy decode → per-(l,t) metrics → pooling in Stage 4). |

### Implementation section

| Statement | Label | Implementation |
|---|---|---|
| Package `energy_llm`, artifact pipeline, every stage independently re-runnable | `sec:implementation` | flat package `energy_llm/`; each stage reads only prior-stage artifacts; per-sample artifacts are the checkpoint unit (`io_artifacts.SampleManifest`). |
| `AutoModelForCausalLM.from_pretrained`; alias registry; dequantise; float32 on CPU; row L2 norms stored; banks.pt with dims/norms/bank id | `subsec:stage1` | `models.load_model_and_tokenizer`, `banks.build_banks` (`_dequantized_weight` for bitsandbytes Params4bit; `.float().cpu()` before normalisation). |
| Value-branch (up_proj) and W2^T banks = capabilities, never evaluated; raw down_proj rejected | `subsec:stage1` | `bank_id ∈ {gate, value, w2}`; `down`/`down_proj` raises `ValueError` (`models.resolve_projection`); no experiment config uses anything but `gate`. |
| `apply_chat_template` (checkpoint default); greedy `model.generate`, `max_new_tokens=50` | `subsec:stage2` | `trajectory.build_prompt_ids`; `capture_sample` (`do_sample=False`); config `generation.max_new_tokens: 50`. |
| `register_forward_pre_hook` on `mlp`; `torch.inference_mode`; hooks observe without rewriting | `subsec:stage2` | `ProbePointHooks`; `@torch.inference_mode()` on all capture paths. |
| logsumexp via scipy (max-shifted) | `subsec:stage2` | `metrics.lse_term`. |
| JS = `jensenshannon(P,Q,base=2)**2 · log 2` (nats, bound exact) | `eq:js_convention` | `metrics.js_nats`, byte-for-byte that expression (exactness tested). |
| Hellinger² directly in NumPy | `subsec:stage2` | `metrics.hellinger_sq`. |
| Hidden states never persisted; compressed per-sample artifacts: [L] ref scalars, [L,T_g] per-token scalars, metadata | `subsec:stage2` | `trajectory.write_sample_artifacts` → `{id}.npz` + `{id}.json`; only scores/scalars leave the hook, on CPU. |
| Score tensors cached for the β sweep (no forward passes) | `subsec:an_sensitivity` | `{id}_retrieval_scores.npz` (float16, compressed) behind `trajectory.cache_score_tensors` (E1 only); `features.build_feature_table_at_beta` recomputes everything per β. |
| Shard interface, order-independent Stage 4 | `subsec:stage2` | `data.shard` stride slicing; `--shard-index/--num-shards`; `hpc/trajectories.sbatch` array job. |
| Judge via OpenAI-compatible chat completions, locked prompt, raw response archived, labels separate from trajectories, relabelling CPU-only | `subsec:stage3_labeling` | `labeling.JUDGE_PROMPT_TEMPLATE` (constant + sha256 stamped on every label), `label_samples` (resumable, exponential backoff), labels in `results/<exp>/labels/`. |
| Pipeline cloned per fold ⇒ no standardisation leakage | `subsec:stage4` | sklearn `Pipeline` + explicit `clone` per fold in `probe.select_c`/`run_probe_protocol`; enforced by `tests/test_probe.py::TestLeakage`. |

### Analysis method

| Statement | Label | Implementation |
|---|---|---|
| Bootstrap test AUROC, B=10000, 95% CI, single-class resamples redrawn | `subsec:an_discrimination` | `stats.bootstrap_auroc_ci` (`_resample_indices` redraws). |
| Paired bootstrap ΔAUROC on identical resamples | `subsec:an_contribution` | `stats.paired_bootstrap_delta_auroc` (one index draw per replicate, both probes scored on it). |
| Baselines: length-normalised seq log-prob + mean token entropy, same decoding pass | `subsec:an_contribution` | computed in `capture_sample` from `generate(output_scores=True)`; `features.BASELINE_COLUMNS = [seq_logprob, token_entropy_mean]`; base vs full probes in `probe.run_base_and_full_probes`. |
| Folded statistic max(AUROC, 1−AUROC); permutation null of the scan max over ALL layer-feature pairs, B_perm=1000, 95th pct | `subsec:an_localisation` | `stats.permutation_scan_threshold` over the 4L φ columns (label-free column ranks precomputed once). |
| Cohen's κ vs 60-sample human subset, threshold ≥ 0.6 | `subsec:an_label_reliability` | `labeling.judge_human_kappa` + `make_human_subset_csv`; notebook 01. |
| β sensitivity from cached tensors, descriptive | `subsec:an_sensitivity` | notebook 02 β-sweep cell; same locked split per β. |
| Confirmatory family = exactly {discrimination, contribution}, Bonferroni α=0.05 | `subsec:an_sensitivity` | each claim's corrected decision reads the 97.5% bootstrap CI (`discriminates_bonferroni` / `contributes_bonferroni`); everything else labelled exploratory/descriptive in outputs. |
| Per-category descriptive AUROC, largest categories only | `subsec:an_discrimination` | `stats.per_category_auroc` (min_n + both-classes filter), notebook 02. |

### Experimental configuration (tab:hyperparameters)

| Parameter | Value | Config key / code |
|---|---|---|
| bank | gate | `bank.bank_id` in all of e1/e2/e3 |
| normalize (row L2) | True | structural in `banks.extract_memory_bank` (always) |
| hook_target | mlp_input | structural in `ProbePointHooks` (pre-hook on `mlp`) |
| score | gate pre-activation, rescaled | structural (`s = r @ m̂ᵀ` in the hook) |
| normalize_query | False | structural (query never normalised) |
| β* target H̄*=0.55 | calibrated | `calibration.target_norm_entropy` |
| β grid | {1,2,5,10,15,20,30,50,100} | `calibration.beta_grid` |
| Operating band H̄∈[0.3,0.7] | design rationale | not a runtime check (the thesis states it as the saturation-motivated band around the target); the calibration curve is archived in `calibration.json` so the band placement is auditable |
| calibration_samples N_c | 30 | `calibration.num_samples` |
| Decoding | greedy | `do_sample=False` (structural) |
| T_max | 50 | `generation.max_new_tokens` |
| Chat template | checkpoint default | `tokenizer.apply_chat_template` (structural) |
| Pooling | mean over all generated tokens | structural in `features.sample_features_from_arrays` |
| Baseline features | seq log-prob (length-norm.), mean token entropy | `features.BASELINE_COLUMNS` |
| Regulariser / C grid | L2 / {0.01,0.1,1,10} | `probe.c_grid` |
| Inner / outer CV | 3-fold / 5-fold stratified | `probe.inner_cv_folds` / `probe.outer_cv_folds` |
| Split | 70/15/15 | `probe.split` |
| Bootstrap / permutation B | 10000 / 1000 | `stats.n_bootstrap` / `stats.n_permutation` |
| Confirmatory family | Bonferroni, α=0.05, 2 tests | `stats.alpha`; 97.5% CI decisions |
| Judge / human subset | gpt-4o-mini / 60 | `labeling.judge_model` / `labeling.human_subset_size` |
| Seed | 42 | `seed` (one seed flows to splits, CV, subsampling, calibration draw, bootstrap, permutation) |
| Models / datasets | Qwen2.5-3B-I + Llama-3.2-3B-I; TruthfulQA 817 gen split; TriviaQA rc.nocontext 500@seed42 | `models.MODEL_REGISTRY`; `data.load_truthfulqa` / `data.load_triviaqa`; configs e1/e2/e3 |
| BF16 single A100; 4-bit local-dev only | `subsec:models` | `model.dtype: bfloat16`, `model.load_in_4bit: false` in all experiment configs; 4-bit path exists only for local dev |

## Implementation notes and flagged decisions

No conflict between the thesis chapters and the rebuild specification was
found. Four places where the code had to fix something the thesis leaves
mechanically open:

1. **Generation-state capture mechanics.** The thesis describes pre-hooks
   retaining "the state of each newly produced token" during decoding. An
   incremental hook inside `model.generate` never sees the *last* generated
   token (it is emitted but never fed back), so it yields only T_g−1 states.
   The implementation instead runs greedy decoding first (hooks off,
   collecting the per-step logits for the baselines) and then ONE hooked
   teacher-forced forward pass over [prompt ++ generated tokens], keeping
   positions `n_prompt−1` (prefill reference) through the end (generated
   tokens 1..T_g). By causal masking these states are mathematically
   identical to the incremental ones, the prompt prefix is identical across
   phases by construction, and all T_g generated positions — including the
   last — enter the pooling, exactly as eqs. delta_energy–mean_hellinger
   sum t = 1..T_g. Verified by
   `tests/test_trajectory.py::TestPrefillReference`.

2. **"Calibration restricted to the training split."** The stratified
   70/15/15 split needs labels, which do not exist at calibration time
   (calibration precedes Stage 2/3). The code therefore draws the N_c=30
   calibration questions deterministically at seed 42, records their ids in
   `calibration.json`, and `probe.make_locked_splits` **forces those ids
   into the train split** before stratified-splitting the remainder to the
   overall 70/15/15 ratios. This guarantees by construction that no
   calibration question reaches validation or test; stratification on the
   remaining ~96% of samples is unaffected.

3. **Bonferroni mechanics.** The thesis fixes the family
   {discrimination, contribution} at α=0.05 and reports 95% bootstrap CIs.
   The code reports both the 95% CI (as in the text) and a 97.5% CI whose
   lower bound provides the Bonferroni-corrected accept/reject decision for
   each of the two confirmatory claims.

4. **Score-matmul precision.** The thesis pins float32 for bank
   construction (row norms) and scipy for the log-sum-exp; it does not pin
   the precision of the s = m̂ᵀr matmul on GPU. Default is
   `trajectory.score_dtype: float32` (used by all experiment configs); a
   `bfloat16` override exists solely for the 10 GB MIG slices in
   `hpc/trajectories.sbatch`, where a 3B BF16 model plus float32 banks does
   not fit. All downstream math (softmax, lse, entropy, divergences) always
   runs in float64 on CPU regardless.

Two additions to the prescribed file layout, both forced by reuse across
stages: `energy_llm/models.py` (model loader + SwiGLU alias registry, needed
by Stages 1 and 2 and calibration) and `energy_llm/data.py` (dataset
adapters + sharding + calibration draw, needed by Stage 2, calibration, and
the probe split).
