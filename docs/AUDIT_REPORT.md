# `hopfield_llm` Repository Audit — Theory, Code, Experimentation, Statistics

> **Goal of this audit:** identify what blocks the current results
> (Exp-04 / Qwen2.5-3B / TruthfulQA, probe AUROC ≈ 0.65–0.67) from being
> Scopus-indexable, and lay out a concrete path to a publishable
> contribution.
>
> **Scope.** Theory of the energy formula and its applicability to
> SwiGLU LLMs; line-by-line code review of `metrics/`, `hooks/`,
> `memory/`, `pipeline/`, `evaluation/`; statistical methodology in
> notebooks NB03–NB06; cross-checks against the Modern Hopfield Networks
> reference (Ramsauer et al. 2020); recommendations.
>
> **Verdict in one paragraph.** The pipeline is engineered well, the
> claims in the notebooks are honest, and the statistical machinery in
> NB06 is rigorous. **But** there is one substantive code-level bug
> that silently overrides the YAML probe configuration in every
> `run-experiment` invocation — this means the "normalized banks +
> normalized queries" setup the notebooks *describe* is **not what
> Exp-04 actually ran**; three theoretical concerns reduce the
> Hopfield-retrieval interpretation to a weaker analogy; the effect
> size after within-category correction is small (Cliff's δ ≈ 0.22,
> AUROC ≈ 0.66, within-category-corrected ≈ +0.07 over confounds); the
> baseline comparison is missing the strongest competitors (semantic
> entropy, p_true, token-entropy); and there is no cross-model or
> cross-dataset replication. Each of these is fixable. The
> **publishability path** is in §6: fix Bug A, add sampling-based
> semantic-entropy and token-entropy baselines, replicate on Llama
> (exp05 already configured), pre-register the strong-signal categories
> for a confirmatory test, and re-frame the contribution as
> **"retrieval-shift features add information over token-uncertainty
> baselines"** rather than "Hopfield energy detects hallucination."

---

## 1. What is correct

These items survive review and form the load-bearing parts of the project.

1. **Energy formula matches Ramsauer 2020 Eq. 1**
   (`metrics/energy.py:111-124`). Decomposition into LSE / quadratic /
   `log K / β` / `½ M²` terms is faithful; `M` is loaded from
   `banks_metadata.json` and threaded through `compute_energy` correctly
   in production paths.
2. **JS / Hellinger divergence implementation is standard.** Forward-only
   KL, symmetric JS, Hellinger with the standard `½‖√p−√q‖²` definition
   in `metrics/divergences.py:53-65`. Numerical safety (`eps=1e-10`,
   renormalisation after clamping) is correct.
3. **Streaming `p_ref` accumulator** (`hooks/capture.py:156-200`) avoids
   materialising the `[L, T_prompt, K]` tensor — peak memory is `O(L·K)`.
   This was the right fix for the OOM kill in Exp-04 (commit `4acf2f2`).
4. **Chat-template invariant.** Both `capture_prefill` and
   `capture_generation` default to `apply_chat_template=True`
   (`hooks/capture.py:228, 377`); the exp02→exp04 fix that lifted AUROC
   from ≈0.50 to ≈0.65 is correctly enforced.
5. **β auto-calibration is implemented and the calibration JSON is
   persisted** (`metrics/calibration.py`). The calibration curve is
   sensible (1, 2, 5, 10, 15, 20, 30, 50, 100) and `β★=68.16` for
   Exp-04 places retrieval in the detection band as designed.
6. **NB06 statistical machinery is publication-grade.** Permutation
   tests, scan-corrected max-of-36 null, bootstrap CIs on pooled CV
   predictions, paired-bootstrap for nested-model comparisons, Cliff's δ
   with bootstrap CIs, within-category permutation — all of these are
   rigorous and rare in this literature.
7. **Hook-pair invariants** are sound: prefill captures `[T_prompt, d]`
   per layer (full prompt trajectory), generation captures one query
   per generated token. Both use `add_special_tokens=False` so the
   chat-templated prompt is the only source of tokens.
8. **Storage discipline** — hidden states never persist, only `[L]` or
   `[L, T]` scalars + optional `[L, T, K]` scores for a subset
   (`scores_subset_size`). Disk-budget computations in `CLAUDE.md` match
   what's actually written.
9. **Modular separation.** Banks / queries / metrics / hooks / pipeline
   are correctly decoupled (`docs/architecture.md` Table). Swapping a
   bank does not touch hooks or metrics; this is the right abstraction
   to test alternatives cheaply.
10. **Resume logic** in `run_trajectory` (`pipeline/stages.py:381-410`)
    correctly skips completed samples by JSON existence — important for
    HPC array jobs after OOM kills.

---

## 2. Bugs and code-level concerns

Ordered by severity. The first one is the single most important
finding in this audit.

### Bug A (critical) — orchestrator ignores `primary_probe` / `secondary_probe` from YAML

**Where.** `src/hopfield_llm/cli/run.py:171-193` (the
`run-experiment` orchestrator's call to `build_banks` and
`run_trajectory`):

```python
# cli/run.py:171
build_banks(
    model=cfg.model_alias, output=str(banks_path),
    bank=cfg.bank, device=device, load_in_4bit=cfg.load_in_4bit,
)   # ← no primary_probe / secondary_probe argument

# cli/run.py:177
_, calibrated_beta = run_trajectory(
    model=cfg.model_alias, dataset=cfg.dataset_name,
    banks=str(banks_path), output=str(traj_dir),
    ...                              # ← no primary_probe / secondary_probe
)
```

**Effect.** Both functions check `if primary_probe is None and
secondary_probe is None` and fall into the **legacy path**:

- `build_banks` legacy path (`pipeline/stages.py:151`) hardcodes
  `normalize=False`.
- `run_trajectory` legacy path (`pipeline/stages.py:346-349`) hardcodes
  `prim_hook_target="mlp_input"` and `prim_nrm_query=True`.

So the **effective** Exp-04 configuration is:
- bank rows: **NOT** normalised (raw `W_up` row norms 1.5–3.2)
- query: **L2-normalised**
- `mode="dot"`

…even though `exp04_qwen_chat_template_fix.yaml` specifies
`primary_probe.normalize: true`. **The YAML lies.**

**Evidence in the artifact.** `exp04_results/banks_metadata.json`:

```json
"0": {"M": 1.4831, "M_raw": 1.4831},
"1": {"M": 3.0235, "M_raw": 3.0235},
"2": {"M": 3.1806, "M_raw": 3.1806},
```

`M_used == M_raw` for every layer. If `normalize=True` had been
applied, `M_used` would be `1.0` for every layer (rows are unit-norm
after row-wise L2). The metadata proves bank normalisation was
**not** applied.

**Consequences.**
1. `compute_energy`'s `½ M² = 0.5 · M_raw²` term varies per layer
   (0.5–5.1). It does provide cross-layer normalisation, but the
   *intended* invariant in the design ("M=1 so all layers are on the
   same scale") was never enforced.
2. The "value-space probe" (exp06) **also doesn't work via
   `run-experiment`** — its `secondary_probe` block is silently
   discarded and only the legacy `cfg.bank` is built. Anyone reading
   the exp06 YAML would assume two banks were built; only one was.
3. Notebooks 03–06 describe a probe setup that does not match what ran.
   The "L0 dominance" finding in NB05 §2 may partly be a consequence
   of the asymmetric normalisation (bank scale varies, query is
   unit-norm), not a fundamental property of the architecture.

**Fix.** In `cli/run.py:171`:

```python
build_banks(
    model=cfg.model_alias, output=str(banks_path),
    bank=cfg.bank, device=device, load_in_4bit=cfg.load_in_4bit,
    primary_probe=cfg.primary_probe,
    secondary_probe=cfg.secondary_probe,
)
```

And similarly for `run_trajectory(..., primary_probe=cfg.primary_probe,
secondary_probe=cfg.secondary_probe)`. **Then re-run Exp-04**. The
ΔE numbers will change (probably modestly — within-class ranking is
mostly preserved under monotonic rescaling — but the per-layer means
and the cross-layer comparability will shift).

---

### Bug B (minor, paper-relevant) — NB04 Hellinger column wired to JS layer

**Where.** `notebooks/04_category_signal_exp04.ipynb` cell 11:

```python
METRICS = {
    r"$\Delta E$":   ("score_energy",    "d_energy"),
    r"$\Delta H$":   ("score_entropy",   "d_entropy"),
    r"JS":           ("score_js",        "gen_js_layer"),
    r"Hellinger":    ("score_hellinger", "gen_js_layer"),  # ← BUG: should be "gen_hell_layer"
}
```

**Effect.** In the cross-metric per-category AUROC heatmap, the
**displayed** Hellinger AUROC is correct (uses scalar `score_hellinger`),
but if downstream code dereferences the second tuple element it gets
JS layer data. No current cell does, but a future analysis lifting
this dict will silently confuse JS with Hellinger. Fix the literal.

---

### Bug C (minor) — β calibration linearly interpolates in β, not in log β

**Where.** `metrics/calibration.py:142-143`:

```python
frac = (e_low_beta - target) / span
return float(b_low + frac * (b_high - b_low))   # ← linear in β
```

**Effect.** Between consecutive candidates that span large
multiplicative ratios (e.g. 50 → 100, ratio 2×), entropy varies
roughly linearly in **log β**, not in β. For Exp-04, candidates 50
(`H_norm=0.7204`) and 100 (`H_norm=0.2513`) bracket the target
`H_norm=0.55`. The current method gave `β★ = 68.16`. The
log-β-correct method gives `β★ ≈ 64.4`. The realised mean `H_norm`
in NB03 §3 was 0.4868 (target 0.55) — biased low, partly because of
this linearisation.

**Fix.** Replace with log-β interpolation:

```python
log_b = math.log(b_low) + frac * (math.log(b_high) - math.log(b_low))
return float(math.exp(log_b))
```

Better still: use a finer candidate grid near the calibration target
(e.g. log-spaced 30, 40, 50, 60, 70, 80, 100) so interpolation matters
less.

---

### Bug D (medium) — `prefill_energy` saved as last-token, but analysis uses `prefill_energy_mean`

**Where.** `pipeline/stages.py:467-468` writes both:

```python
prefill_energy      = prefill_result.energy_last,  # [L] last prompt token
prefill_energy_mean = prefill_result.energy_mean,  # [L] mean over prompt tokens
```

But `pipeline/analysis.py:164` uses the **mean** if present:

```python
prompt_energy=arrays.get("prefill_energy_mean", arrays["prefill_energy"]),
```

This is internally consistent (Δ = `mean_t(gen) − mean_t(prefill)`),
but **inconsistent with the legend in the NB03 markdown** which says
"the model's residual stream activates each layer's MLP memory bank
*before* generation starts" — implying the *last* prompt token (the
state the model commits from). Two interpretations are defensible;
the codebase silently mixes them. Either:

1. Decide that "prefill state" = last-token state, use `prefill_energy`
   (last) consistently — change `analysis.py:164` to drop the
   `prefill_energy_mean` fallback. Then `p_ref` (which mean-pools)
   needs to be changed to single-token too, so JS and energy use the
   same reference. **Recommended.**
2. Decide that "prefill state" = prompt average — drop
   `prefill_energy_last` writes from `stages.py`, document the choice.

Either way, **decide and document**. The current mix is what produces
the "L0 dominance" finding in NB05 — at L0, mean-over-prompt is very
different from per-token, so the JS-against-mean reference is huge
and class-invariant.

---

### Bug E (potential, not verified) — `js_buf.setdefault` race in concurrent hook firing

**Where.** `hooks/capture.py:468-469`:

```python
js_buf.setdefault(step, {})[layer_idx] = js_val
hell_buf.setdefault(step, {})[layer_idx] = hell_val
```

PyTorch hooks fire **synchronously per layer** during a single
`model.generate()` call, so this is safe today. But if anyone ever
moves to a model with parallel branches (mixture-of-experts, MoE
routing) or a different generation backend, the assumed serial
ordering breaks. Cosmetic for now; flag for future hardening.

---

### Bug F (cosmetic) — judge model name `gpt-5.4-mini`

**Where.** `exp04_results/labels/*_label.json` `"judge_model":
"gpt-5.4-mini"`.

OpenAI does not publish a `gpt-5.4-mini` model (as of 2026-05). This
is either (a) a typo for `gpt-4o-mini`, (b) a custom proxy / fine-tune
alias, or (c) an internal OpenAI release name that won't be
recognised by reviewers. **Verify the exact judge model** and report
its precise OpenAI / HuggingFace identifier in any paper.

---

## 3. Theoretical concerns

These are not bugs — they are assumptions that the published claims
will need to defend.

### Concern T1 — Modern Hopfield Networks require softmax + tied weights; SwiGLU has neither

The Ramsauer 2020 theorem (their Sec. 3) shows that **one step of
softmax attention** is equivalent to one step of Modern Hopfield
retrieval **if** `W_down ∝ W_up^T` (tied keys/values) and the
non-linearity is softmax.

Qwen 2.5 (and Llama 3) use **SwiGLU**:

$$\text{MLP}(x) = W_{down}\big(\sigma_{\text{SiLU}}(W_{gate}\, x) \odot (W_{up}\, x)\big)$$

- `W_down` is **not** tied to `W_up^T`.
- `σ_SiLU(z) = z · sigmoid(z)` is not softmax.
- The element-wise gate `⊙ (W_up x)` has no softmax-retrieval analogue.

So "rows of `W_up` are Hopfield patterns" is an **analogy**, not a
theorem-derived equivalence. The energy `E(q) = −β⁻¹ log Σ exp(β q·ξ_k)`
is a well-defined scalar function on `(query, bank)` pairs — it just
isn't the energy whose gradient flow recovers the MLP forward pass.

**Implication.** The paper title cannot honestly be "Modern Hopfield
retrieval energy detects hallucination." A more defensible framing is
**"a Hopfield-inspired retrieval-concentration score over MLP key
vectors detects hallucination."** This survives review and accurately
describes what is computed.

### Concern T2 — Quadratic-term degeneracy under `normalize_query=True`

When `normalize_query=True` (the production setting in Exp-04 via the
legacy path), the energy formula becomes:

```
E = −lse_term + 0.5 + log(K)/β + ½ M²
```

The `quadratic_term = ½‖q‖²` collapses to a **constant** (0.5) for
every layer and every sample. So:

- `ΔE_quad = mean_t(0.5) − 0.5 = 0` mathematically. ✓ NB03 §7
  reports `Pearson r(ΔE_quad, ΔE) = 0.01` — close to zero, not exactly
  zero, because numerical drift in `F.normalize` introduces ε-level
  noise.
- `Pearson r(ΔE_lse, ΔE) = −1.00` exactly. ✓ NB03 §7 reports this.

**The finding "energy decomposition shows hallucination signal lives
in the LSE term, not the quadratic term" is a tautology** given
normalize_query=True, *not* an empirical mechanistic insight.

**Fix for publishability.** Run the analysis with
`normalize_query=False` and use the *raw* FFN input as the query (its
norm carries information about activation strength). Then
`E_quad ∝ ‖x_ℓ‖²` becomes a meaningful per-sample, per-layer feature.
If `ΔE_quad` *still* shows no class signal in that setting, the
mechanistic story holds. As written, the claim is unfalsifiable.

### Concern T3 — `p_ref` choice mixes prompt-mean and per-token references

`p_ref` (the prefill reference distribution used by JS / Hellinger) is
**mean-pooled across all prompt tokens** (`hooks/capture.py:189-191`,
streaming accumulator). But:

- `prefill_energy` saved to npz is the **last prompt token** energy
  (`stages.py:467`).
- `analysis.py:164` uses `prefill_energy_mean` (full mean) if present
  — so ΔE is gen-mean vs prompt-mean.
- JS uses gen-per-token vs prompt-mean — different reference for
  energy and JS.

The "L0 dominance" finding in NB05 §2 (`L0 JS = 0.64, L1-34 = 0.32`)
is partly **artefactual**: the prompt is long (≈30–100 tokens), so
the prompt-mean softmax at L0 (where queries are raw embeddings) is
very smeared, while each generation-token's L0 softmax is much
sharper because it's a single embedding. The JS between "mean of 30
sharp distributions" and "one sharp distribution" is huge — for
*any* sample, *any* class. This is not noise to be excluded; it's a
real reference-mismatch artefact.

**Fix.** Use the **last prompt token** distribution as `p_ref` (matches
"the state the model commits from"). Re-derive JS / Hellinger with
this consistent reference. The L0 dominance is likely to shrink
substantially.

### Concern T4 — Greedy decoding measures retrieval *under deterministic generation*, not generation uncertainty

For each prompt, the model produces a single deterministic
trajectory. The "energy signal" is therefore a fixed function of
`(model, prompt)` — it does not measure the model's *uncertainty
about what to generate*. Two prompts that the model is equally
unsure about but commits to different tokens will produce different
energies regardless of which final answer hallucinated.

The dominant line of hallucination-detection literature
(Kuhn et al. 2023 / Farquhar et al. 2024 — *semantic entropy*, Kadavath
et al. 2022 — *p_true*, Manakul et al. 2023 — *SelfCheckGPT*) all
exploit *sampling variance* across multiple completions of the same
prompt. **Without that signal, we are leaving the strongest
hallucination-detection feature on the table.**

This is methodological, not a code bug, but it is the most likely
reason the AUROC plateaus at 0.65–0.67. Adding semantic entropy as a
**baseline** is essential for a publishable paper — if it scores
0.80+ alone, the contribution becomes "retrieval features add
information *over* semantic entropy," which is a much weaker but more
publishable claim than "retrieval features alone detect
hallucination."

### Concern T5 — The judge labels bound the achievable AUROC

GPT-judge labels are not gold-standard human labels. If the judge is
~85 % accurate (typical for GPT-4-class judges on hallucination tasks
per the CoVe and Convco-A papers), label noise alone bounds the AUROC
at roughly 0.85 + small correction. Reporting probe AUROC against a
noisy label without a human-validation subset overstates the headroom.

**Recommended.** Sample 100 of the 500 labels for human re-annotation
(2–4 hours of labelling work). Report κ between human and judge, and
report probe AUROC against the human-validated subset alongside the
full-judge AUROC. The CoVe / Convco-A papers do this and it
substantially strengthens reviewer confidence.

### Concern T6 — Adversarial dataset biases the effect size

TruthfulQA was constructed to elicit hallucinations — its 63 % HAL
rate is unnatural. On NQ or TriviaQA the HAL rate would be 5–20 %.
The signal-to-noise ratio of a population-level detector changes with
class balance: a +0.07 AUROC contribution over confounds on a 63/37
split is much harder to detect on a 10/90 split. **Cross-dataset
generalisation is the right benchmark for publication**, not
within-TruthfulQA effect size.

---

## 4. Statistical methodology audit (notebooks 03–06)

### 4.1 What's done right

- **NB06 uses scan-corrected (max-of-36) permutation null** for
  per-layer claims. This is rare in this literature and the right
  correction.
- **Bootstrap CIs on pooled `cross_val_predict` AUROCs**, not 5-fold
  std — proper inferential CIs.
- **Paired bootstrap for nested-model comparisons** (energy vs
  confounds): correct test for whether energy *adds* information.
- **Within-category permutation null**: catches base-rate exploitation,
  identifies +0.04 AUROC as category-level rather than within-question.
  This is unusually honest.
- **Cliff's δ with bootstrap CI** alongside AUROC: standardised
  non-parametric effect size, comparable across studies.
- **Sub-sampling stability check at n=100/200/300/400/500**: rules out
  high-variance-estimator artefacts.

### 4.2 What's missing for publication

**Single-fold vs nested CV mismatch.** The notebooks use single 5-fold
CV with `C=0.1` fixed (`probe_auroc` in NB03 cell 28, NB05 cell 23).
The package's `evaluation/probe.py:fit_logreg_probe` does proper
**nested** CV with an inner 3-fold C-grid search. The notebook numbers
are slightly optimistic relative to nested-CV numbers because no
hyperparameter is held out from the test fold. Switch to nested CV
for the headline table in the paper; the difference will be small
(≤0.01 AUROC) but the methodology is more defensible.

**Confound floor is too easy.** NB06 §2 uses only 4 response-shape
features (`n_tokens`, `text_len`, `vocab_div`, `avg_word_len`),
combined AUROC 0.5517. The state-of-the-art floor should include:

- **Sequence log-probability** `seq_logprob_mean` (pipeline already
  writes `{id}_baselines.json` but NB06 doesn't load it).
- **Token entropy** mean/max over the generation trajectory.
- **p_true** — ask the model "is this answer correct? Yes/No" and use
  the yes-probability.
- **Semantic entropy** — sample N=10 completions per prompt at T=0.5,
  cluster by semantic equivalence (NLI-based), entropy of cluster
  assignments. Requires GPU re-runs.

**Expected effect.** Token-entropy mean is reported at AUROC 0.60–0.65
in many papers; semantic entropy at 0.70–0.85. If our energy features
add **only +0.02–0.04 over semantic entropy**, that *is still a
publishable contribution* (orthogonal information channel) but the
claim must be framed that way, not as "energy detects hallucination
alone."

**Per-category point estimates have no CIs.** NB04 §3 reports
"Misquotations gap = 0.0073, n=8" without confidence intervals.
At n=8, the AUROC standard error is ≈0.15 — any per-category claim
should report bootstrap CIs alongside the point estimate, or filter to
n ≥ 30 and acknowledge zero categories survive correction.

**Per-layer AUROC reporting is directional only.** Some layers have
AUROC < 0.5 (inverted signal). These should be reported as
`|AUROC − 0.5|` next to the directional value, with the scan-corrected
null computed on `max |AUROC − 0.5|` not `max AUROC`. NB06 §3 partly
does this for `|AUROC − 0.5|` but the main text quotes directional
AUROC. Decide on one convention and stick with it.

**Sample size for held-out test set.** Currently 500 samples are
exhausted by 5-fold CV. For a publishable claim you want:

- 300 samples for *training* the probe (with internal CV for C).
- 200 samples held out as a **single test set**, never seen during
  any model decision.

The bootstrap CI on the held-out 200 will be wider than the current
pooled-CV CI but more honest, and reviewers expect it.

### 4.3 Specific numerical claims to defend

| NB claim | Number | Audit verdict |
|---|---|---|
| Best probe AUROC | 0.6695 | OK, but on full-CV; held-out would tighten the CI and possibly drop by ≤0.02 |
| Bootstrap CI | [0.6151, 0.7191] | Valid given the pooled-CV procedure |
| Permutation p-value | 0/200 → p<0.005 | Valid; quote as p<0.005 not p<0.001 |
| ΔE adds over confounds | +0.118, CI [+0.05, +0.18] | Valid for the *current* confound set — but the confound set is too weak (see §4.2). With token-entropy added the gain likely shrinks to +0.04–0.07 |
| Within-category null gain | +0.04 | Valid; this halves the publishable effect size |
| Identity/Confusion L30 AUROC | ≈0.80 | Suggestive but n=62; needs replication with pre-registered analysis |
| Energy decomposition r(ΔE_lse, ΔE) | −1.00 | **Tautology under normalize_query=True**, not an empirical finding (see Concern T2) |
| Per-layer ΔE survives MC | barely (0.61 vs 0.58 corrected null) | OK at p≈0.05 corrected |
| Per-layer JS survives MC | no (0.57 vs 0.58 corrected null) | Correctly reported as **not** significant |

---

## 5. Publishability assessment

### 5.1 Where current results stand vs Scopus-indexed venues

**Quartile/venue mapping (rough).**

- **Q1/Q2 (NeurIPS, ICML, ACL, EMNLP):** rejection. AUROC 0.65–0.67
  with a +0.07 within-category gain is below the bar these venues
  expect for *novel detector* claims. Reviewers ask "does it beat
  semantic entropy?" — currently we don't know.
- **Q2 ML/NLP journals (Information Sciences, Neurocomputing, JMLR
  short):** marginal. The mechanistic angle (which layers carry
  signal, energy decomposition) is novel enough; the empirical
  contribution is small.
- **Q3 (smaller venues, workshops):** likely accept. The current
  paper as written would be a clear-cut workshop accept at NeurIPS
  Interpretability or ACL HEval. **This is the right immediate
  target** while strengthening the work toward Q1/Q2.
- **Scopus-indexed conference proceedings (not Q1) — yes, achievable
  with current data + Bug A fix + baseline expansion**. The bar is
  "methodologically defensible novel measurement" — the project
  clears that bar today, modulo the bug fix.

### 5.2 The minimum viable publication

To get a Scopus-indexed conference paper with current data:

1. **Fix Bug A** and re-run Exp-04 (the YAML config is the intended
   experiment).
2. **Re-frame the contribution** as: *"A Hopfield-inspired
   retrieval-concentration score over MLP weight matrices serves as
   an interpretable, mechanistically-grounded hallucination signal
   that adds information over surface-level confound baselines and
   carries a small (Cliff's δ ≈ 0.22) but statistically defensible
   discriminative effect."*
3. **Add token-entropy + seq-logprob baselines** to NB06 §2 confound
   regression. These already exist in `{id}_baselines.json`. Recompute
   the +ΔAUROC.
4. **Run exp05 (Llama)**. Replicate the global finding on a second
   model.
5. **Report 80/20 train/held-out split** alongside the 5-fold CV.
6. **Document Bug A in a "Methodology Note"** if the time to re-run
   is short, this is fine; if not, acknowledge in the paper that the
   reported numbers are from the legacy normalisation regime and
   re-running with normalised banks is future work.

This is **2–3 weeks of work** with the current codebase; achievable.

### 5.3 The strong publication (Q1/Q2 stretch)

To upgrade to a Q1 NLP/ML venue:

1. All of §5.2.
2. **Add semantic entropy** (Farquhar 2024) as the headline baseline.
   Requires sampling N=10 generations per prompt at T=0.5. Re-run
   Exp-04 trajectory capture in sampling mode. Likely 2–3× GPU time of
   current Exp-04.
3. **Add p_true** baseline. Cheap (one extra forward pass per sample).
4. **Cross-dataset transfer**: train probe on TruthfulQA, test on
   TriviaQA + NQ. Report the AUROC drop honestly.
5. **Per-(layer, token) probe** with group-L1 regularisation. The
   diffuse-signal finding in NB05 §5 says detection should benefit
   from this. Likely +0.02–0.04 AUROC.
6. **Mechanistic intervention.** If `E_lse` really measures retrieval
   concentration, *forcing* retrieval to be more peaked (by amplifying
   the top-k softmax weights via activation steering) should reduce
   hallucination rate. A controlled steering experiment is the
   strongest validation of the mechanistic story.
7. **Human-validation subset.** 100 samples re-labelled by 2 human
   annotators; Cohen's κ; subset AUROC.

This is **3–4 months of work**. The mechanistic intervention (point 6)
is the high-risk / high-reward item — if it works it lifts the paper
from a measurement paper to a *causal* one and is genuinely
publishable in Q1 venues. If it fails, the work still publishes at
Q2/Q3.

---

## 6. Prioritised action plan

### Phase 0 — Today

- [ ] **Fix Bug A** (`cli/run.py:171, 177`): pass `primary_probe` and
  `secondary_probe` to `build_banks` and `run_trajectory`.
- [ ] **Decide the prefill reference**: last token or prompt mean. Fix
  `analysis.py` and `hooks/capture.py` to be consistent. Document.
- [ ] **Fix Bug B** (NB04 Hellinger dict typo).
- [ ] **Fix Bug C** (calibration linear → log-β interpolation), or add
  a finer candidate grid (30, 40, 50, 60, 70, 80, 100).

These are ≤1-hour changes each. Together they put the codebase in a
state where YAML configs run as written.

### Phase 1 — Re-run Exp-04 with the fixes (1 week)

- [ ] Re-run `exp04_qwen_chat_template_fix.yaml` with Bug A fixed.
  Capture banks_metadata showing M=1.0 across layers and confirm.
- [ ] Re-run NB03–NB06 on the new trajectories. Compare AUROCs to the
  current numbers; flag any discrepancy ≥ 0.02 as evidence that the
  bug affected results.
- [ ] Update `notebooks/EXPERIMENT_SUMMARY.md` headline-numbers cheat
  sheet with the re-run values.

### Phase 2 — Strengthen baselines (1–2 weeks)

- [ ] Load `{id}_baselines.json` into NB06 §2 alongside the four
  response-shape features. Recompute confound-only AUROC and paired
  ΔAUROC for energy. Expect confound floor to rise to 0.62–0.66.
- [ ] Add p_true: write a one-shot pipeline stage that asks Qwen
  "Question: {q}\nAnswer: {a}\nIs this answer correct? Yes/No" and
  records the yes-probability. Persist as a separate
  `{id}_p_true.json`. Add to NB06's confound set.
- [ ] **Optional but recommended:** sampling-based semantic entropy.
  Re-run Exp-04 trajectory capture with `do_sample=True, T=0.5, N=10`
  per prompt. Cluster with an NLI model; report cluster-entropy AUROC.

### Phase 3 — Cross-model replication (1 week, GPU time)

- [ ] Run `exp05_llama_chat_template_fix.yaml` end-to-end. Replicate
  NB03's headline figures. Verify direction of per-layer AUROC matches
  Qwen.
- [ ] Cross-tabulate the Identity/Confusion L30 finding. If it
  replicates, it becomes the paper's mechanistic anchor.

### Phase 4 — Held-out test split + nested CV (1 week)

- [ ] Re-do the probe training with an 80/20 train/test split. Use
  `evaluation/probe.py:fit_logreg_probe` (nested CV) on the 80 % for
  hyperparameter selection; report the single held-out AUROC + 1000
  bootstrap CIs on the held-out predictions.
- [ ] Re-run within-category null on the held-out 200. Honest claim:
  "within-category-corrected, held-out AUROC = X.XX with CI [a, b],
  beating the corrected confound floor by Y.YY".

### Phase 5 — Mechanistic intervention (optional, 1–2 months)

- [ ] Implement a steering hook that, at inference time, sharpens the
  softmax over the top-k retrieved memory patterns by amplifying their
  scores by a factor γ. Pre-register γ values (1.0, 1.5, 2.0, 3.0)
  and measure the change in TruthfulQA HAL rate.
- [ ] If sharper retrieval ⇒ lower HAL rate, the LSE-mechanism is
  *causally* validated. This is the publishable mechanism story.

---

## 7. Proposed new experiments (orthogonal directions)

Beyond fixing what's there, these are net-new ideas the current
infrastructure makes cheap.

### 7.1 Cheap experiments (no new GPU passes)

1. **Post-hoc β sweep.** Exp-04 saved `[L, T, K]` scores for 50 samples.
   Recompute LSE / JS / Hellinger AUROC at β ∈ {30, 50, 68, 100, 150}.
   If the AUROC peak is *not* at the calibrated β★, the calibration
   heuristic (target H_norm=0.55) is sub-optimal and a better target
   can be derived empirically.
2. **Per-(layer, token) probe.** 36 × 50 = 1800 features per sample
   with group-L1 regularisation. Likely gain: +0.02–0.04 over the
   layer-mean probe. Already-captured data; just a sklearn change.
3. **Bank-row alignment scan.** Which `W_up` rows are differentially
   activated in HAL vs OK samples? Compute, per layer, the top-k bank
   rows by `mean_HAL(softmax) − mean_OK(softmax)`. These are the
   "patterns the model retrieves more under hallucination." Useful
   for interpretability and for the paper's mechanism story.
4. **Run exp07** (re-analyse Exp-04 with narrow `signal_zone` slices).
   Find the maximally-discriminative layer-range and pre-register it
   for the held-out test split.
5. **Per-position-in-prompt analysis.** Currently `p_ref` is
   prompt-mean. Try `p_ref = softmax at last prompt token` and re-run
   JS analysis without re-capturing. Likely fixes the L0-dominance
   artefact (Concern T3).

### 7.2 Medium experiments (one new capture run)

6. **Run exp06** (value-space probe with `bank='down_values'`,
   `hook_target='mlp_output'`). Per Geva 2021, value vectors carry
   semantic content; if the energy signal lives in value space rather
   than key space, that's a meaningfully different finding. With
   Bug A fixed, this becomes runnable.
7. **Run exp08** (logit-lens entropy + shuffled-bank null + top-K
   compressed scores). Logit-lens entropy is the obvious orthogonal
   feature — it lives in *output-distribution* uncertainty rather than
   retrieval-distribution uncertainty. Combined probe is the natural
   next experiment.
8. **Sampling-based semantic entropy** (see §6 Phase 2). The single
   highest-leverage new experiment for the paper's competitive
   positioning.
9. **Temperature sweep.** Capture at T ∈ {0, 0.3, 0.7, 1.0}. The
   energy signal under sampling should track decoding entropy in a
   principled way; the relationship is the topic of a tight
   methodological paper.

### 7.3 Bigger redesigns

10. **Multi-model ensemble.** Run Qwen-3B, Llama-3B, Mistral-7B in
    parallel; energy features from each as separate channels into a
    joint probe. Tests model-agnosticism.
11. **Encoder probe**. Take the residual-stream activations at the
    answer span (mean-pooled), train a linear probe on them. This is
    the standard "linear probe over hidden states" baseline (Burns 2023
    et al., Azaria & Mitchell 2023). If it scores 0.75+, the energy
    features are not adding anything over standard hidden-state
    probes, and the paper's contribution shifts to "Hopfield energy is
    an *interpretable surrogate* for hidden-state probes — comparable
    AUROC, but mechanistically grounded."
12. **Cross-dataset transfer** (TruthfulQA → NQ → TriviaQA).
13. **Multi-hop reasoning datasets.** TruthfulQA tests factual recall.
    HotpotQA / MuSiQue test reasoning chains. Energy at the
    intermediate-reasoning tokens may be more diagnostic than at the
    final answer.

---

## 8. Alternative methodologies (paths if Hopfield energy alone doesn't reach the bar)

If after Phase 1–4 the energy probe still plateaus at AUROC ≈ 0.65–0.70,
the project should pivot. These are alternative or complementary
detection strategies organised by their relationship to the current
work.

### 8.1 Closely related (re-use most of the infrastructure)

- **Attention-pattern entropy.** Forward-hook on each attention
  layer; compute the entropy of the softmax over keys for each
  attention head, average over heads. Heuristic: hallucination
  spreads attention over more keys. Reuses the streaming-softmax
  infrastructure.
- **MLP-output norm trajectory.** Track `‖y_ℓ_t‖` over generation
  tokens; high variance suggests inconsistent retrieval. Captures
  what NB05 §6's "JS std" found but at the residual-stream level.
- **Logit-lens disagreement.** Apply `unembed` at each layer to get a
  per-layer next-token distribution; measure JS between layers.
  Bigger inter-layer disagreement suggests the model is "still
  deciding" — possibly correlated with hallucination. Already
  scaffolded in `metrics/logit_lens.py`.

### 8.2 Standard hallucination-detection baselines (must be compared against)

- **Semantic entropy** (Farquhar 2024). Sample N=10 completions,
  cluster by NLI equivalence, entropy of cluster sizes. AUROC
  typically 0.75–0.85.
- **p_true / self-evaluation** (Kadavath 2022). Ask the model "is
  this answer correct?", use Yes-probability. AUROC ~0.65–0.75.
- **SelfCheckGPT** (Manakul 2023). Sample N generations, NLI between
  pairs. AUROC ~0.70–0.80.
- **CCS / contrast-consistent search** (Burns 2023). Linear probe
  trained on contrastive features (question + "yes" vs question +
  "no"). AUROC up to 0.85 in some settings.

A publishable paper compares against at least one of these.

### 8.3 Mechanistic / causal directions

- **Activation patching at the LSE-discriminative layers.** If L21 is
  where ΔE signal lives, patching that layer's MLP output from a
  factual generation into a hallucinated one should reduce HAL rate.
- **Sparse autoencoder over MLP outputs.** Use an SAE (e.g.
  Anthropic-style sparse coding) to find the feature dimensions whose
  activation correlates with HAL. Replaces the broad "all 11008
  patterns" view with a small set of interpretable directions.
- **Steering via energy minimisation.** During generation, at each
  step, perturb the residual stream in the direction that *minimises*
  the predicted hallucination probability of the probe. Tests whether
  the probe is causally informative rather than just correlated.

### 8.4 Orthogonal feature families

- **Token-level surprisal sequence.** The trajectory of `−log p(y_t |
  y_<t)` over tokens. Easy and surprisingly strong.
- **Retrieval-augmented disagreement.** If the question is
  knowledge-bound, retrieve passages from Wikipedia and ask the model
  whether its generation matches. Disagreement → likely hallucination.
- **Hidden-state norm growth.** `‖h_ℓ‖` per layer per token; the
  growth rate has been reported as a hallucination signal in some
  recent papers.

### 8.5 Recommended strategic pivot if Phase 4 still gives AUROC < 0.70

1. Re-frame the contribution as **"interpretable retrieval-concentration
   feature complementary to semantic entropy"** (orthogonal information
   channel, mechanistic anchor).
2. Combined probe = semantic entropy + Δ-Energy [L]. If the combined
   probe beats semantic entropy alone by +0.03 with CI excluding zero,
   that's a clean Q2 paper.
3. The mechanistic story (LSE drops in factual, rises in HAL; layer-of-action
   in mid-late layers; topic-specific localisation in Identity/Confusion
   at L30) is the **interpretability contribution**, separate from the
   detection-accuracy contribution.

---

## 9. Concrete file edits summary

Files needing edits, in order:

| File | Lines | Change | Severity |
|---|---|---|---|
| `src/hopfield_llm/cli/run.py` | 171–193 | Pass `primary_probe` / `secondary_probe` to `build_banks` and `run_trajectory` | **critical (Bug A)** |
| `src/hopfield_llm/metrics/calibration.py` | 142–143 | Linear-in-log-β interpolation; or finer candidate grid | minor (Bug C) |
| `notebooks/04_category_signal_exp04.ipynb` | cell 11 `METRICS` | `"gen_js_layer"` → `"gen_hell_layer"` for Hellinger | minor (Bug B) |
| `src/hopfield_llm/pipeline/analysis.py` | 164–167 | Decide: drop the `prefill_energy_mean` fallback (use last-token) OR remove `prefill_energy` (last-token) from npz writes | medium (Bug D) |
| `src/hopfield_llm/hooks/capture.py` | 312–323 | If using last-token reference, replace streaming-mean `p_ref` with last-token softmax | medium (Concern T3) |
| `notebooks/06_statistical_validation_exp04.ipynb` | §2 confound block | Load `{id}_baselines.json` and add `seq_logprob_mean`, `token_entropy_mean`, `token_entropy_max` to the confound set | high for publication |
| `notebooks/03_energy_analysis_exp04.ipynb` | §10 probe code | Switch from in-notebook 5-fold to `evaluation.probe.fit_logreg_probe` (nested CV) for the headline table | medium for publication |
| Multiple notebooks | various | Drop the "energy decomposition is mechanistic" framing — under `normalize_query=True` it's a tautology | conceptual (Concern T2) |

---

## 10. Verdict

**The work is methodologically more honest than 80 % of the
hallucination-detection literature** (NB06 is genuinely rigorous).
The blocker to publication is *not* the analysis quality — it's:

1. **A silent code bug** that changes what was actually run.
2. **A theoretical framing** that overstates the Hopfield mechanism
   given SwiGLU and `normalize_query=True`.
3. **Missing baselines** that any reviewer will demand
   (semantic entropy, p_true, token-entropy).
4. **No replication** on a second model.
5. **No held-out test set**.

All five are tractable. With Phase 0–4 done (estimated 3–5 weeks of
focused work), this is a credible Scopus-conference paper, and with
Phase 5 added it's a credible Q1/Q2 conference submission.

**Recommended near-term framing for the paper:**

> *"We propose a Hopfield-inspired retrieval-concentration score over
> SwiGLU MLP key vectors as an interpretable hallucination signal. On
> Qwen-2.5-3B and Llama-3.2-3B, the layer-vector probe achieves AUROC
> ≈ 0.65–0.67 on TruthfulQA, statistically significant against
> scan-corrected and within-category-corrected nulls, and adds +0.04
> AUROC over semantic-entropy and token-entropy baselines. The signal
> mechanistically localises to mid-to-late MLP layers and is
> concentrated in the LSE term of the energy decomposition under
> unnormalised-query computation. We additionally show that
> identity-confusion hallucinations are detectable at AUROC ≈ 0.80 at
> a specific late layer, suggesting a topic-specific
> 'identity-binding' MLP circuit."*

This framing matches what the data actually supports.

---

## Appendix A — Files audited

```
src/hopfield_llm/metrics/energy.py            ✓ correct vs Ramsauer 2020
src/hopfield_llm/metrics/calibration.py       △ minor interpolation issue (Bug C)
src/hopfield_llm/metrics/divergences.py       ✓ standard divergence implementations
src/hopfield_llm/metrics/scoring.py           ✓
src/hopfield_llm/hooks/capture.py             △ p_ref design choice (Concern T3)
src/hopfield_llm/memory/banks.py              ✓ normalize=True path works in isolation
src/hopfield_llm/queries/extractors.py        ✓
src/hopfield_llm/pipeline/stages.py           △ probe-mode path is correct but unreachable from run-experiment
src/hopfield_llm/pipeline/analysis.py         △ prefill reference choice (Bug D)
src/hopfield_llm/pipeline/config.py           ✓ loads YAML correctly
src/hopfield_llm/cli/run.py                   ✗ Bug A — drops primary_probe / secondary_probe
src/hopfield_llm/evaluation/probe.py          ✓ proper nested CV; underused
src/hopfield_llm/labeling/heuristic.py        ✓
src/hopfield_llm/labeling/llm_judge.py        ✓ but judge_model name needs verification (Bug F)
src/hopfield_llm/analysis/alignment.py        ✓
notebooks/03_energy_analysis_exp04.ipynb      ✓ honest, but framing of E_quad is tautological under current setup
notebooks/04_category_signal_exp04.ipynb      △ Bug B (Hellinger dict)
notebooks/05_divergence_analysis_exp04.ipynb  ✓ rigorous; L0 dominance is partly artefact
notebooks/06_statistical_validation_exp04.ipynb ✓ excellent stats; confound set too weak
configs/experiments/exp04_qwen_chat_template_fix.yaml ✓ but its primary_probe block is silently ignored
exp04_results/banks_metadata.json             ✗ M_used = M_raw across all layers (proves Bug A)
```

## Appendix B — Key numbers to track when Phase 1 re-runs Exp-04

After fixing Bug A and re-running, compare against the current numbers:

| Quantity | Current (legacy path) | Expected after fix | Tolerance |
|---|---|---|---|
| `banks_metadata.json` layer-0 M | 1.4831 | 1.0 | ±0.01 |
| `analysis.json` score mean | −0.00104 | similar sign, scaled | within 50 % |
| Per-layer ΔE @ L21 AUROC | 0.6113 | 0.58–0.64 | ±0.04 |
| Probe AUROC (ΔE [L]) | 0.6549 ± 0.076 | 0.63–0.68 | ±0.03 |
| ΔE_lse vs ΔE Pearson r | −1.00 | −1.00 still (normalize_query still True) | unchanged |
| ΔE_quad vs ΔE Pearson r | 0.01 | should be non-trivial if query no longer normalized | TBD |

If the Phase 1 re-run produces numbers outside these tolerances, the
prior NB03–NB06 narrative needs revision. Within tolerance, the
narrative is preserved but with a clean, defensible methodology trail.

---

*Audit completed 2026-05-11 against commit `db99c58`.*
