# Pre-registration: Hopfield Energy Hallucination Detection

**Registered:** 2026-05-21  
**Experiment series:** exp04b (Qwen2.5-3B), exp05 (Llama-3.2-3B)  
**Status:** Pre-registered before exp04b and exp05 results are analysed

This document follows open-science pre-registration conventions.
Hypotheses and success criteria are locked before the confirmatory data are seen.

---

## 1. Background and Motivation

Modern Hopfield Networks (MHN) interpret transformer MLP layers as associative
memories.  The retrieval energy `E = -β⁻¹ log Σ exp(β q·ξ_k) + ½‖q‖²` measures
how peaked the softmax over memory patterns is.  We hypothesise that generating
a hallucination shifts the model's retrieval regime in a detectable way.

Exp-04 (pilot, Qwen2.5-3B, 500 TruthfulQA, normalize_query=True) showed a
signal with nested-CV AUROC ~0.65.  However, normalize_query=True forces
E_quad = 0.5 by construction, making the energy decomposition partially
non-empirical.  Exp-04b uses normalize_query=False to make the result
genuinely falsifiable.

---

## 2. Pre-registered Hypotheses

### Primary Hypothesis (H1)

**Claim:** The layer-vector ΔE[L] probe (36-dim, per-layer mean energy shift
gen−prefill) achieves AUROC > 0.60 on the 75-sample held-out test set of
exp04b (Qwen2.5-3B, TruthfulQA, normalize_query=False, gpt-4o-mini labels).

**Direction:** HAL samples have higher (less negative) ΔE than OK samples,
concentrated in layers L18–L30 (the 50–83% depth band).

**Success criterion:** AUROC > 0.60 AND the peak layer is in [L14, L32].

**Failure would mean:** The exp-04 signal was an artefact of the
normalize_query=True / unit-norm-query regime.

---

### Secondary Hypothesis (H2)

**Claim:** The ΔH_norm[L] probe achieves AUROC within ±0.03 of ΔE[L] on the
same held-out test set.  (Entropy and energy are predicted to be redundant.)

---

### Secondary Hypothesis (H3)

**Claim:** The gated-key bank (W_gate ⊙ W_up rows, exp07b) achieves AUROC
within ±0.03 of the plain W_up bank (exp04b), OR higher, never lower by more
than 0.03 (i.e. gating does not hurt).

**Motivation:** The pilot used a half-key approximation (W_up only).  If the
full SwiGLU key (gated product) captures more signal, it supports the
mechanistic interpretation.  If comparable, W_up is a sufficient proxy.

---

### Secondary Hypothesis (H4 — cross-model)

**Claim:** The ΔE[L] probe trained on exp04b (Qwen) achieves AUROC > 0.55
when evaluated on exp05 (Llama-3.2-3B, same dataset, zero-shot transfer).

---

## 3. Analysis Plan

All analyses use the fixed random seed 42 for train/val/test splits.

### 3.1 Primary analysis (H1)

1. Load exp04b trajectories and gpt-4o-mini labels.
2. Apply 70/15/15 stratified split (seed=42).  **Do not look at test labels
   before this document is published.**
3. On the 425 dev samples: run `fit_logreg_probe` (outer 5-fold, inner
   3-fold C-search from {0.01, 0.1, 1.0, 10.0}).
4. Select best feature set by dev-set nested-CV AUROC.
5. Retrain on 350 train + 75 val with best C from step 3.
6. **Evaluate once** on 75 test.  Report AUROC.

### 3.2 Per-layer analysis

Plot AUROC by layer for ΔE.  Claim the peak layer; report whether it falls in
[L14, L32].  No post-hoc selection of layers for the main claim.

### 3.3 Gated-key comparison (H3)

Repeat step 3.1 with exp07b (gated_key bank).  Report Δ AUROC vs exp04b.
Apply McNemar or DeLong test for statistical significance.

### 3.4 Cross-model transfer (H4)

Train final probe on all 500 exp04b samples with C from nested-CV.  Evaluate
directly on 500 exp05 (Llama) samples.  No fine-tuning on Llama data.

---

## 4. Exploratory (Non-Confirmatory) Analyses

The following are explicitly exploratory and will be labelled as such in the paper:

- Per-TruthfulQA-category AUROC breakdown.
- LAM orthogonality (mean angle between top-K activated bank rows).
- Comparison to token-entropy and seq-logprob baselines.
- Value-space probe (exp06).
- Answer-span restriction vs full-generation pooling.

---

## 5. Exclusion Criteria

- Samples where `generated_text` is empty or fewer than 3 tokens → excluded.
- Samples where gpt-4o-mini returned `is_hallucination=None` (parse failure)
  → excluded.
- Layers where the bank extraction failed (logged in banks_metadata.json) →
  imputed with per-column median in the probe.

---

## 6. Multiple Comparison Correction

Primary H1 is a single test.  Secondary H2, H3, H4 will be reported with
Bonferroni correction (α = 0.05 / 4 = 0.0125) where significance is claimed.

Per-layer AUROC plots are exploratory and not subject to Bonferroni correction
(the paper will note this).

---

## 7. What Would Change the Scope Claim

| Outcome | Interpretation |
|---|---|
| H1 AUROC > 0.65 on test | Confirms and extends pilot; mechanistic claim supported |
| H1 AUROC 0.60–0.65 on test | Signal reproduced with normalize_query=False; scope unchanged |
| H1 AUROC < 0.60 on test | Signal may have been unit-norm artefact; downgrade paper claim to "pilot observation" |
| H3: gated_key > W_up by >0.03 | Upgrade from "half-key" to "full SwiGLU key"; rename main probe |
| H4 cross-model fails | Remove cross-model generality claim; retitle to single-model study |

---

*This document was written and committed before exp04b GPU results were analysed.
Any deviation from this plan during analysis must be documented in `docs/analysis_deviations.md`.*
