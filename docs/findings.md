# Empirical findings & design rationale (Exp-04)

This file consolidates the development-time audit and pipeline review into a
single record of *what the first campaign showed* and *why the final pipeline is
shaped the way it is*. It supersedes the earlier working notes
`AUDIT_REPORT.md` and `FULL_PIPELINE_REVIEW.md`. The authoritative theory and
method are `methodology.tex` / `fundamentals.tex`; the locked analysis plan is
`preregistration.md`; completed-run provenance is `reproducibility.md`.

## Exp-04 at a glance

- **Setup:** Qwen2.5-3B-Instruct, 500 TruthfulQA samples, seed 42, 4-bit NF4,
  bank=`up` (W_up rows), `energy_mode=dot`, `normalize_query=true`,
  β auto-calibrated to 68.16 (target H̄≈0.55), gpt-4o-mini judge.
  Class balance: 317 hallucinated / 183 correct.
- **Headline:** ΔE layer-vector logistic probe **AUROC ≈ 0.65**
  (bootstrap 95% CI ≈ [0.60, 0.72]); Hellinger probe ≈ 0.656; ΔE+JS ≈ 0.659.
- **Effect size is small:** Cliff's δ max ≈ 0.22; 11/36 layers individually
  clear their bootstrap CI; no TruthfulQA category survives Benjamini-Hochberg.

## What carries signal (and what does not)

| Signal | Verdict | Evidence |
|---|---|---|
| **ΔE (delta energy), per-layer [L]** | **Keep — primary** | AUROC 0.6549; mechanistically clean (lives entirely in the LSE/concentration term: r(ΔE_lse, ΔE) = −1.00, r(ΔE_quad, ΔE) ≈ 0). Hallucination ⇒ broader, less-peaked retrieval. |
| **Hellinger, per-layer [L]** | Keep | AUROC 0.6562, equivalent to ΔE. |
| **JS divergence** | Keep, but use per-layer [L] and **exclude layer 0** | L0 dominates scalar means (mean JS 0.64 vs 0.32 elsewhere) and carries no class signal. Scalar JS AUROC 0.52 (sign cancellation); per-layer JS clears little after scan correction. JS adds <0.01 over ΔE. |
| ΔH̄ (normalised entropy) | Keep as alternate | Best layer ≈ 0.60; mathematically tied to ΔH; redundant with ΔE. |
| energy_slope / energy_var / energy_spread | Low signal | Per-layer effect magnitudes tiny; report as diagnostics only. |
| logit-lens entropy / top-1 prob | Diagnostic | Not computed in Exp-04; kept behind `enable_logit_lens` as an orthogonal output-uncertainty probe. |
| shuffled-bank null control | Diagnostic | Kept behind `enable_null_control`; controls "any bank gives this Δ". |

**Depth:** signal concentrates in mid-to-late MLP layers (peak ≈ L21; identity/
confusion categories peak ≈ L30 with local AUROC ≈ 0.80). Layer-vector probes
always beat scalar reductions because per-layer signs are mixed and mean-pooling
cancels them.

**Why the ~0.65 plateau is fundamental, not a missing feature:** small true
effect sizes on an adversarial 63%-hallucination dataset; ~+0.04 of the apparent
gain is category base-rate exploitation (within-category null ≈ 0.59 vs global
≈ 0.55), leaving a true within-question gain over confounds of ≈ +0.07; greedy
decoding removes the sampling variance that semantic-entropy methods exploit.

## Design decisions this motivated (→ the final config)

1. **`normalize_query=false`** in the final config. Under `normalize_query=true`,
   E_quad ≡ ½ is constant, so "signal lives in LSE not norm" is *algebraically
   guaranteed*, not empirical. Releasing the query norm makes the decomposition
   falsifiable (pre-registered H1/T2 → `exp04b`).
2. **`gated_key` bank** (W_gate ⊙ W_up) as the principled SwiGLU key, vs the
   raw `up` bank (pre-registered H3 → `exp07b`) and the `down_values`
   value-space probe (Geva 2021 → `exp06`).
3. **Explicit bank normalisation (M=1)** so per-layer energies are cross-layer
   comparable (Exp-04 ran un-normalised banks; fixed in probe mode).
4. **Last-token prefill reference** for both energy and divergences (removes the
   L0 reference-mismatch artefact).
5. **Answer-span pooling** (`pool_over_answer=true`) to focus on answer-bearing
   tokens rather than connectors.
6. **Baseline suite + nested-model test:** report ΔAUROC of [Hopfield+baselines]
   over [baselines-only] with a paired bootstrap, so the contribution claim is
   the *added* value over token-uncertainty / semantic-entropy / HalluField /
   P(True), not raw AUROC.

## Open items for the final campaign

- Run the `final_*` matrix (Qwen/Llama × TruthfulQA/TriviaQA) and the
  pre-registered ablations (`exp04b`, `exp05`, `exp06`, `exp07b`).
- Sampling-based per-prompt energy **variance** (N completions at T>0) is the
  most promising untested extension (bridges to semantic-entropy gains).
- INTRA is captured as hidden-state sidecars (`{id}_intra.npz`); fitting its
  PCA+LR probe into the HTML report is still TODO (scalar baselines and the
  nested-model test are already integrated).
- Optional ≥100-sample human label-validation subset to bound the judge ceiling.
