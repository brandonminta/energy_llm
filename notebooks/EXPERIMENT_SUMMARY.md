# Energy-Based Hallucination Detection — Full Experimentation Summary

> Companion document for notebooks `01_dataset_analysis`, `02_category_layer_analysis`,
> `02_report`, `03_energy_analysis_exp04`, `04_category_signal_exp04`,
> `05_divergence_analysis_exp04`, `06_statistical_validation_exp04`.
>
> Scope: complete methodology, what the codebase does, deep dive on
> Experiment 04 (Qwen2.5-3B / TruthfulQA, 500 samples), figure-by-figure
> walkthrough of every notebook output, discussion of what hallucination
> detection actually yields with these signals, and proposed next
> methodological directions.

---

## 1. One-paragraph elevator pitch

Treat each transformer MLP block as a **Modern Hopfield memory bank** whose
patterns are the rows of `W_up` (or `W_down.T`). For every layer `ℓ` we
compute (i) the **retrieval energy** `E_ℓ(q) = −β⁻¹ log Σ_k exp(β q·ξ_k) +
½‖q‖² + β⁻¹ log K`, (ii) the **retrieval entropy** of the softmax over the
`K` patterns, and (iii) the **JS / Hellinger divergence** between the
generation-time and prefill-time retrieval distributions. Each sample
becomes a `[L, T]` energy trajectory plus `[L, T]` divergence trajectory
plus prefill baselines. We then test whether `Δ = mean_gen − prefill` — the
per-layer shift induced by autoregressive generation — discriminates
hallucinated from factual answers. Across 500 TruthfulQA samples on
Qwen2.5-3B (Exp-04), the answer is *yes, but barely*: a 36-dim per-layer
linear probe reaches **AUROC ≈ 0.65–0.67** with bootstrap 95 % CI excluding
0.5 and excluding the response-shape confound floor (~0.55), while no
single layer or scalar reduction beats 0.61. The signal is small,
topic-heterogeneous, partly explained by category base rates, and
mechanistically attributable to the **LSE term** (softmax peakedness over
memory patterns), not the quadratic norm term.

---

## 2. Codebase methodology — what `hopfield_llm` does

### 2.1 Pipeline stages

```
[1] build-banks    GPU  →  banks.pt + banks_metadata.json   (W_up rows per layer)
[2] run-trajectory GPU  →  {id}.npz  {id}.json              (per-sample E, H, JS, …)
[3] analyze        CPU  →  analysis.json                    (per-layer summaries)
[4] visualize      CPU  →  plots/*.png                      (baseline plots)
[5] label          CPU  →  {id}_label.json + calibration.json (GPT judge + F1)
[6] evaluate       CPU  →  report.html                      (AUROC, probe, baselines)
```

Stages 1–4 are orchestrated by `run-experiment` from a YAML config that
captures **everything** (model, dataset, β, banks, probe target, sharding,
output dir). Stage 5–6 are opt-in because they consume an API budget.

### 2.2 The energy decomposition

```
E_ℓ(q) =  −β⁻¹ log Σ_k exp(β q·ξ_k)    ← E_lse  (retrieval concentration)
        + ½‖q‖²                          ← E_quad (query norm)
        + β⁻¹ log K − ½ M_ℓ²             ← cross-layer normalisation
```

`M_ℓ = max_k ‖ξ_k‖` is the per-layer maximum row-norm of the bank. The
`−½ M_ℓ²` term makes energies comparable across layers (without it,
deeper layers with bigger row norms produce systematically lower LSE). It
is passed from `banks_metadata.json` into `compute_energy` and is critical
for every per-layer plot in the notebooks to share a single y-axis scale.

### 2.3 β calibration

The pipeline picks β automatically so that mean prefill normalised entropy
`H̄_norm = H̄ / log K` lands at a target fraction (default 0.55) of its
maximum. Why: at β too low (≤10) the softmax over K=11008 patterns is
nearly uniform — no retrieval structure, no signal. At β too high (≥100)
it collapses onto the argmax — retrieval becomes a hard 1-of-K lookup,
also signal-free. Between these regimes lies the **detection band**
(`H_norm ≈ 0.55`) where per-pattern weights span 3–4 orders of magnitude
and the softmax derivative `∂p/∂q` is large enough for small query shifts
to produce visible distribution changes. **Exp-04 calibrated β★ = 68.16
on a 30-sample warm-up** — the YAML seed value `β = 15.0` was overridden
(`H_norm ≈ 0.96` at β=15 would have placed retrieval in the diffuse
regime).

### 2.4 What is captured per sample

| Artifact | Shape | Always saved? | Purpose |
|---|---|---|---|
| `prefill_energy`, `prefill_entropy`, `prefill_norm_entropy`, `prefill_lse`, `prefill_top_activation`, `prefill_quadratic` | `[L]` | yes | Per-layer scalar from one forward pass on the prompt |
| `gen_energy`, `gen_entropy`, `gen_norm_entropy`, `gen_lse`, `gen_top_activation`, `gen_quadratic` | `[L, T]` | yes | Per-(layer, token) scalar over the generated tokens |
| `gen_js`, `gen_hellinger` | `[L, T]` | yes (in Exp-04) | Per-(layer, token) divergence between gen softmax and prefill softmax over the K patterns. Requires `save_distributions=True` |
| `{id}_scores.npz` | `[L, T, K]` | first `scores_subset_size` samples only | Raw softmax-input scores — enables post-hoc β sweep without re-running the forward pass |
| `token_ids`, `generation_config` | — | yes | Bookkeeping for re-tokenisation and reproducibility |

Hidden states are **never persisted** — they live only in GPU memory during
the forward pass. The pipeline computes the layer scalars on-the-fly via
forward hooks (`capture_prefill` and `capture_generation`), both of which
apply the chat template by default to guarantee `Δ = gen − prefill` is
comparable per layer.

### 2.5 Labels

The label stage stamps each sample with:
- a **F1-based heuristic** comparing the generated answer to the gold answers
- a **GPT-5.4-mini judge** call (binary hallucinated / factual)
- Cohen's κ between heuristic and judge is written to `calibration.json`

In Exp-04 the judge produced **317 HAL / 183 OK** (63.4 % hallucinated) —
TruthfulQA is adversarial, so this class imbalance is expected.

---

## 3. The experimentation arc

| Exp | What changed | Status |
|---|---|---|
| `exp01_truthfulqa_baseline` | First run, β=15, no chat-template applied to prefill | Identified the chat-template inconsistency bug |
| `exp02_truthfulqa_500` | 500 samples on Qwen2.5-3B, β=15 | Same prefill bug; energy signal looked diluted |
| `exp03_truthfulqa_500` | 500 samples on Llama-3.2-3B, β=15 | Same prefill bug — Llama counterpart of exp02 |
| **`exp04_qwen_chat_template_fix`** | **Both** prefill and generation observe the chat-templated prompt; β auto-calibrated to 68.16 | **The main result. All NB03–06 analysis is on Exp-04** |
| `exp05_llama_chat_template_fix` | Same fix on Llama-3.2-3B | Two-model contrast; replication test |
| `exp06_qwen_value_space` | Adds a paired value-space probe (`W_down.T` rows) alongside the key-space probe (`W_up` rows) | Tests Geva et al. value-vector hypothesis |
| `exp07_qwen_layer_scan` | Re-analyses Exp-04 trajectories with narrow `signal_zone` layer slices | Zero new GPU work |
| `exp08_qwen_full_metrics` | Adds logit-lens entropy, shuffled-bank null, top-K compressed scores | Audits whether new metrics catch signal Δ-energy misses |

The chat-template fix between exp02→exp04 is the **single most important
methodological change** in the whole project: prefill and generation must
ingest the same prompt for `Δ = gen − prefill` to be meaningful. Without
it, the prefill state encodes a raw question while generation encodes a
chat-formatted one, and the energy difference picks up template tokens
rather than hallucination.

---

## 4. Experiment 04 — full spec

### 4.1 Run config

| Field | Value | Why |
|---|---|---|
| Model | `qwen25_3b` (Qwen 2.5-3B-Instruct), 4-bit | Local + HPC feasible, instruct-tuned for greedy decoding |
| Dataset | TruthfulQA, 500 samples, seed 42 | Adversarial; designed to elicit hallucination |
| Bank | `up` (rows of `W_up`) — key space | Geva-style memory keys |
| β | auto-calibrated to **68.16** (target 0.55·log K) | Forces retrieval into the detection band |
| Decoding | greedy (`do_sample=False`), max 50 new tokens | Deterministic; per-sample energy is a function of input only |
| Hook target | `mlp_input` | Captures query into the MLP before W_up projection |
| Scores subset | 50 (`save_distributions=True`) | Stores `[L, T, K]` softmax scores for post-hoc β sweep |
| Labels | GPT-5.4-mini binary judge + F1 heuristic | Cohen's κ ≈ moderate agreement (see `calibration_beta.json` for the β curve, not κ) |
| Output | `exp04_results/` (analysis, plots, labels, trajectories) | Self-contained run dir |

### 4.2 What is computed, compared, and why

For each of the 500 samples:

1. **Prefill pass** — chat-templated prompt forward through the model; at
   every layer compute the scalar `(E, E_lse, E_quad, H, H_norm,
   top_activation)`. This is the baseline.
2. **Generation pass** — greedy decoding up to 50 tokens. For each generated
   token `t` and layer `ℓ`, store `(E_{ℓ,t}, H_{ℓ,t}, …)` and the
   divergence `JS(p^gen_{ℓ,t} ‖ p^pre_ℓ)`, `Hellinger(p^gen_{ℓ,t}, p^pre_ℓ)`.
3. **Per-sample delta** — `ΔX_ℓ = mean_t(X^gen_{ℓ,t}) − X^pre_ℓ`. This
   subtracts the prompt-level baseline (which differs per sample because
   each question has its own prefill state) and isolates the
   generation-induced shift (which is what we want to attribute to
   hallucination).
4. **Per-sample scalar** — `score = mean_ℓ(ΔX_ℓ)`. Used by `analyze` to
   produce population-level statistics in `analysis.json`. The scalar form
   is the *worst* possible reduction because it mean-pools across layers
   that carry signed-opposite signals (see §5.2 below), but it is the
   simplest detector to report.
5. **Per-sample vector** — the 36-dim per-layer feature `[ΔX_0, …, ΔX_35]`
   fed to a logistic-regression probe (StandardScaler + L2 with `C=0.1`,
   5-fold stratified CV).

### 4.3 What's in `exp04_results/`

```
config.yaml             ← snapshot of the run config
git_info.json           ← commit hash + branch + dirty flag
banks_metadata.json     ← {layer: K, D, M, ...} — feeds m_per_layer into compute_energy
trajectories/           ← 500 × {id}.json, {id}.npz, 50 × {id}_scores.npz, calibration_beta.json
labels/                 ← 500 × {id}_label.json
analysis.json           ← per-layer mean/std for each metric + score_by_category
plots/                  ← 20 layer-profile PNGs from `visualize`
log.txt                 ← stage timings
```

`analysis.json` headline numbers (read from disk):

- 500 samples scored, 36 layers
- Score metric: `delta_energy`, aggregation `mean`
- Score summary: mean −0.00104, std 0.00239, range [−0.0082, +0.0060]
- Global divergences: `energy_shift_l1` mean 0.37, `energy_shift_l2` mean
  0.12, `peak_delta_layer` mean 32.2 (signal concentrates near the network
  top)
- Per-layer `delta_energy` mean: argmax `|·|` at layer 35 (value −0.079)
- Per-layer `delta_entropy` mean: argmax `|·|` at layer 15 (value −2.18) —
  consistent with mid-network being where retrieval concentration shifts

---

## 5. Notebook-by-notebook walkthrough

### 5.1 Notebook 03 — Hopfield Energy Analysis

**Question.** Does `ΔE_ℓ` discriminate hallucinated from factual answers?

**Cells & figures:**

- **§1 Load (cell 3)** — 500 samples / 317 HAL / 183 OK loaded by
  joining `trajectories/*.json`, `*.npz`, `labels/*_label.json`.
- **§2 Dataset overview (cell 5, fig 1)** — three-panel sanity check:
  class balance, generation length distribution (HAL 49.0 ± 5.3 tokens,
  OK 49.3 ± 3.8 — essentially identical, ruling out a trivial-length
  confound at this point), category overview. *Implication:* hallucination
  doesn't change response length materially, so the energy signal can't
  be a hidden token-count proxy.
- **§3 β calibration (cell 7, fig 2)** — plots the calibration curve
  H_norm(β) computed on the 30-sample warm-up, marks β=15 (the YAML seed,
  *not* used) and β★=68.16 (the calibrated value, *actually* used).
  Realised H_norm over the full 500 samples = 0.4868, close to the 0.55
  target. **Establishes that retrieval is in the detection regime** —
  prerequisite for everything downstream.
- **§4 Prefill profile (cell 10, fig 3)** — prefill `E`, `E_gen mean`,
  `H_norm` per layer overlaid for HAL vs OK. *The HAL and OK prefill
  curves are visually identical.* This is the **methodology sanity check**
  — if labels somehow leaked into the prefill state, we'd see separation
  here. We don't, so any discriminative signal must come from
  `Δ = gen − prefill`.
- **§5 Delta metrics (cell 13, fig 4)** — 2×3 grid of `ΔE, ΔH, ΔH_norm,
  ΔE_lse, Δtop_act, ΔE_quad` per layer with ±1 std bands. *HAL bands sit
  slightly above OK in mid-late layers (~L15–30) for energy and entropy;
  Δtop_act is mechanically inverted vs ΔE_lse (top-activation grows as
  retrieval becomes more peaked).* The error bands overlap heavily — the
  signal is small per-sample.
- **§6 Per-layer AUROC (cell 16, fig 5)** — AUROC of each Δ metric at
  each layer. **Best per-layer AUROC: ΔE at L21 = 0.6113, ΔH at L9 =
  0.6027, ΔE_lse at L6 = 0.6039.** No layer crosses 0.62 for any metric;
  the scalar (mean-of-layers) AUROC sits at 0.48–0.53 — *worse* than
  per-layer best, because mean-pooling averages contradictory sign
  signals. **Cross-reference NB06 §3:** the scan-corrected (max-of-36)
  null 95th percentile is 0.58, so ΔE@L21 clears the corrected bar by
  0.03 but JS does not.
- **§7 Energy decomposition (cell 19, fig 6)** — splits `E = E_lse +
  E_quad`. *Pearson r(ΔE_lse, ΔE) = −1.00; r(ΔE_quad, ΔE) = 0.011.* The
  HAL/OK gap lives entirely in the **LSE term** (softmax peakedness over
  memory patterns); the **quadratic-norm term carries no class signal**.
  Mechanistically: hallucinated generation produces a *less-peaked*
  retrieval distribution, not a smaller / bigger query norm. This is the
  publishable mechanistic claim.
- **§8 Hallucination fingerprint (cell 22, fig 7)** — 500-row × 36-col
  heatmap of ΔE per sample × layer, sorted OK-then-HAL with a yellow
  separator. The visual shows *no* clear horizontal band split — both
  classes contain mixtures of red and blue rows. **Most variance in ΔE is
  between samples, not between classes.** Detection is therefore a
  "weight of evidence across many layers" problem, not a single-layer
  threshold.
- **§9 Token-level dynamics (cell 25, fig 8)** — energy and JS as a
  function of token position at probe layers `[0, 8, 16, 24, 32, 35]`.
  *Energy is roughly flat over t once decoding starts; the HAL/OK gap is
  established at t=1 and persists.* JS at gen[t]-vs-prefill is highest
  early then decays. **No temporal structure to exploit** — a
  single-token-position probe could plausibly match the trajectory
  probe.
- **§10 Logistic probe ablation (cells 28–29, fig 9)** — 5-fold CV AUROC
  of probes on different feature sets:

  | Feature set | AUROC | ± std |
  |---|---|---|
  | Δ Energy [L]        | 0.6545 | 0.0755 |
  | Δ Entropy [L]       | 0.6674 | 0.0839 |
  | Δ NormEntropy [L]   | 0.6674 | 0.0839 |
  | JS [L]              | 0.6436 | 0.0601 |
  | Hellinger [L]       | 0.6593 | 0.0688 |
  | All Δ metrics [5×L] | 0.6486 | 0.0723 |
  | All metrics + JS/Hell [7×L] | 0.6417 | — |

  *Combining feature families adds nothing* — they are different views of
  the same underlying retrieval-shift phenomenon. No probe approaches
  0.70 ("useful detector" threshold). The signal is recoverable only by
  the **layer-vector probe**, not by any scalar or any single layer.

**Notebook 03 conclusions:**
1. β★=68.16 places retrieval in the detection band (H_norm ≈ 0.55).
2. The HAL/OK shift is real, statistically detectable, and lives in the
   LSE term (concentration), not the quadratic term (norm).
3. Per-sample signal is small; per-layer AUROC ceiling ≈ 0.61.
4. Linear-pooled across 36 layers, AUROC ≈ 0.65 — "signal exists" not
   "deployable detector."
5. JS and ΔE are largely redundant.
6. No temporal structure — signal established by the first generated
   token.

---

### 5.2 Notebook 04 — Category Signal

**Question.** Is the signal topic-universal or topic-specific?

**Cells & figures:**

- **§1–2 Load + category overview (cell 3, fig 1 in cell 5).** TruthfulQA's
  38 categories are imbalanced (n: 5 to 58) and have wildly different
  hallucination rates (22 % Politics → 94 % Confusion: People). The 6
  macro-groups (Factual/Historical n=130, Social/Institutional n=129,
  Science/Health n=72, Identity/Confusion n=62, Belief/Conspiracy n=60,
  Language/Logic n=38, Other n=9) recover statistical power.
- **§3 Per-category gap (cell 8, fig 2).** `ΔE^HAL − ΔE^OK` per category
  with Mann–Whitney-U p-values and BH (FDR) correction across 33 testable
  categories.
  - **Top positive (HAL > OK) gaps:** Misquotations (+0.0073, n=8),
    Confusion: People (+0.0048, n=17), Indexical Error: Time (+0.0044,
    n=6), Paranormal (+0.0028, n=15), Nutrition (+0.0026, n=8).
  - **Top inverted (OK > HAL) gaps:** Logical Falsehood (−0.0049, n=8),
    Conspiracies (−0.0047, n=13), History (−0.0038, n=13), Politics
    (−0.0032, n=9), Sociology (−0.0021, n=33).
  - **Zero categories survive BH at p<0.05.** The test is *underpowered*
    given 30+ categories and 5–58 samples each.
- **§4 Cross-metric AUROC heatmap (cell 11, fig 3).** Per category, AUROC
  of `score_energy`, `score_entropy`, `score_js`, `score_hellinger`.
  - High-AUROC categories show **metric agreement** (Economics 0.73,
    Paranormal 0.70, Confusion: People 0.69 — multiple metrics catch
    them).
  - Mid-AUROC categories show **metric divergence**: Language (mean 0.625)
    has ΔE/ΔH ≤ 0.43 (inverted) but JS/Hellinger 0.78–0.92 (strongly
    positive). *Language-type hallucinations shift the retrieval
    distribution shape but not its concentration.*
  - Inverted-AUROC categories (Politics 0.18, History 0.28, Religion
    0.28, Logical Falsehood 0.25) are inverted across **all** metrics —
    the inversion is a real topology property of the retrieval signal in
    those topics.
  - **Bug note for paper-grade fix:** `METRICS["Hellinger"]` maps the
    scalar to `score_hellinger` but the layer field to `gen_js_layer` —
    cosmetic in the displayed AUROC but should be fixed.
- **§5 Macro-group layer profiles (cell 14, fig 4).** ΔE per layer × HAL
  vs OK, panel per macro-group:
  - Factual/Historical and Science/Health show the predicted HAL > OK in
    mid-late layers (~L15–30).
  - Identity/Confusion shows the **strongest visual separation, late
    layers (L25+)** — consistent with identity/entity binding in deep
    layers.
  - Belief/Conspiracy shows roughly inverted profile (OK > HAL in early
    layers).
  - Language/Logic ΔE is flat — for that group, JS not ΔE is the right
    view.
- **§6 Per-layer AUROC by macro-group (cell 17, fig 5).**
  - **Identity/Confusion peaks AUROC ≈ 0.80 around L30** — the strongest
    single-layer / single-feature finding in the entire experiment,
    though only on 62 samples.
  - Other macros peak in 0.60–0.65 range, mostly L20–L30.
  - Belief/Conspiracy and parts of Language/Logic dip below 0.5 in mid
    layers (the inversion pattern is layer-specific too).
- **§7 Signal-consistency scatter (cell 20, fig 6).** Gap and AUROC vs
  HAL-rate per category.
  - Pearson r(gap, HAL rate) = +0.386 (p=0.027), Spearman 0.258
    (p=0.147). Small / borderline correlation.
  - AUROC vs HAL-rate is U-shaped — categories near 50 % HAL show modest
    AUROC; extreme-rate categories produce high-variance AUROC because
    of class imbalance (Confusion: People at 94 % has only 1 OK sample —
    the "AUROC 1.0" is computed from 1×16 pairs and is essentially
    uninformative).
- **§8 Category ranking (cells 23–24, table).** Categories ranked by mean
  AUROC across 4 metrics. High-signal (>0.60): Economics, Paranormal,
  Confusion: People, Nutrition, Misquotations, Language, Proverbs.
  Low/inverted (<0.45): Misconceptions, Stereotypes, Religion, Politics,
  Superstitions, Logical Falsehood, Education, History.

**Notebook 04 conclusions:**
1. The global signal is a **population average over heterogeneous
   per-category effects.** Some categories drive most of the average
   (Economics, Identity/Confusion); others invert it (Politics, Religion,
   History, Logical Falsehood, Conspiracies).
2. **Identity/Confusion @ L30 AUROC ≈ 0.80** is the strongest local
   finding; mechanistically suggestive of identity-binding circuits.
3. **Zero categories survive BH correction** — confirmatory per-category
   claims would need pre-registration + a much larger dataset.
4. Some of the global AUROC is **base-rate exploitation** (some
   categories are intrinsically more hallucination-prone). NB06 §8
   quantifies this at ~+0.04 AUROC.

---

### 5.3 Notebook 05 — Divergence Analysis

**Question.** What does JS/Hellinger between gen and prefill retrieval
distributions add over the energy view?

**Cells & figures:**

- **§2 L0 dominance (cell 5, fig 1).** Mean JS at layer 0 = 0.6417 (same
  for HAL and OK). Mean JS at L1–34 = 0.3193. **L0 carries no class
  signal and dominates any all-layer mean.** Cause: L0's prefill query is
  a mean over prompt-token embeddings; L0's generation queries are
  single-token-context embeddings — these objects differ in norm and
  direction independent of hallucination, so their softmaxes diverge a
  lot. **Methodological lesson:** always exclude L0 from scalar layer
  averages, or use a regularised probe that can zero it out.
- **§3 Why scalar JS fails (cell 8, fig 2).** Per-layer JS gap and
  per-layer JS AUROC:
  - **10 layers** carry HAL > OK signal; **13 layers** carry OK > HAL
    signal.
  - Strongest positive: L6 AUROC 0.5714. Strongest inverted: L9 AUROC
    0.3875 (equivalently 0.6125 with sign flip).
  - **Mean-pooling across layers cancels the signed signals**, giving JS
    scalar AUROC ≈ 0.52 — barely above chance.
- **§4 Class-conditional KDE (cell 11, fig 3).** HAL vs OK distributions
  at the most-discriminative layers (L6, L9, L19, L26, L34). Curves
  overlap heavily; the gap is a **small mean shift, not a shape change.**
  No bimodality, no "hallucination cluster."
- **§5 Layer × Token heatmaps (cell 14, fig 4).** Three heatmaps: mean
  JS[L, T] for HAL, for OK, and the difference. Peak |HAL − OK|
  difference at (L=4, T=1), magnitude ~0.046. Discriminative information
  spread diffusely across many (L, T) cells, not concentrated.
- **§6 Temporal dynamics at probe layers (cell 17, fig 5).**
  - JS over token position: HAL stays consistently above OK at L6, gap
    set early and maintained.
  - JS std (within-sample temporal variability) over layers: HAL has
    slightly wider spread, **JS std [L] probe AUROC ≈ 0.6490 —
    independent of JS mean.** HAL trajectories are noisier.
- **§7 Sample-level fingerprints (cell 20, fig 6).** Highest-JS samples
  per class shown as `[L, T]` heatmaps. Visually indistinguishable
  between HAL and OK. **No per-sample mechanistic story** — the signal
  is statistical, not interpretable per generation.
- **§8 Probe comparison (cell 23, fig 7).** 5-fold CV AUROC by feature
  set:

  | Feature set | AUROC | ± std |
  |---|---|---|
  | JS scalar (mean all layers) | 0.5194 | 0.0290 |
  | JS mean [L]                  | 0.6436 | 0.0601 |
  | Hellinger mean [L]           | 0.6593 | 0.0688 |
  | JS max [L]                   | 0.5652 | 0.0676 |
  | JS std [L]                   | 0.6490 | 0.0541 |
  | JS slope [L]                 | 0.6302 | 0.0800 |
  | **JS 4-stat [4×L] (mean+max+std+slope)** | **0.6664** | **0.0494** |
  | ΔE [L]                       | 0.6545 | 0.0755 |
  | JS-mean + ΔE [2×L]           | 0.6618 | 0.0806 |
  | JS-4stat + ΔE [5×L]          | 0.6645 | 0.0513 |

  **Layer-vector always beats scalar** (+0.12 from L-vectorising JS).
  **Temporal moments per layer add value** (best feature family).
  **ΔE + JS combined ≈ ΔE alone** — redundancy.
- **§9 JS vs ΔE per-layer correlation (cell 27, fig 8).** Pearson r per
  layer mostly |r|<0.3, peaks near 0.5 at a few layers. Combined probe
  (JS+ΔE) gains only +0.007 over ΔE alone, +0.018 over JS alone — within
  noise. **JS and ΔE are partially redundant; one signal viewed two
  ways.**
- **§10 Per-category JS vs ΔE (cell 30, fig 9).**
  - Above-diagonal (JS >> ΔE): Language, Distraction — distribution
    shifts without concentration shifts.
  - Below-diagonal (ΔE >> JS): Confusion: People, Misquotations —
    concentration shifts without distribution shape changes.
  - On-diagonal: most categories.

**Notebook 05 conclusions:**
1. L0 always dominates scalar layer-means and carries no class signal —
   exclude or down-weight.
2. Per-layer JS has mixed sign across layers; the L-vector probe is
   essential.
3. **JS 4-stat [4×36] = 0.6664 AUROC** is the strongest single-family
   feature set.
4. JS and ΔE are partially redundant — combining them doesn't beat ΔE
   alone.
5. JS-std as a feature is independently informative — **temporal
   stability of divergence is a real signal**, not just mean level.
6. Topic-conditional feature selection is suggestive but underpowered
   here.

---

### 5.4 Notebook 06 — Statistical Validation

**Question.** Are the AUROCs from NB03–05 statistically defensible?

**Cells & figures:**

- **§2 Confound floor (cell 5).** AUROC of trivial response-shape features:
  `n_tokens` 0.504, `text_len` 0.523, `vocab_div` 0.533, `avg_word_len`
  0.539. **Combined 4-feature probe: AUROC 0.5517 ± 0.0372.** *This is
  the floor any energy claim must beat — not 0.5.*
- **§3 Permutation test (cells 8–9, fig 1).** 1000 label shuffles per
  feature; per-layer null bands and the **max-of-36 scan-corrected null**.
  - ΔE: scan-corrected 95 % null = **0.5800**; observed max 0.6113 (L21) →
    **clears the corrected bar by 0.03**. *p ≈ 0.05 after correction.*
  - JS: scan-corrected 95 % null = **0.5801**; observed max 0.5714 (L6) →
    **does NOT clear**. Per-layer JS claims are not defensible after MC
    correction.
  - All robust claims must therefore come from the L-vector probe, not
    from single layers.
- **§4 Bootstrap CIs + permutation null on the probe (cells 12–14, figs
  2–3).** 5-fold CV with `cross_val_predict` (pooled held-out
  probabilities), then 1000-bootstrap on (y, p̂) pairs:

  | Probe | AUROC | 95 % CI |
  |---|---|---|
  | Confounds only           | 0.5492 | [0.4922, 0.5998] (straddles 0.5) |
  | ΔE [L]                   | 0.6549 | [0.6039, 0.7038] |
  | JS mean [L]              | 0.6424 | [0.5909, 0.6887] |
  | Hellinger mean [L]       | 0.6562 | [0.6055, 0.7029] |
  | ΔE + JS [2·L]            | 0.6594 | [0.6127, 0.7074] |
  | **ΔE + JS + Confounds**  | **0.6695** | **[0.6151, 0.7191]** |

  Permutation null on strongest probe: **0/200 permutations exceed
  observed**, so `p < 0.005`.
- **§5 Cliff's δ effect sizes (cell 17, fig 4).**
  - Max ΔE |δ| = 0.223 → AUROC 0.611 — a **SMALL** effect on Romano's
    scale (negligible <0.147, small <0.33, medium <0.474, large ≥0.474).
  - **11/36 ΔE layers have CIs excluding 0**; **8/36 JS layers**. Versus
    1.8 expected under α=0.05 noise → real distributed effect.
- **§6 Confound regression (cell 20).** Paired-bootstrap nested-probe
  comparison:

  | Comparison | ΔAUROC | 95 % CI | Significant? |
  |---|---|---|---|
  | Conf + ΔE vs Conf only | **+0.1181** | [+0.0509, +0.1865] | **yes** |
  | Conf + ΔE + JS vs Conf + ΔE | +0.0022 | [−0.0229, +0.0267] | no |
  | ΔE alone vs Conf only | +0.1057 | [+0.0272, +0.1754] | yes |

  **Energy adds +0.118 AUROC over response-shape confounds — robust.**
  **JS adds essentially nothing over ΔE + confounds.**
- **§7 Sample-size sensitivity (cell 23, fig 5).** AUROC at n = 100, 200,
  300, 400, 500 (20 random trials each):

  | n | AUROC | ± std | range |
  |---|---|---|---|
  | 100 | 0.6084 | 0.057 | [0.522, 0.703] |
  | 200 | 0.6401 | 0.050 | [0.566, 0.732] |
  | 300 | 0.6601 | 0.039 | [0.598, 0.732] |
  | 400 | 0.6673 | 0.019 | [0.633, 0.700] |
  | 500 | 0.6657 | 0.010 | [0.648, 0.681] |

  **Smooth degradation**, not catastrophic — not a high-variance
  artefact. Future replications need ~150–200 samples to clear the
  corrected null.
- **§8 Within-category permutation (cell 26, fig 6).** Compares two
  nulls:
  - Global label-shuffle null: 95th %ile ≈ 0.55.
  - Within-category label-shuffle null: 95th %ile ≈ **0.59** (+0.04
    above global).
  - **Observed probe AUROC 0.66 exceeds the within-category null by
    +0.07.** *Within-question signal is ~+0.07, not +0.16.*
  - **About +0.04 of the apparent global signal is category base-rate
    exploitation**, not within-question retrieval structure. The honest
    paper claim is the within-category bar.

**Notebook 06 verdict.**
The signal is **statistically defensible** on three fronts:
1. The L-vector probe AUROC (0.65–0.67) has bootstrap 95 % CI clearly
   excluding 0.5 *and* excluding the confound floor 0.55.
2. The +0.118 ΔAUROC of energy over confounds survives paired bootstrap.
3. Permutation null (0/200) → p < 0.005.

But also **honestly bounded**:
1. Per-layer claims survive only marginally after MC correction (ΔE
   barely, JS not at all).
2. Effect sizes are **small** (|δ|<0.33 everywhere).
3. **~+0.04 of the apparent signal is category-level base-rate
   variation**, not within-question retrieval structure. The true
   within-question gain over confounds is closer to +0.07, not +0.12.

---

### 5.5 Notebooks 01–02 (legacy, pre-fix)

- **`01_dataset_analysis.ipynb`** — first-pass exploration on Qwen + Llama
  before the chat-template fix. Reports global ΔE AUROC 0.47–0.54 — much
  weaker than Exp-04, consistent with the prefill bug masking signal.
  Per-category AUROC heatmap shows extreme variance (0.00–1.00) at low n.
- **`02_category_layer_analysis.ipynb`** — refines the per-category
  per-layer view with CIs. Same data as NB01; superseded by NB04.
- **`02_report.ipynb`** — formatted research-report version of NB01/02 for
  reading. Same conclusions, prettier presentation.

These pre-fix notebooks are useful as a **before/after comparison**: the
chat-template fix between exp02→exp04 lifted ΔE AUROC from ≈0.50 to
≈0.65, which is the single largest improvement in the project.

---

## 6. What hallucination detection yields here — discussion

### 6.1 Headline claim and its honest scope

> Modern Hopfield retrieval energy / entropy / divergence at MLP layers
> carry a small but statistically-defensible hallucination signal that is
> recoverable only by a layer-vector linear probe, plateaus at AUROC
> ≈ 0.65–0.67 on Qwen2.5-3B / TruthfulQA, lives mechanistically in the
> LSE term (retrieval concentration) rather than the query-norm term, is
> topic-heterogeneous (Identity/Confusion strong, Politics/Religion
> inverted), and after within-category null correction adds about +0.07
> AUROC of within-question information beyond category base-rate
> exploitation.

This is **not** a deployable detector (AUROC ~0.65 implies ~35 % of
HAL/OK pairs misordered) and **not** a topic-universal mechanism.

### 6.2 Why the signal is small

Two coupled reasons:

1. **TruthfulQA is adversarial.** Questions were chosen because the model
   tends to fail on them — that's the whole construction. So the "hard"
   cases are 63 % of the dataset, not 5–10 % as in natural distributions.
   This compresses the discriminative axis: there's less "easy factual"
   signal to anchor against.
2. **Generation is greedy.** Greedy decoding produces a deterministic
   trajectory per prompt — no per-token sampling variance to amplify the
   class signal. With temperature > 0 and N samples per prompt, the
   per-sample energy *variance* itself becomes a feature (this is what
   the semantic-entropy line of work exploits, and it's the most
   promising "more orthogonal feature" extension).

### 6.3 Mechanistic interpretation

The signal living in **`E_lse`** (LSE term, retrieval peakedness) and not
in `E_quad` (query norm) supports a specific mechanistic picture:

- **Factual generation** — the model retrieves from a small, sharp set of
  MLP memory patterns. Retrieval is peaked; E_lse drops.
- **Hallucinated generation** — the model spreads retrieval weight more
  diffusely across patterns. Retrieval is less peaked; E_lse rises.

The per-category heterogeneity then suggests **different failure modes
have different retrieval signatures**:
- *Identity/Confusion* hallucinations involve over-committing to a
  similar-but-wrong entity → retrieval becomes peaked but on the wrong
  patterns. **ΔE catches this; JS partially misses it** (the distribution
  shape is similar, just shifted).
- *Language/Distraction* hallucinations involve fluent-but-wrong
  continuations by spreading retrieval over a *different* set of
  patterns. **JS catches this; ΔE misses** (concentration unchanged,
  identity changed).
- *Politics/Religion/History* invert the sign — *factual* answers in
  these topics involve more retrieval spread than hallucinations,
  possibly because contested-fact questions force the model to consider
  multiple memory regions before committing.

### 6.4 What this is *not*

- **Not a per-sample explanation.** NB05 §7 confirms extreme HAL/OK
  fingerprints are visually indistinguishable. There's no per-generation
  story to read off the retrieval pattern.
- **Not transfer-tested.** All results are on Qwen2.5-3B / TruthfulQA.
  Exp-05 will test Llama; cross-dataset transfer (NQ, TriviaQA) is
  untested.
- **Not deployable.** AUROC 0.65–0.67 is a research result, not a
  detector. Clinical / production thresholds start at 0.80.

---

## 7. Proposed methodological extensions

Ordered by expected ratio of (likelihood of new signal) / (engineering
cost):

### 7.1 Cheap wins (no new GPU passes)

1. **Re-run the full probe excluding L0 explicitly.** L0 carries 30× the
   mean JS of other layers and no class signal. The L2-regularised probe
   already learns to zero it out, but a clean ablation would tighten
   reported numbers by ~+0.01.
2. **Report per-category AUROC with bootstrap CIs**, not point
   estimates. The "Confusion: People AUROC = 1.0" line in NB04 §8 is on
   1 OK sample and is uninformative; replacing with proper CIs would
   make the heterogeneity story honest.
3. **Add sequence-level baselines** (`seq_logprob_mean`, `token_entropy_mean`,
   `token_entropy_max`, `p_true`) to the confound regression in NB06.
   Pipeline already writes these to `{id}_baselines.json`; NB06 §2 just
   doesn't load them. Including them is **the strongest reviewer
   demand** the current analysis fails to meet. Predict: token entropy
   alone scores ~0.60 (it correlates with model uncertainty); the
   honest gain of ΔE over a *full* baseline including token entropy is
   probably +0.04–0.06, not +0.12.
4. **Fix the NB04 `METRICS["Hellinger"]` bug** (`gen_js_layer` →
   `gen_hell_layer`).
5. **Post-hoc β sweep.** Exp-04 saved `[L, T, K]` scores for 50
   samples; rerun the LSE/JS/Hellinger computations at β ∈ {30, 50,
   68, 100, 150} without a new forward pass. If the AUROC peak isn't at
   β★=68 (the calibration target), the calibration heuristic is
   suboptimal and should be re-derived.

### 7.2 Medium investment (one new run)

6. **Replicate on Llama-3.2-3B (exp05).** Already scheduled. If the
   per-layer-of-action shifts (Identity/Confusion L30 in Qwen), the
   story becomes "retrieval-shift hallucination signal is general but
   localises differently per architecture." If it doesn't shift, the
   story becomes "the L30 effect is Qwen-specific."
7. **Sample N completions per prompt at T > 0** — every prompt gets 5–10
   sampled generations; per-prompt **energy variance** becomes a
   feature. This is the natural bridge to semantic-entropy methods.
   Expected gain: substantial (semantic entropy alone is reported at
   AUROC 0.7+ in the literature).
8. **Topic-conditional probe.** Train per-macro-group probes and
   ensemble. Predict at the per-sample level using the probe whose
   macro-group matches the sample. Pre-register the strong-signal
   topics (Economics, Identity/Confusion, Paranormal) and report on
   held-out runs.
9. **Run exp06 (value-space probe).** `W_down.T` rows as the bank
   (Geva-style "value vectors"). The current `up` (key-space) bank may
   carry less semantic content than the value space. Worth ~+0.02–0.05
   AUROC if Geva is right.
10. **Run exp08 (logit-lens + shuffled-bank null).** Logit-lens entropy
    is a natural orthogonal feature (output-distribution uncertainty,
    not retrieval-distribution uncertainty). Shuffled-bank null
    confirms the signal isn't due to bank-row geometry alone.

### 7.3 Bigger redesigns

11. **Cross-dataset transfer (NQ, TriviaQA).** The TruthfulQA result is
    on adversarial questions; natural-distribution datasets may yield
    different effect sizes. Useful for the paper's generalisation
    claim.
12. **Per-(layer, token) probe with proper regularisation.** A 36×50 =
    1800-feature probe with 500 samples needs stronger regularisation
    (group L1 across tokens, L2 within layers). May find that
    early-token-only features suffice.
13. **Combine with attention-pattern features.** Hallucination may show
    up in *which* tokens the model attends to during generation as much
    as in retrieval. Per-head, per-layer attention entropy is the
    natural feature.
14. **Steering experiment.** If ΔE_lse really tracks retrieval-spread,
    *forcing* the LSE to be more peaked (by amplifying attention onto
    high-activation patterns) should reduce hallucination rates. A
    controlled intervention is the strongest validation of the
    mechanistic story.

### 7.4 Statistical methodology upgrades

15. **Report AUROC with bootstrap CIs everywhere**, not 5-fold std.
    NB06 already does this for headline numbers; extend to all
    notebooks.
16. **Pre-register the next study.** All current per-category and
    per-layer findings are exploratory. A confirmatory study would
    fix the feature set (probe = ΔE [L], C=0.1) and report on a
    held-out dataset.
17. **Replace Cohen's κ judge calibration with judge-vs-human on a
    100-sample subset.** Current pipeline writes κ between F1 heuristic
    and GPT judge — useful but doesn't tell us how the judge compares
    to a human gold standard.

---

## 8. TL;DR conclusions

1. **Energy / divergence in MLP memory banks carries a real, small,
   statistically defensible hallucination signal** on Qwen2.5-3B /
   TruthfulQA. AUROC ≈ 0.65–0.67 with bootstrap CI [0.62, 0.72] for the
   strongest probe.
2. **The signal lives in retrieval *concentration* (LSE), not query
   *norm***. Mechanistically: hallucinated generation spreads memory
   retrieval more diffusely than factual generation.
3. **Layer-vector probes always beat scalar reductions** because per-layer
   signals carry mixed signs across the network (10 positive / 13
   inverted JS layers; similar for ΔE). Mean-pooling cancels them.
4. **JS and ΔE are largely redundant** — two views of the same
   retrieval-shift phenomenon. Combining them adds <0.01 AUROC.
5. **The signal is topic-heterogeneous**: Identity/Confusion strong
   (L30 AUROC ≈ 0.80); Politics/Religion/History inverted; zero
   categories survive BH correction at p<0.05 given small per-category n.
6. **~+0.04 of the apparent global signal is category base-rate
   exploitation**, not within-question retrieval structure (within-category
   permutation null is at 0.59 not 0.55). Honest within-question gain
   over confounds ≈ +0.07.
7. **The signal is established at the first generated token** and persists
   — no temporal structure to exploit.
8. **The single most important methodological change** in the project
   was applying the chat template identically to prefill and generation
   (exp02 → exp04), which raised ΔE AUROC from ≈0.50 to ≈0.65.
9. **Highest-value next step:** add sequence-level baselines
   (token-entropy, seq-logprob, p_true) to the confound regression. The
   current confound floor (0.55) understates how much the energy signal
   adds beyond cheap baselines.
10. **Highest-value mechanistic next step:** repeat the analysis with
    per-prompt sampling (N completions at T > 0) so per-prompt energy
    variance becomes a feature — the natural bridge to semantic entropy.

---

## Appendix A — Headline numbers cheat sheet

| Quantity | Value |
|---|---|
| Model | Qwen 2.5-3B-Instruct, 4-bit |
| Dataset | TruthfulQA, 500 samples, seed 42 |
| Class balance | HAL 317 / OK 183 (63.4 % HAL) |
| Layers | 36 |
| Patterns per bank | K = 11008 |
| Calibrated β★ | 68.16 (target H_norm = 0.55 · log K) |
| Realised H_norm | 0.4868 (mean over 500 × 36) |
| Best per-layer ΔE AUROC | 0.6113 (L21) |
| Best per-layer ΔH AUROC | 0.6027 (L9) |
| Best per-layer JS AUROC | 0.5714 (L6) |
| Scan-corrected null 95 % | 0.5800 |
| Best L-vector probe | ΔE+JS+Confounds AUROC 0.6695 CI [0.615, 0.719] |
| Best JS-family probe | JS 4-stat [4×L] AUROC 0.6664 |
| Energy vs confounds ΔAUROC | +0.118 CI [+0.05, +0.18] |
| JS over (Energy + Confounds) ΔAUROC | +0.002 CI [−0.02, +0.03] (n.s.) |
| Confound-only AUROC | 0.5517 (the right floor, not 0.5) |
| Within-category null 95 % | ≈ 0.59 (vs global null ≈ 0.55) |
| Cliff's δ (max) | 0.223 (Romano: SMALL effect) |
| Layers with Cliff's δ CI ≠ 0 | 11/36 ΔE, 8/36 JS |
| Energy decomposition | r(ΔE_lse, ΔE) = −1.00; r(ΔE_quad, ΔE) = +0.01 |
| Strongest local finding | Identity/Confusion macro AUROC ≈ 0.80 at L30 (n=62) |

## Appendix B — File map for Exp-04

```
exp04_results/
├── config.yaml                          ← run config snapshot
├── git_info.json                        ← commit db99c58
├── banks_metadata.json                  ← {layer: K, D, M, M_raw} × 36
├── analysis.json                        ← per-layer mean/std, score_by_category (38 cats)
├── log.txt                              ← stage timings
├── trajectories/                        ← 500 samples
│   ├── calibration_beta.json            ← β★=68.16, calibration curve
│   ├── truthfulqa_<id>.json             ← question, generation, metadata
│   ├── truthfulqa_<id>.npz              ← prefill_*, gen_* arrays [L]/[L,T]
│   └── truthfulqa_<id>_scores.npz       ← [L,T,K] raw scores (50 samples only)
├── labels/                              ← 500 × judge labels
│   └── truthfulqa_<id>_label.json       ← is_hallucination + judge_raw_output + F1
└── plots/                               ← from `visualize`
    ├── delta_{energy,entropy,...}_by_layer.png
    ├── global_divergences.png
    ├── score_by_category.png
    └── analysis_snapshot.json           ← machine-readable copy of analysis.json
```

The four analysis notebooks (`03–06`) all read from this directory and
write their PNGs to `outputs/nb0X_plots/`.
