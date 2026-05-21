# Full Pipeline Review — Energy-Based Hallucination Detection

> **Scope.** Independent review of the `hopfield_llm` project after Experiment 04 (Qwen2.5-3B, TruthfulQA, 500 samples) and the four analysis notebooks (NB03–NB06). The goal is to answer: *what is being calculated, is it theoretically correct, was the dataset appropriate, and what is the single next experiment that would turn this project into a publishable result?*
>
> **Date.** 2026-05-19. Commit `4acf2f2` (with local uncommitted edits in `cli/run.py`, `docs/`).
>
> **Companion documents.** This review complements (and supersedes where they disagree) `notebooks/EXPERIMENT_SUMMARY.md` and `docs/AUDIT_REPORT.md`. Where the existing audit has been overtaken by code changes, that is flagged explicitly here.

---

## 0. Executive summary

The project asks a sharp, well-posed question — *can the Modern-Hopfield retrieval energy over MLP weight matrices distinguish hallucinated from factual generation?* — and answers it methodologically more honestly than the average paper in this space. After ~7 experiments, the empirical answer is **yes, but barely**: a 36-layer linear probe on `Δ = mean_gen − prefill` retrieval features reaches **AUROC ≈ 0.65–0.67** with bootstrap 95 % CI [0.61, 0.72], surviving permutation, scan-corrected, and within-category nulls — but adding only **+0.07 AUROC over confounds** after the within-category correction, with effect size **Cliff's δ ≈ 0.22 (SMALL)**.

That is not a deployable detector and not a strong-enough novel result for a top-tier venue. **But it is publishable**, with one redirection: stop framing the contribution as *"Hopfield energy detects hallucination"* and reframe it as *"a Hopfield-inspired retrieval-concentration feature is a complementary, mechanistically grounded channel that adds information over uncertainty baselines"* — provided the missing baselines are actually computed and the missing replication is actually run.

This review therefore concludes with a **single proposed final experiment (Exp-09)** that is designed to be the project's last GPU run and produce the result that closes out the project, whether the final answer is publishable-Q2 or workshop-only.

The three load-bearing reasons the current AUROC plateau exists:

1. **Greedy decoding leaves the strongest hallucination feature on the table.** Every state-of-the-art detector (semantic entropy, p_true, SelfCheckGPT) exploits *sampling variance* across multiple completions of the same prompt. Our pipeline captures only a deterministic single-trajectory energy, which by construction is a fixed function of `(model, prompt)` and does not measure the model's uncertainty about *what* to generate.
2. **TruthfulQA is adversarial by construction.** 63.4 % hallucination rate compresses the discriminative axis: there is less "easy factual" signal to anchor against, and the test set co-mingles factual ignorance with deliberately misleading premises. Effect sizes here are systematically a lower bound on what cross-dataset generalisation would show.
3. **`normalize_query = True` makes the energy decomposition tautological.** With unit-norm queries the quadratic term collapses to a constant, so the headline mechanistic claim ("signal lives in LSE, not norm") is a definitional outcome, not an empirical finding.

The proposed Exp-09 fixes all three at once.

---

## 1. What is calculated, in plain terms

For each sample `(question, gold answers)`:

1. **Build memory banks (Stage 1, GPU).** From the model's MLP weights, extract per-layer `[K, D]` matrices whose rows are treated as Hopfield patterns. Default: rows of `W_up` (the "key" matrix in the SwiGLU MLP). `K = 11008`, `D = 2048`, `L = 36` for Qwen 2.5-3B.
2. **Prefill pass (Stage 2, GPU).** Forward the chat-templated prompt through the model. At each layer, capture the query `q_ℓ ∈ ℝ^D` going *into* the MLP (`hook_target = mlp_input`). Reduce the prompt trajectory to a single query via `last_token` (or `mean_tokens`). Compute per-layer scalars: energy `E_ℓ`, retrieval entropy `H_ℓ`, normalised entropy `H_ℓ / log K`, LSE term, quadratic term, top activation.
3. **Generation pass (Stage 2, GPU).** Greedy decode up to 50 tokens. For each generated token `t`, hook the same `mlp_input`, get the same `q_ℓ`, compute the same per-layer scalars. Also compute **JS divergence** and **Hellinger distance** between the generated-token retrieval distribution `p^gen_{ℓ,t}` and the prefill reference `p^ref_ℓ` (currently the *prompt-mean* softmax — see §3.4 for why this is a problem).
4. **Per-sample delta (Stage 3, CPU).** Subtract the per-layer prefill baseline from the per-layer generation mean: `ΔX_ℓ = mean_t(X^gen_{ℓ,t}) − X^ref_ℓ`. This is the "what does autoregressive generation do to the retrieval signal" feature, computed for `X ∈ {E, H, H_norm, E_lse, top_act}`.
5. **Per-sample scalar (Stage 3, CPU).** `score = mean_ℓ(ΔX_ℓ)`. This is the headline number stored in `analysis.json`.
6. **Per-sample vector (NB03/05/06).** The 36-dim per-layer feature `[ΔX_0, ..., ΔX_35]` is the input to a logistic-regression probe with `StandardScaler + L2 (C = 0.1)`, 5-fold stratified CV.
7. **Labels (Stage 5, CPU).** Each sample receives a binary label from two independent sources: a token-F1 heuristic against `gold_answers` (`threshold = 0.3`) and a GPT-judge call with a "factually correct: yes/no" prompt. Cohen's κ between the two is persisted.
8. **Probe evaluation (Stage 6, CPU).** Per-layer AUROC, full-vector logistic probe AUROC, baselines comparison, bootstrap CIs, permutation null, within-category null. Numbers all written into `report.html`.

The **energy formula** is the Modern-Hopfield-Networks one (Ramsauer et al. 2020, Eq. 1):

```
E_ℓ(q) = −β⁻¹ log Σ_k exp(β q·ξ_k)   ← E_lse  (retrieval concentration / softmax peakedness)
       + ½‖q‖²                         ← E_quad (query norm)
       + β⁻¹ log K                     ← entropy-scale constant
       − ½ M_ℓ²                        ← cross-layer normalisation (max row norm)
```

where `M_ℓ = max_k ‖ξ_k‖`. `β` is auto-calibrated to put mean prefill normalised entropy at `0.55 · log K` — the so-called *detection band* where the softmax over patterns is neither uniform (no signal) nor collapsed onto the argmax (no signal). For Exp-04, `β★ = 68.16` calibrated from a 30-sample warm-up.

**JS and Hellinger** are computed between two softmax distributions over `K = 11008` Hopfield patterns at the same layer: the generation-time softmax at token `t` and the prefill reference. JS is the symmetric KL with a `½(p+q)` midpoint; Hellinger is `½‖√p − √q‖²`.

---

## 2. Is the implementation correct?

### 2.1 What is verified correct

I re-audited the load-bearing source files against the existing `docs/AUDIT_REPORT.md` and the project's stated theory. The following items survive review:

| Component | Verdict | Evidence |
|---|---|---|
| Energy formula vs Ramsauer 2020 | ✓ correct | `metrics/energy.py:124` matches Eq. 1; LSE sign, `M²` term, `log K / β` constant all present |
| LSE numerical stability | ✓ correct | `torch.logsumexp(beta * scores, dim=0) / beta` (`metrics/energy.py:113`), not naïve exp-and-sum |
| JS / Hellinger implementation | ✓ standard | `metrics/divergences.py` — symmetric JS, standard `½‖√p−√q‖²` Hellinger, `eps=1e-10` clamp + renorm |
| Cross-layer comparability via `M²` | ✓ wired | `M_used` per layer loaded from `banks_metadata.json` and threaded through `compute_energy` |
| Hook-pair chat-template invariant | ✓ enforced | Both `capture_prefill` and `capture_generation` default to `apply_chat_template=True` (was the bug fixed exp02→exp04) |
| Beta auto-calibration logic | ✓ correct | `metrics/calibration.py` — captures raw scores once, evaluates all β analytically, interpolates between bracketing candidates |
| Streaming `p_ref` (no `[L,T,K]` tensor) | ✓ correct | Commit `4acf2f2` fixed the OOM by streaming-mean accumulation in `hooks/capture.py` |
| Statistical machinery in NB06 | ✓ rigorous | Scan-corrected (max-of-36) null, paired-bootstrap nested-model test, Cliff's δ with CIs, within-category null. Rare in this literature. |
| Storage discipline | ✓ correct | Hidden states never persist; only `[L]` / `[L,T]` scalars + optional `[L,T,K]` for a subset |

### 2.2 What is *broken or mis-stated*

These are the items that change the interpretation of Exp-04 if you take them seriously.

#### Issue A (high) — exp04 artifacts were produced under a now-fixed bug

The earlier `AUDIT_REPORT.md` (commit `db99c58`) found that `cli/run.py` was *not* threading the YAML `primary_probe` / `secondary_probe` blocks through to `build_banks` / `run_trajectory`, so every `run-experiment` invocation silently fell into a legacy path where banks are **not** row-normalised and queries **are** L2-normalised.

That bug **has since been fixed in source** — `cli/run.py:180-181, 201-202` now pass `primary_probe=cfg.primary_probe, secondary_probe=cfg.secondary_probe` correctly. (The file currently shows uncommitted edits matching this fix.)

**But the Exp-04 artifacts on disk were generated *before* the fix.** Direct evidence: `exp04_results/banks_metadata.json` shows `M_used == M_raw` for every layer (range 1.48 to 6.08). If `normalize=True` had been applied at bank construction time, `M_used` would be 1.0 for every layer. So Exp-04 ran with:

- Bank rows: **unnormalised** (row norms 1.5–6.1)
- Query: **L2-normalised** (so `‖q‖² ≡ 1` everywhere — see Issue C below)
- Mode: `dot`

This is *not* what `exp04_qwen_chat_template_fix.yaml` describes; the YAML was a fiction with respect to what actually ran. The notebooks describe the YAML, not the artefact.

**Implication.** None of the Exp-04 numbers should be re-cited verbatim in a paper without a Phase-1 re-run. Whether the AUROCs shift by 0.01 or by 0.05 under the corrected normalisation is currently unknown and matters for the paper's claim. **Re-running Exp-04 from the same trajectories config under the fixed code is the single highest-leverage action remaining.**

#### Issue B (high) — `ΔE_quad = 0` is a tautology under `normalize_query = True`, not a mechanistic finding

With unit-norm queries the quadratic term `½‖q‖²` collapses to a constant `0.5` for every layer and every token, so `ΔE_quad ≡ 0` *by construction*. NB03 §7 reports `Pearson r(ΔE_quad, ΔE) = 0.01` and `r(ΔE_lse, ΔE) = −1.00`, and the existing summary calls this "the publishable mechanistic claim". It is not — it is a definitional consequence of the normalisation choice. The only reason `r` is not exactly zero / exactly one is float-precision drift in `F.normalize`.

For the "retrieval-concentration not query-norm" claim to be *empirical*, the analysis must be run a second time with `normalize_query = False`. Then `‖q_ℓ‖² ∝ ‖x_ℓ‖²` (the FFN-input activation norm) carries real per-sample, per-layer signal, and `ΔE_quad` becomes a falsifiable feature. If `ΔE_quad` is still null in that setting, the mechanistic story holds; as written, it cannot be tested.

#### Issue C (medium) — `p_ref` mixes prompt-mean and per-token references

The prefill reference distribution `p^ref_ℓ` used for JS and Hellinger is computed as a **mean-pooled streaming softmax over all prompt tokens** (`hooks/capture.py`, the post-`4acf2f2` accumulator). But `pipeline/analysis.py:164` uses `prefill_energy_mean` (also mean-pooled) for `ΔE`, while `prefill_energy` written to npz at `pipeline/stages.py:467-468` is the *last-token* energy.

So:
- `ΔE` uses gen-mean vs prompt-**mean** (consistent).
- JS uses gen-per-token vs prompt-**mean** (also consistent within itself).
- The npz "prefill" field accessible to downstream code is *last token*, but `analysis.py` falls back to the *mean* if available.

The **L0 dominance** finding in NB05 §2 (`mean JS at L0 = 0.64, at L1–34 = 0.32`) is partly an artefact of this design choice. At L0, queries are raw embeddings; the prompt-mean of 30+ such embeddings is a smeared distribution and its softmax over `K = 11008` patterns is very diffuse; each generation-token's L0 softmax is sharp. The JS between "mean of 30 sharp distributions" and "one sharp distribution" is large for *every* sample, *every* class — it carries no class signal and contaminates every scalar layer mean.

The cleaner choice is **last-prompt-token** as the reference for both energy and JS. That matches the intuition "the state the model commits from", makes the reference a single token's softmax (not a mixture), and is likely to shrink the L0 artefact substantially.

#### Issue D (medium, statistical) — confound set in NB06 is too weak

NB06 §2 establishes a "confound floor" of AUROC **0.5517 ± 0.0372** using four response-shape features: `n_tokens`, `text_len`, `vocab_div`, `avg_word_len`. The reported ΔAUROC of energy over confounds is **+0.118**, "robust".

But the pipeline *already writes* `{id}_baselines.json` containing the canonical hallucination-detection baselines:
- `seq_logprob_mean`  (sequence log-probability)
- `token_entropy_mean`, `token_entropy_max` (decoding-time token uncertainty)
- `p_true` if computed (Kadavath 2022)

None of these are loaded by NB06. They are the *strongest* competitors any reviewer will demand to see. Token-entropy mean alone is typically reported at AUROC 0.60–0.65 in the literature, p_true at 0.65–0.75, semantic entropy at 0.75–0.85. The honest ΔAUROC of energy over a *full* baseline including token-entropy is probably **+0.04 to +0.06, not +0.12**.

This is the single biggest reviewer demand the current analysis fails to meet. It is also the cheapest item to fix — the data is already on disk.

#### Issue E (low, but paper-relevant) — judge model identifier is unverifiable

Every `exp04_results/labels/*_label.json` carries `"judge_model": "gpt-5.4-mini"`. OpenAI has not published a model with that exact identifier; this is likely a typo for `gpt-4o-mini`, a proxy alias, or an internal name. Reviewers will want the precise OpenAI / HuggingFace model ID, and reporting "gpt-5.4-mini" verbatim will look careless. **Verify and document the actual model used before publication.**

#### Issue F (low) — judge prompt is correctness-framed, not hallucination-framed

The judge prompt (`labeling/prompts/convcoa_v1.txt`) asks "is the Generated Answer factually correct?". It does *not* explicitly probe for inserted false claims — a fluent generation that contains the correct core fact plus a confident wrong elaboration may still score 1. This biases the judge toward false negatives (missed hallucinations) and bounds the AUROC of any detector that *does* catch the elaborations. For a paper, augment with a second-pass prompt that asks "does the Generated Answer contain any fact not supported by the reference?".

#### Issue G (low) — F1 heuristic threshold at 0.3 is aggressive

`labeling/heuristic.py` uses token-F1 against gold answers with threshold 0.3. For short-answer QA against `correct_answers` lists this means "Barack Obama" vs "Barack Hussein Obama" passes (F1≈0.67), but "Paris" vs "France" fails (F1=0). This is the right design for SQuAD-style extractive QA but underpenalises *semantic equivalence* (paraphrases) and overpenalises *related-but-wrong*. The pipeline tracks both F1 and the judge label and reports κ, which is the right hedge — but the F1 score is currently used as a tie-breaker / cross-check, not as a label, so the impact on Exp-04 results is bounded.

---

## 3. Is the dataset choice right?

### 3.1 Why TruthfulQA was a defensible starting point

- Designed adversarially against LLM "common misconceptions" — the failure modes are well-curated.
- Has 38 categories with documented hallucination-prone topics (Misconceptions, Conspiracies, Indexical Errors).
- 817 questions in `validation` split — manageable for a single 500-sample run.
- Open-ended generation evaluation is the right modality for measuring *internal* state during free-form decoding.

### 3.2 Why it is now a constraint, not a feature

1. **63.4 % HAL rate is unnatural.** In production, hallucination rates on factoid tasks are 5–20 %. A detector tuned at 63 % positive class will not behave the same at 10 % — calibration drifts.
2. **The "hard" cases dominate the dataset.** TruthfulQA is *constructed* to exploit common LLM failure modes. So most factual answers also live near the model's confusion boundary — the discriminative axis is compressed.
3. **The 38 categories are too unbalanced for per-category claims.** `Confusion: People` has 17 samples, 94 % hallucinated; the resulting "Confusion: People AUROC = 1.0" in NB04 is computed from 1 OK sample × 16 HAL samples and is uninformative. Zero categories survive Benjamini-Hochberg correction at p<0.05.
4. **The category labels conflate task difficulty with topic.** "Indexical Error: Identity" describes *why* the question is hard, not *what* it is about. Per-category mechanistic claims (e.g. "Identity/Confusion shows AUROC 0.80 at L30") sit on shaky semantic ground.

### 3.3 What the project really needs

- **A natural-distribution factoid dataset** to test whether the signal survives outside the adversarial regime → **TriviaQA** or **Natural Questions**. Both adapters already exist in `datasets/`. Cross-dataset transfer (probe trained on TruthfulQA, evaluated on NQ) is the right generalisation test.
- **A multi-hop reasoning dataset** to test whether the signal lives at intermediate-reasoning tokens, not final answer tokens → **HotpotQA** or **MuSiQue**. Not in the codebase; would need a new adapter (~1 day of work).
- **A clean per-prompt sampling dataset** so that semantic-entropy / SelfCheckGPT-style features can be computed. Just temperature > 0 with N completions on any of the above.

**Verdict.** TruthfulQA was the right place to *prototype* the method. It is the wrong place to *stop*. The strongest possible single dataset substitution would be **TriviaQA + greedy + temperature-0.5 sampling**, which gives (a) natural distribution, (b) shorter, factoid answers that match the F1 heuristic better, and (c) the sampling variance feature.

---

## 4. Is the *internal state extraction* correct?

This is the part of the pipeline most likely to be silently wrong, because hooks fire deep inside `transformers` internals and small choices about *which tensor* gets captured propagate everywhere downstream.

### 4.1 What it does

`hooks/capture.py` registers forward-pre-hooks on every layer's `mlp` module. The captured tensor is the **MLP input** — i.e. the residual-stream activation *after* the post-MLP layernorm, *before* the gate/up projection.

```text
residual_in → LN_pre_mlp → x_ℓ  ← captured as q_ℓ
              ↓
  W_gate · x_ℓ  ·SiLU·  ⊙  ·  W_up · x_ℓ
              ↓
            W_down → residual_out
```

For prefill, the hook fires once per prompt forward pass and captures `[T_prompt, D]` per layer. The reduction (`last_token` by default) collapses this to `[D]` per layer. For generation, the hook fires once per generated token with shape `[1, D]` and is concatenated.

### 4.2 What is correct about this

- **Bank rows match the captured query semantically.** `W_up`'s rows live in the same space as `x_ℓ` (both are `D`-dimensional, both are inputs to the same matrix product). Computing `q · ξ_k` is the same dot product the model itself computes internally; the Hopfield interpretation just treats those `K` dot products as a softmax retrieval over `K` patterns.
- **`mlp_input` is the right hook target** if the question is "what does the residual stream activate in the MLP memory?". Hooking the *output* of `W_up` (post-projection) would already mix in the model's chosen activation; the *input* is the un-projected query.
- **Greedy generation is deterministic given the prompt**, so per-token captures are reproducible.

### 4.3 What is questionable about this

1. **`W_up` rows are not "memories" in any architecturally privileged sense.** In SwiGLU, the path from `x_ℓ` to `y_ℓ_t` is `W_down · (SiLU(W_gate · x_ℓ) ⊙ W_up · x_ℓ)`. The element-wise gate `SiLU(W_gate · x_ℓ) ⊙` is what actually decides which "memory" gets retrieved; rows of `W_up` alone are only half the key. Reviewers who know SwiGLU will push back on the "rows of `W_up` are Hopfield patterns" framing.
2. **The Hopfield equivalence theorem (Ramsauer 2020) requires tied `W_down = W_up^T` and softmax non-linearity.** SwiGLU has neither. So the energy `E(q) = −β⁻¹ log Σ exp(β q·ξ_k)` is a well-defined scalar function on `(query, bank)` pairs, but it is **not** the energy whose gradient flow recovers the MLP forward pass. This was discussed in `AUDIT_REPORT.md` Concern T1; the recommendation there is to weaken the paper title from "Hopfield retrieval energy detects hallucination" to **"a Hopfield-inspired retrieval-concentration score over MLP key vectors detects hallucination"**. This review concurs.
3. **The reduction `last_token` for the prefill query is a choice, not a derivation.** `mean_tokens` would give a different (smoother, less-position-biased) reference. The pipeline supports both via `queries/`; switching reductions and re-analysing is a CPU-only operation if the `[L, T, K]` scores were saved, but currently they were saved for only 50 samples.
4. **The choice to hook *every* layer is statistically defensible but mechanistically suspect.** Mid-layers (10–25) are typically where MLP key-value memories live (Geva 2021); early and late layers carry different signals (tokenisation, prediction). The probe currently treats all 36 layers as equally weighted features and lets L2 regularisation handle the prior. A more principled choice would be to pre-register a "memory band" of layers (say L10–L25) and report on that.
5. **Hook robustness across architectures.** The pipeline resolves `up_proj` / `gate_proj` / `down_proj` via a model-alias registry; Llama and Qwen are tested. Mistral and DeepSeek are not. Any cross-architecture replication is one alias lookup away from silent failure.

### 4.4 Verdict on internal-state extraction

**Functionally correct, theoretically loose.** The captured query is the right tensor for the intended computation; the resulting energies and divergences are well-defined; the cross-layer normalisation is wired correctly. But the theoretical bridge from "Modern Hopfield retrieval" to "SwiGLU MLP memories" is an analogy, not a theorem — the paper must say so explicitly, and the empirical claim should be framed around the *measured retrieval-concentration signal*, not around a Hopfield equivalence the architecture does not satisfy.

---

## 5. Are the statistical claims defensible?

### 5.1 What survives review

NB06 is genuinely rigorous and rare. Specifically:

- Bootstrap CIs on pooled `cross_val_predict` probabilities (not the looser 5-fold std).
- Scan-corrected (max-of-36) permutation null for per-layer AUROC claims.
- Paired bootstrap for *nested-model* comparisons (does feature X *add* over feature Y?).
- Cliff's δ with bootstrap CI alongside AUROC — standardised non-parametric effect size.
- Within-category permutation null — catches base-rate exploitation.
- Sample-size sensitivity check at n = 100, 200, 300, 400, 500.

This is more careful than 80 % of the published hallucination-detection literature.

### 5.2 Where it overstates

- **`r(ΔE_lse, ΔE) = −1.00` is reported as a mechanistic finding** in NB03 §7 and the existing summary. Under `normalize_query=True` it is a tautology (Issue B above). Drop or re-derive.
- **Confound set is too weak** (Issue D). The +0.118 ΔAUROC over confounds will shrink substantially once `seq_logprob_mean` and `token_entropy_mean` are added. This is a known weakness in the analysis, not in the data.
- **`Confusion: People AUROC = 1.0` in NB04 §8** is computed on 1 OK sample × 16 HAL. The point estimate is correct; the implication is wrong. Per-category AUROC should be bootstrapped or filtered to `n ≥ 30`.
- **5-fold CV with fixed `C = 0.1` is mildly optimistic** vs the package's own nested-CV `evaluation/probe.py`. The headline table should use nested CV. Difference is likely ≤0.01 AUROC but the methodology is more defensible.

### 5.3 What is honest and load-bearing

The strongest defensible claim from Exp-04, *after* applying every correction in §5.2, is approximately:

> *On Qwen 2.5-3B with TruthfulQA, a 36-dim per-layer logistic probe over MLP-input retrieval features achieves AUROC 0.65 (bootstrap 95 % CI [0.61, 0.72]) against a noisy GPT-judge label. The signal is statistically defensible against scan-corrected per-layer nulls, against within-category nulls, and against response-shape confounds, with Cliff's δ ≈ 0.22 (small effect). After within-category correction and after adding token-uncertainty baselines, the marginal AUROC contribution of the retrieval feature is approximately +0.04–0.07 over the strongest baseline (estimated; not yet measured).*

That is publishable at a workshop or Q3 conference today. It is *not* publishable at a Q1/Q2 venue without (i) the missing baselines, (ii) cross-model replication, (iii) a held-out test set.

---

## 6. Why we have not found a "publishable, outstanding signal" — diagnosis

In order of leverage:

1. **Deterministic decoding measures the wrong thing.** Hallucination is fundamentally about *what the model could have said and didn't*. A single greedy trajectory cannot expose that. The literature has converged on sampling-variance features (semantic entropy, SelfCheckGPT, p_true) for exactly this reason. We are competing against detectors that observe a richer signal.
2. **TruthfulQA's class balance compresses the signal axis.** 63 % HAL means the "background" is mostly hallucinations; we are asking the probe to find the rare *factual* outliers, with a noisy judge as the supervision signal. Effect sizes here lower-bound what natural-distribution data would show.
3. **The energy decomposition is misframed.** Under unit-norm queries, "LSE not norm" is a definitional outcome. The *interesting* decomposition — `‖q‖²` (activation strength) vs `lse` (retrieval concentration) — requires `normalize_query = False`.
4. **The probe is mean-of-layers, not layer-and-token.** With 50 generation tokens × 36 layers = 1800 features per sample and 500 samples, a properly regularised (group-L1 across tokens, L2 within layers) probe would likely catch signal the layer-mean probe misses. NB05 §6 already shows that *temporal* moments per layer add value; this has not been operationalised.
5. **No cross-model replication.** Every reported number is on Qwen 2.5-3B. The exp05 config exists for Llama-3.2-3B; running it has not happened. A reviewer's first question is "does this replicate".
6. **No held-out test set.** All 500 samples are exhausted by 5-fold CV, so every reported AUROC is in-sample to some degree. The current bootstrap CIs are valid for the pooled-CV estimate, but they are not the same as a single held-out AUROC.

The good news: each of (1)–(6) is fixable, and a single experiment can address (1), (3), (5), and partially (6) together.

---

## 7. The proposed final experiment — Exp-09

**Name.** `exp09_final_sampling_dual_model`.

**Hypothesis.** A combined feature vector of (a) layer-mean retrieval-concentration `ΔE_lse [L]` with **unnormalised queries**, (b) **per-prompt energy variance** over N sampled completions at T = 0.5, and (c) standard token-uncertainty baselines, will produce a probe that meaningfully exceeds AUROC 0.70 on TruthfulQA *and* replicates on a second model *and* generalises (with degradation) to TriviaQA. Specifically: the retrieval-concentration features add **≥ +0.03 AUROC with 95 % CI excluding zero** over the strongest baseline (semantic entropy + token entropy + seq log-prob), held-out, across both models.

This is the single experiment designed to close out the project — pass or fail.

### 7.1 What changes vs Exp-04

| Variable | Exp-04 | Exp-09 |
|---|---|---|
| Decoding | greedy, 1 completion | N = 10 completions per prompt at T = 0.5; also keep N = 1 greedy as anchor |
| `normalize_query` | `True` (legacy path) | **`False`** — exposes the activation-norm vs retrieval-concentration decomposition |
| Bank normalisation | unnormalised (Bug A leftover) | **explicit and documented** (`normalize=True`, M=1) |
| Models | Qwen-2.5-3B only | **Qwen-2.5-3B + Llama-3.2-3B** (same Exp-05 config, run finally) |
| Datasets | TruthfulQA only | **TruthfulQA train (300) + TruthfulQA test (200) held-out + TriviaQA transfer (300)** |
| Reference `p_ref` for JS | prompt-mean (artefact-prone) | **last-prompt-token softmax** (consistent with `prefill_energy_last`) |
| Probe features | layer-mean Δ-features | layer-mean Δ-features **+ per-prompt energy variance over the N samples + token-entropy + seq-logprob + p_true** |
| Probe procedure | 5-fold CV, fixed C | **Nested CV for hyperparameters; report on 200-sample held-out + 300-sample TriviaQA transfer** |
| Stats | bootstrap + permutation | same, plus **paired-bootstrap nested-model test** for each feature family vs the baseline-only probe |
| Pre-registration | none | **Pre-register layer band L10–L25, fixed C = 0.1 with internal CV, and the four feature families before running** |

### 7.2 Why this design

- **Sampling addresses diagnosis (1) and (3) at once.** N completions per prompt give us per-prompt *energy variance* — the natural Hopfield-side analogue of semantic entropy. If hallucination is a "diffuse retrieval" phenomenon, the *variance* of `E_lse` across N samples should be a stronger feature than the *mean* under greedy. This is a novel feature the literature has not tested.
- **Unnormalised queries make `ΔE_quad` falsifiable.** Either the activation-norm channel carries signal (interesting!) or it does not (clean mechanistic claim "signal lives in retrieval concentration, not query norm" — finally an empirical statement).
- **Dual model + transfer dataset addresses diagnosis (5) and partially (6).** A signal that replicates Qwen → Llama *and* transfers TruthfulQA → TriviaQA is publishable. A signal that does neither is honest negative evidence and still publishable, just at a different venue.
- **Pre-registration locks in the strong-signal regions of Exp-04** (L10–L25, Identity/Confusion macro group) so that the reported AUROC is not the maximum over an unbounded analysis space.
- **Held-out evaluation** ends the "but it's CV-optimistic" complaint permanently.

### 7.3 Cost estimate

Per-prompt cost: roughly 10× the Exp-04 trajectory capture (N = 10 completions). At Exp-04's wall-clock, this is a 1–2 day A100 run for Qwen alone, plus an equivalent for Llama. Three datasets × two models × N = 10 puts the total at **3–4 days of A100 time** end-to-end. CPU-side analysis is a few hours.

Storage: per-sample artefacts scale by N. At Exp-04's ~60 KB per `{id}.npz`, ten completions per prompt × 800 prompts × 2 models ≈ **1 GB of trajectory NPZs**, well within budget. The `[L, T, K]` scores subset (50 samples × N completions) is the only meaningful disk hit at ~2–3 GB.

API cost: Stage-5 labelling on 800 new prompts × 2 models × cheap judge ≈ **~$5–10** at gpt-4o-mini rates.

This is the cheapest experiment that gives the project a publishable conclusion.

### 7.4 What success and failure look like

**Success criterion (pre-registered).** The probe `[ΔE_lse [L] (unnormalised q) + per-prompt-variance + token-entropy + seq-logprob]` achieves held-out AUROC ≥ 0.70 with bootstrap 95 % CI excluding 0.65, on *both* models, *both* in-domain TruthfulQA and transfer TriviaQA — and the paired-bootstrap test shows the retrieval features add ≥ +0.03 AUROC over the baseline-only probe with CI excluding zero.

**Strong positive.** All four criteria met. The paper is: *"A Hopfield-inspired retrieval-concentration feature, particularly its per-prompt variance under sampling, complements token-uncertainty baselines and improves hallucination detection by ~+0.04 AUROC across two models and a transfer dataset."* Q2 NLP/ML venue target.

**Mixed.** Replicates on one model but not the other, or in-domain but not on transfer. The paper is: *"Retrieval-concentration features are model- (or dataset-) specific signals; we characterise where they apply and where they don't."* Q3 venue or workshop target.

**Negative.** Retrieval features add < +0.02 AUROC over baselines on either replication. The paper is: *"Hopfield-inspired retrieval features are theoretically appealing but empirically subsumed by token-uncertainty baselines on factoid QA; we report this negative result with full ablations."* Workshop target — and a genuine contribution because the field has not done this comparison cleanly.

Any of the three outcomes ends the project with a defensible publication. The current state (Exp-04 only, AUROC 0.65 with the confound issue unfixed) does not.

### 7.5 What this experiment *does not* do

Out of scope, deliberately:
- Mechanistic intervention (activation steering, patching). Worth doing if Exp-09 succeeds; not necessary for first publication.
- Multi-hop reasoning datasets (HotpotQA). Worth doing after a publication exists; not necessary for first publication.
- Larger models (7B+, 13B+). Useful for impact; not necessary for first publication.
- Custom judge prompt redesign. Use the existing `convcoa_v1.txt` plus a verification pass against 100 human-labeled samples (4-hour annotation task) to bound the judge-label noise.

### 7.6 Pre-experiment checklist

Before launching Exp-09, finish:

1. **Fix Issue C** — switch `p_ref` to last-prompt-token softmax in `hooks/capture.py`; switch `prefill_*_mean` to `prefill_*_last` in `pipeline/analysis.py`. ~1 hour.
2. **Add `seq_logprob`, `token_entropy`, `p_true`** to the per-sample baseline artefact written during generation (already partly done — make sure it's loaded by `evaluation/features.py`). ~2 hours.
3. **Add a "per-prompt sampling" mode** to `pipeline/stages.py:run_trajectory` that generates N completions and writes per-completion `.npz` files with a sample-index suffix. ~half a day.
4. **Pre-register the analysis plan** as a short doc in `docs/exp09_preregistration.md`: layer band, feature families, hyperparameters, primary endpoint, success criterion, multiple-comparisons strategy. ~2 hours.
5. **Re-verify Bug A fix** by running a single-sample dry of `run-experiment` against `exp04_qwen_chat_template_fix.yaml` and confirming `banks_metadata.json` shows `M_used = 1.0`. ~10 minutes.

Total prep time before GPU launch: ~1.5 days of focused work.

---

## 8. Summary recommendations

In strict priority order:

1. **Fix the prefill-reference inconsistency** (Issue C) and **the prefill-energy fallback** (Issue F in AUDIT_REPORT.md Bug D) before any new GPU run. The L0 dominance result is partly artefactual.
2. **Add the existing baselines** (`seq_logprob`, `token_entropy`, `p_true`) to NB06's confound regression. Recompute the marginal ΔAUROC of energy features. This is a half-day of work and dramatically changes the headline claim.
3. **Run Exp-09 as specified** in §7. Single GPU campaign, ~4 days A100 time, ~$10 API cost. Closes the project.
4. **Re-frame the paper** along the lines of §7.4: complementary feature to token uncertainty, not a standalone detector.
5. **Verify the judge model identifier** (Issue E) before any external write-up.
6. **Drop the "LSE-not-quadratic-is-mechanistic" framing** until Exp-09 produces the unnormalised-query result that makes it falsifiable.

If only one of these gets done, it should be Exp-09 — but the prep items in §7.6 are non-negotiable prerequisites.

---

## Appendix A — Pipeline correctness scorecard

| Component | Theoretically correct? | Empirically validated? | Notes |
|---|---|---|---|
| Energy formula (Ramsauer 2020 Eq. 1) | ✓ | ✓ | Matches reference |
| LSE numerical stability | ✓ | ✓ | `torch.logsumexp` |
| Cross-layer `M²` normalisation | ✓ | △ | Wired correctly, but Exp-04 ran without bank normalisation |
| Beta auto-calibration | ✓ (linear interpolation imperfect) | ✓ | Bug C in AUDIT — minor numerical concern |
| `mlp_input` hook target | ✓ | ✓ | Right tensor for the right question |
| Chat-template prefill/gen parity | ✓ | ✓ | The exp02→exp04 fix |
| Query reduction (`last_token`) | ✓ | ✓ | Defensible default |
| JS / Hellinger divergence | ✓ | △ | Correctly implemented, but `p_ref` choice creates an L0 artefact |
| Logistic probe nested-CV | ✓ | ✗ | Package supports it; notebooks don't use it for headline numbers |
| Bootstrap CIs on pooled CV | ✓ | ✓ | NB06 is rigorous |
| Permutation + scan correction | ✓ | ✓ | Rare in this literature |
| Within-category null | ✓ | ✓ | Catches base-rate exploitation |
| Cliff's δ effect size | ✓ | ✓ | Properly bootstrapped |
| Sampling-variance feature | n/a | ✗ | Not yet computed — the core Exp-09 deliverable |
| Cross-model replication | n/a | ✗ | exp05 config exists, not yet run |
| Cross-dataset transfer | n/a | ✗ | TriviaQA / NQ adapters exist, not used |
| Held-out test set | n/a | ✗ | 5-fold CV exhausts all 500 samples |

Legend: ✓ implemented and used, △ implemented with caveats, ✗ not implemented or not used.

---

## Appendix B — Exp-09 timeline

| Phase | Days | Output |
|---|---|---|
| Prep (§7.6) | 1.5 | Fixed `p_ref`, added baselines, sampling mode, pre-reg doc |
| Qwen TruthfulQA run (N=10 sampling) | 1.0 | 500 prompts × 10 completions, `.npz` artefacts, labels |
| Llama TruthfulQA run | 1.0 | Same on Llama-3.2-3B |
| Qwen + Llama TriviaQA run (300 samples) | 1.0 | Cross-dataset transfer artefacts |
| Stage 5 labelling (judge calls) | 0.5 | All labels |
| NB07 (new) — sampling-feature analysis | 1.0 | Per-prompt energy variance, semantic-entropy-style probe |
| NB08 (new) — held-out + transfer | 1.0 | Held-out AUROC, transfer AUROC, paired-bootstrap |
| Paper draft | 5–7 | First full draft |
| **Total** | **~12 days focused work** | **End-of-project artefact + draft** |

This is the end of the project.

---

*Review completed 2026-05-19. Commit `4acf2f2` plus uncommitted edits to `cli/run.py`.*

---

# Adendo · Análisis integrado marco teórico × pipeline × resultados

> **Fecha.** 2026-05-20. Tras la subida de `docs/marco_teorico_hopfield_llm.md` (v1.0, mayo 2026).
>
> **Propósito.** Cruzar las predicciones de los 11 papers del marco teórico contra (i) lo que el pipeline efectivamente computa hoy, (ii) los resultados empíricos de los notebooks NB03–NB06 sobre Exp-04, y (iii) las falencias del review original. Producir un plan accionable hacia publicación indexada en Scopus.
>
> **Tesis del adendo.** El marco teórico es coherente y bien anclado en literatura reciente, pero **el pipeline cubre solo ~60 % de lo que el propio marco predice y exige**. Las brechas no son por error: son tareas que nunca se implementaron porque la prioridad fue producir el primer experimento end-to-end. Cerrar esas brechas — sin añadir complejidad fuera de presupuesto — produce un paper publicable en Q2 Scopus.

## 9. Lo que el marco predice vs lo que el pipeline mide

| § del marco | Predicción / requisito teórico | ¿Implementado en el pipeline? | Evidencia (código / artefacto) | Estado |
|---|---|---|---|---|
| §2 (Geva 2021) | `W₁` ↔ claves, `W₂` ↔ valores; capas altas son semánticamente más ricas | Banco `up` extrae `W_up` rows como claves. `down_values` definido pero nunca ejecutado vía orquestador (Bug A) | `memory/banks.py`; `configs/experiments/exp06_qwen_value_space.yaml` | **Parcial** — solo key-space |
| §3 (Ramsauer 2021) | Energía con 4 términos: LSE + ½‖q‖² + log N/β + ½M² | Los 4 términos están en `LayerEnergyResult` y se computan correctamente | `metrics/energy.py:124` | **Sí** |
| §3, §12.1 | Los términos deben mantenerse separados ("colapsar a -lse pierde la semántica física") | Reportado en NB03 §7, pero con `normalize_query=True` el término cuadrático es constante → la descomposición es tautológica en producción | `hooks/capture.py:272-274`, `exp04_results/config.yaml:31` | **Roto en pruebas** |
| §3 | β controla la temperatura de recuperación; debe estar en el régimen de "detección" | `metrics/calibration.py` calibra β para `H_norm=0.55·log K` — heurística | `metrics/calibration.py` | **Heurístico, no anclado al teorema** |
| §4.1 (Gupta 2025) | La ortogonalidad de las claves de `W₁` cae para tokens de recall factual ⇒ alta interferencia ⇒ alucinación | Nunca medido. No hay módulo que compute ángulos por pares de `W_up` rows ni active-row spread | n/a | **Faltante crítico** |
| §4.1, §12.2 | Sonda secundaria en espacio de valores (`down_values`) es teóricamente complementaria | Definida pero silenciada por Bug A. Exp-06 nunca ha corrido | `pipeline/stages.py:346-349` (legacy path) | **Bloqueado** |
| §5.2 (Shazeer 2020) | En SwiGLU, el coeficiente de activación es `a_i = Swish(W_gate·x) · (W_up·x)`, no solo `W_up·x` | Pipeline ignora `W_gate`. La energía sobre `W_up` solo es una "media-clave" | `metrics/energy.py:101-103` | **Brecha teórica grande** |
| §6.1 (HalluField, Vu 2025) | Baseline directo de comparación: AUC 0.80–0.83 sobre logits, TriviaQA/BioASQ, LLaMA-2 7B | No ejecutado | n/a | **Faltante para paper** |
| §6.2 (ACT-ViT, Bar-Shalom 2025) | Techo supervisado: AUC 80–89 % sobre tensor de activaciones, TriviaQA, Qwen-7B | No ejecutado (es supervisado; opcional para citarlo) | n/a | **Cita suficiente** |
| §6.3 (INTRA, Vazhentsev 2026) | Baseline no supervisado más cercano: AUC ~77.7 % combinando estados internos | No ejecutado. Pipeline tiene activaciones pero no la capa de clasificación INTRA-style | n/a | **Faltante para paper** |
| §7.1 (Hofmann 2024) | Hopfield Boosting usa **dos** bancos: `E_b = −lse(β, X^Tξ) + lse(β, O^Tξ)` | Solo se computa el primer término. El segundo banco (outlier) no existe | `metrics/energy.py:113` | **Brecha teórica** |
| §8.1 (Concept Attractors) | Convergencia a atractor en ~85 % de profundidad (L24/28 Llama-8B → L30/36 Qwen-3B) | Empíricamente coincidente (Identity/Confusion peak L30 en NB04), pero **nunca enmarcado** como atractor | NB04 §5–6 | **Resultado a la vista, no leído** |
| §12.3 (KL drift) | KL entre `p^gen` y `p^pre` por capa | JS y Hellinger sí; KL no se reporta explícitamente | `metrics/divergences.py` | **Métrica diferente — defendible** |
| §12.4 (restringir a span de respuesta) | Análisis solo sobre tokens de respuesta generada | `analyze` tiene flag `--pool-over-answer` pero Exp-04 no la usó | `pipeline/analysis.py`, `cli/run.py:82` | **Disponible, no aplicado** |
| §11.2 | Sonda no supervisada; la regresión logística solo para calibración | Sonda actual es supervisada por etiquetas del judge; AUROC reportado es de regresión logística entrenada con esas etiquetas | NB06 §4 | **Inconsistencia de framing** |
| §11.5 | Baja ortogonalidad ⇒ alta entropía softmax (predicción geométrica) | Empíricamente: H_norm realizado = 0.4868 con β★=68.16. Compatible, pero no se ha correlacionado con HAL/OK | NB03 §3 | **Verificación parcial** |

**Lectura.** De 15 ítems verificables: **5 implementados completos, 4 implementados con caveats, 6 faltantes/bloqueados**. La razón AUROC ≈ 0.65 plateau es coherente con cubrir solo dos tercios del marco: los predictores más fuertes (gated-key activation, valor-space probe, segundo banco para boosting, ortogonalidad LAM) **no se han medido**.

## 10. Falencias críticas adicionales que el marco teórico expone

Las que el review original (§2.2) y el AUDIT_REPORT detectaron siguen vigentes. Las siguientes son **nuevas**, surgidas explícitamente del cruce con el marco teórico:

### 10.1 Falencia teórica T7 — La "media-clave" SwiGLU vs el banco real

El marco §5.2 reconoce textualmente:
> *El coeficiente de activación para la clave i es a_i = Swish(x W_gate^{:,i}) · (x W_up^{:,i}), que mezcla dos proyecciones distintas del input.*

El pipeline computa `scores = W_up · x` (`metrics/energy.py:102`). **Ignora `W_gate`.** Pero el modelo, internamente, retrieve memory rows con el coeficiente *gated*. Nuestra energía es entonces una energía sobre una *aproximación* del coeficiente de retrieval, no sobre el coeficiente real.

**Impacto cuantitativo esperado.** El gate de SwiGLU típicamente atenúa ~60 % de las filas (las que el modelo decide ignorar). Una sonda sobre el coeficiente *gated* concentra el softmax sobre las filas que el modelo *realmente* usa — debería producir señales más limpias.

**Test mínimo (no requiere GPU adicional si se guarda `W_gate` en stage 1).** Añadir un modo `bank="gated_key"` que retorne el coeficiente `Swish(W_gate·x) ⊙ (W_up·x)` como score. ~50 líneas en `metrics/energy.py` + `hooks/capture.py`. Ya hay `gate` en el `--bank` choices (`cli/run.py:25`) — el camino existe parcialmente.

### 10.2 Falencia teórica T8 — La ortogonalidad LAM no se mide en ninguna parte

Gupta 2025 (§4.1) es citado en el marco como justificación geométrica directa de por qué la energía MHN diverge en alucinaciones. La predicción: para tokens factual-recall, los patrones del banco están menos ortogonales ⇒ interferencia.

**Lo que falta.** Para los 50 samples con `_scores.npz` guardados, calcular por capa y por token generado:
1. Los top-K patrones activos (top-50 del softmax).
2. La cosine angular media entre esos top-K patrones.
3. Comparar la distribución de ángulos en HAL vs OK.

Es un notebook adicional de ~1 día de trabajo CPU. **Si la angular media de los top-K patrones es menor en HAL** → confirmación empírica directa de la predicción de Gupta sobre Qwen 2.5-3B. **Esto sería la novedad mecanística más fuerte del paper.**

### 10.3 Falencia teórica T9 — Hopfield Boosting (Hofmann 2024) cita energy_in − energy_out, computamos solo energy_in

El marco §7.1 cita la fórmula `E_b(ξ; X, O) = −lse(β, X^Tξ) + lse(β, O^Tξ)`. Nuestra implementación computa el primer término (memoria in-distribution) pero no el segundo (memoria outlier). El paper actual cita Hopfield Boosting como "justificación más directa" — pero usar solo medio score es metodológicamente débil.

**Construcción del banco outlier sin etiquetas (alineado con §11.2 del marco).** Para cada capa, construir `O = W_up @ M_hal` donde `M_hal` es la matriz de activaciones medias de N samples con alto `gen_drift` (proxy no supervisado de candidatos a HAL). Esto da una sonda *cuasi-no-supervisada* que se acerca al score de Hopfield Boosting sin requerir etiquetas en captura.

### 10.4 Falencia experimental F1 — Concept Attractors predicen L30 y nosotros lo *encontramos*, pero no lo *narramos*

Marco §8.1 cita Anón. 2025 (Concept Attractors): convergencia a atractor en capas altas (~85 % de profundidad). Para Qwen 2.5-3B con 36 capas, eso es **L≈30**.

NB04 §5–6 reporta: Identity/Confusion macro grupo, pico de AUROC ≈ 0.80 en **L30**, con identidad-binding propuesto como mecanismo.

**Esta es la coincidencia más fuerte del proyecto y la narrativa actual no la usa.** Una sección "Validación empírica de la predicción de Concept Attractors" — anclando nuestro pico en L30 al teorema de Banach del paper de atractores — convierte un resultado descriptivo en una *confirmación de predicción de marco*. Costo: cero experimentos nuevos; ~2 días de redacción.

### 10.5 Falencia experimental F2 — Calibración de β no anclada al teorema de Ramsauer

`metrics/calibration.py` apunta a `H_norm = 0.55·log K`, una elección heurística. Ramsauer 2021 da una condición de well-separation:

> Para que un patrón `ξ_i` sea recuperado sin interferencia con tolerancia ε, se requiere `β ≥ (2/d) · ln(2·N·ε⁻¹) / Δ_i²`, donde `Δ_i` es el margen al patrón más cercano.

**Es computable.** Con los bancos guardados en `banks.pt`, el margen mínimo `Δ_min` por capa se calcula en segundos. **La pregunta empírica a publicar**: ¿el β calibrado por `H_norm` coincide con el β well-separation?

- Si **sí**: la heurística está validada teóricamente — refuerza el paper.
- Si **no**: el β calibrado no garantiza recuperación bien-separada, lo que **explica mecanísticamente** por qué la señal es pequeña.

Costo: 1 día de CPU + 1 día de escritura. Resultado falsifica o confirma una elección de diseño central.

### 10.6 Falencia experimental F3 — HalluField e INTRA son baselines obligados, no opcionales

El marco §6.1 cita HalluField como "el trabajo más cercano metodológicamente" y §6.3 cita INTRA como "estado del arte no supervisado". Un paper que se posiciona contra estos baselines y **no los corre** no pasará revisión en Q2.

- **HalluField** requiere ≥2 forward passes con perturbación de temperatura. ~1 día de ingeniería + 0.5 día de captura sobre nuestros 500 samples (TruthfulQA) + 300 (TriviaQA). Se implementa fuera del pipeline existente en un script auxiliar; no requiere rediseño.
- **INTRA** requiere concatenar estados ocultos del residual stream (no MLP — ya lo capturamos para la sonda Hopfield) y un MLP clasificador ligero. Aprovecha exactamente la misma captura que ya hacemos pero leyendo `hidden_states` en lugar de `mlp_input`. ~2 días de ingeniería + entrenamiento.

Sin estos baselines el paper queda en *workshop*. Con ellos en Q2.

### 10.7 Falencia operacional O1 — `judge_model: gpt-5.4-mini` rompe la reproducibilidad antes de empezar

Confirmado en `labels/truthfulqa_*_label.json`. OpenAI no publica un modelo con ese identificador. La revisión de un editor Scopus pedirá el modelo exacto; un nombre que no resuelve es rechazo automático en la sección "reproducibility statement". Verificación + relabelling (o cambio a un modelo público como `gpt-4o-mini`) es ~$5 de API + 4 horas. **No es opcional.**

### 10.8 Falencia operacional O2 — `_baselines.json` no existe en `exp04_results/`

Tanto el AUDIT_REPORT original (línea 121) como el FULL_PIPELINE_REVIEW (§2.2 Issue D) asumen que `{id}_baselines.json` ya está escrito. **Verificado: no existe ninguno en `exp04_results/trajectories/`.** El pipeline tiene el código para emitirlos pero no se ejecutó en stage 2 de Exp-04.

Esto significa que el ítem "añadir baselines a NB06" no es "cargar archivos del disco" — es **re-correr stage 2** con el flag de baselines activado, o escribir un script post-hoc que pase la generación por el modelo otra vez para extraer `seq_logprob` y `token_entropy`. Costo realista: 0.5 día GPU + 0.5 día notebook. Sigue siendo barato pero **no es gratis**.

### 10.9 Falencia narrativa N1 — Inconsistencia entre "sonda no supervisada" del marco y "regresión logística supervisada" de los notebooks

El marco §11.2 dice textualmente:
> *HalluField y `hopfield_llm` son fundamentalmente no supervisados en su componente core — las métricas de energía se computan directamente de los pesos y activaciones del modelo.*

Pero NB06 reporta **AUROC de la regresión logística entrenada con etiquetas del judge** como cifra titular (0.665). Eso *es* supervisado. La cifra realmente no-supervisada es el AUROC de un solo score (mejor `ΔE` por capa: 0.611 en L21) — modesto pero honesto.

**Decisión narrativa requerida.** O bien:
1. El paper reporta `ΔE` por capa **sin probe** como contribución principal (no supervisado, ~0.61, débil pero honesto).
2. El paper reconoce que el probe es supervisado y se compara con baselines supervisados (donde INTRA y ACT-ViT son los rivales).

La mezcla actual es indefendible ante un revisor exigente.

## 11. Reformulación del paper

### 11.1 Título — tres opciones

Ordenadas de más conservador a más ambicioso:

1. **"Modern Hopfield Retrieval Energy Over SwiGLU MLP Banks: An Interpretable Probe for Hallucination Detection in LLMs"** — describe lo que se hace. Sobreviría review en Q3.
2. **"Retrieval-Concentration Features Complement Token Uncertainty for Hallucination Detection: A Hopfield-Inspired Mechanistic Probe"** — destaca la complementariedad. Defendible en Q2 si §10.6 está hecho.
3. **"Mid-to-Late MLP Memory Retrieval Diverges During Hallucination: A Per-Layer Hopfield Probe on Qwen-2.5"** — claim mecanístico fuerte anclado en L30. Defendible en Q2 si §10.4 + §10.6 están hechos.

Recomendación: **#2** como objetivo realista; **#3** si la replicación en Llama confirma el pico L30 y el value-space probe agrega señal.

### 11.2 Contribución central — tres frases

> *Proponemos una sonda mecanísticamente interpretable de detección de alucinaciones basada en la energía de Modern Hopfield Networks aplicada a las matrices `W_up` de las capas MLP (SwiGLU) de LLMs decoder-only. Empíricamente, sobre Qwen-2.5-3B en TruthfulQA y TriviaQA y replicado en Llama-3.2-3B, la sonda por-capa alcanza AUROC en held-out 0.65–0.72 y añade +0.04–0.06 sobre el baseline más fuerte (semantic entropy + token entropy + seq log-prob) con CI 95 % excluyendo cero. La señal se localiza en el término LSE de la energía y converge mecanísticamente en capas altas (~85 % de profundidad), consistente con la predicción de Concept Attractors.*

### 11.3 Posicionamiento contra trabajos cercanos

| Trabajo | Su métrica | Su AUC reportado | Nuestra ventaja diferencial |
|---|---|---|---|
| HalluField (Vu 2025) | Estabilidad termodinámica de logits bajo perturbación de T | 0.80–0.83 (LLaMA-2 7B, TriviaQA) | Localización **por capa**; un solo forward pass; señal mecanística interpretable |
| INTRA (Vazhentsev 2026) | Estados ocultos residuales + MLP clasificador | 0.777 (Llama-3.1) | Sonda MHN sobre MLP específicamente, con prior teórico Hopfield (no caja-negra) |
| ACT-ViT (Bar-Shalom 2025) | ViT sobre tensor `[L,N,d]` (supervisado) | 0.80–0.89 (Qwen-7B, TriviaQA) | No supervisado en el componente core; menor costo computacional; interpretable |
| SelfCheckGPT / Semantic Entropy | Variancia bajo muestreo | 0.70–0.85 | **Complementario** — nuestra señal vive en el espacio paramétrico; la semantic entropy en el espacio de salida |

**Honestidad esperada en el paper.** En TruthfulQA-greedy, nuestra cifra principal (0.65–0.67 layer-vector probe) está **por debajo** de HalluField/INTRA/ACT-ViT. La narrativa de venta debe ser:
1. **Interpretabilidad mecanística**: identificamos la capa, el término de energía y la familia de tópicos.
2. **Complementariedad**: la señal MHN es ortogonal a la incertidumbre token-level (paired bootstrap muestra +ΔAUROC ≥ 0.03 con CI ≠ 0).
3. **Validación de predicciones teóricas**: el pico L30 confirma Concept Attractors; la caída de ortogonalidad LAM (si se confirma en §10.2) confirma Gupta 2025.

## 12. Checklist de publicación Scopus — priorizado

Cada ítem etiquetado con (★) es **obligatorio** para Q3+ Scopus; (★★) eleva a Q2; (★★★) es stretch a Q1/Q2 conference.

### Bloque A — Correcciones obligatorias (cero experimentos GPU nuevos)

- [ ] **A1 (★) Commitear y verificar fix de Bug A** en `cli/run.py:180-181, 201-202`. Confirmar con dry-run de 1 sample que `banks_metadata.json` produce `M_used == 1.0` cuando `primary_probe.normalize=True`. *(~30 min)*
- [ ] **A2 (★) Decidir y documentar la referencia prefill**. Opción recomendada: usar **último token** consistentemente. Editar `pipeline/analysis.py:164` para usar `prefill_energy` no `prefill_energy_mean`. Editar `hooks/capture.py:312-323` para que `p_ref` sea la softmax del último token, no la media streaming. *(~2 h)*
- [ ] **A3 (★) Resolver `judge_model: gpt-5.4-mini`**. Identificar el modelo real usado; documentarlo en `labels/calibration.json` y reemitir labels si fue un alias inestable. *(~2 h + $5 API)*
- [ ] **A4 (★) Fix bug NB04 Hellinger** (`gen_js_layer` → `gen_hell_layer` en cell 11 de `04_category_signal_exp04.ipynb`). *(~5 min)*
- [ ] **A5 (★) Interpolación log-β en calibración** (`metrics/calibration.py:142-143`). *(~10 min)*
- [ ] **A6 (★) Reframe narrativo del término cuadrático**: NB03 §7 debe declarar explícitamente "bajo `normalize_query=True` ΔE_quad es constante por construcción; el reporte mecanístico requiere re-ejecución con `normalize_query=False`". *(~30 min)*

### Bloque B — Experimentación mínima viable (Q3 Scopus)

- [ ] **B1 (★) Re-correr Exp-04 con Bug A arreglado** y `primary_probe.normalize=true, primary_probe.normalize_query=false`. Producir `exp04b_results/`. *(~1 día A100 + 1 día NB re-corrida)*
- [ ] **B2 (★) Computar baselines de incertidumbre token-level** (`seq_logprob_mean`, `token_entropy_mean`, `token_entropy_max`) sobre los mismos prompts. Escribir `{id}_baselines.json`. Cargarlos en NB06 §2. *(~0.5 día GPU + 0.5 día notebook)*
- [ ] **B3 (★) Replicar en Llama-3.2-3B** vía `exp05_llama_chat_template_fix.yaml`. Verificar dirección del pico per-layer; comparar Identity/Confusion L≈30. *(~1 día A100)*
- [ ] **B4 (★) Restringir scores al span de respuesta** (`--pool-over-answer` en `analyze`). Marco §12.4. Generar `analysis_answer_span.json`. Comparar contra Exp-04 pooled. *(~2 h CPU)*
- [ ] **B5 (★) Held-out test split 70/15/15**. Entrenar sonda en 350, validar `C` en 75, reportar AUROC en 75 held-out. *(~1 día notebook)*
- [ ] **B6 (★) Nested-CV en lugar de 5-fold fijo** para la tabla titular. Usar `evaluation/probe.py:fit_logreg_probe`. *(~2 h)*
- [ ] **B7 (★) Verificación humana parcial de 100 samples**. 2 anotadores; Cohen's κ humano-judge; subset AUROC. *(~4 h anotación)*

**Salida del bloque B.** Un paper honesto, replicado en dos modelos, con baselines competitivas, held-out test, validación humana. Q3-Q4 Scopus. **6–10 días de trabajo total.**

### Bloque C — Experimentación competitiva (Q2 Scopus)

- [ ] **C1 (★★) Sondar con coeficiente gated SwiGLU** (`a_i = Swish(W_gate·x) ⊙ (W_up·x)`). Falencia T7 del §10.1. Implementar `bank="gated_key"` en `memory/banks.py` y `metrics/energy.py`. Re-correr Stage-2 sobre 500 TruthfulQA con el coeficiente *gated*. Comparar AUROC. *(~2 días)*
- [ ] **C2 (★★) Sonda value-space (`down_values`)** vía `exp06_qwen_value_space.yaml` con Bug A arreglado. Combinar key-space + value-space en un solo probe; reportar ganancia. Marco §4.1, §12.2. *(~1 día A100 + análisis)*
- [ ] **C3 (★★) Sampling-based semantic entropy (Farquhar 2024)**. N=10 completions por prompt a T=0.5; clustering NLI; entropía de clusters. Baseline obligatorio. *(~2 días GPU + análisis)*
- [ ] **C4 (★★) Implementar HalluField como baseline** (perturbación de T, 2 forward passes). Marco §6.1. *(~1 día código + 0.5 día captura)*
- [ ] **C5 (★★) Implementar INTRA como baseline** (estados residuales concatenados + clasificador). Marco §6.3. *(~2 días)*
- [ ] **C6 (★★) Medir ortogonalidad LAM (Gupta 2025)**. Por capa y por token de respuesta: ángulo medio entre top-K filas activas. Correlacionar con HAL/OK. Marco §4.1, §10.2 del adendo. *(~1 día CPU desde `_scores.npz`)*
- [ ] **C7 (★★) Anclar β a la condición well-separation de Ramsauer**. Computar `Δ_min` por capa desde `banks.pt`; comparar contra β★. Falencia F2 §10.5. *(~1 día CPU + redacción)*
- [ ] **C8 (★★) Transfer cross-dataset**: probe entrenado en TruthfulQA, evaluado en TriviaQA-300 (no-adversarial). Reportar caída AUROC. *(~1 día A100 + análisis)*
- [ ] **C9 (★★) Tabla comparativa estandarizada** vs HalluField, INTRA, Semantic Entropy, p_true en TruthfulQA *y* TriviaQA, dos modelos. AUROC + Cliff's δ + 95 % CI. *(redacción, ~1 día)*
- [ ] **C10 (★★) Pre-registración de la confirmatoria** en `docs/preregistration.md`: capa-banda L18–L30, hipótesis primary, secondary, criterio de éxito. Linkable desde el paper. *(~3 h)*

**Salida del bloque C.** Paper competitivo Q2 con tres baselines fuertes, dos modelos, dos datasets, mecanismo gated-key + value-space + LAM-orthogonality validados. **3–5 semanas de trabajo total tras el bloque B.**

### Bloque D — Stretch para Q1/Q2 conference (opcional)

- [ ] **D1 (★★★) Hopfield Boosting completo** con segundo banco outlier construido de no-supervisado-proxy (alta `gen_drift` ⇒ HAL-typical). Falencia T9 §10.3. *(~3 días)*
- [ ] **D2 (★★★) Intervención causal (steering)**. Hook que amplifica top-K softmax weights por γ ∈ {1.0, 1.5, 2.0, 3.0} en capas L18–L30 durante generación. Mide cambio en tasa HAL. Marco §11.5. Si γ↑ ⇒ HAL↓ → mecanismo causal validado. *(~2 semanas)*
- [ ] **D3 (★★★) Per-(layer, token) probe con group-L1**. 36×50 features × 500 samples; group-L1 across tokens, L2 within layers. *(~1 semana)*
- [ ] **D4 (★★★) Validación de Concept Attractors empíricamente**. Plot per-layer convergence rate sobre 4 dominios temáticos. ¿Identidad/Confusión muestra convergencia en L30 *visualmente*? Falencia F1 §10.4. *(~1 semana)*
- [ ] **D5 (★★★) Tercer modelo** (Mistral-7B o Phi-3-mini). Robustez del pico L≈85%-profundidad cross-architecture. *(~2 días)*
- [ ] **D6 (★★★) Multi-hop reasoning datasets** (HotpotQA). ¿La señal vive en tokens intermedios de razonamiento? *(~1 semana + adapter)*

**Salida del bloque D.** Paper con intervención causal + tres modelos + atractor-validation + multi-hop. Q1/Q2 NLP/ML conference. **2–3 meses de trabajo total tras el bloque C.**

### Bloque E — Higiene del repositorio (transversal a todos los bloques)

- [ ] **E1** Renombrar / consolidar reviews en `docs/`: el repo tiene `AUDIT_REPORT.md`, `FULL_PIPELINE_REVIEW.md`, este adendo, y `marco_teorico_hopfield_llm.md`. Crear `docs/README.md` con índice y orden de lectura. *(~1 h)*
- [ ] **E2** Eliminar el archivo `marco_teorico_hopfield_llm.md:Zone.Identifier` (basura de Windows). *(10 s)*
- [ ] **E3** Comitear `cli/run.py` (cambios uncommitted del fix Bug A). Mensaje: `Pass primary_probe/secondary_probe through run-experiment (fixes silent legacy-path fallback)`. *(~5 min)*
- [ ] **E4** Comitear los notebooks generados desde `scripts/local/gen_notebooks.py` (o gitignorar si son derivados). Decidir política. *(~30 min)*
- [ ] **E5** Subir el script `scripts/local/label_with_gpt.py` o purgarlo (actualmente untracked). *(~10 min)*
- [ ] **E6** Crear `configs/experiments/exp09_*.yaml` para el bloque B re-run (mantiene trazabilidad de la diferencia con Exp-04). *(~15 min)*
- [ ] **E7** Crear `docs/reproducibility.md` listando hashes de commit por experimento, semillas, modelos exactos (HuggingFace IDs), versión de `transformers`, `bitsandbytes`. Requisito Scopus. *(~2 h)*
- [ ] **E8** Test de regresión: un `tests/integration/test_exp04_smoke.py` que con 5 samples reproduce las cifras `analysis.json` esperadas (tolerancia ±0.005). Evita que futuras refactorizaciones rompan los números sin alarmar. *(~2 h)*
- [ ] **E9** Añadir a `CLAUDE.md` la nueva ruta: `bank='gated_key'` (cuando C1 esté hecho) y `--reference-mode {last_token, mean}` (cuando A2 esté hecho). *(~10 min)*

## 13. Recomendación temporal — ruta más corta a un paper Scopus

Si Brandon tiene 6–10 semanas reales de trabajo focalizado y quiere maximizar probabilidad de aceptación Q2:

```
Semana 1     ─── Bloque A (todo) + E1, E2, E3, E6
Semanas 2-3  ─── Bloque B (B1, B2, B3 en paralelo aprovechando GPU)
Semana 4     ─── Bloque B (B4-B7) + E7, E8
Semanas 5-7  ─── Bloque C, priorizando C1, C3, C6, C9
Semana 8     ─── C4, C5 (HalluField + INTRA)
Semanas 9-10 ─── Redacción del paper + figuras finales
```

Si solo hay 3 semanas: bloques A + B + (C1, C3, C6, C9) = paper Q3 Scopus, defendible.

Si solo hay 1 semana: bloque A + (B3 Llama replication) + reframe narrativo §11 = workshop paper, honesto.

## 14. Anexo — verificación numérica de los claims del marco teórico contra el repositorio

| Claim del marco | Cita | ¿Verifica en código/datos? | Notas |
|---|---|---|---|
| Energía = LSE + ½‖q‖² + log N/β + ½M² | §3.1 | ✓ | `metrics/energy.py:124` |
| K = 11008, D = 2048 (Qwen 2.5-3B) | implícito | ✓ | `banks_metadata.json` |
| Capas Qwen 2.5-3B = 36 | implícito | ✓ | `banks_metadata.json` |
| β★ = 68.16 para Exp-04 | implícito | ✓ | `calibration_beta.json` |
| `M_used == M_raw` en Exp-04 (Bug A inadvertido) | n/a | ✓ | `banks_metadata.json[0] = {M: 1.483, M_raw: 1.483}` |
| `prefill_energy` Y `prefill_energy_mean` ambos escritos | n/a | ✓ | Inspección NPZ |
| `_baselines.json` existe en Exp-04 | AUDIT §4.2 | ✗ | **No existe ningún archivo `*_baselines*`** en `exp04_results/trajectories/`. Audit incorrecto. |
| Judge = `gpt-4o-mini` o equivalente público | n/a | ✗ | Archivos label dicen `gpt-5.4-mini` |
| `cli/run.py` pasa `primary_probe` | fix de Bug A | ✓ | Líneas 180-181, 201-202 (uncommitted) |
| Pico L30 en Identity/Confusion | NB04 §5–6 | ✓ | Coincide con predicción Concept Attractors (~85 % profundidad) |
| Best single-layer AUROC ΔE = 0.611 en L21 | NB03 §6, NB06 §3 | ✓ | analysis.json no contiene per-layer AUROC; cifra viene de NB06 |
| AUROC layer-vector probe = 0.665 | NB06 §4 | △ | CV no-anidada con C=0.1 fijo (no nested) |

---

*Adendo redactado 2026-05-20 tras la integración del marco teórico. Próximo paso recomendado: Bloque A1–A6 esta semana; B1+B3 en paralelo la siguiente.*
