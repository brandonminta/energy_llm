# docs/ — Reading Guide

This directory contains the research documentation for `hopfield_llm`.
Read in the order below to build context progressively.

---

## 1. Theoretical Framework

**`marco_teorico_hopfield_llm.md`**
The theoretical foundation: 11 source papers synthesised into the research programme.
Covers Modern Hopfield Networks, SwiGLU memory interpretation, Concept Attractors,
LAM orthogonality, and competitive positioning vs HalluField / INTRA / Semantic Entropy.
*Start here if you are new to the project.*

---

## 2. Architecture and Usage

**`../CLAUDE.md`** (project root)
Authoritative reference for the CLI, pipeline stages, config schema, artifact formats,
and critical implementation invariants.  Updated as new features are added.

**`architecture.md`**
System design overview: how the 6-stage pipeline fits together, data flow,
module responsibilities.

**`usage.md`**
End-to-end walkthrough with example commands for each pipeline stage.

**`quickref.md`**
One-page command cheat-sheet.

---

## 3. Experimental Record

**`../exp04_results/config.yaml`**
Exact configuration used for Exp-04 (Qwen2.5-3B, 500 TruthfulQA, β★≈68.16).
Canonical record of what was actually run.

**`reproducibility.md`**
Commit hashes, model IDs, library versions, and seeds for every completed
experiment.  Required for Scopus submission.

---

## 4. Quality and Publication Readiness

**`AUDIT_REPORT.md`**
Original pipeline audit (pre-publication review).  Identifies structural issues
in the first pass; some findings superseded by FULL_PIPELINE_REVIEW.md.

**`FULL_PIPELINE_REVIEW.md`**
Deep retrospective analysis against the theoretical framework.
§1–§8:  Pre-existing pipeline review.
§9:     Theory-vs-implementation gap table (15 rows).
§10:    Nine new critical falencias (T7 SwiGLU half-key, T8 LAM unmeasured, etc.).
§11:    Paper reformulation — title options, contribution statement, competitive
        positioning table.
§12:    Scopus publication checklist — Blocks A through E with time estimates.
§13:    Temporal roadmap (1 week → workshop, 3 weeks → Q3, 6–10 weeks → Q2).
§14:    Numeric verification annex.

**`preregistration.md`**
Pre-registered hypotheses (H1–H4) locked before exp04b and exp05 GPU results
are analysed.  Analysis plan, success criteria, exclusion criteria.
*Do not modify after exp04b is run.*

---

## 5. Deviation Log

**`analysis_deviations.md`** *(create when needed)*
Any deviation from `preregistration.md` during analysis must be documented here
with a rationale.  Required for open-science compliance.

---

## Key File Relationships

```
marco_teorico_hopfield_llm.md
    └─► FULL_PIPELINE_REVIEW.md  (gaps identified against the theory)
            └─► preregistration.md  (confirmatory hypotheses locked)
                    └─► notebooks/03–06  (analysis, results)
                            └─► paper draft  (§ citations back to preregistration)
```
