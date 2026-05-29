# docs/ — Reading Guide

This directory contains the research documentation for `hopfield_llm`.
Read in the order below to build context progressively.

---

## 1. Theoretical Framework

**`fundamentals.tex`** — the authoritative thesis chapter (Theoretical
Framework): Transformer/SwiGLU internals, FFN-as-memory, Modern Hopfield
retrieval, and the information-theoretic divergences. *Start here.*

**`methodology.tex`** — the authoritative thesis chapter (Methodology): the
problem, the probe design, the six-stage pipeline, the proposed model, the
analysis method, and the final experimental configuration.

**`marco_teorico_hopfield_llm.md`** — earlier Spanish synthesis of the source
papers; kept as a reading aid. *Superseded by `fundamentals.tex` for the final
framework.*

---

## 2. Architecture and Usage

**`../CLAUDE.md`** (project root) — authoritative reference for the CLI,
pipeline stages, config schema, artifact formats, and implementation invariants.

**`architecture.md`** — system design: how the 6-stage pipeline fits together,
data flow, module responsibilities.

**`usage.md`** — end-to-end walkthrough with example commands per stage.

**`quickref.md`** — one-page command cheat-sheet.

**`setup.md`** — local + HPC installation.

---

## 3. Experimental Record

**`findings.md`** — consolidated empirical findings of the first campaign
(Exp-04) and the design rationale they motivated. Supersedes the former
`AUDIT_REPORT.md` / `FULL_PIPELINE_REVIEW.md` working notes.

**`preregistration.md`** — pre-registered hypotheses (H1–H4) and the locked
analysis plan, success criteria, and exclusion criteria. *Do not modify after
the corresponding GPU runs are analysed.*

**`reproducibility.md`** — commit hashes, model IDs, library versions, and seeds
for every completed experiment. The completed Exp-04 artefacts live under
`outputs/exp04_qwen_truthfulqa/` (gitignored).

**`analysis_deviations.md`** *(create when needed)* — any deviation from
`preregistration.md` during analysis, documented with a rationale, for
open-science compliance.

---

## Key File Relationships

```
fundamentals.tex / methodology.tex   (authoritative theory + method)
    └─► preregistration.md           (confirmatory hypotheses locked)
            └─► configs/experiments/final_*.yaml + exp0*.yaml  (runs)
                    └─► findings.md + notebooks/03–06          (results)
                            └─► reproducibility.md             (provenance)
```
