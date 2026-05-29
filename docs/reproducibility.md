# Reproducibility Record

Locked commit hashes, model IDs, library versions, seeds, and key hyperparameters
for every completed experiment.  Required for Scopus submission (open-science annex).

---

## Exp-04 — Qwen2.5-3B / TruthfulQA (primary)

| Field | Value |
|-------|-------|
| **Commit** | `db99c582d473ed9b2b09d636388b70b357f938d2` |
| **Branch** | `main` |
| **Config** | `configs/experiments/exp04_qwen_chat_template_fix.yaml` (snapshot in `outputs/exp04_qwen_truthfulqa/config.yaml`) |
| **Model** | `Qwen/Qwen2.5-3B-Instruct` (4-bit NF4 via bitsandbytes) |
| **Dataset** | TruthfulQA MC (HuggingFace `truthful_qa`, `multiple_choice` config), `max_samples=500`, `seed=42` |
| **β** | auto-calibrated from 30 calibration samples, converged at **β★ ≈ 68.16** |
| **Bank** | `up` (W_up rows, key-space probe) |
| **Hook target** | `mlp_input` |
| **normalize_query** | `True` (‼ see preregistration.md §caveats — E_quad ≡ 0 under unit-norm) |
| **Labels** | GPT-4o-mini judge (`gpt-4o-mini-2024-07-18`), heuristic F1 fallback |
| **Labeling** | same commit — `hopfield-llm label` (stage 5) |
| **Hallucination rate** | ≈ 33 % (gpt-4o-mini) |

### Software environment

| Package | Version |
|---------|---------|
| Python | 3.12 |
| torch | 2.5.1+cu121 |
| transformers | 5.0.0 |
| bitsandbytes | 0.46.1 |
| datasets | 4.5.0 |
| scikit-learn | 1.8.0 |
| numpy | 2.4.2 |

### Hardware

| Field | Value |
|-------|-------|
| GPU | NVIDIA RTX 2050 (4 GB VRAM) |
| RAM | 16 GB |
| OS | WSL2 / Linux 6.6 |

### Key artifacts (not tracked in git)

| Artifact | Location |
|----------|----------|
| Trajectory `.npz` + `.json` | `outputs/exp04_qwen_truthfulqa/trajectories/` (500 files) |
| Labels | `outputs/exp04_qwen_truthfulqa/labels/` (500 `_label.json` files) |
| Analysis | `outputs/exp04_qwen_truthfulqa/analysis.json` |
| Config snapshot | `outputs/exp04_qwen_truthfulqa/config.yaml` |

---

## Exp-04b — Qwen2.5-3B / normalize_query=False (pending)

| Field | Value |
|-------|-------|
| **Commit** | *pending* |
| **Config** | `configs/experiments/exp04b_qwen_normalize_query_false.yaml` |
| **Purpose** | Falsification check — replicate exp-04 with unnormalized queries so E_quad ≠ 0; pre-registered in `docs/preregistration.md` |
| **Status** | Awaiting GPU run |

---

## Exp-05 — Llama-3.2-1B-Instruct / TruthfulQA (pending)

| Field | Value |
|-------|-------|
| **Commit** | *pending* |
| **Config** | `configs/experiments/exp05_llama_chat_template_fix.yaml` |
| **Purpose** | Cross-model replication; pre-registered H4 (AUROC > 0.55) |
| **Status** | Awaiting GPU run |

---

## Exp-06 — Qwen2.5-3B / value-space probe (pending)

| Field | Value |
|-------|-------|
| **Config** | `configs/experiments/exp06_qwen_value_space.yaml` |
| **Bank** | `down_values` (rows of W_down.T) |
| **Status** | Awaiting GPU run |

---

## Exp-07b — Qwen2.5-3B / gated-key probe (pending)

| Field | Value |
|-------|-------|
| **Config** | `configs/experiments/exp07b_qwen_gated_key.yaml` |
| **Bank** | `gated_key` (W_gate ⊙ W_up, element-wise) |
| **Purpose** | Pre-registered H3 (gated_key AUROC ≥ W_up − 0.03) |
| **Status** | Awaiting GPU run |

---

## Analysis deviations

Any deviation from `preregistration.md` during analysis of exp-04b / exp-05 confirmatory
results **must** be documented in `docs/analysis_deviations.md` before the analysis notebook
is run, per open-science protocol.
