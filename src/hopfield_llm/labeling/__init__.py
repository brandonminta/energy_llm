"""Hallucination labeling for hopfield_llm.

Two labelers are provided:

heuristic  — SQuAD F1 against gold answers; fast, no API calls.
llm_judge  — binary 0/1 judgment from an external LLM; higher accuracy.

Typical usage:

    from hopfield_llm.labeling.heuristic import label_sample_heuristic
    from hopfield_llm.labeling.llm_judge import LLMJudge, label_sample_with_judge
"""

from hopfield_llm.labeling.heuristic import (
    compute_cohen_kappa,
    compute_f1_score,
    label_sample_heuristic,
)
from hopfield_llm.labeling.llm_judge import (
    LLMJudge,
    build_judge_prompt,
    label_sample_with_judge,
)

__all__ = [
    "compute_f1_score",
    "compute_cohen_kappa",
    "label_sample_heuristic",
    "LLMJudge",
    "build_judge_prompt",
    "label_sample_with_judge",
]
