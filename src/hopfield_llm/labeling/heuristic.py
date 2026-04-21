"""F1-based heuristic labeler and calibration utilities.

compute_f1_score  — SQuAD-style token F1 of prediction vs gold answers.
label_sample_heuristic — stamps a DataSample with F1-derived correctness label.
compute_cohen_kappa    — inter-annotator agreement for calibration.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import string
from collections import Counter
from dataclasses import replace

from hopfield_llm.datasets.base import DataSample


# ------------------------------------------------------------------
# Text normalisation (SQuAD convention)
# ------------------------------------------------------------------

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase, strip articles, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = _ARTICLE_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ------------------------------------------------------------------
# SQuAD-style token F1
# ------------------------------------------------------------------


def compute_f1_score(prediction: str, gold_answers: list[str]) -> float:
    """Compute SQuAD-style token F1 of prediction against a list of gold answers.

    Both sides are normalised (lowercase, strip articles, remove punctuation,
    collapse whitespace) before tokenisation.  Returns the maximum F1 across
    all gold strings.

    Returns 0.0 when the prediction or all gold answers normalise to empty.
    """
    pred_norm = _normalize(prediction)
    pred_tokens = pred_norm.split()
    if not pred_tokens:
        return 0.0

    best = 0.0
    for gold in gold_answers:
        gold_norm = _normalize(gold)
        gold_tokens = gold_norm.split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        precision = n_common / len(pred_tokens)
        recall    = n_common / len(gold_tokens)
        f1        = (2 * precision * recall) / (precision + recall)
        best = max(best, f1)

    return best


# ------------------------------------------------------------------
# Label version — derived from this function's source at import time
# ------------------------------------------------------------------


def _compute_label_version() -> str:
    src = inspect.getsource(compute_f1_score)
    return "f1v1_" + hashlib.md5(src.encode()).hexdigest()[:8]


_F1_LABEL_VERSION: str = _compute_label_version()


# ------------------------------------------------------------------
# Per-sample labeler
# ------------------------------------------------------------------


def label_sample_heuristic(
    sample: DataSample,
    threshold: float = 0.3,
) -> DataSample:
    """Return a new DataSample labelled with SQuAD F1 correctness score.

    Does NOT mutate the input sample.

    Args:
        sample:    DataSample with generated_text and gold_answers set.
        threshold: Samples with F1 < threshold are marked as hallucinations.

    Returns:
        New DataSample with correctness_score, is_hallucination, label_source,
        and label_version populated.
    """
    max_f1 = compute_f1_score(sample.generated_text, sample.gold_answers)
    return replace(
        sample,
        correctness_score=max_f1,
        is_hallucination=(max_f1 < threshold),
        label_source="f1_heuristic_v1",
        label_version=_F1_LABEL_VERSION,
    )


# ------------------------------------------------------------------
# Inter-annotator calibration
# ------------------------------------------------------------------


def compute_cohen_kappa(
    labels_a: list[bool | int],
    labels_b: list[bool | int],
) -> float:
    """Compute Cohen's kappa between two binary label sequences.

    Args:
        labels_a: First rater's binary labels (bool or 0/1 int).
        labels_b: Second rater's binary labels (same length).

    Returns:
        Cohen's kappa in [-1, 1].  Returns 0.0 when expected agreement is 1.0.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Label lists must have the same length: "
            f"{len(labels_a)} vs {len(labels_b)}"
        )
    n = len(labels_a)
    if n == 0:
        raise ValueError("Cannot compute kappa on empty label lists")

    po = sum(int(a) == int(b) for a, b in zip(labels_a, labels_b)) / n
    a_pos = sum(int(x) for x in labels_a) / n
    b_pos = sum(int(x) for x in labels_b) / n
    pe    = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)

    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1 - pe)
