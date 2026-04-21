"""Token-level answer span alignment utilities.

find_answer_span    — locate a gold answer substring in generated_text and return
                      the corresponding token-index range.
content_token_mask  — boolean mask excluding leading/trailing whitespace tokens.
extract_answer_features — mean-pool a [L, T] metric array over the answer span
                          (or content tokens as fallback).
"""

from __future__ import annotations

import numpy as np


def find_answer_span(
    generated_text: str,
    gold_answers: list[str],
    tokenizer,
) -> tuple[int, int] | None:
    """Find token indices [start, end) of the first gold answer in generated_text.

    Matching is case-insensitive on the first gold string that appears as a
    substring.  Character offsets are mapped to token indices via
    return_offsets_mapping=True.

    Args:
        generated_text: Model output string.
        gold_answers:   Candidate reference answers; first match wins.
        tokenizer:      HuggingFace tokenizer (must support offset_mapping).

    Returns:
        (tok_start, tok_end) with exclusive end, or None if no match found.
    """
    text_lower = generated_text.lower()
    char_start = char_end = None
    for gold in gold_answers:
        idx = text_lower.find(gold.lower())
        if idx >= 0:
            char_start = idx
            char_end = idx + len(gold)
            break
    if char_start is None:
        return None

    encoding = tokenizer(
        generated_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoding["offset_mapping"]

    tok_start: int | None = None
    tok_end: int | None = None
    for i, (s, e) in enumerate(offsets):
        if tok_start is None and s <= char_start < e:
            tok_start = i
        if s < char_end <= e:
            tok_end = i + 1

    if tok_start is None or tok_end is None:
        return None
    return (tok_start, tok_end)


def content_token_mask(
    generated_text: str,
    tokenizer,
) -> np.ndarray:
    """Boolean mask: True for tokens that map to non-whitespace character positions.

    Excludes tokens whose entire character span falls inside the leading or
    trailing whitespace of generated_text.

    Args:
        generated_text: Model output string.
        tokenizer:      HuggingFace tokenizer (must support offset_mapping).

    Returns:
        Boolean ndarray of shape [T].  All-False when text is blank.
    """
    encoding = tokenizer(
        generated_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoding["offset_mapping"]
    n = len(offsets)

    stripped = generated_text.strip()
    if not stripped:
        return np.zeros(n, dtype=bool)

    content_start = len(generated_text) - len(generated_text.lstrip())
    content_end = len(generated_text.rstrip())

    mask = np.array(
        [not (e <= content_start or s >= content_end) for s, e in offsets],
        dtype=bool,
    )
    return mask


def extract_answer_features(
    metrics: np.ndarray,
    span: tuple[int, int] | None,
    fallback_mask: np.ndarray,
) -> np.ndarray:
    """Mean-pool a [L, T] metric array over the answer span or content tokens.

    Args:
        metrics:       float array [L, T] — per-layer per-token energy / entropy etc.
        span:          (start, end) token index range from find_answer_span, or None.
        fallback_mask: [T] boolean mask from content_token_mask; used when span is None
                       or out-of-bounds.

    Returns:
        [L] array of nan-mean values over the selected token positions.
    """
    L, T = metrics.shape

    if span is not None:
        start, end = span
        if 0 <= start < end <= T:
            return np.nanmean(metrics[:, start:end], axis=1)

    if fallback_mask.any():
        return np.nanmean(metrics[:, fallback_mask], axis=1)

    return np.nanmean(metrics, axis=1)
