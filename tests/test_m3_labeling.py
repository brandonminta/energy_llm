"""Tests for M3: hallucination labeling.

Case 1: Heuristic F1 on exact match returns 1.0.
Case 2: Heuristic F1 on no overlap returns 0.0 and is_hallucination=True.
Case 3: Judge output parsing robustness.
Case 4: Cohen's kappa on mock calibration data.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hopfield_llm.datasets.base import DataSample
from hopfield_llm.labeling.heuristic import (
    compute_cohen_kappa,
    compute_f1_score,
    label_sample_heuristic,
)
from hopfield_llm.labeling.llm_judge import (
    LLMJudge,
    _parse_judge_output,
    label_sample_with_judge,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_sample(**kwargs) -> DataSample:
    defaults = dict(
        id="test_0",
        source="test",
        question="What is the capital of France?",
        gold_answers=["Paris", "Paris, France"],
        generated_text="",
    )
    defaults.update(kwargs)
    return DataSample(**defaults)


# ------------------------------------------------------------------
# Case 1: Exact match → F1 = 1.0, not a hallucination
# ------------------------------------------------------------------


def test_heuristic_exact_match_f1():
    sample = _make_sample(generated_text="Paris")
    label  = label_sample_heuristic(sample)
    assert label.correctness_score == pytest.approx(1.0)
    assert label.is_hallucination is False
    assert label.label_source == "f1_heuristic_v1"
    assert label.label_version != ""


def test_heuristic_does_not_mutate_input():
    sample = _make_sample(generated_text="Paris")
    label  = label_sample_heuristic(sample)
    assert sample.correctness_score is None
    assert sample.is_hallucination  is None
    assert label is not sample


def test_heuristic_article_normalisation():
    sample = _make_sample(
        generated_text="the Paris",
        gold_answers=["Paris"],
    )
    label = label_sample_heuristic(sample)
    assert label.correctness_score == pytest.approx(1.0)


def test_heuristic_multi_gold_best_match():
    """Returns max F1 across all gold answers."""
    sample = _make_sample(
        generated_text="city of Paris",
        gold_answers=["London", "Paris"],
    )
    label = label_sample_heuristic(sample)
    assert label.correctness_score > 0.0


# ------------------------------------------------------------------
# Case 2: No overlap → F1 = 0.0, is_hallucination=True
# ------------------------------------------------------------------


def test_heuristic_no_overlap():
    sample = _make_sample(
        generated_text="Berlin",
        gold_answers=["Paris", "Paris, France"],
    )
    label = label_sample_heuristic(sample)
    assert label.correctness_score == pytest.approx(0.0)
    assert label.is_hallucination is True


def test_heuristic_empty_generated_text():
    sample = _make_sample(generated_text="")
    label  = label_sample_heuristic(sample)
    assert label.correctness_score == pytest.approx(0.0)
    assert label.is_hallucination is True


def test_heuristic_threshold_boundary():
    """F1 exactly at threshold is still a hallucination (strict <)."""
    sample = _make_sample(
        generated_text="Paris",
        gold_answers=["Paris"],
    )
    # F1 will be 1.0; with threshold=1.01 it should be flagged
    label = label_sample_heuristic(sample, threshold=1.01)
    assert label.is_hallucination is True


# ------------------------------------------------------------------
# Case 3: Judge output parsing robustness
# ------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("[Output]: 1",                    1),
    ("Output: 0",                      0),
    ("1",                              1),
    ("reasoning... so the answer is: 0", 0),
    ("  0  ",                          0),
    ("The answer is correct. 1",       1),
    ("score=0",                        0),
])
def test_parse_judge_output_valid(raw, expected):
    assert _parse_judge_output(raw) == expected


def test_parse_judge_output_parse_failed():
    assert _parse_judge_output("unparseable garbage xyz") is None


def test_parse_judge_output_empty():
    assert _parse_judge_output("") is None


def test_parse_judge_output_multi_line_last_wins():
    raw = "First line: 1\nSecond line: 0"
    assert _parse_judge_output(raw) == 0


def test_label_sample_judge_success_correct():
    mock_judge = MagicMock(spec=LLMJudge)
    mock_judge.model_id    = "gpt-4o-mini"
    mock_judge._prompt_hash = "abc123"
    mock_judge.judge.return_value = {"raw": "[Output]: 1", "parsed": 1, "success": True}

    sample = _make_sample(generated_text="Paris")
    result = label_sample_with_judge(sample, mock_judge)

    assert result.is_hallucination is False
    assert result.correctness_score == pytest.approx(1.0)
    assert result.judge_raw_output == "[Output]: 1"
    assert "parse_failed" not in result.label_source
    assert "error"        not in result.label_source


def test_label_sample_judge_success_incorrect():
    mock_judge = MagicMock(spec=LLMJudge)
    mock_judge.model_id    = "gpt-4o-mini"
    mock_judge._prompt_hash = "abc123"
    mock_judge.judge.return_value = {"raw": "Output: 0", "parsed": 0, "success": True}

    sample = _make_sample(generated_text="Berlin")
    result = label_sample_with_judge(sample, mock_judge)

    assert result.is_hallucination is True
    assert result.correctness_score == pytest.approx(0.0)


def test_label_sample_judge_parse_failure_leaves_none():
    mock_judge = MagicMock(spec=LLMJudge)
    mock_judge.model_id    = "gpt-4o-mini"
    mock_judge._prompt_hash = "abc123"
    # parsed=None means the judge's internal parser already failed
    mock_judge.judge.return_value = {
        "raw": "unparseable garbage", "parsed": None, "success": True
    }

    sample = _make_sample(generated_text="some answer")
    result = label_sample_with_judge(sample, mock_judge)

    assert result.is_hallucination is None, \
        "is_hallucination must remain None on parse failure"
    assert "parse_failed" in result.label_source
    assert result.judge_raw_output == "unparseable garbage"


def test_label_sample_judge_error_leaves_none():
    mock_judge = MagicMock(spec=LLMJudge)
    mock_judge.model_id    = "gpt-4o-mini"
    mock_judge._prompt_hash = "abc123"
    mock_judge.judge.side_effect = RuntimeError("API unreachable")

    sample = _make_sample(generated_text="some answer")
    result = label_sample_with_judge(sample, mock_judge)

    assert result.is_hallucination is None, \
        "is_hallucination must remain None when judge raises"
    assert "error" in result.label_source


# ------------------------------------------------------------------
# Case 4: Cohen's kappa on mock calibration data
# ------------------------------------------------------------------


def test_cohen_kappa_perfect_agreement():
    labels = [True, False, True, True, False]
    assert compute_cohen_kappa(labels, labels) == pytest.approx(1.0, abs=1e-9)


def test_cohen_kappa_known_value():
    """Analytical example with known kappa."""
    # po = 4/6, a_pos = 3/6 = 0.5, b_pos = 3/6 = 0.5
    # pe = 0.5*0.5 + 0.5*0.5 = 0.5
    # kappa = (4/6 - 0.5) / (1 - 0.5) = (0.6667 - 0.5) / 0.5 ≈ 0.3333
    labels_a = [True,  True,  False, False, True,  False]
    labels_b = [True,  False, True,  False, True,  False]
    kappa = compute_cohen_kappa(labels_a, labels_b)
    assert abs(kappa - 1 / 3) < 0.05, f"Expected ~0.333, got {kappa:.4f}"


def test_cohen_kappa_symmetric():
    labels_a = [True, True, False, True, False, False]
    labels_b = [True, False, False, True, True, False]
    assert compute_cohen_kappa(labels_a, labels_b) == pytest.approx(
        compute_cohen_kappa(labels_b, labels_a)
    )


def test_cohen_kappa_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        compute_cohen_kappa([True, False], [True])


def test_cohen_kappa_int_labels():
    """Accepts 0/1 int labels as well as bool."""
    labels_a = [1, 0, 1, 0]
    labels_b = [1, 0, 0, 1]
    result = compute_cohen_kappa(labels_a, labels_b)
    assert -1.0 <= result <= 1.0


# ------------------------------------------------------------------
# DataSample extension sanity checks
# ------------------------------------------------------------------


def test_datasample_new_fields_have_defaults():
    """Existing construction signatures must still work."""
    s = DataSample(
        id="test_0",
        source="triviaqa",
        question="What?",
        gold_answers=["Answer"],
    )
    assert s.generated_text    == ""
    assert s.generation_config == {}
    assert s.correctness_score is None
    assert s.is_hallucination  is None
    assert s.label_source      == ""
    assert s.judge_raw_output  == ""


def test_datasample_labeling_fields_settable():
    s = DataSample(
        id="x", source="test", question="Q?", gold_answers=["A"],
        generated_text="answer",
        correctness_score=0.8,
        is_hallucination=False,
        label_source="f1_heuristic_v1",
    )
    assert s.correctness_score == pytest.approx(0.8)
    assert s.is_hallucination  is False
