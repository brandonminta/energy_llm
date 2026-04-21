"""Abstract base class and common data structures for datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class DataSample:
    """Uniform sample structure produced by all dataset adapters.

    Fields are intentionally general to support future datasets, tasks, and
    evaluation modes without requiring a schema change.

    Core fields (required at construction):
        id, source, question, gold_answers

    Pipeline fields (filled at run time):
        model, metadata

    Labeling fields (M3, all optional — default to empty/None):
        generated_text, generation_config, correctness_score,
        is_hallucination, label_source, label_version,
        judge_raw_output, judge_model, judge_prompt_hash
    """
    id:           str            # "{source}_{original_id}"
    source:       str            # "triviaqa" | "nq" | "truthfulqa" | ...
    question:     str            # prompt / question text
    gold_answers: list[str]      # reference answer(s) for evaluation
    model:        str = ""       # filled at run time by the pipeline
    metadata:     dict[str, Any] = field(default_factory=dict)

    # M3 labeling fields — optional; existing code paths ignore them
    generated_text:    str              = ""
    generation_config: dict[str, Any]   = field(default_factory=dict)
    correctness_score: float | None     = None   # in [0, 1]
    is_hallucination:  bool  | None     = None   # derived from correctness_score
    label_source:      str              = ""     # e.g. "gpt4o_judge_v1", "f1_heuristic_v1"
    label_version:     str              = ""     # hash or versioned id of labeling code
    judge_raw_output:  str              = ""     # full judge output for re-labeling
    judge_model:       str              = ""
    judge_prompt_hash: str              = ""


class BaseDataset(ABC):
    """Minimal interface for dataset adapters.

    Adding a new dataset: subclass this, implement __iter__, __len__, and
    name.  The pipeline and evaluation code only depend on this interface.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[DataSample]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in artifact filenames, e.g. 'triviaqa'."""
        ...
