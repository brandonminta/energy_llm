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
    """
    id:           str            # "{source}_{original_id}"
    source:       str            # "triviaqa" | "nq" | "truthfulqa" | ...
    question:     str            # prompt / question text
    gold_answers: list[str]      # reference answer(s) for evaluation
    model:        str = ""       # filled at run time by the pipeline
    metadata:     dict[str, Any] = field(default_factory=dict)
    # Future fields (judge scores, labels, etc.) go in metadata to avoid
    # frequent schema changes.


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
