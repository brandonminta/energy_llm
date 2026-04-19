"""Abstract base class and common data structures for datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class DataSample:
    """Uniform sample structure produced by all dataset adapters."""
    id: str               # "{source}_{original_id}"
    source: str           # "triviaqa" | "nq" | "truthfulqa"
    question: str
    gold_answers: list[str]
    model: str = ""       # filled in by the pipeline at run time


class BaseDataset(ABC):
    """Minimal interface for dataset adapters."""

    @abstractmethod
    def __iter__(self) -> Iterator[DataSample]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'triviaqa'."""
        ...
