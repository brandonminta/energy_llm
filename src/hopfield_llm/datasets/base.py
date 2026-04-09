"""Abstract base class and common data structures for hallucination datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class DataSample:
    """Uniform sample structure that all datasets must produce."""
    sample_id: str
    question: str
    answer: str
    label: int              # 1 = factual, 0 = hallucinated
    source_dataset: str
    metadata: dict = field(default_factory=dict)


class BaseHallucinationDataset(ABC):
    """Every dataset adapter must implement this interface.

    Yields paired samples: for each question, one factual (label=1) and one
    hallucinated (label=0) sample sharing the same ``sample_id`` prefix.

    Minimal implementation example::

        class MyDataset(BaseHallucinationDataset):
            def __init__(self, path):
                self._samples = load_my_data(path)

            def __iter__(self):
                return iter(self._samples)

            def __len__(self):
                return len(self._samples)

            @property
            def name(self):
                return "my_dataset"
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
        """Short identifier for this dataset (e.g. 'truthfulqa')."""
        ...
