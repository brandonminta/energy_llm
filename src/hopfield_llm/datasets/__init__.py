"""Dataset adapters."""

from hopfield_llm.datasets.base import BaseHallucinationDataset, DataSample
from hopfield_llm.datasets.truthfulqa import TruthfulQADataset

__all__ = [
    "BaseHallucinationDataset",
    "DataSample",
    "TruthfulQADataset",
]
