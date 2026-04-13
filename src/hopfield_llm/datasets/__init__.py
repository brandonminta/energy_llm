"""Dataset adapters."""

from hopfield_llm.datasets.base import BaseDataset, DataSample
from hopfield_llm.datasets.nq import NaturalQuestionsDataset
from hopfield_llm.datasets.triviaqa import TriviaQADataset
from hopfield_llm.datasets.truthfulqa import TruthfulQADataset

__all__ = [
    "BaseDataset",
    "DataSample",
    "NaturalQuestionsDataset",
    "TriviaQADataset",
    "TruthfulQADataset",
]
