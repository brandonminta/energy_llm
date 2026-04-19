"""TruthfulQA dataset adapter."""

from __future__ import annotations

import random
from typing import Iterator

from datasets import load_dataset

from hopfield_llm.datasets.base import BaseDataset, DataSample
from hopfield_llm.utils.logging import get_logger

log = get_logger("datasets.truthfulqa")


class TruthfulQADataset(BaseDataset):
    """Load TruthfulQA (generation split) and produce one DataSample per question.

    Each sample's ``gold_answers`` contains all correct answers for the question.
    """

    def __init__(
        self,
        max_samples: int | None = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        self._max_samples = max_samples
        self._seed = seed
        self._samples: list[DataSample] = []
        self._load(verbose)

    @property
    def name(self) -> str:
        return "truthfulqa"

    def __iter__(self) -> Iterator[DataSample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def _load(self, verbose: bool) -> None:
        rng = random.Random(self._seed)
        raw = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

        samples: list[DataSample] = []
        skipped = 0

        for idx, row in enumerate(raw):
            question = row["question"].strip()
            best = row["best_answer"].strip()
            correct = [a.strip() for a in row["correct_answers"] if a.strip()]

            gold = correct if correct else []
            if best and best not in gold:
                gold = [best] + gold

            if not gold:
                skipped += 1
                continue

            samples.append(DataSample(
                id=f"truthfulqa_{idx}",
                source="truthfulqa",
                question=question,
                gold_answers=gold,
            ))

        if self._max_samples is not None and self._max_samples < len(samples):
            rng.shuffle(samples)
            samples = samples[: self._max_samples]
            samples.sort(key=lambda s: int(s.id.split("_")[1]))

        self._samples = samples

        if verbose:
            log.info(
                "TruthfulQA: loaded %d samples (skipped %d with no gold answers)",
                len(self._samples), skipped,
            )
