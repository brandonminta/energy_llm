"""TriviaQA dataset adapter."""

from __future__ import annotations

import random
from typing import Iterator

from datasets import load_dataset

from hopfield_llm.datasets.base import BaseDataset, DataSample
from hopfield_llm.utils.logging import get_logger

log = get_logger("datasets.triviaqa")


class TriviaQADataset(BaseDataset):
    """Load TriviaQA (rc config, validation split) and produce one DataSample per question.

    HuggingFace source: ``mandarjoshi/trivia_qa``, config ``rc``.
    Each sample's ``gold_answers`` is the list of normalised answer aliases.
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
        return "triviaqa"

    def __iter__(self) -> Iterator[DataSample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def _load(self, verbose: bool) -> None:
        rng = random.Random(self._seed)
        raw = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")

        samples: list[DataSample] = []
        skipped = 0

        for idx, row in enumerate(raw):
            question = row["question"].strip()
            # answer.aliases contains all valid answer strings
            aliases: list[str] = row.get("answer", {}).get("aliases", [])
            gold = [a.strip() for a in aliases if a.strip()]

            if not gold:
                skipped += 1
                continue

            samples.append(DataSample(
                id=f"triviaqa_{idx}",
                source="triviaqa",
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
                "TriviaQA: loaded %d samples (skipped %d with no gold answers)",
                len(self._samples), skipped,
            )
