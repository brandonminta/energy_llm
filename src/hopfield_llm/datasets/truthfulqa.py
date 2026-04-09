"""TruthfulQA dataset adapter."""

from __future__ import annotations

import random
from typing import Iterator, Literal

from datasets import load_dataset

from hopfield_llm.datasets.base import BaseHallucinationDataset, DataSample
from hopfield_llm.utils.logging import get_logger

log = get_logger("datasets.truthfulqa")


class TruthfulQADataset(BaseHallucinationDataset):
    """Load TruthfulQA and produce paired factual/hallucinated DataSamples.

    For each question, a factual answer (label=1) and a hallucinated answer
    (label=0) are emitted as two separate DataSamples that share the same
    ``sample_id`` prefix.
    """

    def __init__(
        self,
        factual_strategy: Literal["best", "random_correct", "all_correct"] = "best",
        hallucinated_strategy: Literal["random_incorrect", "all_incorrect"] = "random_incorrect",
        categories: list[str] | None = None,
        max_samples: int | None = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        self._factual_strategy = factual_strategy
        self._hallucinated_strategy = hallucinated_strategy
        self._categories = categories
        self._max_samples = max_samples
        self._seed = seed
        self._samples: list[DataSample] = []
        self._load(verbose)

    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "truthfulqa"

    def __iter__(self) -> Iterator[DataSample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    # ------------------------------------------------------------------
    # Paired accessors (convenience for experiment 1)
    # ------------------------------------------------------------------

    def paired_samples(self) -> list[tuple[DataSample, DataSample]]:
        """Return (factual, hallucinated) pairs grouped by sample_id prefix."""
        by_prefix: dict[str, dict[int, DataSample]] = {}
        for s in self._samples:
            prefix = s.sample_id.rsplit("_", 1)[0]
            by_prefix.setdefault(prefix, {})[s.label] = s
        pairs = []
        for group in by_prefix.values():
            if 1 in group and 0 in group:
                pairs.append((group[1], group[0]))
        return pairs

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, verbose: bool) -> None:
        rng = random.Random(self._seed)
        raw = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        if self._categories is not None:
            raw = raw.filter(lambda x: x["category"] in self._categories)

        pairs: list[tuple[DataSample, DataSample]] = []
        skipped = 0

        for idx, row in enumerate(raw):
            question = row["question"].strip()
            best_answer = row["best_answer"].strip()
            correct_list = [a.strip() for a in row["correct_answers"] if a.strip()]
            incorrect_list = [a.strip() for a in row["incorrect_answers"] if a.strip()]
            category = row["category"]

            if not incorrect_list:
                skipped += 1
                continue
            if not correct_list and self._factual_strategy != "best":
                skipped += 1
                continue

            factual_candidates = self._resolve_factual(
                best_answer, correct_list, rng
            )
            hallucinated_candidates = self._resolve_hallucinated(
                incorrect_list, rng
            )

            for ans_f, src_f in factual_candidates:
                for ans_h, src_h in hallucinated_candidates:
                    sid = f"tqa_{idx}"
                    factual = DataSample(
                        sample_id=f"{sid}_factual",
                        question=question,
                        answer=ans_f,
                        label=1,
                        source_dataset="truthfulqa",
                        metadata={
                            "category": category,
                            "source_answer": src_f,
                            "original_idx": idx,
                        },
                    )
                    hallucinated = DataSample(
                        sample_id=f"{sid}_hallucinated",
                        question=question,
                        answer=ans_h,
                        label=0,
                        source_dataset="truthfulqa",
                        metadata={
                            "category": category,
                            "source_answer": src_h,
                            "original_idx": idx,
                        },
                    )
                    pairs.append((factual, hallucinated))

        if self._max_samples is not None:
            rng.shuffle(pairs)
            pairs = pairs[: self._max_samples]
            pairs.sort(key=lambda p: p[0].metadata["original_idx"])

        for f, h in pairs:
            self._samples.append(f)
            self._samples.append(h)

        if verbose:
            n_pairs = len(pairs)
            cats = sorted({s.metadata["category"] for s in self._samples})
            log.info("Loaded %d pairs (%d samples)", n_pairs, len(self._samples))
            log.info("Skipped: %d (missing incorrect/correct answers)", skipped)
            log.info("Categories: %d", len(cats))
            if self._categories is not None:
                log.info("Filter: %s", self._categories)

    def _resolve_factual(self, best, correct_list, rng):
        if self._factual_strategy == "best":
            return [(best, "best_answer")]
        if self._factual_strategy == "random_correct":
            return [(rng.choice(correct_list), "correct_answers")]
        if self._factual_strategy == "all_correct":
            return [(a, "correct_answers") for a in correct_list]
        raise ValueError(f"Unknown factual_strategy: '{self._factual_strategy}'")

    def _resolve_hallucinated(self, incorrect_list, rng):
        if self._hallucinated_strategy == "random_incorrect":
            return [(rng.choice(incorrect_list), "incorrect_answers[random]")]
        if self._hallucinated_strategy == "all_incorrect":
            return [
                (a, f"incorrect_answers[{i}]")
                for i, a in enumerate(incorrect_list)
            ]
        raise ValueError(
            f"Unknown hallucinated_strategy: '{self._hallucinated_strategy}'"
        )
