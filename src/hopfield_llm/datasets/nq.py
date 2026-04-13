"""Natural Questions dataset adapter."""

from __future__ import annotations

import random
from typing import Iterator

from datasets import load_dataset

from hopfield_llm.datasets.base import BaseDataset, DataSample
from hopfield_llm.utils.logging import get_logger

log = get_logger("datasets.nq")


def _extract_short_answers(row: dict) -> list[str]:
    """Pull distinct short-answer strings from an NQ row.

    NQ annotations are stored as a struct-of-arrays:
        row['annotations']['short_answers']  →  list[list[dict]]
    where the outer list is over annotations and the inner list is over
    individual short answers within that annotation.

    Each short-answer dict may carry a ``text`` field directly.  If that is
    absent or empty, the answer text is reconstructed from document tokens.
    """
    try:
        ann = row["annotations"]
        # struct-of-arrays: short_answers is a list (one entry per annotation)
        sa_lists = ann.get("short_answers", [])
        seen: set[str] = set()
        texts: list[str] = []

        for sa_list in sa_lists:
            if not sa_list:
                continue
            for sa in sa_list:
                # Prefer the pre-computed text field
                text = sa.get("text", "").strip()
                if not text:
                    # Fall back: extract from document tokens
                    start = sa.get("start_token")
                    end = sa.get("end_token")
                    if start is not None and end is not None:
                        doc_tokens = row["document"]["tokens"]
                        toks = doc_tokens["token"][start:end]
                        html = doc_tokens["is_html"][start:end]
                        text = " ".join(t for t, h in zip(toks, html) if not h).strip()
                if text and text not in seen:
                    seen.add(text)
                    texts.append(text)

        return texts
    except (KeyError, IndexError, TypeError):
        return []


class NaturalQuestionsDataset(BaseDataset):
    """Load Natural Questions (validation split) and produce one DataSample per question.

    HuggingFace source: ``google-research-datasets/natural_questions``.
    Only questions with at least one short answer are kept.
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
        return "nq"

    def __iter__(self) -> Iterator[DataSample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def _load(self, verbose: bool) -> None:
        rng = random.Random(self._seed)
        raw = load_dataset(
            "google-research-datasets/natural_questions",
            split="validation",
        )

        samples: list[DataSample] = []
        skipped = 0

        for idx, row in enumerate(raw):
            question = row["question"]["text"].strip()
            gold = _extract_short_answers(row)

            if not gold:
                skipped += 1
                continue

            samples.append(DataSample(
                id=f"nq_{idx}",
                source="nq",
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
                "NaturalQuestions: loaded %d samples (skipped %d with no short answers)",
                len(self._samples), skipped,
            )
