"""Dataset adapters and sharding (subsec:datasets).

- TruthfulQA: generation split, all 817 questions, category metadata kept for
  the descriptive per-category breakdown.
- TriviaQA: ``rc.nocontext`` validation split, ``num_samples`` questions
  drawn at the fixed seed.

Sample ids are stable functions of the source row index, so sharded workers
and resumed runs always agree on identity.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Sample:
    sample_id: str
    question: str
    gold_answers: list[str] = field(default_factory=list)
    category: str | None = None
    source: str = ""


def load_truthfulqa(num_samples: int | None = None, seed: int = 42) -> list[Sample]:
    from datasets import load_dataset
    raw = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    samples: list[Sample] = []
    for idx, row in enumerate(raw):
        correct = [a.strip() for a in row["correct_answers"] if a.strip()]
        best = (row.get("best_answer") or "").strip()
        gold = ([best] if best and best not in correct else []) + correct
        if not gold:
            continue
        cat = row.get("category")
        samples.append(Sample(
            sample_id=f"truthfulqa_{idx}",
            question=row["question"].strip(),
            gold_answers=gold,
            category=cat.strip() if isinstance(cat, str) and cat.strip() else None,
            source="truthfulqa",
        ))
    return _subsample(samples, num_samples, seed)


def load_triviaqa(num_samples: int | None = 500, seed: int = 42) -> list[Sample]:
    from datasets import load_dataset
    raw = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    samples: list[Sample] = []
    for idx, row in enumerate(raw):
        aliases = [a.strip() for a in row.get("answer", {}).get("aliases", []) if a.strip()]
        if not aliases:
            continue
        samples.append(Sample(
            sample_id=f"triviaqa_{idx}",
            question=row["question"].strip(),
            gold_answers=aliases,
            source="triviaqa",
        ))
    return _subsample(samples, num_samples, seed)


def _subsample(samples: list[Sample], num_samples: int | None, seed: int) -> list[Sample]:
    if num_samples is not None and num_samples < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, num_samples)
        samples.sort(key=lambda s: int(s.sample_id.rsplit("_", 1)[1]))
    return samples


def load_samples(name: str, num_samples: int | None = None, seed: int = 42) -> list[Sample]:
    loaders = {"truthfulqa": load_truthfulqa, "triviaqa": load_triviaqa}
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choices: {sorted(loaders)}")
    return loaders[name](num_samples=num_samples, seed=seed)


def shard(samples: list[Sample], shard_index: int, num_shards: int) -> list[Sample]:
    """Non-overlapping stride slice ``samples[shard_index::num_shards]`` for
    SLURM array jobs (subsec:stage2). Stage 4 is order-independent."""
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index {shard_index} out of range for {num_shards} shards")
    return samples[shard_index::num_shards]


def calibration_sample_ids(samples: list[Sample], n_calibration: int, seed: int = 42) -> list[str]:
    """Deterministic warm-up subset for beta calibration (N_c questions).

    These ids are recorded in the calibration artifact and forced into the
    probe's train split, which enforces 'calibration restricted to the
    training split' (subsec:beta_calibration) by construction.
    """
    rng = random.Random(seed)
    ids = [s.sample_id for s in samples]
    return sorted(rng.sample(ids, min(n_calibration, len(ids))))
