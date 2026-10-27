"""Stage 3 — labelling via an external judge (subsec:stage3_labeling).

An OpenAI-compatible chat-completions client queries the judge
(``gpt-4o-mini``) with ONE locked prompt template — a constant in this
module, never edited per run — presenting the question, the generated
answer, and the gold answers, and requesting a binary supported/unsupported
verdict. The raw judge response is archived verbatim next to the parsed
label, so the parsing rule can be audited and re-applied without re-querying
the service. Labels live in their own directory, separate from the
trajectories, so relabelling is CPU-only.

Resumable: already-labelled samples are skipped. Retries with exponential
backoff. Judge agreement against the human-labelled subset is reported as
Cohen's kappa with threshold kappa >= 0.6 (landis1977measurement).
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from pathlib import Path
from typing import Any

from energy_llm.io_artifacts import SampleManifest, append_jsonl, load_json, save_json_atomic

log = logging.getLogger("energy_llm.labeling")

# LOCKED judge prompt template (sec:problem_description). Do not edit per
# run: every label artifact records its hash so any drift is detectable.
JUDGE_PROMPT_TEMPLATE = """\
You are grading whether a model's answer to a question is factually supported by a set of gold reference answers.

Question: {question}

Gold answers (the answer is correct if it matches ANY of these):
{gold_answers}

Model answer: {answer}

Decide whether the model answer is SUPPORTED or UNSUPPORTED by the gold answers. SUPPORTED means the model answer states, paraphrases, or is entailed by at least one gold answer. UNSUPPORTED means the model answer contradicts the gold answers, asserts something the gold answers do not support, or fails to answer the question.

Reply with exactly one word: SUPPORTED or UNSUPPORTED."""

JUDGE_PROMPT_SHA256 = hashlib.sha256(JUDGE_PROMPT_TEMPLATE.encode()).hexdigest()
LABEL_SUFFIX = "_label.json"


def render_judge_prompt(question: str, answer: str, gold_answers: list[str]) -> str:
    gold = "\n".join(f"- {a}" for a in gold_answers)
    return JUDGE_PROMPT_TEMPLATE.format(question=question, gold_answers=gold, answer=answer)


def parse_verdict(raw: str) -> int | None:
    """Parse the binary verdict; 1 = hallucination (UNSUPPORTED).

    Checks UNSUPPORTED before SUPPORTED because the former contains the
    latter as a substring. Returns None when unparseable.
    """
    text = raw.strip().upper()
    if "UNSUPPORTED" in text:
        return 1
    if "SUPPORTED" in text:
        return 0
    return None


def make_judge_client(api_base: str | None = None):
    """OpenAI-compatible chat-completions client (api key from OPENAI_API_KEY)."""
    from openai import OpenAI
    return OpenAI(base_url=api_base) if api_base else OpenAI()


def query_judge(
    client: Any,
    judge_model: str,
    prompt: str,
    max_retries: int = 6,
    base_delay: float = 2.0,
) -> str:
    """One judged completion with exponential backoff + jitter."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            delay = base_delay * (2**attempt) * (1 + random.random())
            log.warning("judge call failed (attempt %d/%d): %r — retrying in %.1fs",
                        attempt + 1, max_retries, exc, delay)
            time.sleep(delay)
    raise RuntimeError(f"judge failed after {max_retries} retries: {last_exc!r}")


def label_samples(
    trajectories_dir: str | Path,
    labels_dir: str | Path,
    client: Any,
    judge_model: str = "gpt-4o-mini",
    max_retries: int = 6,
    limit: int | None = None,
) -> dict[str, int]:
    """Label every completed trajectory sample that has no label yet."""
    trajectories_dir = Path(trajectories_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    traj_manifest = SampleManifest(trajectories_dir)
    label_manifest = SampleManifest(labels_dir, suffix=LABEL_SUFFIX)
    done = label_manifest.complete_ids()
    todo = sorted(sid for sid in traj_manifest.complete_ids() if sid not in done)
    if limit is not None:
        todo = todo[:limit]
    log.info("labelling: %d to do, %d already labelled", len(todo), len(done))

    counts = {"labelled": 0, "skipped_failed": 0, "unparseable": 0, "errors": 0}
    for sid in todo:
        meta = load_json(trajectories_dir / f"{sid}.json")
        if meta.get("failed") or not meta.get("generated_text"):
            counts["skipped_failed"] += 1
            continue
        prompt = render_judge_prompt(
            meta["question"], meta["generated_text"], meta["gold_answers"]
        )
        try:
            raw = query_judge(client, judge_model, prompt, max_retries=max_retries)
        except Exception as exc:
            counts["errors"] += 1
            append_jsonl(labels_dir / "failures.jsonl",
                         {"sample_id": sid, "error": repr(exc), "time": time.time()})
            continue
        verdict = parse_verdict(raw)
        if verdict is None:
            counts["unparseable"] += 1
        save_json_atomic(labels_dir / f"{sid}{LABEL_SUFFIX}", {
            "sample_id": sid,
            "is_hallucination": verdict,            # 1 = UNSUPPORTED, None = unparseable
            "judge_raw_response": raw,              # archived verbatim
            "judge_model": judge_model,
            "judge_prompt_sha256": JUDGE_PROMPT_SHA256,
            "labelled_at": time.time(),
        })
        counts["labelled"] += 1
    return counts


def load_labels(labels_dir: str | Path) -> dict[str, int]:
    """{sample_id -> is_hallucination} for all parseable labels."""
    labels_dir = Path(labels_dir)
    out: dict[str, int] = {}
    for path in sorted(labels_dir.glob(f"*{LABEL_SUFFIX}")):
        record = load_json(path)
        if record.get("is_hallucination") is not None:
            out[record["sample_id"]] = int(record["is_hallucination"])
    return out


# ----------------------------------------------------------------------
# Human agreement (subsec:an_label_reliability)
# ----------------------------------------------------------------------

def make_human_subset_csv(
    labels_dir: str | Path, output_csv: str | Path, n: int = 60, seed: int = 42
) -> Path:
    """Write the template CSV of n random labelled samples for hand-labelling.
    Fill the ``human_label`` column with 0 (supported) or 1 (hallucination)."""
    import csv

    labels = load_labels(labels_dir)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(sorted(labels), min(n, len(labels))))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_id", "human_label"])
        for sid in chosen:
            writer.writerow([sid, ""])
    return output_csv


def judge_human_kappa(labels_dir: str | Path, human_csv: str | Path) -> dict[str, Any]:
    """Cohen's kappa between judge and the human-labelled subset.
    kappa >= 0.6 is the acceptance condition (landis1977measurement)."""
    import csv

    from sklearn.metrics import cohen_kappa_score

    judge = load_labels(labels_dir)
    human: dict[str, int] = {}
    with open(human_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            value = (row.get("human_label") or "").strip()
            if value in ("0", "1"):
                human[row["sample_id"].strip()] = int(value)
    common = sorted(set(judge) & set(human))
    if len(common) < 2:
        raise ValueError(f"only {len(common)} overlapping labelled samples in {human_csv}")
    y_judge = [judge[sid] for sid in common]
    y_human = [human[sid] for sid in common]
    kappa = float(cohen_kappa_score(y_human, y_judge))
    agreement = float(sum(a == b for a, b in zip(y_judge, y_human)) / len(common))
    return {
        "kappa": kappa,
        "n_subset": len(common),
        "raw_agreement": agreement,
        "passes_threshold": kappa >= 0.6,
    }
