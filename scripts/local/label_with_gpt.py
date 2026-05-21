"""Label trajectory samples using GPT as judge.

Supports one or multiple experiment folders in a single run.
Safe to interrupt and re-run: already-labeled samples are always skipped.

Usage (single folder):
    export OPENAI_API_KEY=sk-...
    python scripts/local/label_with_gpt.py \\
        --traj-dir /path/to/experiment/trajectories

Usage (multiple folders):
    python scripts/local/label_with_gpt.py \\
        --traj-dir /path/to/qwen/trajectories /path/to/llama/trajectories

Labels are written to <traj-dir>/../labels/{sample_id}_label.json
Cost estimate with gpt-4o-mini: ~$0.10 per 500 TruthfulQA samples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


# ── Helpers ─────────────────────────────────────────────────────────────────

def _check_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        sys.exit(
            "ERROR: OPENAI_API_KEY not set.\n"
            "Run:  export OPENAI_API_KEY=sk-..."
        )
    return key


def _collect_samples(traj_dir: Path) -> list[Path]:
    """Return sorted trajectory .json files, excluding side-files."""
    return sorted(
        f for f in traj_dir.glob("*.json")
        if not any(tag in f.name for tag in ("_label", "_scores", "_hpre"))
    )


def _label_dir_for(traj_dir: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    return traj_dir.parent / "labels"


def _print_header(traj_dir: Path, label_dir: Path, n: int, model: str) -> None:
    print(f"\n{'═'*56}")
    print(f"  Experiment : {traj_dir.parent.name}")
    print(f"  Samples    : {n}")
    print(f"  Model      : {model}")
    print(f"  Labels →   : {label_dir}")
    print(f"{'═'*56}")


# ── Per-folder labeling ──────────────────────────────────────────────────────

def _label_folder(
    traj_dir: Path,
    label_dir: Path,
    judge,
    label_sample_with_judge,
    DataSample,
    max_samples: int | None,
    delay: float,
) -> dict:
    label_dir.mkdir(parents=True, exist_ok=True)
    sample_files = _collect_samples(traj_dir)

    if not sample_files:
        print(f"  WARNING: no trajectory .json files found in {traj_dir}")
        return {"total": 0, "success": 0, "skipped": 0, "failed": 0, "hallucinations": 0}

    if max_samples:
        sample_files = sample_files[:max_samples]

    total     = len(sample_files)
    skipped   = 0
    success   = 0
    failed    = 0
    hal_count = 0

    _print_header(traj_dir, label_dir, total, judge.model_id)

    for i, jf in enumerate(sample_files, 1):
        # ── Load metadata ────────────────────────────────────────────────
        try:
            meta = json.loads(jf.read_text())
        except Exception as exc:
            print(f"  [{i:>4}/{total}]  SKIP (bad JSON: {exc})  {jf.name}")
            failed += 1
            continue

        sample_id  = meta.get("id", jf.stem)
        label_path = label_dir / f"{sample_id}_label.json"

        # ── Resume: skip already labeled ─────────────────────────────────
        if label_path.exists():
            skipped += 1
            continue

        sample = DataSample(
            id=sample_id,
            source=meta.get("source", ""),
            question=meta.get("question", ""),
            gold_answers=meta.get("gold_answers", []),
            generated_text=meta.get("generated_text", ""),
        )

        # ── Call judge ───────────────────────────────────────────────────
        try:
            labeled = label_sample_with_judge(sample, judge)
        except Exception as exc:
            print(f"  [{i:>4}/{total}]  ERR  {sample_id}  ({exc})")
            # Save an error record so the slot is filled and won't retry
            # on re-run unless the user deletes it manually.
            label_path.write_text(json.dumps({
                "id":               sample_id,
                "is_hallucination": None,
                "correctness_score": None,
                "label_source":     "error",
                "judge_model":      judge.model_id,
                "judge_prompt_hash": "",
                "judge_raw_output": str(exc),
            }, indent=2, ensure_ascii=False))
            failed += 1
            continue

        # ── Persist label ────────────────────────────────────────────────
        label_path.write_text(json.dumps({
            "id":               labeled.id,
            "is_hallucination": labeled.is_hallucination,
            "correctness_score": labeled.correctness_score,
            "label_source":     labeled.label_source,
            "judge_model":      labeled.judge_model,
            "judge_prompt_hash": labeled.judge_prompt_hash,
            "judge_raw_output": labeled.judge_raw_output,
        }, indent=2, ensure_ascii=False))

        if labeled.is_hallucination is None:
            tag = " ??"
        elif labeled.is_hallucination:
            tag = "HAL"
            hal_count += 1
        else:
            tag = " OK"

        success += 1
        print(f"  [{i:>4}/{total}]  {tag}  {sample_id}")

        if delay > 0:
            time.sleep(delay)

    hal_rate = (hal_count / success * 100) if success else 0.0
    print(f"\n  Labeled={success}  Skipped={skipped}  Failed={failed}  "
          f"Hallucination rate={hal_count}/{success} ({hal_rate:.1f}%)")

    return {
        "total":         total,
        "success":       success,
        "skipped":       skipped,
        "failed":        failed,
        "hallucinations": hal_count,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label trajectory samples with a GPT judge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--traj-dir", required=True, nargs="+",
        metavar="DIR",
        help="One or more trajectory directories to process",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model ID",
    )
    parser.add_argument(
        "--label-dir", default=None,
        help="Override output dir for labels (only valid with a single --traj-dir)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Process at most N samples per folder (useful for quick tests)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.1,
        help="Seconds between API calls to stay under rate limits",
    )
    args = parser.parse_args()

    if args.label_dir and len(args.traj_dir) > 1:
        sys.exit("ERROR: --label-dir can only be used with a single --traj-dir.")

    api_key = _check_api_key()

    # ── Import project code ──────────────────────────────────────────────────
    try:
        from hopfield_llm.datasets.base import DataSample
        from hopfield_llm.labeling import LLMJudge, label_sample_with_judge
    except ImportError:
        sys.exit("ERROR: hopfield_llm not installed.\nRun:  pip install -e .")

    judge = LLMJudge(
        model_id=args.model,
        api_key=api_key,
        temperature=0.0,
        max_retries=3,
    )

    # ── Process each folder ──────────────────────────────────────────────────
    totals = {"total": 0, "success": 0, "skipped": 0, "failed": 0, "hallucinations": 0}

    for traj_dir_str in args.traj_dir:
        traj_dir  = Path(traj_dir_str)
        label_dir = _label_dir_for(traj_dir, args.label_dir if len(args.traj_dir) == 1 else None)

        stats = _label_folder(
            traj_dir=traj_dir,
            label_dir=label_dir,
            judge=judge,
            label_sample_with_judge=label_sample_with_judge,
            DataSample=DataSample,
            max_samples=args.max_samples,
            delay=args.delay,
        )
        for k in totals:
            totals[k] += stats[k]

    # ── Global summary (shown when processing multiple folders) ──────────────
    if len(args.traj_dir) > 1:
        hal_rate = (totals["hallucinations"] / totals["success"] * 100) if totals["success"] else 0.0
        print(f"\n{'═'*56}")
        print(f"  TOTAL ACROSS ALL FOLDERS")
        print(f"  Total={totals['total']}  Labeled={totals['success']}  "
              f"Skipped={totals['skipped']}  Failed={totals['failed']}")
        print(f"  Hallucination rate: {totals['hallucinations']}/{totals['success']} "
              f"= {hal_rate:.1f}%")
        print(f"{'═'*56}")


if __name__ == "__main__":
    main()
