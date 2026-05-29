"""Logit-based baselines for hallucination detection.

compute_logit_baselines  — per-token log-probability and entropy from a
                           single teacher-forced forward pass.
compute_p_true           — Kadavath et al. 2022 P(True) probe.
"""

from __future__ import annotations

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

from hopfield_llm.datasets.base import DataSample
from hopfield_llm.utils.logging import get_logger

log = get_logger("evaluation.baselines")


@torch.no_grad()
def compute_logit_baselines(
    llm,
    sample: DataSample,
    span: Optional[tuple[int, int]] = None,
) -> dict:
    """Compute seq_logprob_mean, token_entropy_mean, token_entropy_max.

    Runs a single teacher-forced forward pass over (prompt + generated_text)
    and extracts per-token log-probabilities and entropies.

    Args:
        llm:    HFLLM instance.
        sample: DataSample with generated_text populated.
        span:   Optional (start, end) token span into generated_text.
                When provided, statistics are restricted to that span.
                Span indices are relative to the generated tokens (0-based).

    Returns:
        Dict with keys: seq_logprob_mean, token_entropy_mean, token_entropy_max.
    """
    nan_result = {
        "seq_logprob_mean":   float("nan"),
        "token_entropy_mean": float("nan"),
        "token_entropy_max":  float("nan"),
    }

    if not sample.generated_text:
        return nan_result

    prompt_ids = llm.tokenizer(
        sample.question,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]
    gen_ids = llm.tokenizer(
        sample.generated_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    gen_len = gen_ids.shape[1]
    if gen_len == 0:
        return nan_result

    full_ids = torch.cat([prompt_ids, gen_ids], dim=1).to(llm.device)
    prompt_len = prompt_ids.shape[1]

    outputs = llm.model(full_ids, output_hidden_states=False)
    logits = outputs.logits[0]  # [T_full, vocab]

    # Shift: logits[prompt_len-1 : prompt_len+gen_len-1] predict gen tokens
    gen_logits = logits[prompt_len - 1 : prompt_len + gen_len - 1]  # [gen_len, vocab]
    gen_target = full_ids[0, prompt_len:]                            # [gen_len]

    log_probs  = F.log_softmax(gen_logits.float(), dim=-1)  # [gen_len, vocab]
    token_lp   = log_probs[torch.arange(gen_len), gen_target]  # [gen_len]

    probs      = torch.exp(log_probs)
    entropies  = -(probs * log_probs).sum(dim=-1)  # [gen_len]

    if span is not None:
        start, end = max(0, span[0]), min(gen_len, span[1])
        if start < end:
            token_lp  = token_lp[start:end]
            entropies = entropies[start:end]

    return {
        "seq_logprob_mean":   float(token_lp.mean().cpu()),
        "token_entropy_mean": float(entropies.mean().cpu()),
        "token_entropy_max":  float(entropies.max().cpu()),
    }


@torch.no_grad()
def compute_p_true(
    llm,
    sample: DataSample,
    prompt_template: str = "Question: {q}\nAnswer: {a}\nIs the answer true? ",
) -> float:
    """Kadavath et al. 2022 P(True) baseline.

    Constructs a prompt ending in "Is the answer true? " and reads
    p(" True") / (p(" True") + p(" False")) from the next-token distribution.

    Args:
        llm:              HFLLM instance.
        sample:           DataSample with question and generated_text.
        prompt_template:  Format string with {q} and {a} placeholders.

    Returns:
        P(True) in [0, 1], or nan on tokenization failure.
    """
    prompt = prompt_template.format(q=sample.question, a=sample.generated_text)
    inputs = llm.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(llm.device) for k, v in inputs.items()}

    outputs = llm.model(**inputs)
    last_logits = outputs.logits[0, -1, :].float()  # [vocab]

    true_ids  = llm.tokenizer.encode(" True",  add_special_tokens=False)
    false_ids = llm.tokenizer.encode(" False", add_special_tokens=False)

    if not true_ids or not false_ids:
        log.warning("compute_p_true: tokenizer failed to encode ' True' or ' False'")
        return float("nan")

    true_logit  = last_logits[true_ids[0]]
    false_logit = last_logits[false_ids[0]]

    pair_probs = torch.softmax(torch.stack([true_logit, false_logit]), dim=0)
    return float(pair_probs[0].cpu())


def save_baselines(
    baselines: dict,
    artifact_dir: Path,
    sample_id: str,
) -> Path:
    """Write baseline dict to {artifact_dir}/{sample_id}_baselines.json."""
    path = Path(artifact_dir) / f"{sample_id}_baselines.json"
    path.write_text(json.dumps(baselines, indent=2), encoding="utf-8")
    return path


# ------------------------------------------------------------------
# Batch runner + CLI entrypoint
# ------------------------------------------------------------------

_SIDECAR_TAGS = (
    "calibration", "_label", "_baselines", "_scores", "_hpre",
    "_hallufield", "_intra", "_semmentropy", "_lam",
)


def run_baselines_batch(
    llm,
    traj_dir: Path,
    output_dir: Optional[Path] = None,
    with_p_true: bool = True,
    max_samples: Optional[int] = None,
) -> int:
    """Compute token-uncertainty + P(True) baselines for every trajectory sample.

    Writes ``{sample_id}_baselines.json`` next to each trajectory file with keys
    ``seq_logprob_mean``, ``token_entropy_mean``, ``token_entropy_max`` and
    (when ``with_p_true``) ``p_true``.  Resume-safe: existing files are skipped.

    Returns:
        Number of samples newly processed.
    """
    traj_dir = Path(traj_dir)
    out_dir  = Path(output_dir) if output_dir else traj_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(
        f for f in traj_dir.glob("*.json")
        if not any(t in f.name for t in _SIDECAR_TAGS)
    )
    if max_samples:
        sample_files = sample_files[:max_samples]

    total = len(sample_files)
    done = skipped = failed = 0
    log.info("run_baselines_batch: %d samples (p_true=%s)", total, with_p_true)

    for i, jf in enumerate(sample_files, 1):
        try:
            meta = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping %s: %s", jf.name, exc)
            failed += 1
            continue

        sample_id = meta.get("id", jf.stem)
        out_path  = out_dir / f"{sample_id}_baselines.json"
        if out_path.exists():
            skipped += 1
            continue

        sample = DataSample(
            id=sample_id,
            source=meta.get("source", ""),
            question=meta.get("question", ""),
            gold_answers=meta.get("gold_answers", []),
            generated_text=meta.get("generated_text", ""),
        )
        try:
            bl = compute_logit_baselines(llm, sample)
            if with_p_true:
                bl["p_true"] = compute_p_true(llm, sample)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%d/%d] FAILED %s: %s", i, total, sample_id, exc)
            failed += 1
            continue

        bl["id"] = sample_id
        save_baselines(bl, out_dir, sample_id)
        done += 1
        if i % 50 == 0:
            log.info("[%d/%d] done=%d skipped=%d failed=%d", i, total, done, skipped, failed)

    log.info("run_baselines_batch: done=%d skipped=%d failed=%d total=%d",
             done, skipped, failed, total)
    return done


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m hopfield_llm.evaluation.baselines --trajectories DIR``."""
    import argparse

    from hopfield_llm.models.loader import HFLLM
    from hopfield_llm.utils.logging import setup_logging

    parser = argparse.ArgumentParser(
        description="Token-uncertainty + P(True) baselines over a trajectory directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trajectories", required=True, help="Trajectory directory")
    parser.add_argument("--model", default=None,
                        help="Model alias; defaults to the model in the first trajectory JSON")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-p-true", action="store_true", help="Skip the P(True) baseline")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args(argv)

    setup_logging()
    traj_dir = Path(args.trajectories)
    model_alias = args.model
    if model_alias is None:
        first = next(iter(sorted(traj_dir.glob("*.json"))), None)
        if first is None:
            parser.error(f"No trajectory JSON files in {traj_dir}")
        model_alias = json.loads(first.read_text(encoding="utf-8")).get("model", "qwen25_3b")
        log.info("Resolved model alias from trajectory metadata: %s", model_alias)

    llm = HFLLM(model_id=model_alias, device=args.device, load_in_4bit=not args.no_4bit)
    llm.model.eval()
    run_baselines_batch(
        llm, traj_dir, with_p_true=not args.no_p_true, max_samples=args.max_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
