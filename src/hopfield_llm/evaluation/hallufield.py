"""HalluField baseline: temperature-perturbation sensitivity score.

Bertsch et al. (2023) / the "HalluField" framing: a model that is uncertain
about the answer will produce distributions that are more sensitive to a small
change in the sampling temperature.  We quantify this as the Jensen–Shannon
divergence between the per-token output distribution at T=1.0 and T=1+ε.

High JS → temperature-sensitive → likely hallucinating.

This module provides:
  compute_hallufield_score  — single sample, returns scalar + per-token array.
  run_hallufield_batch      — loop over a trajectory directory (resume-safe).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from hopfield_llm.utils.logging import get_logger

log = get_logger("evaluation.hallufield")

_EPS_DEFAULT = 0.1       # temperature perturbation: T2 = T1 + eps
_T1_DEFAULT  = 1.0       # base temperature


def compute_hallufield_score(
    llm,
    sample,
    t1: float = _T1_DEFAULT,
    eps: float = _EPS_DEFAULT,
) -> dict:
    """Compute the HalluField score for one sample.

    Runs two teacher-forced forward passes (T=t1 and T=t1+eps) over the
    concatenated (prompt + generated_text) sequence and returns the mean
    JS divergence between the two per-token output distributions.

    Args:
        llm:    HFLLM instance.
        sample: DataSample with question and generated_text populated.
        t1:     Base temperature (default 1.0).
        eps:    Perturbation magnitude (default 0.1).

    Returns:
        Dict with keys:
          hallufield_score   — scalar mean JS (higher = more hallucination)
          js_per_token       — numpy float32 array [T_gen]
          n_gen_tokens       — int
    """
    nan_result = {
        "hallufield_score": float("nan"),
        "js_per_token":     np.array([], dtype=np.float32),
        "n_gen_tokens":     0,
    }

    if not getattr(sample, "generated_text", ""):
        return nan_result

    t2 = t1 + eps

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

    with torch.no_grad():
        logits = llm.model(full_ids, output_hidden_states=False).logits[0]
        # Logits for the gen tokens: logits[prompt_len-1 : prompt_len+gen_len-1]
        gen_logits = logits[prompt_len - 1 : prompt_len + gen_len - 1]  # [T, V]

    # Per-temperature softmax distributions
    p1 = F.softmax(gen_logits.float() / t1, dim=-1)  # [T, V]
    p2 = F.softmax(gen_logits.float() / t2, dim=-1)  # [T, V]

    # Jensen–Shannon divergence per token
    m  = 0.5 * (p1 + p2)
    # clamp for numerical stability
    js = 0.5 * (
        (p1 * (p1.clamp(1e-10) / m.clamp(1e-10)).log()).sum(-1) +
        (p2 * (p2.clamp(1e-10) / m.clamp(1e-10)).log()).sum(-1)
    )  # [T]  values in [0, log2] nats

    js_np = js.cpu().float().numpy()
    score  = float(js_np.mean())

    return {
        "hallufield_score": score,
        "js_per_token":     js_np,
        "n_gen_tokens":     gen_len,
    }


def run_hallufield_batch(
    llm,
    traj_dir: Path,
    output_dir: Optional[Path] = None,
    t1: float = _T1_DEFAULT,
    eps: float = _EPS_DEFAULT,
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Compute HalluField scores for all trajectory samples.

    Writes {sample_id}_hallufield.json next to each trajectory file.
    Resume-safe: already-computed files are skipped.

    Returns:
        List of result dicts for non-skipped samples.
    """
    traj_dir = Path(traj_dir)
    out_dir  = Path(output_dir) if output_dir else traj_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from hopfield_llm.datasets.base import DataSample
    except ImportError as exc:
        raise RuntimeError("hopfield_llm not installed") from exc

    sample_files = sorted(
        f for f in traj_dir.glob("*.json")
        if not any(t in f.name for t in ("calibration", "_label", "_baselines",
                                          "_hallufield", "_scores", "_hpre"))
    )
    if max_samples:
        sample_files = sample_files[:max_samples]

    results = []
    total = len(sample_files)
    log.info("run_hallufield_batch: %d samples, t1=%.1f, eps=%.2f", total, t1, eps)

    for i, jf in enumerate(sample_files, 1):
        try:
            meta = json.loads(jf.read_text())
        except Exception as exc:
            log.warning("Skipping %s: %s", jf.name, exc)
            continue

        sample_id = meta.get("id", jf.stem)
        out_path  = out_dir / f"{sample_id}_hallufield.json"

        if out_path.exists():
            continue

        sample = DataSample(
            id=sample_id,
            source=meta.get("source", ""),
            question=meta.get("question", ""),
            gold_answers=meta.get("gold_answers", []),
            generated_text=meta.get("generated_text", ""),
        )

        try:
            res = compute_hallufield_score(llm, sample, t1=t1, eps=eps)
        except Exception as exc:
            log.warning("[%d/%d] FAILED %s: %s", i, total, sample_id, exc)
            continue

        record = {
            "id":               sample_id,
            "hallufield_score": res["hallufield_score"],
            "n_gen_tokens":     res["n_gen_tokens"],
            "t1":               t1,
            "eps":              eps,
        }
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        results.append(record)
        log.debug("[%d/%d] %s  score=%.4f", i, total, sample_id, res["hallufield_score"])

    log.info("run_hallufield_batch: done=%d", len(results))
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m hopfield_llm.evaluation.hallufield --trajectories DIR``."""
    import argparse

    from hopfield_llm.models.loader import HFLLM
    from hopfield_llm.utils.logging import setup_logging

    parser = argparse.ArgumentParser(
        description="HalluField free-energy baseline (Vu et al., 2025) over a trajectory dir",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trajectories", required=True, help="Trajectory directory")
    parser.add_argument("--model", default=None, help="Model alias")
    parser.add_argument("--t1", type=float, default=_T1_DEFAULT)
    parser.add_argument("--eps", type=float, default=_EPS_DEFAULT)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    args = parser.parse_args(argv)

    setup_logging()
    traj_dir = Path(args.trajectories)
    alias = args.model
    if alias is None:
        first = next(iter(sorted(traj_dir.glob("*.json"))), None)
        if first is None:
            parser.error(f"No trajectory JSON files in {traj_dir}")
        alias = json.loads(first.read_text(encoding="utf-8")).get("model", "qwen25_3b")

    llm = HFLLM(model_id=alias, device=args.device, load_in_4bit=not args.no_4bit)
    llm.model.eval()
    run_hallufield_batch(
        llm, traj_dir, t1=args.t1, eps=args.eps, max_samples=args.max_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
