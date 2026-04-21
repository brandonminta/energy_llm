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
