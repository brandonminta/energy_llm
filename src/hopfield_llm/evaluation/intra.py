"""INTRA baseline: per-layer residual-stream hidden states as hallucination features.

Vazhentsev et al. (2026) show that the residual stream across layers achieves
ROC-AUC ~77.7% on Llama 3.1 across 9 datasets, outperforming logit-based
methods.  INTRA is the strongest unsupervised-representation baseline and the
primary comparator for the Hopfield energy probe.

This module captures the last-generated-token hidden state at every transformer
layer via a teacher-forced forward pass and saves [L, d] arrays to
{sample_id}_intra.npz.  A separate analysis step (NB_intra) applies PCA +
logistic regression to these features.

Usage (GPU required):
    from hopfield_llm.evaluation.intra import run_intra_batch
    from hopfield_llm.models.loader import HFLLM
    llm = HFLLM("qwen25_3b", device="cuda", load_in_4bit=True)
    llm.model.eval()
    run_intra_batch(llm, traj_dir="exp04_results/trajectories")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from hopfield_llm.utils.logging import get_logger

log = get_logger("evaluation.intra")


def compute_intra_features(
    llm,
    question: str,
    generated_text: str,
    apply_chat_template: bool = True,
) -> dict:
    """Run one teacher-forced forward pass and extract per-layer hidden states.

    Captures the residual stream at the output of each transformer block for:
      - The last generated token (most information-dense for hallucination detection)
      - Mean over all generated tokens (robust to position effects)

    Args:
        llm:                HFLLM instance (must be loaded with output_hidden_states=True,
                            which is the HFLLM default).
        question:           The original question / prompt string.
        generated_text:     The model's full generated answer string.
        apply_chat_template: Apply the tokenizer's chat template to the prompt
                            (must match how the trajectory was captured).

    Returns:
        Dict with:
          gen_hidden_last   — float32 [L, d]  last-gen-token hidden state per layer
          gen_hidden_mean   — float32 [L, d]  mean over all gen-token hidden states
          n_gen_tokens      — int
          n_layers          — int
    """
    nan_result = {
        "gen_hidden_last": None,
        "gen_hidden_mean": None,
        "n_gen_tokens":    0,
        "n_layers":        0,
    }

    if not generated_text:
        return nan_result

    # Build prompt string (mirror what capture_prefill does)
    if apply_chat_template and hasattr(llm.tokenizer, "apply_chat_template"):
        prompt_str = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_str = question

    prompt_ids = llm.tokenizer(
        prompt_str,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]
    gen_ids = llm.tokenizer(
        generated_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    gen_len = gen_ids.shape[1]
    if gen_len == 0:
        return nan_result

    full_ids = torch.cat([prompt_ids, gen_ids], dim=1).to(llm.device)
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = llm.model(full_ids, output_hidden_states=True)

    # outputs.hidden_states: tuple of [1, T_full, d] — index 0 = embedding,
    # index l+1 = output of transformer block l.
    hidden_states = outputs.hidden_states  # length = n_layers + 1

    n_layers = len(hidden_states) - 1  # exclude embedding layer
    d = hidden_states[1].shape[-1]

    # Extract generation-token hidden states for each layer
    # Gen tokens occupy positions [prompt_len : prompt_len+gen_len]
    gen_hidden_last = np.zeros((n_layers, d), dtype=np.float32)
    gen_hidden_mean = np.zeros((n_layers, d), dtype=np.float32)

    for l in range(n_layers):
        # hidden_states[l+1]: [1, T_full, d]
        hs_gen = hidden_states[l + 1][0, prompt_len:prompt_len + gen_len, :]  # [T_gen, d]
        gen_hidden_last[l] = hs_gen[-1].float().cpu().numpy()
        gen_hidden_mean[l] = hs_gen.float().mean(dim=0).cpu().numpy()

    return {
        "gen_hidden_last": gen_hidden_last,
        "gen_hidden_mean": gen_hidden_mean,
        "n_gen_tokens":    gen_len,
        "n_layers":        n_layers,
    }


def run_intra_batch(
    llm,
    traj_dir: Path,
    output_dir: Optional[Path] = None,
    apply_chat_template: bool = True,
    max_samples: Optional[int] = None,
) -> int:
    """Compute and save INTRA hidden-state features for all trajectory samples.

    Writes {sample_id}_intra.npz next to each trajectory file.
    Resume-safe: already-computed files are skipped.

    Args:
        llm:                 HFLLM instance.
        traj_dir:            Directory containing {id}.json / .npz files.
        output_dir:          Output directory (defaults to traj_dir).
        apply_chat_template: Match the chat-template setting used in the run.
        max_samples:         Stop after N samples (for quick tests).

    Returns:
        Number of samples processed.
    """
    traj_dir = Path(traj_dir)
    out_dir  = Path(output_dir) if output_dir else traj_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(
        f for f in traj_dir.glob("*.json")
        if not any(t in f.name for t in ("calibration", "_label", "_baselines",
                                          "_hallufield", "_intra", "_scores", "_hpre",
                                          "_lam", "_semmentropy"))
    )
    if max_samples:
        sample_files = sample_files[:max_samples]

    total     = len(sample_files)
    done      = 0
    skipped   = 0

    log.info("run_intra_batch: %d samples", total)

    for i, jf in enumerate(sample_files, 1):
        try:
            meta = json.loads(jf.read_text())
        except Exception as exc:
            log.warning("Skipping %s: %s", jf.name, exc)
            continue

        sample_id = meta.get("id", jf.stem)
        out_path  = out_dir / f"{sample_id}_intra.npz"

        if out_path.exists():
            skipped += 1
            continue

        question       = meta.get("question", "")
        generated_text = meta.get("generated_text", "")

        try:
            feat = compute_intra_features(
                llm, question, generated_text, apply_chat_template=apply_chat_template
            )
        except Exception as exc:
            log.warning("[%d/%d] FAILED %s: %s", i, total, sample_id, exc)
            continue

        if feat["gen_hidden_last"] is None:
            log.warning("[%d/%d] SKIP (no gen tokens) %s", i, total, sample_id)
            continue

        np.savez_compressed(
            out_path,
            gen_hidden_last=feat["gen_hidden_last"],
            gen_hidden_mean=feat["gen_hidden_mean"],
        )
        done += 1
        log.debug("[%d/%d] %s  L=%d d=%d T=%d",
                  i, total, sample_id,
                  feat["n_layers"], feat["gen_hidden_last"].shape[1], feat["n_gen_tokens"])

        if i % 50 == 0:
            log.info("[%d/%d] done=%d skipped=%d", i, total, done, skipped)

    log.info("run_intra_batch: done=%d skipped=%d total=%d", done, skipped, total)
    return done
