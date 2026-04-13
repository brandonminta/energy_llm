"""Two-stage pipeline: bank extraction and trajectory collection.

    Stage 1: build_banks   — W_down weights → .pt artifact
    Stage 2: run_trajectory — prefill + generation passes → .npz + .json per sample
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm

from hopfield_llm.datasets.nq import NaturalQuestionsDataset
from hopfield_llm.datasets.triviaqa import TriviaQADataset
from hopfield_llm.datasets.truthfulqa import TruthfulQADataset
from hopfield_llm.extraction.generation import run_generation_pass
from hopfield_llm.extraction.memory_bank import extract_mlp_memory_bank
from hopfield_llm.extraction.prefill import run_prefill_pass
from hopfield_llm.models.loader import HFLLM
from hopfield_llm.utils.io import load_torch_artifact, save_torch_artifact
from hopfield_llm.utils.logging import get_logger

log = get_logger("pipeline.stages")


# ------------------------------------------------------------------
# Stage 1 — Bank extraction
# ------------------------------------------------------------------


def build_banks(
    model: str,
    output: str | Path,
    device: str | None = None,
    load_in_4bit: bool = True,
) -> Path:
    """Extract W_down[l] for all layers and save as a torch artifact.

    Saves ``{model_alias}_banks.pt`` (or the path specified by *output*).
    Banks are 0-indexed: ``banks[l]`` has shape ``[d, d_m]``.

    Args:
        model: Model alias or HuggingFace model ID.
        output: Output path for the .pt artifact.
        device: Target device (None = auto).
        load_in_4bit: Use 4-bit NF4 quantization when available.

    Returns:
        Path to the saved artifact.
    """
    llm = HFLLM(
        model_id=model,
        device=device,
        load_in_4bit=load_in_4bit,
        output_hidden_states=False,
    )
    banks = extract_mlp_memory_bank(
        llm, bank="down", normalize=False, strict=True
    )
    artifact = {
        "artifact_type": "banks",
        "model": llm.summary(),
        "banks": banks,
    }
    path = save_torch_artifact(artifact, output)
    log.info(
        "build_banks: saved %d layers to %s (model=%s)",
        len(banks), path, llm.alias,
    )
    return path


# ------------------------------------------------------------------
# Stage 2 — Trajectory collection
# ------------------------------------------------------------------


def run_trajectory(
    model: str,
    dataset: str,
    banks: str | Path,
    output: str | Path,
    beta: float = 15.0,
    threshold: float = 0.1,
    energy_mode: str = "dot",
    max_samples: int | None = None,
    seed: int = 42,
    max_new_tokens: int = 50,
    diagnostic_subset: int = 20,
    shard_id: int = 0,
    num_shards: int = 1,
    device: str | None = None,
    load_in_4bit: bool = True,
) -> Path:
    """Run prefill + generation for each sample; persist .npz and .json artifacts.

    For each sample the following files are written to *output*:
        ``{sample_id}.npz``  — compressed metric arrays (prefill + generation)
        ``{sample_id}.json`` — metadata (question, gold_answers, generated_text, …)

    For the first *diagnostic_subset* samples (default 20) an additional
        ``{sample_id}_hpre.npz`` — raw prefill h_pre, shape [L, d_m]
    is saved for diagnostic inspection.

    Sharding: set ``num_shards > 1`` and pass each worker a unique ``shard_id``
    in ``[0, num_shards)`` to split the dataset across SLURM array tasks.
    All shards write to the same *output* directory (per-sample filenames are
    unique so there are no collisions).

    Args:
        model: Model alias or HuggingFace model ID.
        dataset: Dataset name — 'triviaqa', 'nq', or 'truthfulqa'.
        banks: Path to the .pt artifact produced by build_banks.
        output: Directory for output artifacts.
        beta: Hopfield inverse temperature.
        threshold: Active-neuron threshold.
        energy_mode: 'dot' (default) or 'cosine'.
        max_samples: Cap on dataset size before sharding (None = all).
        seed: RNG seed for deterministic subsampling.
        max_new_tokens: Maximum tokens to generate per sample.
        diagnostic_subset: Number of samples (shard-local index) for which
            raw h_pre tensors are saved.
        shard_id: 0-indexed shard index for this worker.
        num_shards: Total shards; 1 = no sharding (all samples).
        device: Target device (None = auto).
        load_in_4bit: Use 4-bit NF4 quantization when available.

    Returns:
        Path to the output directory.
    """
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load banks
    banks_artifact = load_torch_artifact(banks)
    banks_dict: dict[int, object] = banks_artifact["banks"]
    model_summary = banks_artifact["model"]
    model_alias = model_summary.get("alias") or model

    # Load dataset then shard (stride-based so load is even across GPUs)
    dataset_name = dataset.lower()
    if dataset_name == "triviaqa":
        ds = TriviaQADataset(max_samples=max_samples, seed=seed)
    elif dataset_name == "nq":
        ds = NaturalQuestionsDataset(max_samples=max_samples, seed=seed)
    elif dataset_name == "truthfulqa":
        ds = TruthfulQADataset(max_samples=max_samples, seed=seed)
    else:
        raise ValueError(
            f"Unknown dataset: '{dataset}'. Use 'triviaqa', 'nq', or 'truthfulqa'."
        )

    samples = list(ds)
    if num_shards > 1:
        samples = samples[shard_id::num_shards]

    log.info(
        "run_trajectory: %d samples (shard %d/%d), model=%s, dataset=%s",
        len(samples), shard_id, num_shards, model_alias, dataset_name,
    )

    # Load model
    llm = HFLLM(
        model_id=model,
        device=device,
        load_in_4bit=load_in_4bit,
        output_hidden_states=False,
    )
    L = llm.n_layers

    n_ok = 0
    n_err = 0

    bar = tqdm(
        samples,
        desc=f"trajectory[{shard_id}/{num_shards}]",
        unit="sample",
        dynamic_ncols=True,
    )

    for i, sample in enumerate(bar):
        save_hpre = i < diagnostic_subset
        try:
            # Prefill pass
            prefill_out = run_prefill_pass(
                llm, sample.question, banks_dict,
                beta=beta, threshold=threshold, energy_mode=energy_mode,
                return_h_pre=save_hpre,
            )
            if save_hpre:
                prefill_result, h_pre_raw = prefill_out
            else:
                prefill_result = prefill_out
                h_pre_raw = None

            # Generation pass
            gen_result = run_generation_pass(
                llm, sample.question, banks_dict,
                beta=beta, threshold=threshold, energy_mode=energy_mode,
                max_new_tokens=max_new_tokens,
            )

            # Save .npz
            np.savez_compressed(
                out_dir / f"{sample.id}.npz",
                prefill_energy=prefill_result.energy,
                prefill_entropy=prefill_result.entropy,
                prefill_norm_entropy=prefill_result.norm_entropy,
                prefill_lse=prefill_result.lse,
                prefill_quadratic=prefill_result.quadratic,
                prefill_top_act=prefill_result.top_act,
                prefill_n_active=prefill_result.n_active,
                gen_energy=gen_result.energy,
                gen_entropy=gen_result.entropy,
                gen_norm_entropy=gen_result.norm_entropy,
                gen_lse=gen_result.lse,
                gen_quadratic=gen_result.quadratic,
                gen_top_act=gen_result.top_act,
                gen_n_active=gen_result.n_active,
                token_ids=gen_result.token_ids,
            )

            # Save .json
            meta = {
                "id": sample.id,
                "source": sample.source,
                "question": sample.question,
                "gold_answers": sample.gold_answers,
                "generated_text": gen_result.generated_text,
                "model": model_alias,
                "n_layers": L,
                "n_tokens_generated": int(len(gen_result.token_ids)),
            }
            with open(out_dir / f"{sample.id}.json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, ensure_ascii=False)

            # Save diagnostic h_pre
            if save_hpre and h_pre_raw is not None:
                np.savez_compressed(
                    out_dir / f"{sample.id}_hpre.npz",
                    h_pre=h_pre_raw,
                )

            n_ok += 1
            bar.set_postfix(ok=n_ok, err=n_err, refresh=False)

        except Exception as exc:
            n_err += 1
            bar.set_postfix(ok=n_ok, err=n_err, refresh=False)
            # Log the error type, message, and the single most relevant stack frame
            tb = traceback.extract_tb(exc.__traceback__)
            origin = tb[-1] if tb else None
            loc = f"{origin.filename.split('/')[-1]}:{origin.lineno}" if origin else "?"
            log.warning(
                "SKIP %s — %s: %s  (at %s)",
                sample.id, type(exc).__name__, exc, loc,
            )

    bar.close()
    log.info(
        "run_trajectory: finished — %d ok, %d errors → %s",
        n_ok, n_err, out_dir,
    )
    return out_dir
