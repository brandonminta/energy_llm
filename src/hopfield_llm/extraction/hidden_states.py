"""Hidden-state extraction with single-sample and batched modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from hopfield_llm.utils.logging import get_logger

log = get_logger("extraction.hidden_states")


@dataclass
class SegmentHiddenStates:
    h_question: torch.Tensor   # [n_layers, D]
    h_answer: torch.Tensor     # [n_layers, D]
    n_layers: int
    hidden_size: int
    t_question: int            # token count
    t_separator: int
    t_answer: int
    t_total: int
    split_idx: int             # token index where answer segment begins
    layer_indices: list[int]   # 1-indexed transformer layer numbers
    pooling: str
    normalized: bool


def segment_hidden_states_to_dict(states: SegmentHiddenStates) -> dict:
    return {
        "h_question": states.h_question,
        "h_answer": states.h_answer,
        "n_layers": states.n_layers,
        "hidden_size": states.hidden_size,
        "t_question": states.t_question,
        "t_separator": states.t_separator,
        "t_answer": states.t_answer,
        "t_total": states.t_total,
        "split_idx": states.split_idx,
        "layer_indices": states.layer_indices,
        "pooling": states.pooling,
        "normalized": states.normalized,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _pool_tokens(
    h: torch.Tensor,
    pooling: Literal["mean", "last", "max", "first"],
) -> torch.Tensor:
    """Pool over the token dimension.

    Args:
        h: Tensor of shape [n_layers, T, D].

    Returns:
        Tensor of shape [n_layers, D].
    """
    if h.shape[1] == 0:
        raise ValueError("Empty segment: no tokens to pool.")
    if pooling == "mean":
        return h.mean(dim=1)
    if pooling == "last":
        return h[:, -1, :]
    if pooling == "first":
        return h[:, 0, :]
    if pooling == "max":
        return h.max(dim=1).values
    raise ValueError(f"Unknown pooling: '{pooling}'. Use: mean, last, max, first")


def align_banks_to_layers(
    banks: dict[int, torch.Tensor],
    layer_indices: list[int],
) -> dict[int, torch.Tensor]:
    """Build a position-indexed bank dict aligned to 1-indexed layer_indices.

    Banks are keyed by 0-indexed layer number.  Hidden states use 1-indexed
    layer_indices (1 = first transformer layer).  This function maps position
    *i* in the layer_indices list to ``banks[layer_indices[i] - 1]``.

    Returns:
        Dict mapping position index to bank tensor, only for layers present
        in both *banks* and *layer_indices*.
    """
    return {
        i: banks[li - 1]
        for i, li in enumerate(layer_indices)
        if (li - 1) in banks
    }


# ------------------------------------------------------------------
# Single-sample extraction
# ------------------------------------------------------------------


@torch.no_grad()
def extract_segment_hidden_states(
    llm,
    question: str,
    answer: str,
    separator: str = "\n",
    pooling: Literal["mean", "last", "max", "first"] = "mean",
    layers: list[int] | None = None,
    normalize: bool = False,
    to_cpu: bool = True,
) -> SegmentHiddenStates:
    """Extract pooled hidden states for question and answer segments.

    Tokenises ``question + separator + answer`` as a single sequence, runs one
    forward pass, and pools each segment's hidden states independently.

    Args:
        llm: HFLLM instance (must have output_hidden_states=True).
        question: Question text.
        answer: Answer text.
        separator: Token separator between question and answer.
        pooling: Pooling strategy over tokens.
        layers: 1-indexed layer numbers to extract (None = all).
        normalize: Apply L2 normalization after pooling.
        to_cpu: Move result tensors to CPU.

    Returns:
        SegmentHiddenStates with h_question and h_answer of shape [n_layers, D].
    """
    if not llm.output_hidden_states:
        raise RuntimeError("HFLLM must be initialised with output_hidden_states=True.")
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if not answer.strip():
        raise ValueError("Answer must not be empty.")

    ids_q = llm.tokenizer(question, return_tensors="pt", add_special_tokens=True).input_ids[0]
    ids_sep = (
        llm.tokenizer(separator, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if separator
        else torch.tensor([], dtype=torch.long)
    )
    ids_a = llm.tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(ids_a) == 0:
        raise ValueError("Answer tokenized to an empty sequence.")

    ids_full = torch.cat([ids_q, ids_sep, ids_a], dim=0).unsqueeze(0).to(llm.device)
    attention_mask = torch.ones_like(ids_full, device=llm.device)

    split_idx = len(ids_q) + len(ids_sep)
    t_question = len(ids_q)
    t_separator = len(ids_sep)
    t_answer = len(ids_a)
    t_total = t_question + t_separator + t_answer

    outputs = llm.model(
        input_ids=ids_full, attention_mask=attention_mask, output_hidden_states=True
    )
    all_hs = outputs.hidden_states
    n_transformer_layers = len(all_hs) - 1  # index 0 is the embedding layer

    if layers is None:
        layer_indices = list(range(1, n_transformer_layers + 1))
    else:
        invalid = [l for l in layers if l < 1 or l > n_transformer_layers]
        if invalid:
            raise ValueError(
                f"Invalid layer indices: {invalid}. Valid range: [1, {n_transformer_layers}]"
            )
        layer_indices = sorted(layers)

    stacked = torch.stack([all_hs[l][0] for l in layer_indices], dim=0).float()

    # Free VRAM before pooling
    del outputs, all_hs
    if llm.device == "cuda":
        torch.cuda.empty_cache()

    if to_cpu:
        stacked = stacked.cpu()

    h_question = _pool_tokens(stacked[:, :t_question, :], pooling)
    h_answer = _pool_tokens(stacked[:, split_idx:, :], pooling)

    if normalize:
        h_question = F.normalize(h_question, dim=1)
        h_answer = F.normalize(h_answer, dim=1)

    return SegmentHiddenStates(
        h_question=h_question,
        h_answer=h_answer,
        n_layers=len(layer_indices),
        hidden_size=stacked.shape[2],
        t_question=t_question,
        t_separator=t_separator,
        t_answer=t_answer,
        t_total=t_total,
        split_idx=split_idx,
        layer_indices=layer_indices,
        pooling=pooling,
        normalized=normalize,
    )


# ------------------------------------------------------------------
# Batched extraction (Task 2 — parallelisation)
# ------------------------------------------------------------------


@torch.no_grad()
def extract_hidden_states_batch(
    llm,
    samples: list[tuple[str, str]],
    separator: str = "\n",
    pooling: Literal["mean", "last", "max", "first"] = "mean",
    layers: list[int] | None = None,
    normalize: bool = False,
    batch_size: int = 1,
) -> list[SegmentHiddenStates]:
    """Extract hidden states for multiple (question, answer) pairs with batching.

    When batch_size > 1, sequences are padded and processed in a single forward
    pass.  This dramatically reduces wall time on GPUs with sufficient VRAM
    (e.g. A100) while remaining safe on constrained devices (RTX 2050) by
    defaulting to batch_size=1.

    Args:
        llm: HFLLM instance (output_hidden_states=True).
        samples: List of (question, answer) tuples.
        separator: Token separator.
        pooling: Pooling strategy.
        layers: 1-indexed layer numbers (None = all).
        normalize: L2-normalise after pooling.
        batch_size: Number of samples per forward pass.

    Returns:
        List of SegmentHiddenStates, one per input sample.
    """
    if not llm.output_hidden_states:
        raise RuntimeError("HFLLM must be initialised with output_hidden_states=True.")

    results: list[SegmentHiddenStates] = []

    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start : batch_start + batch_size]

        if len(batch) == 1:
            # Fast path — no padding overhead
            q, a = batch[0]
            results.append(
                extract_segment_hidden_states(
                    llm, q, a,
                    separator=separator,
                    pooling=pooling,
                    layers=layers,
                    normalize=normalize,
                    to_cpu=True,
                )
            )
            continue

        # Tokenise each sample and track boundaries
        all_ids: list[torch.Tensor] = []
        boundaries: list[tuple[int, int, int]] = []  # (t_q, t_sep, t_a)

        for question, answer in batch:
            ids_q = llm.tokenizer(
                question, return_tensors="pt", add_special_tokens=True
            ).input_ids[0]
            ids_sep = (
                llm.tokenizer(
                    separator, return_tensors="pt", add_special_tokens=False
                ).input_ids[0]
                if separator
                else torch.tensor([], dtype=torch.long)
            )
            ids_a = llm.tokenizer(
                answer, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
            all_ids.append(torch.cat([ids_q, ids_sep, ids_a], dim=0))
            boundaries.append((len(ids_q), len(ids_sep), len(ids_a)))

        # Pad to same length
        pad_id = llm.tokenizer.pad_token_id or 0
        padded = pad_sequence(all_ids, batch_first=True, padding_value=pad_id).to(
            llm.device
        )
        attention_mask = (padded != pad_id).long().to(llm.device)

        outputs = llm.model(
            input_ids=padded,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        all_hs = outputs.hidden_states
        n_transformer_layers = len(all_hs) - 1

        if layers is None:
            layer_indices = list(range(1, n_transformer_layers + 1))
        else:
            layer_indices = sorted(layers)

        # Stack selected layers: [n_layers, batch, T, D]
        stacked = torch.stack([all_hs[l] for l in layer_indices], dim=0).float()

        del outputs, all_hs
        if llm.device == "cuda":
            torch.cuda.empty_cache()

        stacked = stacked.cpu()

        # Extract per-sample results
        for idx, (t_q, t_sep, t_a) in enumerate(boundaries):
            t_total = t_q + t_sep + t_a
            split_idx = t_q + t_sep
            sample_hs = stacked[:, idx, :t_total, :]  # [n_layers, T, D]

            h_question = _pool_tokens(sample_hs[:, :t_q, :], pooling)
            h_answer = _pool_tokens(sample_hs[:, split_idx:t_total, :], pooling)

            if normalize:
                h_question = F.normalize(h_question, dim=1)
                h_answer = F.normalize(h_answer, dim=1)

            results.append(
                SegmentHiddenStates(
                    h_question=h_question,
                    h_answer=h_answer,
                    n_layers=len(layer_indices),
                    hidden_size=int(stacked.shape[3]),
                    t_question=t_q,
                    t_separator=t_sep,
                    t_answer=t_a,
                    t_total=t_total,
                    split_idx=split_idx,
                    layer_indices=layer_indices,
                    pooling=pooling,
                    normalized=normalize,
                )
            )

        del stacked

    log.info(
        "Extracted hidden states for %d samples (batch_size=%d, pooling=%s)",
        len(results), batch_size, pooling,
    )
    return results
