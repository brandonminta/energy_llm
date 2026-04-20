"""Hook-based activation capture during prefill and generation.

This module is responsible for *instrumentation* only:
- registering forward pre-hooks on down_proj layers,
- capturing the pre-activation tensor (h_pre) at each layer,
- invoking a configurable query extractor to select the query vector,
- computing energy metrics via metrics.energy.compute_energy.

What this module does NOT decide:
- which weights form the memory bank (→ memory.banks)
- which position in the activation becomes the query (→ queries.extractors)
- how energy is computed (→ metrics.energy)

Changing any of those concerns does not require touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from hopfield_llm.metrics.energy import LayerEnergyResult, compute_energy
from hopfield_llm.queries.extractors import last_token
from hopfield_llm.utils.logging import get_logger

log = get_logger("hooks.capture")


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------


@dataclass
class PrefillResult:
    """Per-layer energy metrics from a single forward pass on the question."""
    energy:       np.ndarray  # float32 [L]
    entropy:      np.ndarray  # float32 [L]
    norm_entropy: np.ndarray  # float32 [L]
    lse:          np.ndarray  # float32 [L]
    quadratic:    np.ndarray  # float32 [L]
    top_act:      np.ndarray  # float32 [L]
    n_active:     np.ndarray  # int32   [L]


@dataclass
class GenerationResult:
    """Per-layer per-token energy metrics from an autoregressive generation pass."""
    energy:         np.ndarray  # float32 [L, T]
    entropy:        np.ndarray  # float32 [L, T]
    norm_entropy:   np.ndarray  # float32 [L, T]
    lse:            np.ndarray  # float32 [L, T]
    quadratic:      np.ndarray  # float32 [L, T]
    top_act:        np.ndarray  # float32 [L, T]
    n_active:       np.ndarray  # int32   [L, T]
    token_ids:      np.ndarray  # int32   [T]
    generated_text: str


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------


def _fill_result_arrays(
    h_pre_buf: dict,
    banks: dict[int, torch.Tensor],
    L: int,
    beta: float,
    threshold: float,
    mode: str,
) -> tuple[np.ndarray, ...]:
    """Compute per-layer energy metrics from a captured h_pre buffer.

    Returns:
        Tuple of 7 arrays: energy, entropy, norm_entropy, lse,
        quadratic, top_act, n_active.  Shape [L] each.
    """
    energy_arr       = np.full(L, np.nan, dtype=np.float32)
    entropy_arr      = np.full(L, np.nan, dtype=np.float32)
    norm_entropy_arr = np.full(L, np.nan, dtype=np.float32)
    lse_arr          = np.full(L, np.nan, dtype=np.float32)
    quadratic_arr    = np.full(L, np.nan, dtype=np.float32)
    top_act_arr      = np.full(L, np.nan, dtype=np.float32)
    n_active_arr     = np.zeros(L, dtype=np.int32)

    for l_idx in range(L):
        if l_idx not in h_pre_buf or l_idx not in banks:
            continue
        r: LayerEnergyResult = compute_energy(
            h_pre_buf[l_idx], banks[l_idx],
            beta=beta, threshold=threshold, mode=mode,
        )
        energy_arr[l_idx]       = r.energy
        entropy_arr[l_idx]      = r.entropy
        norm_entropy_arr[l_idx] = r.norm_entropy
        lse_arr[l_idx]          = r.lse
        quadratic_arr[l_idx]    = r.quadratic
        top_act_arr[l_idx]      = r.top_act
        n_active_arr[l_idx]     = r.n_active

    return energy_arr, entropy_arr, norm_entropy_arr, lse_arr, quadratic_arr, top_act_arr, n_active_arr


# ------------------------------------------------------------------
# Prefill capture
# ------------------------------------------------------------------


@torch.no_grad()
def capture_prefill(
    llm,
    question: str,
    banks: dict[int, torch.Tensor],
    beta: float = 15.0,
    threshold: float = 0.1,
    energy_mode: str = "dot",
    query_fn: Callable[[torch.Tensor], torch.Tensor] = last_token,
    return_h_pre: bool = False,
) -> PrefillResult | tuple[PrefillResult, np.ndarray]:
    """Forward pass on the question; capture h_pre per layer via pre-hooks.

    Registers ``register_forward_pre_hook`` on each ``layer.mlp.down_proj``.
    The hook captures the full [T, d_m] activation and passes it to
    ``query_fn`` to select the query vector.  Default: last-token selection.

    Args:
        llm:         HFLLM instance.
        question:    Prompt text (tokenised with add_special_tokens=True).
        banks:       Per-layer memory banks {layer_idx → Tensor[K, D]}.
        beta:        Hopfield inverse temperature.
        threshold:   Active-neuron count threshold.
        energy_mode: 'dot' (default) or 'cosine'.
        query_fn:    Maps [T, d_m] → [d_m].  Default: last_token.
        return_h_pre: If True, also return raw query vectors stacked as [L, d_m].

    Returns:
        PrefillResult, or (PrefillResult, h_pre_array) when return_h_pre=True.
    """
    L = len(llm.layers)
    h_pre_buf: dict[int, torch.Tensor] = {}
    hooks = []

    for l_idx in range(L):
        def _make_hook(layer_idx: int):
            def _hook(module, args):
                # args[0]: [batch, T, d_m]; extract batch=0, apply query_fn
                h = args[0][0].detach().to(torch.float32).cpu()  # [T, d_m]
                h_pre_buf[layer_idx] = query_fn(h)               # [d_m]
            return _hook
        hooks.append(
            llm.layers[l_idx].mlp.down_proj.register_forward_pre_hook(_make_hook(l_idx))
        )

    try:
        inputs = llm.tokenizer(question, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(llm.device) for k, v in inputs.items()}
        llm.model(**inputs, output_hidden_states=False)
    finally:
        for h in hooks:
            h.remove()

    arrays = _fill_result_arrays(h_pre_buf, banks, L, beta, threshold, energy_mode)
    result = PrefillResult(
        energy=arrays[0], entropy=arrays[1], norm_entropy=arrays[2],
        lse=arrays[3], quadratic=arrays[4], top_act=arrays[5], n_active=arrays[6],
    )

    if return_h_pre:
        if h_pre_buf:
            d_m = next(iter(h_pre_buf.values())).shape[0]
            h_pre_stacked = np.zeros((L, d_m), dtype=np.float32)
            for l_idx, vec in h_pre_buf.items():
                h_pre_stacked[l_idx] = vec.numpy()
        else:
            h_pre_stacked = np.empty((L, 0), dtype=np.float32)
        return result, h_pre_stacked

    return result


# ------------------------------------------------------------------
# Generation capture
# ------------------------------------------------------------------


def capture_generation(
    llm,
    question: str,
    banks: dict[int, torch.Tensor],
    beta: float = 15.0,
    threshold: float = 0.1,
    energy_mode: str = "dot",
    max_new_tokens: int = 50,
) -> GenerationResult:
    """Greedy generation; capture h_pre per layer per generated token.

    Registers pre-hooks on ``layer.mlp.down_proj``.  During the prefill phase
    (input length > 1) the hook is skipped.  During autoregressive generation
    energy metrics are computed inline inside the hook, storing only 7 scalars
    per (step, layer) instead of a full [d_m] activation tensor.

    Stop conditions: ``eos_token_id`` and the newline token.

    Args:
        llm:            HFLLM instance.
        question:       Prompt text.
        banks:          Per-layer memory banks {layer_idx → Tensor[K, D]}.
        beta:           Hopfield inverse temperature.
        threshold:      Active-neuron count threshold.
        energy_mode:    'dot' (default) or 'cosine'.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        GenerationResult with metric arrays of shape [L, T].
    """
    L = len(llm.layers)
    step_counter: list[int] = [0]
    # Stores energy results computed inline: step → layer → LayerEnergyResult.
    # Holds only 7 scalars per (step, layer) — not the full activation vector.
    result_buf: dict[int, dict[int, LayerEnergyResult]] = {}
    hooks = []

    for l_idx in range(L):
        def _make_hook(layer_idx: int):
            def _hook(module, args):
                h = args[0]           # [batch, T, d_m]
                if h.shape[1] != 1:
                    return            # prefill phase — skip
                step = step_counter[0]
                if layer_idx in banks:
                    q = h[0, 0, :].detach().to(torch.float32).cpu()
                    r = compute_energy(q, banks[layer_idx],
                                       beta=beta, threshold=threshold, mode=energy_mode)
                    if step not in result_buf:
                        result_buf[step] = {}
                    result_buf[step][layer_idx] = r
                if layer_idx == L - 1:
                    step_counter[0] += 1
            return _hook
        hooks.append(
            llm.layers[l_idx].mlp.down_proj.register_forward_pre_hook(_make_hook(l_idx))
        )

    try:
        step_counter[0] = 0
        inputs = llm.tokenizer(question, return_tensors="pt", add_special_tokens=True)
        prompt_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(llm.device) for k, v in inputs.items()}

        newline_ids = llm.tokenizer.encode("\n", add_special_tokens=False)
        stop_ids: list[int] = []
        if llm.tokenizer.eos_token_id is not None:
            stop_ids.append(llm.tokenizer.eos_token_id)
        stop_ids.extend(newline_ids)

        with torch.no_grad():
            output_ids = llm.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=stop_ids if stop_ids else None,
                pad_token_id=llm.tokenizer.pad_token_id,
            )
    finally:
        for h in hooks:
            h.remove()

    gen_ids = output_ids[0, prompt_len:].cpu()
    token_ids = gen_ids.numpy().astype(np.int32)
    generated_text = llm.tokenizer.decode(gen_ids, skip_special_tokens=True)
    T = len(token_ids)

    energy_arr       = np.full((L, T), np.nan, dtype=np.float32)
    entropy_arr      = np.full((L, T), np.nan, dtype=np.float32)
    norm_entropy_arr = np.full((L, T), np.nan, dtype=np.float32)
    lse_arr          = np.full((L, T), np.nan, dtype=np.float32)
    quadratic_arr    = np.full((L, T), np.nan, dtype=np.float32)
    top_act_arr      = np.full((L, T), np.nan, dtype=np.float32)
    n_active_arr     = np.zeros((L, T), dtype=np.int32)

    for t, step_results in result_buf.items():
        if t >= T:
            continue
        for l_idx, r in step_results.items():
            energy_arr[l_idx, t]       = r.energy
            entropy_arr[l_idx, t]      = r.entropy
            norm_entropy_arr[l_idx, t] = r.norm_entropy
            lse_arr[l_idx, t]          = r.lse
            quadratic_arr[l_idx, t]    = r.quadratic
            top_act_arr[l_idx, t]      = r.top_act
            n_active_arr[l_idx, t]     = r.n_active

    log.debug(
        "Generation: %d tokens, step_counter=%d, captured_steps=%d",
        T, step_counter[0], len(result_buf),
    )

    return GenerationResult(
        energy=energy_arr, entropy=entropy_arr, norm_entropy=norm_entropy_arr,
        lse=lse_arr, quadratic=quadratic_arr, top_act=top_act_arr, n_active=n_active_arr,
        token_ids=token_ids, generated_text=generated_text,
    )
