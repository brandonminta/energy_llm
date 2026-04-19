"""Prefill pass: extract h_pre per layer at the last token position."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hopfield_llm.metrics.hopfield import hopfield_energy_mlp
from hopfield_llm.utils.logging import get_logger

log = get_logger("extraction.prefill")


@dataclass
class PrefillResult:
    energy:       np.ndarray  # float32 [L]
    entropy:      np.ndarray  # float32 [L]
    norm_entropy: np.ndarray  # float32 [L]
    lse:          np.ndarray  # float32 [L]
    quadratic:    np.ndarray  # float32 [L]
    top_act:      np.ndarray  # float32 [L]
    n_active:     np.ndarray  # int32   [L]


@torch.no_grad()
def run_prefill_pass(
    llm,
    question: str,
    banks: dict[int, torch.Tensor],
    beta: float = 15.0,
    threshold: float = 0.1,
    energy_mode: str = "dot",
    return_h_pre: bool = False,
) -> PrefillResult | tuple[PrefillResult, np.ndarray]:
    """Forward pass on question tokens; extract h_pre at the last token position.

    Registers ``register_forward_pre_hook`` on ``layers[l].mlp.down_proj``.
    The hook captures ``args[0][0, -1, :]`` — the last-token slice of the
    input to down_proj, i.e. ``h_pre = SiLU(gate) ⊙ up`` of shape ``[d_m]``.

    Args:
        llm: HFLLM instance.
        question: Question text (tokenised with add_special_tokens=True).
        banks: dict mapping layer index (0-indexed) to W_down [d, d_m].
        beta: Hopfield inverse temperature.
        threshold: Active-neuron threshold.
        energy_mode: 'dot' (default) or 'cosine'.
        return_h_pre: If True, also return raw h_pre stacked as [L, d_m] float32.

    Returns:
        PrefillResult, or (PrefillResult, h_pre_array) when return_h_pre=True.
    """
    L = len(llm.layers)
    h_pre_buf: dict[int, torch.Tensor] = {}
    hooks = []

    for l_idx in range(L):
        def _make_hook(layer_idx: int):
            def _hook(module, args):
                # args[0]: [batch, T, d_m] — capture last token, cast to float32
                # (model may run in BFloat16/Float16; cast here avoids downstream errors)
                h_pre_buf[layer_idx] = args[0][0, -1, :].detach().to(torch.float32).cpu()
            return _hook

        hooks.append(
            llm.layers[l_idx].mlp.down_proj.register_forward_pre_hook(
                _make_hook(l_idx)
            )
        )

    try:
        inputs = llm.tokenizer(
            question, return_tensors="pt", add_special_tokens=True
        )
        inputs = {k: v.to(llm.device) for k, v in inputs.items()}
        llm.model(**inputs, output_hidden_states=False)
    finally:
        for h in hooks:
            h.remove()

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
        r = hopfield_energy_mlp(
            h_pre_buf[l_idx], banks[l_idx],
            beta=beta, threshold=threshold, energy_mode=energy_mode,
        )
        energy_arr[l_idx]       = r.energy
        entropy_arr[l_idx]      = r.entropy
        norm_entropy_arr[l_idx] = r.norm_entropy
        lse_arr[l_idx]          = r.lse
        quadratic_arr[l_idx]    = r.quadratic
        top_act_arr[l_idx]      = r.top_act
        n_active_arr[l_idx]     = r.n_active

    result = PrefillResult(
        energy=energy_arr,
        entropy=entropy_arr,
        norm_entropy=norm_entropy_arr,
        lse=lse_arr,
        quadratic=quadratic_arr,
        top_act=top_act_arr,
        n_active=n_active_arr,
    )

    if return_h_pre:
        if h_pre_buf:
            sample_l = next(iter(h_pre_buf))
            d_m = h_pre_buf[sample_l].shape[0]
            h_pre_stacked = np.zeros((L, d_m), dtype=np.float32)
            for l_idx in range(L):
                if l_idx in h_pre_buf:
                    h_pre_stacked[l_idx] = h_pre_buf[l_idx].numpy()
        else:
            h_pre_stacked = np.empty((L, 0), dtype=np.float32)
        return result, h_pre_stacked

    return result
