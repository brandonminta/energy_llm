"""Generation pass: extract h_pre per layer per generated token."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hopfield_llm.metrics.hopfield import hopfield_energy_mlp
from hopfield_llm.utils.logging import get_logger

log = get_logger("extraction.generation")


@dataclass
class GenerationResult:
    energy:         np.ndarray  # float32 [L, T]
    entropy:        np.ndarray  # float32 [L, T]
    norm_entropy:   np.ndarray  # float32 [L, T]
    lse:            np.ndarray  # float32 [L, T]
    quadratic:      np.ndarray  # float32 [L, T]
    top_act:        np.ndarray  # float32 [L, T]
    n_active:       np.ndarray  # int32   [L, T]
    token_ids:      np.ndarray  # int32   [T]
    generated_text: str


def run_generation_pass(
    llm,
    question: str,
    banks: dict[int, torch.Tensor],
    beta: float = 15.0,
    threshold: float = 0.1,
    energy_mode: str = "dot",
    max_new_tokens: int = 50,
) -> GenerationResult:
    """Greedy generation capturing h_pre per layer per token.

    Registers ``register_forward_pre_hook`` on ``layers[l].mlp.down_proj``.
    A step counter (list of one int) in the closure tracks the current
    generation step and is reset to 0 before each generate call.

    During the initial prefill the hook fires with sequence length T_prompt > 1
    and is skipped.  During autoregressive generation each step fires with
    sequence length 1, which is captured and indexed by step_counter[0].

    Stop conditions: model's eos_token_id and the ``\\n`` token id.

    Args:
        llm: HFLLM instance.
        question: Question text.
        banks: dict mapping layer index (0-indexed) to W_down [d, d_m].
        beta: Hopfield inverse temperature.
        threshold: Active-neuron threshold.
        energy_mode: 'dot' (default) or 'cosine'.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        GenerationResult with metric arrays of shape [L, T].
    """
    L = len(llm.layers)
    step_counter: list[int] = [0]
    # h_pre_buf[step][layer] = captured h_pre tensor
    h_pre_buf: dict[int, dict[int, torch.Tensor]] = {}
    hooks = []

    for l_idx in range(L):
        def _make_hook(layer_idx: int):
            def _hook(module, args):
                h = args[0]  # [batch, T, d_m]
                if h.shape[1] != 1:
                    return  # Prefill phase — skip
                step = step_counter[0]
                if step not in h_pre_buf:
                    h_pre_buf[step] = {}
                # Cast to float32 at capture time — model may run in BFloat16/Float16
                h_pre_buf[step][layer_idx] = h[0, 0, :].detach().to(torch.float32).cpu()
                # Increment step after the last layer processes this token
                if layer_idx == L - 1:
                    step_counter[0] += 1
            return _hook

        hooks.append(
            llm.layers[l_idx].mlp.down_proj.register_forward_pre_hook(
                _make_hook(l_idx)
            )
        )

    try:
        step_counter[0] = 0
        inputs = llm.tokenizer(
            question, return_tensors="pt", add_special_tokens=True
        )
        prompt_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(llm.device) for k, v in inputs.items()}

        # Stop on eos_token_id and "\n"
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

    for t in range(T):
        if t not in h_pre_buf:
            continue
        for l_idx in range(L):
            if l_idx not in h_pre_buf[t] or l_idx not in banks:
                continue
            r = hopfield_energy_mlp(
                h_pre_buf[t][l_idx], banks[l_idx],
                beta=beta, threshold=threshold, energy_mode=energy_mode,
            )
            energy_arr[l_idx, t]       = r.energy
            entropy_arr[l_idx, t]      = r.entropy
            norm_entropy_arr[l_idx, t] = r.norm_entropy
            lse_arr[l_idx, t]          = r.lse
            quadratic_arr[l_idx, t]    = r.quadratic
            top_act_arr[l_idx, t]      = r.top_act
            n_active_arr[l_idx, t]     = r.n_active

    log.debug(
        "Generation: %d tokens, step_counter=%d, captured_steps=%d",
        T, step_counter[0], len(h_pre_buf),
    )

    return GenerationResult(
        energy=energy_arr,
        entropy=entropy_arr,
        norm_entropy=norm_entropy_arr,
        lse=lse_arr,
        quadratic=quadratic_arr,
        top_act=top_act_arr,
        n_active=n_active_arr,
        token_ids=token_ids,
        generated_text=generated_text,
    )
