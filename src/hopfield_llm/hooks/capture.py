"""Hook-based activation capture during prefill and generation.

This module is responsible for *instrumentation* only:
- registering forward hooks on layer.mlp (pre or post),
- capturing the FFN input / output at each layer,
- computing energy metrics via metrics.energy.compute_energy.

What this module does NOT decide:
- which weights form the memory bank (→ memory.banks)
- which position in the activation becomes the query (→ queries.extractors)
- how energy is computed (→ metrics.energy)

Hook targets
------------
``hook_target="mlp_input"`` (default):
    Registers register_forward_pre_hook on layer.mlp.
    Captures args[0] shape [B, T, d] — the FFN input x^l.
    Correct query for key-space probe (bank="up").

``hook_target="mlp_output"``:
    Registers register_forward_hook on layer.mlp.
    Captures output tensor shape [B, T, d] — the FFN output y^l.
    Correct query for value-space probe (bank="down_values").

Full-trajectory prefill (M4)
-----------------------------
``capture_prefill`` captures the complete [T_prompt, d] activation at every
layer — no reduction inside the hook.  ``PrefillResult`` fields are therefore
[L, T_prompt] shaped.  Reduced [L] scalars ``energy_last`` and ``energy_mean``
are provided for backward compatibility.

Pass ``reduce_in_hook=True`` (deprecated) to revert to the old single-vector
behaviour using ``query_fn``.

Tokenization note
-----------------
Both capture functions use add_special_tokens=False so that last_token
selection points to a content token deterministically regardless of whether
the tokenizer adds a leading BOS token.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F

from hopfield_llm.metrics.energy import LayerEnergyResult, compute_energy
from hopfield_llm.queries.extractors import last_token
from hopfield_llm.utils.logging import get_logger

log = get_logger("hooks.capture")


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------


@dataclass
class PrefillResult:
    """Per-layer per-token energy metrics from a single forward pass on the prompt."""
    energy:       np.ndarray  # float32 [L, T_prompt]
    entropy:      np.ndarray  # float32 [L, T_prompt]
    norm_entropy: np.ndarray  # float32 [L, T_prompt]
    lse:          np.ndarray  # float32 [L, T_prompt]
    quadratic:    np.ndarray  # float32 [L, T_prompt]
    top_act:      np.ndarray  # float32 [L, T_prompt]
    n_active:     np.ndarray  # int32   [L, T_prompt]
    # Backward-compat reduced scalars derived from the 2-D fields above
    energy_last:  np.ndarray  # float32 [L] = energy[:, -1]
    energy_mean:  np.ndarray  # float32 [L] = nanmean(energy, axis=1)
    # Optional dense outputs — None unless save_distributions / save_scores
    distributions: np.ndarray | None = None  # float16 [L, T_prompt, K]
    scores:        np.ndarray | None = None  # float16 [L, T_prompt, K]


@dataclass
class GenerationResult:
    """Per-layer per-token energy metrics from an autoregressive generation pass."""
    energy:          np.ndarray  # float32 [L, T]
    entropy:         np.ndarray  # float32 [L, T]
    norm_entropy:    np.ndarray  # float32 [L, T]
    lse:             np.ndarray  # float32 [L, T]
    quadratic:       np.ndarray  # float32 [L, T]
    top_act:         np.ndarray  # float32 [L, T]
    n_active:        np.ndarray  # int32   [L, T]
    token_ids:       np.ndarray  # int32   [T]
    generated_text:  str
    generation_config: dict = field(default_factory=dict)
    # Optional dense outputs — None unless save_distributions / save_scores
    distributions:   np.ndarray | None = None  # float16 [L, T, K]
    scores:          np.ndarray | None = None  # float16 [L, T, K]


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------


def _fill_result_arrays(
    x_buf: dict[int, torch.Tensor],   # layer_idx → [T, d]
    banks: dict[int, torch.Tensor],
    L: int,
    beta: float,
    threshold: float,
    mode: str,
    save_distributions: bool = False,
    save_scores: bool = False,
    m_per_layer: dict[int, float] = {},
) -> tuple[tuple[np.ndarray, ...], np.ndarray | None, np.ndarray | None]:
    """Compute per-layer per-token energy metrics from a captured query buffer.

    Each value in x_buf is expected to be [T, d] (full prompt trajectory).

    Returns:
        (metrics_tuple, dists, raw_scores)

        metrics_tuple contains 7 arrays each of shape [L, T]:
            energy, entropy, norm_entropy, lse, quadratic, top_act, n_active.

        dists:      float16 [L, T, K] or None.
        raw_scores: float16 [L, T, K] or None.
    """
    # Determine T from x_buf values
    T = 1
    for v in x_buf.values():
        if v.ndim == 2:
            T = max(T, v.shape[0])

    K = next(iter(banks.values())).shape[0] if banks else 0

    energy_arr       = np.full((L, T), np.nan, dtype=np.float32)
    entropy_arr      = np.full((L, T), np.nan, dtype=np.float32)
    norm_entropy_arr = np.full((L, T), np.nan, dtype=np.float32)
    lse_arr          = np.full((L, T), np.nan, dtype=np.float32)
    quadratic_arr    = np.full((L, T), np.nan, dtype=np.float32)
    top_act_arr      = np.full((L, T), np.nan, dtype=np.float32)
    n_active_arr     = np.zeros((L, T), dtype=np.int32)

    dists_arr:  np.ndarray | None = (
        np.zeros((L, T, K), dtype=np.float16) if save_distributions and K > 0 else None
    )
    scores_arr: np.ndarray | None = (
        np.zeros((L, T, K), dtype=np.float16) if save_scores and K > 0 else None
    )

    for l_idx in range(L):
        if l_idx not in x_buf or l_idx not in banks:
            continue
        x = x_buf[l_idx]            # [T, d] or [d]
        if x.ndim == 1:
            x = x.unsqueeze(0)      # [1, d]
        t_actual = min(x.shape[0], T)
        for t in range(t_actual):
            r: LayerEnergyResult = compute_energy(
                x[t], banks[l_idx],
                beta=beta, threshold=threshold, mode=mode,
                M=m_per_layer.get(l_idx),
                return_distribution=save_distributions,
                return_scores=save_scores,
            )
            energy_arr[l_idx, t]       = r.energy
            entropy_arr[l_idx, t]      = r.entropy
            norm_entropy_arr[l_idx, t] = r.norm_entropy
            lse_arr[l_idx, t]          = r.lse
            quadratic_arr[l_idx, t]    = r.quadratic
            top_act_arr[l_idx, t]      = r.top_act
            n_active_arr[l_idx, t]     = r.n_active
            if dists_arr is not None and r.distribution is not None:
                dists_arr[l_idx, t] = r.distribution.numpy().astype(np.float16)
            if scores_arr is not None and r.scores is not None:
                scores_arr[l_idx, t] = r.scores.numpy().astype(np.float16)

    metrics = (
        energy_arr, entropy_arr, norm_entropy_arr,
        lse_arr, quadratic_arr, top_act_arr, n_active_arr,
    )
    return metrics, dists_arr, scores_arr


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
    return_x: bool = False,
    hook_target: Literal["mlp_input", "mlp_output"] = "mlp_input",
    normalize_query: bool = True,
    save_distributions: bool = False,
    save_scores: bool = False,
    reduce_in_hook: bool = False,
    m_per_layer: dict[int, float] = {},
) -> PrefillResult | tuple[PrefillResult, np.ndarray]:
    """Forward pass on the question; capture full FFN activation trajectories per layer.

    Args:
        llm:               HFLLM instance.
        question:          Prompt text. Tokenised with add_special_tokens=False.
        banks:             Per-layer memory banks {layer_idx → Tensor[K, D]}.
        beta:              Hopfield inverse temperature.
        threshold:         Active-neuron count threshold.
        energy_mode:       'dot' (default) or 'cosine'.
        query_fn:          [Deprecated] Only used when reduce_in_hook=True.
        return_x:          If True, also return raw query tensor stacked as [L, T, d].
        hook_target:       'mlp_input'  — pre-hook; captures FFN input x^l.
                           'mlp_output' — post-hook; captures FFN output y^l.
        normalize_query:   If True, L2-normalise each token's query vector.
        save_distributions: If True, attach float16 [L, T_prompt, K] distributions
                           to result.distributions.
        save_scores:       If True, attach float16 [L, T_prompt, K] raw dot-product
                           scores to result.scores.
        reduce_in_hook:    [Deprecated] When True, applies query_fn inside the hook
                           to collapse [T, d] → [d] (old single-vector behaviour).
                           Default False captures the full [T_prompt, d] trajectory.

    Returns:
        PrefillResult with 2-D metric arrays [L, T_prompt] and backward-compat
        scalars energy_last / energy_mean [L].
        When return_x=True, returns (PrefillResult, x_array [L, T_prompt, d]).
    """
    if reduce_in_hook:
        warnings.warn(
            "reduce_in_hook=True is deprecated. The default full-trajectory capture "
            "(reduce_in_hook=False) provides richer [L, T_prompt] metrics.",
            DeprecationWarning,
            stacklevel=2,
        )

    L = len(llm.layers)
    x_buf: dict[int, torch.Tensor] = {}  # layer_idx → [T, d] or [d] (deprecated path)
    hooks = []

    def _make_pre_hook(layer_idx: int):
        def _hook(module, args):
            h = args[0]                                          # [B, T, d]
            x = h[0].detach().to(torch.float32).cpu()           # [T, d]
            if reduce_in_hook:
                q = query_fn(x)                                  # [d]
                if normalize_query:
                    q = F.normalize(q, dim=-1)
                x_buf[layer_idx] = q
            else:
                if normalize_query:
                    x = F.normalize(x, dim=-1)                   # row-wise
                x_buf[layer_idx] = x                             # [T, d]
        return _hook

    def _make_post_hook(layer_idx: int):
        def _hook(module, args, output):
            h = output if isinstance(output, torch.Tensor) else output[0]
            x = h[0].detach().to(torch.float32).cpu()           # [T, d]
            if reduce_in_hook:
                q = query_fn(x)
                if normalize_query:
                    q = F.normalize(q, dim=-1)
                x_buf[layer_idx] = q
            else:
                if normalize_query:
                    x = F.normalize(x, dim=-1)
                x_buf[layer_idx] = x
        return _hook

    for l_idx in range(L):
        if hook_target == "mlp_input":
            hooks.append(
                llm.layers[l_idx].mlp.register_forward_pre_hook(_make_pre_hook(l_idx))
            )
        else:
            hooks.append(
                llm.layers[l_idx].mlp.register_forward_hook(_make_post_hook(l_idx))
            )

    try:
        inputs = llm.tokenizer(question, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(llm.device) for k, v in inputs.items()}
        llm.model(**inputs, output_hidden_states=False)
    finally:
        for h in hooks:
            h.remove()

    metrics, dists_arr, scores_arr = _fill_result_arrays(
        x_buf, banks, L, beta, threshold, energy_mode,
        save_distributions=save_distributions,
        save_scores=save_scores,
        m_per_layer=m_per_layer,
    )

    energy_arr = metrics[0]  # [L, T]
    result = PrefillResult(
        energy=energy_arr,
        entropy=metrics[1],
        norm_entropy=metrics[2],
        lse=metrics[3],
        quadratic=metrics[4],
        top_act=metrics[5],
        n_active=metrics[6],
        energy_last=energy_arr[:, -1].copy(),
        energy_mean=np.nanmean(energy_arr, axis=1),
        distributions=dists_arr,
        scores=scores_arr,
    )

    if return_x:
        if x_buf:
            sample = next(iter(x_buf.values()))
            if sample.ndim == 2:
                T_p, d = sample.shape
                x_stacked = np.zeros((L, T_p, d), dtype=np.float32)
                for l_idx, v in x_buf.items():
                    x_stacked[l_idx] = v.numpy()
            else:
                d = sample.shape[0]
                x_stacked = np.zeros((L, 1, d), dtype=np.float32)
                for l_idx, v in x_buf.items():
                    x_stacked[l_idx, 0] = v.numpy()
        else:
            x_stacked = np.empty((L, 0, 0), dtype=np.float32)
        return result, x_stacked

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
    max_new_tokens: int = 128,
    hook_target: Literal["mlp_input", "mlp_output"] = "mlp_input",
    normalize_query: bool = True,
    save_distributions: bool = False,
    save_scores: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    seed: int | None = None,
    apply_chat_template: bool = True,
    m_per_layer: dict[int, float] = {},
) -> GenerationResult:
    """Autoregressive generation; capture FFN activations per layer per token.

    Registers hooks on ``layer.mlp``.  During the prefill phase (input length > 1)
    hooks are skipped.  During autoregressive steps energy metrics are computed
    inline, storing only 7 scalars per (step, layer).

    Stop conditions: ``eos_token_id`` and the newline token (only when newline
    encodes to a single token; otherwise newline-stopping is skipped with a
    debug log).

    Args:
        llm:               HFLLM instance.
        question:          Question text.
        banks:             Per-layer memory banks {layer_idx → Tensor[K, D]}.
        beta:              Hopfield inverse temperature.
        threshold:         Active-neuron count threshold.
        energy_mode:       'dot' (default) or 'cosine'.
        max_new_tokens:    Maximum tokens to generate.
        hook_target:       'mlp_input' or 'mlp_output'.
        normalize_query:   If True, L2-normalise each query vector before energy.
        save_distributions: If True, attach float16 [L, T, K] distributions.
        save_scores:       If True, attach float16 [L, T, K] raw scores.
        temperature:       Sampling temperature (only used when do_sample=True).
        top_p:             Nucleus sampling probability (only used when do_sample=True).
        do_sample:         If True, use sampling; otherwise greedy decoding.
        seed:              Optional RNG seed set before generation when do_sample=True.
        apply_chat_template: If True, wrap question in the tokenizer's chat template
                           (single user turn).  If False, tokenise question directly.

    Returns:
        GenerationResult with metric arrays of shape [L, T] and generation_config dict.
    """
    L = len(llm.layers)
    step_counter: list[int] = [0]
    result_buf: dict[int, dict[int, LayerEnergyResult]] = {}
    dist_buf:   dict[int, dict[int, np.ndarray]] = {}
    score_buf:  dict[int, dict[int, np.ndarray]] = {}
    hooks = []

    def _make_pre_hook(layer_idx: int):
        def _hook(module, args):
            h = args[0]          # [B, T, d]
            if h.shape[1] != 1:
                return           # prefill phase — skip
            step = step_counter[0]
            if layer_idx in banks:
                q = h[0, 0, :].detach().to(torch.float32).cpu()
                if normalize_query:
                    q = F.normalize(q, dim=-1)
                r = compute_energy(q, banks[layer_idx],
                                   beta=beta, threshold=threshold, mode=energy_mode,
                                   M=m_per_layer.get(layer_idx),
                                   return_distribution=save_distributions,
                                   return_scores=save_scores)
                if step not in result_buf:
                    result_buf[step] = {}
                result_buf[step][layer_idx] = r
                if save_distributions and r.distribution is not None:
                    if step not in dist_buf:
                        dist_buf[step] = {}
                    dist_buf[step][layer_idx] = r.distribution.numpy().astype(np.float16)
                if save_scores and r.scores is not None:
                    if step not in score_buf:
                        score_buf[step] = {}
                    score_buf[step][layer_idx] = r.scores.numpy().astype(np.float16)
            if layer_idx == L - 1:
                step_counter[0] += 1
        return _hook

    def _make_post_hook(layer_idx: int):
        def _hook(module, args, output):
            h = output if isinstance(output, torch.Tensor) else output[0]
            if h.shape[1] != 1:
                return           # prefill phase — skip
            step = step_counter[0]
            if layer_idx in banks:
                q = h[0, 0, :].detach().to(torch.float32).cpu()
                if normalize_query:
                    q = F.normalize(q, dim=-1)
                r = compute_energy(q, banks[layer_idx],
                                   beta=beta, threshold=threshold, mode=energy_mode,
                                   M=m_per_layer.get(layer_idx),
                                   return_distribution=save_distributions,
                                   return_scores=save_scores)
                if step not in result_buf:
                    result_buf[step] = {}
                result_buf[step][layer_idx] = r
                if save_distributions and r.distribution is not None:
                    if step not in dist_buf:
                        dist_buf[step] = {}
                    dist_buf[step][layer_idx] = r.distribution.numpy().astype(np.float16)
                if save_scores and r.scores is not None:
                    if step not in score_buf:
                        score_buf[step] = {}
                    score_buf[step][layer_idx] = r.scores.numpy().astype(np.float16)
            if layer_idx == L - 1:
                step_counter[0] += 1
        return _hook

    for l_idx in range(L):
        if hook_target == "mlp_input":
            hooks.append(
                llm.layers[l_idx].mlp.register_forward_pre_hook(_make_pre_hook(l_idx))
            )
        else:
            hooks.append(
                llm.layers[l_idx].mlp.register_forward_hook(_make_post_hook(l_idx))
            )

    try:
        step_counter[0] = 0

        if apply_chat_template and hasattr(llm.tokenizer, "apply_chat_template"):
            prompt = llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = question

        inputs = llm.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(llm.device) for k, v in inputs.items()}

        newline_ids = llm.tokenizer.encode("\n", add_special_tokens=False)
        stop_ids: list[int] = []
        if llm.tokenizer.eos_token_id is not None:
            stop_ids.append(llm.tokenizer.eos_token_id)
        if len(newline_ids) == 1:
            stop_ids.extend(newline_ids)
        else:
            log.debug(
                "Newline encodes to %d tokens; skipping newline-stopping. "
                "TODO: implement StoppingCriteria-based sequence matcher.",
                len(newline_ids),
            )

        if do_sample and seed is not None:
            torch.manual_seed(seed)

        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=stop_ids if stop_ids else None,
            pad_token_id=llm.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_ids = llm.model.generate(**inputs, **gen_kwargs)
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

    gen_dists:      np.ndarray | None = None
    gen_raw_scores: np.ndarray | None = None

    if (save_distributions or save_scores) and banks and T > 0:
        K = next(iter(banks.values())).shape[0]
        if save_distributions:
            gen_dists = np.zeros((L, T, K), dtype=np.float16)
            for t, layer_dists in dist_buf.items():
                if t >= T:
                    continue
                for l_idx, d in layer_dists.items():
                    gen_dists[l_idx, t] = d
        if save_scores:
            gen_raw_scores = np.zeros((L, T, K), dtype=np.float16)
            for t, layer_scores in score_buf.items():
                if t >= T:
                    continue
                for l_idx, s in layer_scores.items():
                    gen_raw_scores[l_idx, t] = s

    gen_cfg = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
        "seed": seed,
        "apply_chat_template": apply_chat_template,
    }

    return GenerationResult(
        energy=energy_arr, entropy=entropy_arr, norm_entropy=norm_entropy_arr,
        lse=lse_arr, quadratic=quadratic_arr, top_act=top_act_arr, n_active=n_active_arr,
        token_ids=token_ids, generated_text=generated_text,
        generation_config=gen_cfg,
        distributions=gen_dists,
        scores=gen_raw_scores,
    )
