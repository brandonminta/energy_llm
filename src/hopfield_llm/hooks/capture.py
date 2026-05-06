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

Full-trajectory prefill
-----------------------
``capture_prefill`` captures the complete [T_prompt, d] activation at every
layer — no reduction inside the hook.  ``PrefillResult`` fields are therefore
[L, T_prompt] shaped.  Reduced [L] scalars ``energy_last`` and ``energy_mean``
are provided for downstream convenience.

Tokenization note
-----------------
Both capture functions use add_special_tokens=False.  Chat-template wrapping
is on by default so prefill and generation observe the same tokenized prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from hopfield_llm.metrics.energy import LayerEnergyResult, compute_energy
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
    p_ref:         np.ndarray | None = None  # float32 [L, K] mean-pooled prefill reference distribution


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
    js_per_layer:        np.ndarray | None = None   # float32 [L, T_gen]
    hellinger_per_layer: np.ndarray | None = None   # float32 [L, T_gen]


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
    accumulate_mean_dist: bool = False,
) -> tuple[tuple[np.ndarray, ...], np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Compute per-layer per-token energy metrics from a captured query buffer.

    Each value in x_buf is expected to be [T, d] (full prompt trajectory).

    Args:
        accumulate_mean_dist: When True, compute the mean softmax distribution
            [L, K] via a streaming accumulator instead of materialising the full
            [L, T, K] tensor.  Peak memory is O(L×K) regardless of prompt length.
            The result is returned as the fourth element of the return tuple.

    Returns:
        (metrics_tuple, dists, raw_scores, mean_dist)

        metrics_tuple: 7 arrays each of shape [L, T]:
            energy, entropy, norm_entropy, lse, quadratic, top_act, n_active.
        dists:     float16 [L, T, K] or None  (only when save_distributions=True).
        raw_scores: float16 [L, T, K] or None (only when save_scores=True).
        mean_dist: float32 [L, K] or None     (only when accumulate_mean_dist=True).
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

    # Streaming mean-distribution accumulators: [L, K] float64, O(L×K) memory.
    # Never materialises the full [L, T, K] tensor even for long prompts.
    _need_dist = save_distributions or accumulate_mean_dist
    mean_dist_sum:   np.ndarray | None = (
        np.zeros((L, K), dtype=np.float64) if accumulate_mean_dist and K > 0 else None
    )
    mean_dist_count: np.ndarray | None = (
        np.zeros(L, dtype=np.int64) if accumulate_mean_dist else None
    )

    for l_idx in range(L):
        if l_idx not in x_buf or l_idx not in banks:
            continue
        x = x_buf[l_idx]            # [T, d]
        t_actual = min(x.shape[0], T)
        for t in range(t_actual):
            r: LayerEnergyResult = compute_energy(
                x[t], banks[l_idx],
                beta=beta, threshold=threshold, mode=mode,
                M=m_per_layer.get(l_idx),
                return_distribution=_need_dist,
                return_scores=save_scores,
            )
            energy_arr[l_idx, t]       = r.energy
            entropy_arr[l_idx, t]      = r.entropy
            norm_entropy_arr[l_idx, t] = r.norm_entropy
            lse_arr[l_idx, t]          = r.lse
            quadratic_arr[l_idx, t]    = r.quadratic
            top_act_arr[l_idx, t]      = r.top_act
            n_active_arr[l_idx, t]     = r.n_active
            if r.distribution is not None:
                if dists_arr is not None:
                    dists_arr[l_idx, t] = r.distribution.cpu().numpy().astype(np.float16)
                if mean_dist_sum is not None and mean_dist_count is not None:
                    mean_dist_sum[l_idx] += r.distribution.cpu().numpy().astype(np.float64)
                    mean_dist_count[l_idx] += 1
            if scores_arr is not None and r.scores is not None:
                scores_arr[l_idx, t] = r.scores.cpu().numpy().astype(np.float16)

    # Normalise accumulated mean distribution → [L, K] float32
    mean_dist: np.ndarray | None = None
    if mean_dist_sum is not None and mean_dist_count is not None:
        counts = mean_dist_count[:, None].clip(1)
        mean_dist = (mean_dist_sum / counts).astype(np.float32)
        mean_dist /= mean_dist.sum(axis=-1, keepdims=True).clip(1e-10, None)

    metrics = (
        energy_arr, entropy_arr, norm_entropy_arr,
        lse_arr, quadratic_arr, top_act_arr, n_active_arr,
    )
    return metrics, dists_arr, scores_arr, mean_dist


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
    return_x: bool = False,
    hook_target: Literal["mlp_input", "mlp_output"] = "mlp_input",
    normalize_query: bool = True,
    save_distributions: bool = False,
    save_scores: bool = False,
    m_per_layer: dict[int, float] = {},
    apply_chat_template: bool = True,
    compute_p_ref: bool = False,
) -> PrefillResult | tuple[PrefillResult, np.ndarray]:
    """Forward pass on the question; capture full FFN activation trajectories per layer.

    Prefill must observe the *same* tokenized prompt that generation will see;
    otherwise prefill→generation deltas mix two different forward passes.  By
    default the question is wrapped in the tokenizer's chat template, matching
    capture_generation.  Pass apply_chat_template=False to tokenize the raw
    question (e.g. for base, non-instruct models).

    Args:
        llm:               HFLLM instance.
        question:          Prompt text. Tokenised with add_special_tokens=False.
        banks:             Per-layer memory banks {layer_idx → Tensor[K, D]}.
        beta:              Hopfield inverse temperature.
        threshold:         Active-neuron count threshold.
        energy_mode:       'dot' (default) or 'cosine'.
        return_x:          If True, also return raw query tensor stacked as [L, T, d].
        hook_target:       'mlp_input'  — pre-hook; captures FFN input x^l.
                           'mlp_output' — post-hook; captures FFN output y^l.
        normalize_query:   If True, L2-normalise each token's query vector.
        save_distributions: If True, attach float16 [L, T_prompt, K] distributions
                           to result.distributions.
        save_scores:       If True, attach float16 [L, T_prompt, K] raw dot-product
                           scores to result.scores.
        apply_chat_template: If True, wrap question in the tokenizer's chat template
                           (single user turn, with assistant generation prompt).
                           Defaults to True so prefill matches the prompt observed
                           by capture_generation.

    Returns:
        PrefillResult with 2-D metric arrays [L, T_prompt] and backward-compat
        scalars energy_last / energy_mean [L].
        When return_x=True, returns (PrefillResult, x_array [L, T_prompt, d]).
    """
    L = len(llm.layers)
    x_buf: dict[int, torch.Tensor] = {}  # layer_idx → [T, d]
    hooks = []

    def _make_pre_hook(layer_idx: int):
        def _hook(module, args):
            h = args[0]                                          # [B, T, d]
            x = h[0].detach().to(torch.float32)                  # [T, d] on GPU
            if normalize_query:
                x = F.normalize(x, dim=-1)                       # row-wise
            x_buf[layer_idx] = x                                 # [T, d]
        return _hook

    def _make_post_hook(layer_idx: int):
        def _hook(module, args, output):
            h = output if isinstance(output, torch.Tensor) else output[0]
            x = h[0].detach().to(torch.float32)                  # [T, d] on GPU
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
        if apply_chat_template and hasattr(llm.tokenizer, "apply_chat_template"):
            prompt = llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = question
        inputs = llm.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(llm.device) for k, v in inputs.items()}
        llm.model(**inputs, output_hidden_states=False)
    finally:
        for h in hooks:
            h.remove()

    # Use streaming accumulator for p_ref so we never materialise [L, T, K].
    # dists_arr is only allocated when the caller explicitly wants it
    # (save_distributions=True); p_ref always uses the O(L×K) path.
    metrics, dists_arr, scores_arr, mean_dist = _fill_result_arrays(
        x_buf, banks, L, beta, threshold, energy_mode,
        save_distributions=save_distributions,
        save_scores=save_scores,
        m_per_layer=m_per_layer,
        accumulate_mean_dist=compute_p_ref,
    )

    p_ref: np.ndarray | None = mean_dist  # [L, K] float32 or None

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
        p_ref=p_ref,
    )

    if return_x:
        if x_buf:
            T_p, d = next(iter(x_buf.values())).shape
            x_stacked = np.zeros((L, T_p, d), dtype=np.float32)
            for l_idx, v in x_buf.items():
                x_stacked[l_idx] = v.cpu().numpy()
        else:
            x_stacked = np.empty((L, 0, 0), dtype=np.float32)
        del x_buf
        return result, x_stacked

    del x_buf
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
    p_ref: np.ndarray | None = None,
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
    js_buf:     dict[int, dict[int, float]] = {}
    hell_buf:   dict[int, dict[int, float]] = {}
    hooks = []

    p_ref_torch: dict[int, torch.Tensor] = {}
    if p_ref is not None:
        for l_idx in range(L):
            if l_idx < p_ref.shape[0]:
                p_ref_torch[l_idx] = torch.from_numpy(p_ref[l_idx]).to(llm.device)

    _need_dist = save_distributions or (p_ref is not None)

    def _make_pre_hook(layer_idx: int):
        def _hook(module, args):
            h = args[0]          # [B, T, d]
            if h.shape[1] != 1:
                return           # prefill phase — skip
            step = step_counter[0]
            if layer_idx in banks:
                q = h[0, 0, :].detach().to(torch.float32)
                if normalize_query:
                    q = F.normalize(q, dim=-1)
                r = compute_energy(q, banks[layer_idx],
                                   beta=beta, threshold=threshold, mode=energy_mode,
                                   M=m_per_layer.get(layer_idx),
                                   return_distribution=_need_dist,
                                   return_scores=save_scores)
                if step not in result_buf:
                    result_buf[step] = {}
                result_buf[step][layer_idx] = r
                if save_distributions and r.distribution is not None:
                    if step not in dist_buf:
                        dist_buf[step] = {}
                    dist_buf[step][layer_idx] = r.distribution.cpu().numpy().astype(np.float16)
                if save_scores and r.scores is not None:
                    if step not in score_buf:
                        score_buf[step] = {}
                    score_buf[step][layer_idx] = r.scores.cpu().numpy().astype(np.float16)
                if p_ref is not None and r.distribution is not None and layer_idx in p_ref_torch:
                    _p = r.distribution
                    _q = p_ref_torch[layer_idx]
                    eps = 1e-10
                    _p_c = _p.clamp(min=eps)
                    _q_c = _q.clamp(min=eps)
                    _m = 0.5 * (_p_c + _q_c)
                    js_val = float(
                        0.5 * (_p_c * (_p_c / _m).log()).sum() +
                        0.5 * (_q_c * (_q_c / _m).log()).sum()
                    )
                    hell_val = float((0.5 * (_p.sqrt() - _q.sqrt()).pow(2).sum()).sqrt())
                    js_buf.setdefault(step, {})[layer_idx] = js_val
                    hell_buf.setdefault(step, {})[layer_idx] = hell_val
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
                q = h[0, 0, :].detach().to(torch.float32)
                if normalize_query:
                    q = F.normalize(q, dim=-1)
                r = compute_energy(q, banks[layer_idx],
                                   beta=beta, threshold=threshold, mode=energy_mode,
                                   M=m_per_layer.get(layer_idx),
                                   return_distribution=_need_dist,
                                   return_scores=save_scores)
                if step not in result_buf:
                    result_buf[step] = {}
                result_buf[step][layer_idx] = r
                if save_distributions and r.distribution is not None:
                    if step not in dist_buf:
                        dist_buf[step] = {}
                    dist_buf[step][layer_idx] = r.distribution.cpu().numpy().astype(np.float16)
                if save_scores and r.scores is not None:
                    if step not in score_buf:
                        score_buf[step] = {}
                    score_buf[step][layer_idx] = r.scores.cpu().numpy().astype(np.float16)
                if p_ref is not None and r.distribution is not None and layer_idx in p_ref_torch:
                    _p = r.distribution
                    _q = p_ref_torch[layer_idx]
                    eps = 1e-10
                    _p_c = _p.clamp(min=eps)
                    _q_c = _q.clamp(min=eps)
                    _m = 0.5 * (_p_c + _q_c)
                    js_val = float(
                        0.5 * (_p_c * (_p_c / _m).log()).sum() +
                        0.5 * (_q_c * (_q_c / _m).log()).sum()
                    )
                    hell_val = float((0.5 * (_p.sqrt() - _q.sqrt()).pow(2).sum()).sqrt())
                    js_buf.setdefault(step, {})[layer_idx] = js_val
                    hell_buf.setdefault(step, {})[layer_idx] = hell_val
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
    del output_ids  # release GPU memory for the full sequence before post-processing
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

    n_captured_steps = len(result_buf)
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
    del result_buf

    log.debug(
        "Generation: %d tokens, step_counter=%d, captured_steps=%d",
        T, step_counter[0], n_captured_steps,
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
    del dist_buf, score_buf

    js_arr = hell_arr = None
    if p_ref is not None and T > 0:
        js_arr   = np.full((L, T), np.nan, dtype=np.float32)
        hell_arr = np.full((L, T), np.nan, dtype=np.float32)
        for t, layer_js in js_buf.items():
            if t < T:
                for l_idx, v in layer_js.items():
                    js_arr[l_idx, t] = v
        for t, layer_hell in hell_buf.items():
            if t < T:
                for l_idx, v in layer_hell.items():
                    hell_arr[l_idx, t] = v
    del js_buf, hell_buf

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
        js_per_layer=js_arr,
        hellinger_per_layer=hell_arr,
    )
