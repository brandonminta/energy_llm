"""Per-question capture: greedy generation, one teacher-forced pass with
residual hooks, and every per-layer metric.

1. The chat-templated prompt is greedy-decoded (clean config, batch 1).
   Generated token ids, per-token log-probability and predictive entropy
   (float32, from the decoding logits) and a truncation flag are kept.
2. One forward pass over prompt ++ generated tokens. The argmax of its
   logits must reproduce the greedy tokens; any mismatch is recorded.
3. A forward pre-hook on every ``post_attention_layernorm`` reads the
   residual x at its INPUT. x_hat, scores and all metrics are float32.
4. Per layer (layers never mixed), per-token traces over every position
   (prompt and response; the role mask separates them downstream) at every
   beta of the grid (beta* last). No aggregation over tokens: that is
   analysis, not extraction. Divergences compare each position with three
   reference positions: question_last (last question token), prompt_last
   (last prompt token) and response_first (first response token).
   Response positions = generated non-stop tokens. An empty response (the
   first generated token is a stop token) is a valid sample: traces are
   computed, the response_first divergences and residuals are NaN by
   design, and ``empty_response`` is set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from energy_llm.core import metrics
from energy_llm.core.bank import Bank
from energy_llm.core.model import ModelBundle, decoder_layers
from energy_llm.core.prompting import ROLE_QUESTION, ROLE_RESPONSE, ROLE_TEMPLATE, Prompt, PromptError

FP16_MAX = float(np.finfo(np.float16).max)
REFERENCES = ("question_last", "prompt_last", "response_first")


class NonFiniteError(RuntimeError):
    """A NaN/inf where none is expected: a real failure."""


# ----------------------------------------------------------------------
# Hooks
# ----------------------------------------------------------------------

class ResidualHooks:
    """Forward pre-hooks on each layer's post_attention_layernorm. They only
    copy the input tensor; the computation graph is untouched."""

    def __init__(self, model):
        self.enabled = False
        self.x: dict[int, torch.Tensor] = {}
        self._handles = [
            layer.post_attention_layernorm.register_forward_pre_hook(
                self._make_hook(l), with_kwargs=True)
            for l, layer in enumerate(decoder_layers(model))
        ]

    def _make_hook(self, l: int):
        def hook(_module, args, kwargs):
            if self.enabled:
                x = args[0] if args else kwargs["hidden_states"]
                self.x[l] = x[0].detach().clone()   # [T, D], model dtype
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


# ----------------------------------------------------------------------
# Generation + teacher forcing
# ----------------------------------------------------------------------

@dataclass
class ForwardTrace:
    prompt: Prompt
    gen_ids: list[int]
    token_logprobs: np.ndarray        # [T_g] float32
    token_entropies: np.ndarray       # [T_g] float32, nats
    truncated: bool
    stop_token_id: int | None
    response_positions: list[int]     # absolute positions, stop tokens excluded
    tf_mismatch_positions: list[int]  # indices j into gen_ids
    residuals: dict[int, torch.Tensor] = field(repr=False)

    @property
    def n_prompt(self) -> int:
        return len(self.prompt.input_ids)

    @property
    def empty_response(self) -> bool:
        return not self.response_positions

    @property
    def reference_positions(self) -> dict[str, int | None]:
        q = [i for i, r in enumerate(self.prompt.roles) if r == ROLE_QUESTION]
        if not q:
            raise PromptError("no question token in the prompt")
        return {"question_last": q[-1], "prompt_last": self.n_prompt - 1,
                "response_first": self.response_positions[0] if self.response_positions else None}

    @property
    def roles(self) -> list[int]:
        stops = set() if self.stop_token_id is None else {self.stop_token_id}
        gen_roles = [ROLE_TEMPLATE if t in stops else ROLE_RESPONSE for t in self.gen_ids]
        return self.prompt.roles + gen_roles


@torch.inference_mode()
def forward_item(bundle: ModelBundle, hooks: ResidualHooks, prompt: Prompt) -> ForwardTrace:
    model, dev = bundle.model, bundle.device
    stops = set(bundle.stop_token_ids)
    ids = torch.tensor([prompt.input_ids], device=dev)
    n = ids.shape[1]

    hooks.enabled = False
    out = model.generate(
        input_ids=ids, attention_mask=torch.ones_like(ids),
        generation_config=bundle.generation_config,
    )
    gen = out.sequences[0, n:].tolist()
    if not gen:
        raise RuntimeError("generation returned no tokens")
    logits = torch.stack(out.logits, dim=0)[:, 0, :].float()   # [T_g, V]
    del out
    logp = torch.log_softmax(logits, dim=-1)
    chosen = logp.gather(1, torch.tensor(gen, device=logp.device)[:, None])[:, 0]
    ent = -(logp.exp() * logp).nan_to_num(0.0).sum(dim=-1)
    token_logprobs = chosen.cpu().numpy().astype(np.float32)
    token_entropies = ent.cpu().numpy().astype(np.float32)
    del logits, logp

    max_new = bundle.generation_config.max_new_tokens
    stop_id = gen[-1] if gen[-1] in stops else None
    truncated = len(gen) >= max_new and stop_id is None
    resp_pos = [n + j for j, t in enumerate(gen) if t not in stops]

    full = torch.tensor([prompt.input_ids + gen], device=dev)
    hooks.x = {}
    hooks.enabled = True
    try:
        tf = model(input_ids=full, attention_mask=torch.ones_like(full),
                   use_cache=False, logits_to_keep=len(gen) + 1)
    finally:
        hooks.enabled = False
    tf_argmax = tf.logits[0, : len(gen)].argmax(dim=-1).tolist()
    del tf
    mismatches = [j for j, (a, b) in enumerate(zip(tf_argmax, gen)) if a != b]
    residuals, hooks.x = hooks.x, {}

    return ForwardTrace(
        prompt=prompt, gen_ids=gen, token_logprobs=token_logprobs,
        token_entropies=token_entropies, truncated=truncated, stop_token_id=stop_id,
        response_positions=resp_pos, tf_mismatch_positions=mismatches,
        residuals=residuals,
    )


# ----------------------------------------------------------------------
# Metrics per item
# ----------------------------------------------------------------------

@dataclass
class ItemResult:
    item_id: str
    answer: dict
    baselines: dict
    betas: list[float]                      # grid + [beta*]
    reference_positions: dict[str, int | None]
    # name -> [L, T, n_beta] float32 (energy, norm_entropy, topk_mass, mean_score,
    # var_score, js/<ref>, dkl/<ref>), [L, T] float32 (margin, xhat_top1_frac),
    # [L, T, k_ids] int32 (topk_ids). Item files store the floats as float16.
    traces: dict[str, np.ndarray]
    role_mask: np.ndarray                   # [T] int8
    hidden: dict[str, np.ndarray]           # position name -> [L_h, D] float16


def _layer_scores(bank: Bank, x: torch.Tensor, l: int, eps: float):
    x_hat = metrics.rms_normalise(x.to(bank.device), eps)
    return x_hat, bank.scores(l, x_hat)


@torch.inference_mode()
def compute_item(bundle: ModelBundle, bank: Bank, trace: ForwardTrace, item_id: str,
                 betas: list[float], hidden_layers: list[int], topk_mass_k: int,
                 topk_ids_k: int) -> ItemResult:
    """``betas``: the extraction grid with beta* last (``ExtractionConfig.betas``).
    One s [T, D_ff] per layer; the grid is looped inside the layer."""
    dims = bundle.dims
    L, T, nb = dims.n_layers, trace.n_prompt + len(trace.gen_ids), len(betas)
    refs = trace.reference_positions
    nan = float("nan")

    per_beta = ["energy", "norm_entropy", "topk_mass", "mean_score", "var_score"] + \
        [f"{d}/{r}" for d in ("js", "dkl") for r in REFERENCES]
    dev = bank.device
    tr = {k: torch.full((L, T, nb), nan, device=dev) for k in per_beta}
    tr["margin"] = torch.empty(L, T, device=dev)
    tr["xhat_top1_frac"] = torch.empty(L, T, device=dev)
    tr["topk_ids"] = torch.empty(L, T, topk_ids_k, dtype=torch.int32, device=dev)
    max_dev = 0.0
    for l in range(L):
        x_hat, s = _layer_scores(bank, trace.residuals[l], l, dims.rms_norm_eps)
        max_dev = max(max_dev, (x_hat.norm(dim=-1) / math.sqrt(dims.d_model) - 1.0).abs().max().item())
        tr["margin"][l] = metrics.margin(s)
        tr["xhat_top1_frac"][l] = metrics.xhat_top1_frac(x_hat)
        # the ranking of p is the ranking of s at any beta > 0
        tr["topk_ids"][l] = metrics.topk_indices(s, topk_ids_k)
        for bi, beta in enumerate(betas):
            logp = metrics.log_distribution(s, beta)
            tr["energy"][l, :, bi] = metrics.energy(s, beta)
            tr["norm_entropy"][l, :, bi] = metrics.norm_entropy(logp)
            tr["topk_mass"][l, :, bi] = metrics.topk_mass(logp, topk_mass_k)
            tr["mean_score"][l, :, bi] = metrics.mean_score(logp, s)
            tr["var_score"][l, :, bi] = metrics.var_score(logp, s)
            for r, pos in refs.items():
                if pos is None:      # empty response: response_first undefined (NaN)
                    continue
                lq = logp[pos]
                tr[f"js/{r}"][l, :, bi] = metrics.js_divergence(logp, lq)
                tr[f"dkl/{r}"][l, :, bi] = (metrics.kl_divergence(logp, lq)
                                            - metrics.kl_divergence(lq, logp))
            del logp
        del x_hat, s

    empty = trace.empty_response
    hidden_pos = refs | {"response_last": trace.response_positions[-1] if not empty else None}
    hidden = {}
    for name, pos in hidden_pos.items():
        rows = [trace.residuals[l][pos].float().cpu() if pos is not None
                else torch.full((dims.d_model,), nan) for l in hidden_layers]
        hidden[name] = torch.stack(rows).numpy()
    if np.nanmax(np.abs(np.concatenate(list(hidden.values())))) > FP16_MAX:
        raise NonFiniteError("residual exceeds the float16 range")

    traces = {k: v.cpu().numpy() for k, v in tr.items()}
    for name, arr in traces.items():
        if empty and name.endswith("/response_first"):
            continue          # NaN by design
        if not np.isfinite(arr).all():
            raise NonFiniteError(f"non-finite values in {name}")
        if arr.dtype == np.float32 and np.abs(arr).max() > FP16_MAX:
            raise NonFiniteError(f"{name} exceeds the float16 range")

    gen = trace.gen_ids
    resp_j = [p - trace.n_prompt for p in trace.response_positions]   # indices into gen
    tok = bundle.tokenizer
    answer = {
        "id": item_id,
        "prompt": trace.prompt.text,
        "response": tok.decode([gen[j] for j in resp_j], skip_special_tokens=True),
        "prompt_token_ids": list(trace.prompt.input_ids),
        "generated_token_ids": list(gen),
        "n_prompt_tokens": trace.n_prompt,
        "n_generated_tokens": len(gen),
        "n_response_tokens": len(resp_j),
        "empty_response": empty,
        "truncated": bool(trace.truncated),
        "stop_token_id": trace.stop_token_id,
        "token_logprobs": trace.token_logprobs.tolist(),
        "token_entropies": trace.token_entropies.tolist(),
        "tf_mismatch_count": len(trace.tf_mismatch_positions),
        "tf_mismatch_positions": list(trace.tf_mismatch_positions),
        "xhat_norm_max_rel_dev": float(max_dev),
    }
    lp = trace.token_logprobs[resp_j]
    nan = float("nan")
    baselines = {   # over response tokens; NaN for an empty response
        "id": item_id,
        "mean_logprob": float(lp.mean()) if not empty else nan,
        "min_logprob": float(lp.min()) if not empty else nan,
        "mean_token_entropy": float(trace.token_entropies[resp_j].mean()) if not empty else nan,
        "n_response_tokens": len(resp_j),
        "empty_response": empty,
    }
    return ItemResult(
        item_id=item_id, answer=answer, baselines=baselines, betas=list(betas),
        reference_positions=refs, traces=traces,
        role_mask=np.asarray(trace.roles, dtype=np.int8),
        hidden={k: v.astype(np.float16) for k, v in hidden.items()},
    )


@torch.inference_mode()
def norm_entropy_curve(bundle: ModelBundle, bank: Bank, trace: ForwardTrace,
                       grid: list[float]) -> np.ndarray:
    """Mean H_bar over layers and the calibration positions (last prompt
    token + response tokens, stop tokens excluded) for every beta on the grid."""
    dims = bundle.dims
    sel_idx = torch.tensor([trace.n_prompt - 1] + trace.response_positions, device=bank.device)
    total = torch.zeros(len(grid), dtype=torch.float64)
    for l in range(dims.n_layers):
        _, s = _layer_scores(bank, trace.residuals[l], l, dims.rms_norm_eps)
        s = s[sel_idx]
        for i, beta in enumerate(grid):
            total[i] += metrics.norm_entropy(metrics.log_distribution(s, beta)).mean().item()
    curve = (total / dims.n_layers).numpy()
    if not np.isfinite(curve).all():
        raise NonFiniteError("non-finite calibration curve")
    return curve
