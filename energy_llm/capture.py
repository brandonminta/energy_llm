"""Per-question capture: greedy generation, one teacher-forced pass with
residual hooks, and every per-layer metric.

1. The chat-templated prompt is greedy-decoded (clean config, batch 1).
   Generated token ids, per-token log-probability and predictive entropy
   (float32, from the decoding logits) and a truncation flag are kept.
2. One forward pass over prompt ++ generated tokens. The argmax of its
   logits must reproduce the greedy tokens; any mismatch is recorded.
3. A forward pre-hook on every ``post_attention_layernorm`` reads the
   residual x at its INPUT. x_hat, scores and all metrics are float32.
4. Phases, per layer (layers never mixed):
   - prompt   = last prompt position (the state that emits the first answer token)
   - response = positions whose input token is a generated non-stop token
   E and H_bar are averaged over response positions; JS, Hellinger^2 and
   top-k Jaccard compare the AVERAGED response distribution with the prompt
   one. Variant ``first`` uses the first response position instead of the
   mean. All of it at every beta in beta* x multipliers (scores reused).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from energy_llm import metrics
from energy_llm.bank import Bank
from energy_llm.model import ModelBundle, decoder_layers
from energy_llm.prompting import ROLE_RESPONSE, ROLE_TEMPLATE, Prompt

FEATURES = (
    "energy_prompt", "energy_resp", "delta_energy",
    "norm_entropy_prompt", "norm_entropy_resp", "delta_norm_entropy",
    "js", "hellinger_sq", "jaccard_topk",
    "margin_prompt", "margin_resp",
)
VARIANTS = ("mean", "first")
FP16_MAX = float(np.finfo(np.float16).max)


class EmptyResponseError(RuntimeError):
    """The model produced no non-stop token: response features are undefined."""


class NonFiniteError(RuntimeError):
    pass


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
        return_dict_in_generate=True, output_logits=True,
    )
    gen = out.sequences[0, n:].tolist()
    if not gen:
        raise EmptyResponseError("generation returned no tokens")
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
    if not resp_pos:
        raise EmptyResponseError("the response consists only of stop tokens")

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
    features: np.ndarray        # [L, n_beta, n_variant, n_feature] float32
    betas: list[float]
    energy: np.ndarray          # [L, T] float32 at beta*
    norm_entropy: np.ndarray    # [L, T] float32 at beta*
    margin: np.ndarray          # [L, T] float32
    role_mask: np.ndarray       # [T] int8
    hidden_prompt_last: np.ndarray    # [L_h, D] float16
    hidden_response_last: np.ndarray  # [L_h, D] float16


def _layer_scores(bank: Bank, x: torch.Tensor, l: int, eps: float):
    x_hat = metrics.rms_normalise(x.to(bank.device), eps)
    return x_hat, metrics.cosine_scores(x_hat, bank.rows[l])


def _phase_features(s_sel: torch.Tensor, m_sel: torch.Tensor, beta: float, k: int,
                    variant: str) -> list[torch.Tensor]:
    """s_sel: [1 + n_resp, D_ff] (row 0 = prompt); m_sel: margins of those rows."""
    logp = metrics.log_distribution(s_sel, beta)
    e = metrics.energy(s_sel, beta)
    h = metrics.norm_entropy(logp)
    p = logp.exp()
    if variant == "mean":
        e_r, h_r, m_r = e[1:].mean(), h[1:].mean(), m_sel[1:].mean()
        p_r = p[1:].mean(dim=0)
    else:
        e_r, h_r, m_r, p_r = e[1], h[1], m_sel[1], p[1]
    p_0 = p[0]
    return [
        e[0], e_r, e_r - e[0],
        h[0], h_r, h_r - h[0],
        metrics.js_divergence(p_r, p_0), metrics.hellinger_sq(p_r, p_0),
        metrics.jaccard_topk(p_r, p_0, k),
        m_sel[0], m_r,
    ]


@torch.inference_mode()
def compute_item(bundle: ModelBundle, bank: Bank, trace: ForwardTrace, item_id: str,
                 beta_star: float, multipliers: list[float], top_k: int,
                 hidden_layers: list[int]) -> ItemResult:
    dims = bundle.dims
    betas = [beta_star * m for m in multipliers]
    sel = [trace.n_prompt - 1] + trace.response_positions
    sel_idx = torch.tensor(sel, device=bank.device)
    L, T = dims.n_layers, trace.n_prompt + len(trace.gen_ids)

    feats = torch.empty(L, len(betas), len(VARIANTS), len(FEATURES))
    energy_tr = torch.empty(L, T)
    entropy_tr = torch.empty(L, T)
    margin_tr = torch.empty(L, T)
    max_dev = 0.0
    for l in range(L):
        x_hat, s = _layer_scores(bank, trace.residuals[l], l, dims.rms_norm_eps)
        dev = (x_hat.norm(dim=-1) / math.sqrt(dims.d_model) - 1.0).abs().max().item()
        max_dev = max(max_dev, dev)
        m_all = metrics.margin(s)
        s_sel, m_sel = s[sel_idx], m_all[sel_idx]
        for bi, beta in enumerate(betas):
            for vi, variant in enumerate(VARIANTS):
                feats[l, bi, vi] = torch.stack(
                    _phase_features(s_sel, m_sel, beta, top_k, variant)).cpu()
        energy_tr[l] = metrics.energy(s, beta_star).cpu()
        entropy_tr[l] = metrics.norm_entropy(metrics.log_distribution(s, beta_star)).cpu()
        margin_tr[l] = m_all.cpu()
        del x_hat, s

    hp, hr = [], []
    last_resp = trace.response_positions[-1]
    for l in hidden_layers:
        x = trace.residuals[l].float()
        hp.append(x[trace.n_prompt - 1].cpu())
        hr.append(x[last_resp].cpu())
    hidden_p = torch.stack(hp).numpy()
    hidden_r = torch.stack(hr).numpy()
    if max(np.abs(hidden_p).max(), np.abs(hidden_r).max()) > FP16_MAX:
        raise NonFiniteError("residual exceeds the float16 range")

    out = {
        "features": feats.numpy(), "energy": energy_tr.numpy(),
        "norm_entropy": entropy_tr.numpy(), "margin": margin_tr.numpy(),
    }
    for name, arr in out.items():
        if not np.isfinite(arr).all():
            raise NonFiniteError(f"non-finite values in {name}")

    gen = trace.gen_ids
    stops = set(bundle.stop_token_ids)
    resp_j = [j for j, t in enumerate(gen) if t not in stops]
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
        "truncated": bool(trace.truncated),
        "stop_token_id": trace.stop_token_id,
        "token_logprobs": trace.token_logprobs.tolist(),
        "token_entropies": trace.token_entropies.tolist(),
        "tf_mismatch_count": len(trace.tf_mismatch_positions),
        "tf_mismatch_positions": list(trace.tf_mismatch_positions),
        "xhat_norm_max_rel_dev": float(max_dev),
    }
    lp = trace.token_logprobs[resp_j]
    baselines = {
        "id": item_id,
        "mean_logprob": float(lp.mean()),
        "min_logprob": float(lp.min()),
        "mean_token_entropy": float(trace.token_entropies[resp_j].mean()),
        "n_response_tokens": len(resp_j),
    }
    return ItemResult(
        item_id=item_id, answer=answer, baselines=baselines, betas=betas,
        role_mask=np.asarray(trace.roles, dtype=np.int8),
        hidden_prompt_last=hidden_p.astype(np.float16),
        hidden_response_last=hidden_r.astype(np.float16), **out,
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
